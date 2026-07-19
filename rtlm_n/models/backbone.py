"""Frozen foundation backbones.

A deterministic pure-NumPy transformer encoder plays the role of the frozen
foundation model f_theta. It is intentionally small so that every RTLM-N
claim is verifiable end-to-end on CPU, but it is a real multi-head
attention + MLP + LayerNorm stack: the adaptation surfaces (output heads,
canonical low-rank adapters inside MLP layers, projected Jacobians) exercise
exactly the structures a production transformer exposes.

The backbone is frozen for a "representation epoch"; its weight hash is the
`base_model_id` that gates state compatibility.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

import numpy as np

from rtlm_n.core.linalg import softmax


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


class TinyTransformer:
    """A frozen transformer encoder with mean pooling.

    forward(tokens) -> h in R^{B x d_model}

    Adapter injection: `mlp_deltas` maps layer index -> Delta W1 of shape
    (d_ff, d_model), added to that layer's first MLP weight. This is the
    surface used by canonical low-rank adapters (Delta W = U C V^T) and
    projected Jacobian adaptation.
    """

    def __init__(
        self,
        seed: int,
        vocab: int = 64,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 64,
        seq_len: int = 16,
        name: str = "tiny",
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab, self.d_model, self.n_heads = vocab, d_model, n_heads
        self.n_layers, self.d_ff, self.seq_len = n_layers, d_ff, seq_len
        self.name = name
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.embed = rng.standard_normal((vocab, d_model)) * s
        self.pos = rng.standard_normal((seq_len, d_model)) * s
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(
                {
                    "Wq": rng.standard_normal((d_model, d_model)) * s,
                    "Wk": rng.standard_normal((d_model, d_model)) * s,
                    "Wv": rng.standard_normal((d_model, d_model)) * s,
                    "Wo": rng.standard_normal((d_model, d_model)) * s,
                    "W1": rng.standard_normal((d_ff, d_model)) * s,
                    "b1": np.zeros(d_ff),
                    "W2": rng.standard_normal((d_model, d_ff)) * (1.0 / np.sqrt(d_ff)),
                    "b2": np.zeros(d_model),
                }
            )
        self.model_hash = self._compute_hash()

    # ------------------------------------------------------------------ id
    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(
            f"{self.name}:{self.vocab}:{self.d_model}:{self.n_heads}:"
            f"{self.n_layers}:{self.d_ff}:{self.seq_len}".encode()
        )
        h.update(np.ascontiguousarray(self.embed).tobytes())
        h.update(np.ascontiguousarray(self.pos).tobytes())
        for layer in self.layers:
            for k in sorted(layer):
                h.update(np.ascontiguousarray(layer[k]).tobytes())
        return h.hexdigest()[:16]

    @property
    def base_model_id(self) -> str:
        return f"{self.name}-{self.model_hash}"

    # ------------------------------------------------------------- forward
    def forward(
        self,
        tokens: np.ndarray,
        mlp_deltas: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """tokens: (B, T) int array. Returns pooled hidden states (B, d_model)."""
        tokens = np.asarray(tokens)
        B, T = tokens.shape
        if T != self.seq_len:
            raise ValueError(f"expected sequence length {self.seq_len}, got {T}")
        x = self.embed[tokens] + self.pos[None, :T, :]
        dh = self.d_model // self.n_heads
        for li, layer in enumerate(self.layers):
            xn = _layer_norm(x)
            q = xn @ layer["Wq"].T
            k = xn @ layer["Wk"].T
            v = xn @ layer["Wv"].T
            q = q.reshape(B, T, self.n_heads, dh).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, self.n_heads, dh).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, self.n_heads, dh).transpose(0, 2, 1, 3)
            att = softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh), axis=-1)
            ctx = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
            x = x + ctx @ layer["Wo"].T
            xn = _layer_norm(x)
            W1 = layer["W1"]
            if mlp_deltas is not None and li in mlp_deltas:
                W1 = W1 + mlp_deltas[li]
            hpre = xn @ W1.T + layer["b1"]
            hact = np.tanh(hpre)
            x = x + hact @ layer["W2"].T + layer["b2"]
        return _layer_norm(x).mean(axis=1)

    def mlp_inputs(self, tokens: np.ndarray, layer: int) -> np.ndarray:
        """Per-token normalized inputs to the MLP of `layer`, flattened to
        (B*T, d_model). Used to build activation-SVD canonical bases from a
        public calibration set."""
        tokens = np.asarray(tokens)
        B, T = tokens.shape
        x = self.embed[tokens] + self.pos[None, :T, :]
        dh = self.d_model // self.n_heads
        for li, lyr in enumerate(self.layers):
            xn = _layer_norm(x)
            q = (xn @ lyr["Wq"].T).reshape(B, T, self.n_heads, dh).transpose(0, 2, 1, 3)
            k = (xn @ lyr["Wk"].T).reshape(B, T, self.n_heads, dh).transpose(0, 2, 1, 3)
            v = (xn @ lyr["Wv"].T).reshape(B, T, self.n_heads, dh).transpose(0, 2, 1, 3)
            att = softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh), axis=-1)
            ctx = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
            x = x + ctx @ lyr["Wo"].T
            xn = _layer_norm(x)
            if li == layer:
                return xn.reshape(B * T, self.d_model)
            hact = np.tanh(xn @ lyr["W1"].T + lyr["b1"])
            x = x + hact @ lyr["W2"].T + lyr["b2"]
        raise ValueError(f"layer {layer} out of range")

    # ------------------------------------------------------------- costing
    @property
    def flops_per_example(self) -> float:
        """Analytic forward-pass FLOPs for one sequence (2*m*n*k per matmul)."""
        T, d, f = self.seq_len, self.d_model, self.d_ff
        per_layer = (
            4 * 2 * T * d * d  # Wq, Wk, Wv, Wo projections
            + 2 * 2 * T * T * d  # QK^T and att@V
            + 2 * T * d * f  # MLP up
            + 2 * T * f * d  # MLP down
        )
        return float(self.n_layers * per_layer + 2 * T * d)  # + embedding lookup/pool


class BaseModelRegistry:
    """Registry of frozen backbones keyed by base_model_id.

    A representation epoch pins one backbone; states created under different
    base_model_ids (or different representation versions) must never merge.
    """

    def __init__(self) -> None:
        self._models: Dict[str, TinyTransformer] = {}

    def register(self, model: TinyTransformer) -> str:
        self._models[model.base_model_id] = model
        return model.base_model_id

    def get(self, base_model_id: str) -> TinyTransformer:
        return self._models[base_model_id]

    def __contains__(self, base_model_id: str) -> bool:
        return base_model_id in self._models
