"""Torch transformer backbones for RTLM-N.

`TorchTransformer` exposes the exact duck-typed interface the NumPy
machinery consumes (`forward(tokens, mlp_deltas)` -> pooled features,
`mlp_inputs`, `flops_per_example`, `base_model_id`, dims), so EdgeNode,
CanonicalBasis, MergeEngine, and ModelFactory run on it unchanged. Two
upgrades over the toy backbone:

- projected Jacobians come from torch autograd (exact, one backward per
  canonical direction) instead of finite differences;
- `mlm_pretrain` learns real language representations from a real corpus
  with masked-language-modeling, after which the backbone is FROZEN for a
  representation epoch.

`from_hf_checkpoint` loads any HuggingFace encoder (e.g. BERT/MiniLM)
behind the same interface when the environment can reach the hub; the
mathematics does not change, only the extractor.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from rtlm_n.real.agnews import MASK, PAD

torch.set_num_threads(max(1, torch.get_num_threads()))


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, d_ff)
        self.w2 = nn.Linear(d_ff, d)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        w1_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        W1 = self.w1.weight if w1_delta is None else self.w1.weight + w1_delta
        h = torch.nn.functional.gelu(h @ W1.T + self.w1.bias)
        return x + self.w2(h)


class TorchTransformer(nn.Module):
    def __init__(
        self,
        vocab: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        seq_len: int = 24,
        seed: int = 0,
        name: str = "torch-mini",
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.vocab, self.d_model, self.n_heads = vocab, d_model, n_heads
        self.n_layers, self.d_ff, self.seq_len = n_layers, d_ff, seq_len
        self.name = name
        self.embed = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([_Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self._hash: Optional[str] = None

    # ----------------------------------------------------------- identity
    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"{self.name}:{self.vocab}:{self.d_model}:{self.n_layers}:{self.d_ff}".encode())
        for k, v in sorted(self.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().astype(np.float32).tobytes())
        return h.hexdigest()[:16]

    def freeze(self) -> "TorchTransformer":
        """End of the representation epoch: freeze and pin base_model_id."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self._hash = self._compute_hash()
        return self

    @property
    def base_model_id(self) -> str:
        if self._hash is None:
            raise RuntimeError("backbone must be frozen (freeze()) before use as a base model")
        return f"{self.name}-{self._hash}"

    # ------------------------------------------------------------ forward
    def _encode(
        self, tokens: torch.Tensor, deltas: Optional[Dict[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        pad_mask = tokens == PAD
        x = self.embed(tokens) + self.pos[None, : tokens.shape[1], :]
        for i, blk in enumerate(self.blocks):
            x = blk(x, pad_mask, w1_delta=None if deltas is None or i not in deltas else deltas[i])
        x = self.ln_f(x)
        keep = (~pad_mask).float().unsqueeze(-1)
        return (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)

    def forward(
        self, tokens: np.ndarray, mlp_deltas: Optional[Dict[int, np.ndarray]] = None
    ) -> np.ndarray:
        """NumPy-in / NumPy-out pooled features (RTLM-N interface)."""
        t = torch.as_tensor(np.asarray(tokens), dtype=torch.long)
        deltas = None
        if mlp_deltas:
            deltas = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in mlp_deltas.items()}
        with torch.no_grad():
            return self._encode(t, deltas).double().numpy()

    def mlp_inputs(self, tokens: np.ndarray, layer: int) -> np.ndarray:
        """Normalized MLP inputs of `layer`, flattened over non-pad tokens
        (for activation-SVD canonical bases)."""
        t = torch.as_tensor(np.asarray(tokens), dtype=torch.long)
        pad_mask = t == PAD
        with torch.no_grad():
            x = self.embed(t) + self.pos[None, : t.shape[1], :]
            for i, blk in enumerate(self.blocks):
                if i == layer:
                    h = blk.ln2(x + blk.attn(blk.ln1(x), blk.ln1(x), blk.ln1(x),
                                             key_padding_mask=pad_mask, need_weights=False)[0])
                    return h[~pad_mask].double().numpy()
                x = blk(x, pad_mask)
        raise ValueError(f"layer {layer} out of range")

    # ----------------------------------------------------------- jacobian
    def jacobian_c(
        self, tokens: np.ndarray, U: np.ndarray, V: np.ndarray, layer: int, head_W: np.ndarray
    ) -> tuple:
        """Exact d logits / d c at c = 0 via autograd, in the canonical
        basis Delta W1 = U C V^T. Returns (logits0 (B,k), J (B,k,r))."""
        t = torch.as_tensor(np.asarray(tokens), dtype=torch.long)
        Ut = torch.as_tensor(U, dtype=torch.float32)
        Vt = torch.as_tensor(V, dtype=torch.float32)
        Wh = torch.as_tensor(head_W, dtype=torch.float32)
        ru, rv = Ut.shape[1], Vt.shape[1]
        c = torch.zeros(ru, rv, requires_grad=True)

        def logits_fn(cmat: torch.Tensor) -> torch.Tensor:
            delta = Ut @ cmat @ Vt.T
            return self._encode(t, {layer: delta}) @ Wh.T

        with torch.no_grad():
            logits0 = logits_fn(c.detach())
        # vectorized exact Jacobian: (B, k, ru, rv) in one reverse sweep set
        J4 = torch.func.jacrev(logits_fn)(c.detach())
        B, k = logits0.shape
        J = J4.reshape(B, k, ru * rv)
        return logits0.double().numpy(), J.double().numpy()

    # ------------------------------------------------------------- costing
    @property
    def flops_per_example(self) -> float:
        T, d, f = self.seq_len, self.d_model, self.d_ff
        per_layer = 4 * 2 * T * d * d + 2 * 2 * T * T * d + 2 * T * d * f + 2 * T * f * d
        return float(self.n_layers * per_layer + 2 * T * d)


def mlm_pretrain(
    model: TorchTransformer,
    tokens: np.ndarray,
    steps: int = 600,
    batch: int = 96,
    lr: float = 3e-3,
    mask_prob: float = 0.15,
    seed: int = 0,
    verbose: bool = False,
) -> TorchTransformer:
    """Masked-language-model pretraining on a real corpus, then freeze.

    A tied-embedding output head predicts masked tokens; the head is
    discarded afterwards - only the representations matter.
    """
    rng = np.random.default_rng(seed)
    head = nn.Linear(model.d_model, model.vocab, bias=False)
    head.weight = model.embed.weight  # weight tying
    opt = torch.optim.Adam(list(model.parameters()), lr=lr)
    model.train()
    n = len(tokens)
    for step in range(steps):
        idx = rng.integers(0, n, size=batch)
        x = tokens[idx].copy()
        target = torch.tensor(x, dtype=torch.long)  # snapshot BEFORE masking
        maskable = x != PAD
        mask = (rng.random(x.shape) < mask_prob) & maskable
        if not mask.any():
            continue
        x[mask] = MASK
        t = torch.tensor(x, dtype=torch.long)
        pad_mask = t == PAD
        h = model.embed(t) + model.pos[None, : t.shape[1], :]
        for blk in model.blocks:
            h = blk(h, pad_mask)
        h = model.ln_f(h)
        logits = head(h[torch.as_tensor(mask)])
        loss = nn.functional.cross_entropy(logits, target[torch.as_tensor(mask)])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (step + 1) % 100 == 0:
            print(f"  mlm step {step + 1}/{steps} loss={loss.item():.3f}")
    return model.freeze()


def pretrained_backbone(
    vocab: int,
    tokens: np.ndarray,
    cache_key: str,
    steps: int = 4000,
    batch: int = 128,
    lr: float = 2e-3,
    seed: int = 0,
    verbose: bool = False,
    **cfg,
) -> "TorchTransformer":
    """MLM-pretrain (or load from cache) a frozen backbone.

    Pretraining is deterministic given (seed, corpus, config), so the
    cached weights are reproducible; delete ~/.cache/rtlmn/*.pt to force
    a re-run.
    """
    import os

    from rtlm_n.real.agnews import CACHE

    model = TorchTransformer(vocab=vocab, seed=seed, **cfg)
    path = os.path.join(CACHE, f"{cache_key}.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.freeze()
    mlm_pretrain(model, tokens, steps=steps, batch=batch, lr=lr, seed=seed, verbose=verbose)
    os.makedirs(CACHE, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def from_hf_checkpoint(name: str, seq_len: int = 32) -> "TorchTransformer":
    """Load a pretrained HuggingFace encoder behind the RTLM-N interface.

    Requires network access to huggingface.co and `transformers`. The
    returned wrapper exposes forward/mlp_inputs/flops_per_example and a
    weight-hash base_model_id, so all RTLM-N machinery applies unchanged.
    """
    from rtlm_n.real.hf_backbone import HFBackbone  # deferred: optional dep

    return HFBackbone(name, seq_len=seq_len)
