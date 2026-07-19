"""HuggingFace pretrained encoders behind the RTLM-N backbone interface.

Usable wherever huggingface.co is reachable:

    from rtlm_n.real.hf_backbone import HFBackbone
    edge = HFBackbone("sentence-transformers/all-MiniLM-L6-v2")
    node = EdgeNode("n0", edge, basis, keys, b"secret")

Mean-pooled last-hidden-state features feed the exact same state algebra;
`base_model_id` hashes the checkpoint weights so representation epochs
stay pinned. Internal adapter surfaces target the intermediate (MLP up)
projection of a chosen encoder layer, mirroring the toy backbone.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

import numpy as np
import torch


class HFBackbone:
    def __init__(self, name: str, seq_len: int = 32) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.name = name
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.d_model = cfg.hidden_size
        self.d_ff = cfg.intermediate_size
        self.n_layers = cfg.num_hidden_layers
        h = hashlib.sha256()
        for k, v in sorted(self.model.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().astype(np.float32).tobytes())
        self.base_model_id = f"hf:{name}-{h.hexdigest()[:16]}"

    # texts (list[str]) or pre-tokenized input_ids both accepted
    def _batch(self, inputs) -> Dict[str, torch.Tensor]:
        if isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], str):
            return self.tokenizer(
                list(inputs), padding=True, truncation=True,
                max_length=self.seq_len, return_tensors="pt",
            )
        ids = torch.as_tensor(np.asarray(inputs), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": (ids != self.tokenizer.pad_token_id).long()}

    def forward(self, inputs, mlp_deltas: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
        enc = self._batch(inputs)
        handles = []
        if mlp_deltas:
            for layer_idx, delta in mlp_deltas.items():
                lin = self.model.encoder.layer[layer_idx].intermediate.dense
                d = torch.as_tensor(delta, dtype=lin.weight.dtype)

                def hook(mod, inp, out, d=d):
                    return out + inp[0] @ d.T

                handles.append(lin.register_forward_hook(hook))
        try:
            with torch.no_grad():
                out = self.model(**enc).last_hidden_state
        finally:
            for h in handles:
                h.remove()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled.double().numpy()

    def mlp_inputs(self, inputs, layer: int) -> np.ndarray:
        enc = self._batch(inputs)
        captured = {}
        lin = self.model.encoder.layer[layer].intermediate.dense

        def hook(mod, inp, out):
            captured["x"] = inp[0]

        h = lin.register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(**enc)
        finally:
            h.remove()
        x = captured["x"]
        keep = enc["attention_mask"].bool()
        return x[keep].double().numpy()

    @property
    def flops_per_example(self) -> float:
        T, d, f = self.seq_len, self.d_model, self.d_ff
        per_layer = 4 * 2 * T * d * d + 2 * 2 * T * T * d + 2 * T * d * f + 2 * T * f * d
        return float(self.n_layers * per_layer + 2 * T * d)
