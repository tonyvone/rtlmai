"""Real SGD-trained baselines (torch) for the real-data benchmarks.

These are the honest competitors, actually optimized with Adam on real
features - not closed-form stand-ins:

- `train_lora_head`: a true LoRA-style adapter W = W0 + B A with the
  node's OWN learned A, B (random init per node -> misaligned subspaces
  by construction, exactly like independent federated LoRA);
- `train_linear_head`: full-head SGD training (FedAvg's local step);
- averaging helpers for the federated variants.

All training happens on frozen backbone features, mirroring how these
baselines are typically deployed for cheap specialization.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _to_t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def train_lora_head(
    h: np.ndarray,
    targets: np.ndarray,
    W0: np.ndarray,
    rank: int = 4,
    steps: int = 300,
    lr: float = 5e-2,
    seed: int = 0,
    weight_decay: float = 1e-4,
) -> np.ndarray:
    """True LoRA on the head: W = W0 + B A, A/B trained with Adam from a
    per-node random init. Returns Delta W = B A (k, d)."""
    torch.manual_seed(seed)
    k, d = W0.shape
    ht, yt, W0t = _to_t(h), _to_t(targets), _to_t(W0)
    A = nn.Parameter(torch.randn(rank, d) * 0.05)
    B = nn.Parameter(torch.zeros(k, rank))
    opt = torch.optim.Adam([A, B], lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        pred = ht @ (W0t + B @ A).T
        loss = nn.functional.mse_loss(pred, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (B @ A).detach().double().numpy()


def train_linear_head(
    h: np.ndarray,
    targets: np.ndarray,
    steps: int = 300,
    lr: float = 5e-2,
    seed: int = 0,
    init_W: Optional[np.ndarray] = None,
    weight_decay: float = 1e-4,
) -> np.ndarray:
    """Full linear head trained with Adam on frozen features. Returns W
    (k, d+1) in augmented-column convention (bias last)."""
    torch.manual_seed(seed)
    n, d = h.shape
    k = targets.shape[1]
    ht, yt = _to_t(h), _to_t(targets)
    lin = nn.Linear(d, k)
    if init_W is not None:
        with torch.no_grad():
            lin.weight.copy_(_to_t(init_W[:, :d]))
            lin.bias.copy_(_to_t(init_W[:, d]))
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        loss = nn.functional.mse_loss(lin(ht), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    W = torch.cat([lin.weight, lin.bias[:, None]], dim=1)
    return W.detach().double().numpy()


def fedavg_heads(
    parts: List[np.ndarray],
    h: np.ndarray,
    targets: np.ndarray,
    steps: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """FedAvg one-shot: each node trains a full head locally with SGD,
    server averages the weights."""
    Ws = []
    for i, idx in enumerate(parts):
        if len(idx) < 8:
            continue
        Ws.append(train_linear_head(h[idx], targets[idx], steps=steps, seed=seed + i))
    return np.mean(Ws, axis=0)


def federated_lora_avg(
    parts: List[np.ndarray],
    h: np.ndarray,
    targets: np.ndarray,
    W0: np.ndarray,
    rank: int = 4,
    steps: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """Independent federated LoRA: per-node adapters in per-node learned
    subspaces, averaged. The misalignment failure mode RTLM-N replaces."""
    deltas = []
    for i, idx in enumerate(parts):
        if len(idx) < 8:
            continue
        deltas.append(
            train_lora_head(h[idx], targets[idx], W0, rank=rank, steps=steps, seed=seed + i)
        )
    return W0 + np.mean(deltas, axis=0)


def distill_head(
    h: np.ndarray,
    teacher_logits: np.ndarray,
    steps: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """Standard distillation baseline: SGD head fit to teacher logits."""
    return train_linear_head(h, teacher_logits, steps=steps, seed=seed)


def accuracy_W(W_aug_cols: np.ndarray, h: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy for a (k, d+1) augmented-column head on raw features."""
    logits = h @ W_aug_cols[:, :-1].T + W_aug_cols[:, -1]
    return float((logits.argmax(axis=1) == labels).mean())
