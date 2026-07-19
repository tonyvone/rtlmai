"""Synthetic distributed world for the foundational experiments.

Construction:
- a SMALL edge backbone and a LARGE teacher backbone (both frozen);
- ground-truth labels come from a fixed linear map over the LARGE
  backbone's representation, so the teacher is well-specified (a stand-in
  for "verified workflow / large verified model") while the small backbone
  must learn corrections it cannot express perfectly;
- non-IID partitions via Dirichlet label skew and per-node token-domain
  bias, so merge-vs-average comparisons are honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.models.backbone import TinyTransformer

N_CLASSES = 5


@dataclass
class World:
    small: TinyTransformer
    large: TinyTransformer
    W_true: np.ndarray  # (d_large+1, k) augmented-row convention
    n_classes: int = N_CLASSES

    def true_logits(self, tokens: np.ndarray) -> np.ndarray:
        return augment_bias(self.large.forward(tokens)) @ self.W_true

    def labels(self, tokens: np.ndarray) -> np.ndarray:
        return self.true_logits(tokens).argmax(axis=1)

    def one_hot(self, labels: np.ndarray) -> np.ndarray:
        return np.eye(self.n_classes)[labels]


def make_world(seed: int = 0, n_classes: int = N_CLASSES) -> World:
    small = TinyTransformer(seed=seed, d_model=32, n_layers=2, d_ff=64, name="edge-small")
    large = TinyTransformer(seed=seed + 1000, d_model=64, n_heads=4, n_layers=4, d_ff=128, name="teacher-large")
    rng = np.random.default_rng(seed + 7)
    W_true = rng.standard_normal((large.d_model + 1, n_classes)) * 2.0
    W_true[-1, :] = 0.0
    return World(small=small, large=large, W_true=W_true, n_classes=n_classes)


def gen_tokens(
    rng: np.random.Generator,
    n: int,
    world: World,
    domain: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Token sequences; `domain` restricts the vocabulary range to create
    per-node covariate shift (non-IID input distributions)."""
    lo, hi = domain if domain is not None else (0, world.small.vocab)
    return rng.integers(lo, hi, size=(n, world.small.seq_len))


def partition_iid(n: int, n_nodes: int, rng: np.random.Generator) -> List[np.ndarray]:
    idx = rng.permutation(n)
    return [np.sort(part) for part in np.array_split(idx, n_nodes)]


def partition_dirichlet(
    labels: np.ndarray, n_nodes: int, alpha: float, rng: np.random.Generator
) -> List[np.ndarray]:
    """Label-skewed partition: each node's class mix ~ Dirichlet(alpha).
    Small alpha -> severe non-IID."""
    n_classes = int(labels.max()) + 1
    node_idx: List[List[int]] = [[] for _ in range(n_nodes)]
    for c in range(n_classes):
        cls = np.flatnonzero(labels == c)
        rng.shuffle(cls)
        props = rng.dirichlet([alpha] * n_nodes)
        cuts = (np.cumsum(props) * len(cls)).astype(int)[:-1]
        for node, chunk in enumerate(np.split(cls, cuts)):
            node_idx[node].extend(chunk.tolist())
    return [np.sort(np.array(ix, dtype=int)) for ix in node_idx]


def fit_small_head(
    world: World, tokens: np.ndarray, targets: np.ndarray, lam: float = 1e-3
) -> np.ndarray:
    """Ridge-fit head over the small backbone (augmented-row convention)."""
    h = augment_bias(world.small.forward(tokens))
    return ridge_solve(h.T @ h, h.T @ targets, lam)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())
