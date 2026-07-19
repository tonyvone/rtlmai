"""Numerical core: ridge materialization, orthonormal bases, diagnostics.

All arrays are float64. Materialization from a sufficient state is a single
regularized linear solve; its cost is independent of the number of raw
examples that built the state.
"""

from __future__ import annotations

import numpy as np

#: Condition-number ceiling above which a state is numerically unsafe and
#: materialization must return UNSUPPORTED.
COND_LIMIT = 1e12


def ridge_solve(H: np.ndarray, R: np.ndarray, lam: float) -> np.ndarray:
    """Solve (H + lam I) W = R for W.

    H is (d, d) PSD (a sum of outer products), R is (d, k).
    Returns W of shape (d, k) such that predictions are h @ W.
    """
    d = H.shape[0]
    A = H + lam * np.eye(d)
    return np.linalg.solve(A, R)


def orthonormal_columns(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    """Random matrix with orthonormal columns (rows >= cols)."""
    if cols > rows:
        raise ValueError(f"cannot build {cols} orthonormal columns in R^{rows}")
    A = rng.standard_normal((rows, cols))
    Q, _ = np.linalg.qr(A)
    return Q[:, :cols]


def condition_number(H: np.ndarray, lam: float) -> float:
    """Condition number of the regularized normal matrix H + lam I."""
    eigs = np.linalg.eigvalsh(H)
    lo, hi = float(eigs[0]) + lam, float(eigs[-1]) + lam
    if lo <= 0:
        return np.inf
    return hi / lo


def is_finite_psd(H: np.ndarray, tol: float = 1e-8) -> bool:
    """Check that H is finite, symmetric, and PSD up to tolerance."""
    if not np.all(np.isfinite(H)):
        return False
    if not np.allclose(H, H.T, atol=1e-9 * (1.0 + np.abs(H).max())):
        return False
    eigs = np.linalg.eigvalsh(H)
    return bool(eigs[0] >= -tol * max(1.0, float(eigs[-1])))


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def augment_bias(h: np.ndarray) -> np.ndarray:
    """Append a constant-1 feature so heads learn a bias term."""
    return np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)
