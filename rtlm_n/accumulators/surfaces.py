"""Adaptation surfaces: how raw local experience becomes state.

Stage A  OutputHeadSurface        EXACT-FROZEN
         y ~ W h on frozen pooled features (classification, regression,
         reward, confidence, routing heads all reduce to this).

Stage B  ResidualHeadSurface      EXACT-FROZEN (for the declared objective)
         r = teacher_logits - student_logits; the merged state materializes
         a correction head that moves the small model toward verified
         teacher behavior.

Stage C  CanonicalLowRankSurface  EXACT-FROZEN (within the constrained space)
         W = W0 + U C V^T with U, V globally fixed. Nodes accumulate
         evidence for C only:
             z = V^T h,  e_U = U^T (y - W0 h)
             Hz = sum z z^T,  A = sum e_U z^T,  C* = A (Hz + lam I)^{-1}

Stage D  ProjectedJacobianSurface LINEARIZED-BOUNDED
         f(theta + Pc, x) ~ f(theta, x) + J_P(x) c inside the network.
         Gauss-Newton statistics G = sum J^T J, g = sum J^T r solved in
         c-space; linearization error is measured and reported.

All surfaces emit plain sums, so every stage inherits the exact merge /
remove / reorder algebra of the state object.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import numpy as np

from rtlm_n.bases.basis import CanonicalBasis, adapter_delta
from rtlm_n.core.guarantees import GuaranteeClass
from rtlm_n.core.hashing import array_bytes, digest
from rtlm_n.core.linalg import augment_bias
from rtlm_n.models.backbone import TinyTransformer


def batch_commitment(tokens: np.ndarray, targets: np.ndarray) -> str:
    """Content hash of a raw event batch (kept for audit/deletion; the raw
    data itself never leaves the node)."""
    return digest(array_bytes(np.asarray(tokens, dtype=np.float64)), array_bytes(targets))[:16]


# --------------------------------------------------------------- Stage A/B
def head_stats(h: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray]:
    """H = sum h h^T, R = sum h r^T on bias-augmented frozen features."""
    ha = augment_bias(h)
    return {"H": ha.T @ ha, "R": ha.T @ targets}


# ----------------------------------------------------------------- Stage C
def lowrank_head_stats(
    basis: CanonicalBasis,
    h: np.ndarray,
    targets: np.ndarray,
    W0: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Evidence for C in W = W0 + U C V^T.

    minimize sum || (y - W0 h) - U C V^T h ||^2 over C. With orthonormal U
    the normal equations depend only on
        Hz = sum z z^T (z = V^T h)   and   A = sum (U^T e) z^T.
    """
    m = basis.matrices("head")
    U, V = m["U"], m["V"]
    e = targets - h @ W0.T          # (n, k) residual against the base head
    z = h @ V                        # (n, rv) input coordinates
    eU = e @ U                       # (n, ru) output coordinates
    return {"H": z.T @ z, "R": z.T @ eU}   # C* = ((H+lam)^-1 R)^T -> (ru, rv)


# ----------------------------------------------------------------- Stage D
def projected_jacobian(
    model: TinyTransformer,
    basis: CanonicalBasis,
    surface: str,
    tokens: np.ndarray,
    head_W: np.ndarray,
    eps: float = 1e-5,
    c0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """J_P(x) = d logits / d c at c = c0 for an internal MLP adapter surface.

    Returns (logits at c0 (B,k), J (B,k,r)) with r = ru*rv. c0 defaults to
    the reference model; iterated Gauss-Newton passes the previously
    materialized coordinates to re-linearize there. Computed by exact
    forward differencing along each canonical direction (r+1 forward
    passes) or delegated to the backbone's autograd Jacobian when exposed.
    """
    layer = int(surface.split(":")[1])
    m = basis.matrices(surface)
    if hasattr(model, "jacobian_c"):
        # torch backbones provide exact vectorized autograd Jacobians
        return model.jacobian_c(tokens, m["U"], m["V"], layer, head_W, c0=c0)
    ru, rv = basis.ru, basis.rv
    r = ru * rv
    B = tokens.shape[0]
    k = head_W.shape[0]
    C0 = np.zeros((ru, rv)) if c0 is None else np.asarray(c0).reshape(ru, rv)

    def logits_at(c: np.ndarray) -> np.ndarray:
        C = C0 + c.reshape(ru, rv)
        delta = adapter_delta(basis, surface, C)
        h = model.forward(tokens, mlp_deltas={layer: delta})
        return h @ head_W.T

    base = logits_at(np.zeros(r))
    J = np.empty((B, k, r))
    for j in range(r):
        c = np.zeros(r)
        c[j] = eps
        J[:, :, j] = (logits_at(c) - base) / eps
    return base, J


def jacobian_stats(J: np.ndarray, residuals: np.ndarray) -> Dict[str, np.ndarray]:
    """Gauss-Newton state: G = sum_j J_j^T J_j, g = sum_j J_j^T r_j."""
    B, k, r = J.shape
    G = np.einsum("bkr,bks->rs", J, J)
    g = np.einsum("bkr,bk->r", J, residuals)
    return {"G": G, "g": g.reshape(r, 1)}


def linearization_error(
    model: TinyTransformer,
    basis: CanonicalBasis,
    surface: str,
    tokens: np.ndarray,
    head_W: np.ndarray,
    c_star: np.ndarray,
    c0: Optional[np.ndarray] = None,
) -> float:
    """Measured relative error of the linearization at c0, evaluated at the
    materialized step dc = c_star (relative to c0):

        || f(c0+dc) - (f(c0) + J(c0) dc) || / || f(c0+dc) - f(c0) ||

    with real forward passes. Required reporting for every
    LINEARIZED-BOUNDED result; iterated Gauss-Newton reports it per round.
    """
    layer = int(surface.split(":")[1])
    ru, rv = basis.matrices(surface)["U"].shape[1], basis.matrices(surface)["V"].shape[1]
    C0 = np.zeros((ru, rv)) if c0 is None else np.asarray(c0).reshape(ru, rv)
    C = C0 + c_star.reshape(ru, rv)
    delta = adapter_delta(basis, surface, C)
    h_true = model.forward(tokens, mlp_deltas={layer: delta})
    logits_true = h_true @ head_W.T
    logits0, J = projected_jacobian(model, basis, surface, tokens, head_W, c0=C0.ravel())
    logits_lin = logits0 + np.einsum("bkr,r->bk", J, c_star)
    num = float(np.linalg.norm(logits_true - logits_lin))
    den = float(np.linalg.norm(logits_true - logits0)) + 1e-12
    return num / den


def objective_id_for(kind: str, extra: Optional[Dict] = None) -> str:
    """Stable identifier of the declared objective a state was built for."""
    h = hashlib.sha256(repr((kind, sorted((extra or {}).items()))).encode())
    return f"{kind}-{h.hexdigest()[:8]}"


SURFACE_GUARANTEES = {
    "head": GuaranteeClass.EXACT_FROZEN,
    "residual-head": GuaranteeClass.EXACT_FROZEN,
    "lowrank-head": GuaranteeClass.EXACT_FROZEN,
    "projected-jacobian": GuaranteeClass.LINEARIZED_BOUNDED,
}
