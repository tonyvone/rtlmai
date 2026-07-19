"""Canonical adaptation bases: the shared neural coordinate system.

This is the load-bearing idea of RTLM-N. Instead of every node optimizing
its own low-rank adapter (whose subspace is an accident of its local
trajectory, making independently trained adapters geometrically misaligned
and non-mergeable), all nodes receive the SAME globally fixed basis and
only accumulate evidence for coordinates inside it:

    Delta W_l = U_l C_l V_l^T      with U_l, V_l fixed and shared
    Delta theta = P c              in the general projected form

Because U/V/P are identical everywhere, per-node statistics live in one
coordinate system and sum exactly. The basis carries a construction
commitment (hash of model id, method, seed, ranks, layer map, version);
states referencing different basis_ids are refused at merge time.

Implemented constructions:
- random-orthogonal: seeded orthonormal directions (no data needed);
- activation-svd: V from the SVD of layer inputs on a PUBLIC calibration
  set (data-aware input directions), U random orthonormal;
- teacher-residual: U from the SVD of teacher-student residuals on the
  calibration set (output directions that actually need correcting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from rtlm_n.core.hashing import canonical_json, digest, array_bytes
from rtlm_n.core.linalg import orthonormal_columns
from rtlm_n.models.backbone import TinyTransformer

BASIS_VERSION = 1


@dataclass(frozen=True)
class CanonicalBasis:
    """A globally shared adaptation basis for one backbone.

    surfaces maps surface names to their fixed matrices:
      - "head": {"U": (k, ru), "V": (d, rv)}         for low-rank head updates
      - "mlp:<layer>": {"U": (d_ff, ru), "V": (d, rv)} for internal adapters
    """

    base_model_id: str
    method: str
    seed: int
    ranks: Tuple[int, int]
    layer_map: Tuple[str, ...]
    version: int
    surfaces: Dict[str, Dict[str, np.ndarray]] = field(repr=False)
    basis_id: str = ""

    @property
    def ru(self) -> int:
        return self.ranks[0]

    @property
    def rv(self) -> int:
        return self.ranks[1]

    def matrices(self, surface: str) -> Dict[str, np.ndarray]:
        return self.surfaces[surface]

    def c_dim(self) -> int:
        return self.ru * self.rv


def _basis_id(
    base_model_id: str,
    method: str,
    seed: int,
    ranks: Tuple[int, int],
    layer_map: Tuple[str, ...],
    version: int,
    surfaces: Dict[str, Dict[str, np.ndarray]],
) -> str:
    """Construction commitment: hash of provenance AND the actual matrices."""
    meta = canonical_json(
        {
            "base_model_id": base_model_id,
            "method": method,
            "seed": seed,
            "ranks": list(ranks),
            "layer_map": list(layer_map),
            "version": version,
        }
    )
    parts = [meta]
    for sname in sorted(surfaces):
        for mname in sorted(surfaces[sname]):
            parts.append(f"{sname}/{mname}".encode())
            parts.append(array_bytes(surfaces[sname][mname]))
    return digest(*parts)[:16]


def build_basis(
    model: TinyTransformer,
    method: str = "random-orthogonal",
    seed: int = 0,
    ru: int = 4,
    rv: int = 4,
    n_classes: int = 5,
    adapter_layers: Tuple[int, ...] = (1,),
    calibration_tokens: Optional[np.ndarray] = None,
    calibration_residuals: Optional[np.ndarray] = None,
) -> CanonicalBasis:
    """Construct a canonical basis for `model`.

    calibration_tokens: public (non-sensitive) token batch used only for
    data-aware constructions; it never contains node-local data.
    calibration_residuals: (n, k) teacher-student residuals on the
    calibration set, used by the teacher-residual method for the head U.
    """
    rng = np.random.default_rng(seed)
    d, k, dff = model.d_model, n_classes, model.d_ff
    surfaces: Dict[str, Dict[str, np.ndarray]] = {}

    # ---- head surface: Delta W_head = U C V^T, W_head: (k, d) ----------
    if method == "teacher-residual":
        if calibration_residuals is None:
            raise ValueError("teacher-residual basis requires calibration_residuals")
        ru_eff = min(ru, k)
        _, _, Vt = np.linalg.svd(calibration_residuals, full_matrices=False)
        U_head = Vt[:ru_eff].T  # principal residual output directions
    else:
        U_head = orthonormal_columns(rng, k, min(ru, k))
    if method in ("activation-svd", "teacher-residual"):
        if calibration_tokens is None:
            raise ValueError(f"{method} basis requires calibration_tokens")
        H = model.forward(calibration_tokens)
        _, _, Vt = np.linalg.svd(H - H.mean(axis=0), full_matrices=False)
        V_head = Vt[:rv].T
    else:
        V_head = orthonormal_columns(rng, d, rv)
    surfaces["head"] = {"U": U_head, "V": V_head}

    # ---- internal MLP adapter surfaces ---------------------------------
    layer_names = []
    for li in adapter_layers:
        U_l = orthonormal_columns(rng, dff, ru)
        if method in ("activation-svd", "teacher-residual") and calibration_tokens is not None:
            X = model.mlp_inputs(calibration_tokens, li)
            _, _, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
            V_l = Vt[:rv].T
        else:
            V_l = orthonormal_columns(rng, d, rv)
        surfaces[f"mlp:{li}"] = {"U": U_l, "V": V_l}
        layer_names.append(f"mlp:{li}")

    layer_map = tuple(["head"] + layer_names)
    bid = _basis_id(model.base_model_id, method, seed, (ru, rv), layer_map, BASIS_VERSION, surfaces)
    return CanonicalBasis(
        base_model_id=model.base_model_id,
        method=method,
        seed=seed,
        ranks=(ru, rv),
        layer_map=layer_map,
        version=BASIS_VERSION,
        surfaces=surfaces,
        basis_id=bid,
    )


def adapter_delta(basis: CanonicalBasis, surface: str, C: np.ndarray) -> np.ndarray:
    """Materialize Delta W = U C V^T for a surface from coordinates C."""
    m = basis.matrices(surface)
    return m["U"] @ C @ m["V"].T
