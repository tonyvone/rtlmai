"""Neural materializer and multi-model factory.

The merged state is the durable asset; models are materialized views. A
materialization is a regularized linear solve in the canonical coordinate
system - its cost depends on state dimensions, never on how many raw
examples built the state, and it requires NO new representation passes.
That asymmetry is what lets one state produce a thousand models for less
than the cost of a thousand fine-tuning runs.

Variants supported from one state:
- any regularization value (per-tenant, per-policy, per-noise-regime);
- reduced adaptation rank (power/latency budgets) by exact sub-solves in
  leading canonical coordinates;
- quantized parameters (edge deployment);
- any subset of contributors (tenant/domain/geography slices, deletions)
  via the state algebra before materialization.

Numerically unsafe states (non-finite, non-PSD, ill-conditioned beyond
COND_LIMIT) are refused with an UNSUPPORTED error - never silently solved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from rtlm_n.accumulators.state import NeuralIntelligenceState
from rtlm_n.bases.basis import CanonicalBasis, adapter_delta
from rtlm_n.core.guarantees import GuaranteeClass, weaker
from rtlm_n.core.linalg import COND_LIMIT, augment_bias, condition_number, ridge_solve
from rtlm_n.metering.meter import CENTRAL, DeviceProfile, EnergyMeter


class UnsupportedMaterializationError(RuntimeError):
    """Raised when a state may not be materialized. Carries the label."""

    guarantee = GuaranteeClass.UNSUPPORTED


@dataclass
class MaterializedModel:
    kind: str
    guarantee: GuaranteeClass
    lam: float
    params: Dict[str, np.ndarray] = field(repr=False)
    meta: Dict[str, object] = field(default_factory=dict)

    def predict(self, h: np.ndarray) -> np.ndarray:
        """Head-style prediction from frozen pooled features."""
        if "W" not in self.params:
            raise ValueError(f"{self.kind} model does not predict from features directly")
        return augment_bias(h) @ self.params["W"]

    def mlp_deltas(self) -> Dict[int, np.ndarray]:
        """Adapter weights to inject into the backbone forward pass."""
        if "delta" not in self.params:
            return {}
        return {int(self.meta["layer"]): self.params["delta"]}

    def nbytes(self) -> int:
        return int(sum(p.nbytes for p in self.params.values()))

    def quantized(self, bits: int = 8) -> "MaterializedModel":
        """Uniform per-tensor quantization for low-power deployment."""
        qparams = {}
        for k, p in self.params.items():
            scale = float(np.abs(p).max()) / (2 ** (bits - 1) - 1) or 1.0
            q = np.round(p / scale).astype(np.float64) * scale
            qparams[k] = q
        return MaterializedModel(
            kind=self.kind,
            guarantee=weaker(self.guarantee, GuaranteeClass.EMPIRICAL_NEURAL),
            lam=self.lam,
            params=qparams,
            meta={**self.meta, "quantized_bits": bits},
        )


class ModelFactory:
    def __init__(
        self,
        basis: Optional[CanonicalBasis] = None,
        device: DeviceProfile = CENTRAL,
        meter: Optional[EnergyMeter] = None,
    ) -> None:
        self.basis = basis
        self.device = device
        self.meter = meter if meter is not None else EnergyMeter()

    # ------------------------------------------------------------ guards
    def _guard(self, state: NeuralIntelligenceState, surface: str, lam: float) -> None:
        if state.guarantee == GuaranteeClass.UNSUPPORTED:
            raise UnsupportedMaterializationError("state is labeled UNSUPPORTED")
        if not state.verify():
            raise UnsupportedMaterializationError(
                "state failed numerical safety verification: UNSUPPORTED"
            )
        if surface not in state.accumulators:
            raise UnsupportedMaterializationError(
                f"state has no accumulator for surface {surface!r}"
            )
        gram_key = "H" if "H" in state.accumulators[surface] else "G"
        cond = condition_number(state.accumulators[surface][gram_key], lam)
        if cond > COND_LIMIT:
            raise UnsupportedMaterializationError(
                f"normal matrix condition {cond:.3e} exceeds limit {COND_LIMIT:.1e}: UNSUPPORTED"
            )
        if self.basis is not None and state.basis_id and self.basis.basis_id != state.basis_id:
            if surface.startswith(("lowrank", "jacobian")):
                raise UnsupportedMaterializationError(
                    "factory basis does not match state basis: UNSUPPORTED"
                )

    def _charge_solve(self, d: int, k: int) -> None:
        self.meter.charge_flops("materialization", (2.0 / 3.0) * d**3 + 2.0 * d * d * k, self.device)

    # ------------------------------------------------------- materializers
    def materialize_head(
        self,
        state: NeuralIntelligenceState,
        lam: float = 1e-3,
        surface: str = "head",
        variant: str = "default",
    ) -> MaterializedModel:
        """W* = (H + lam I)^{-1} R on bias-augmented frozen features."""
        self._guard(state, surface, lam)
        H, R = state.accumulators[surface]["H"], state.accumulators[surface]["R"]
        W = ridge_solve(H, R, lam)
        self._charge_solve(H.shape[0], R.shape[1])
        return MaterializedModel(
            kind=surface,
            guarantee=state.guarantee,
            lam=lam,
            params={"W": W},
            meta={
                "variant": variant,
                "source_state": state.state_hash,
                "sample_count": state.sample_count,
                "tenant_id": state.tenant_id,
            },
        )

    def materialize_lowrank_head(
        self,
        state: NeuralIntelligenceState,
        W0: np.ndarray,
        lam: float = 1e-3,
        rank: Optional[int] = None,
        variant: str = "default",
    ) -> MaterializedModel:
        """Solve C in W = W0 + U C V^T from merged z-space statistics.

        `rank` truncates to the leading canonical input coordinates - an
        exact solve of the lower-rank constrained problem, for power- or
        latency-budgeted variants.
        """
        self._guard(state, "lowrank-head", lam)
        if self.basis is None:
            raise UnsupportedMaterializationError("low-rank materialization requires the basis")
        Hz = state.accumulators["lowrank-head"]["H"]
        Rz = state.accumulators["lowrank-head"]["R"]
        m = self.basis.matrices("head")
        ru_eff, rv_eff = m["U"].shape[1], m["V"].shape[1]
        rv = rv_eff if rank is None else min(rank, rv_eff)
        C_t = ridge_solve(Hz[:rv, :rv], Rz[:rv, :], lam)  # (rv', ru_eff)
        C = np.zeros((ru_eff, rv_eff))
        C[:, :rv] = C_t.T
        self._charge_solve(rv, Rz.shape[1])
        W_full = W0 + m["U"] @ C @ m["V"].T  # (k, d)
        # store in augmented-row convention used by predict(): (d+1, k)
        W_aug = np.zeros((W0.shape[1] + 1, W0.shape[0]))
        W_aug[:-1, :] = W_full.T
        return MaterializedModel(
            kind="lowrank-head",
            guarantee=state.guarantee,
            lam=lam,
            params={"W": W_aug, "C": C},
            meta={
                "variant": variant,
                "rank": rv,
                "source_state": state.state_hash,
                "sample_count": state.sample_count,
            },
        )

    def materialize_adapter(
        self,
        state: NeuralIntelligenceState,
        surface: str,
        lam: float = 1e-2,
        variant: str = "default",
        linearization_error: Optional[float] = None,
    ) -> MaterializedModel:
        """c* = (G + lam I)^{-1} g for a projected-Jacobian surface.

        LINEARIZED-BOUNDED: callers must measure and attach the
        linearization error before deploying (see
        surfaces.linearization_error); it is stored in model meta.
        """
        key = f"jacobian:{surface}"
        self._guard(state, key, lam)
        if self.basis is None:
            raise UnsupportedMaterializationError("adapter materialization requires the basis")
        G = state.accumulators[key]["G"]
        g = state.accumulators[key]["g"]
        c = ridge_solve(G, g, lam).ravel()
        self._charge_solve(G.shape[0], 1)
        C = c.reshape(self.basis.ru, self.basis.rv)
        delta = adapter_delta(self.basis, surface, C)
        layer = int(surface.split(":")[1])
        return MaterializedModel(
            kind=key,
            guarantee=weaker(state.guarantee, GuaranteeClass.LINEARIZED_BOUNDED),
            lam=lam,
            params={"c": c, "delta": delta},
            meta={
                "variant": variant,
                "layer": layer,
                "surface": surface,
                "source_state": state.state_hash,
                "linearization_error": linearization_error,
            },
        )

    # ------------------------------------------------------- factory sweeps
    def materialize_many(
        self,
        state: NeuralIntelligenceState,
        lams: Iterable[float],
        surface: str = "head",
        quantize_bits: Optional[int] = None,
    ) -> List[MaterializedModel]:
        """Materialize one model per regularization value from one state -
        no representation passes, no data replay."""
        out = []
        for lam in lams:
            m = self.materialize_head(state, lam=lam, surface=surface, variant=f"lam={lam:g}")
            if quantize_bits is not None:
                m = m.quantized(quantize_bits)
            out.append(m)
        return out
