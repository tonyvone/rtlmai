"""NeuralIntelligenceState: the compact, mergeable learning object.

A state is a set of sufficient statistics for declared adaptation surfaces
of a frozen backbone, expressed in a canonical basis:

    frozen-feature surfaces:   H = sum h h^T,   R = sum h r^T
    projected-Jacobian:        G = sum J^T J,   g = sum J^T r

Because these are plain sums, the state algebra is:

    MERGE(a, b)  = elementwise sum          (associative, commutative)
    REMOVE(a, b) = elementwise difference   (exact contributor deletion)
    SCALE(a, w)  = elementwise scaling      (importance weighting)

and materialization is a regularized solve whose result depends only on the
final sums - not on partition, merge order, or arrival time. That is what
makes distributed training exactly equivalent to centralized training for
supported surfaces.

States are content-hashed for duplicate rejection and signed by their
originating node for tamper detection.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np

from rtlm_n.core.guarantees import GuaranteeClass, weaker
from rtlm_n.core.hashing import state_hash
from rtlm_n.core.linalg import is_finite_psd


class StateError(ValueError):
    pass


@dataclass(frozen=True)
class NeuralIntelligenceState:
    node_id: str
    base_model_id: str
    representation_version: int
    basis_id: str
    objective_id: str
    task_family: str
    guarantee: GuaranteeClass
    #: surface name -> {"H": (d,d), "R": (d,k)} or {"G": (r,r), "g": (r,)}
    accumulators: Dict[str, Dict[str, np.ndarray]] = field(repr=False)
    sample_count: int = 0
    effective_weight: float = 1.0
    tenant_id: str = "default"
    time_range: Tuple[float, float] = (0.0, 0.0)
    #: hashes of the raw event batches absorbed (audit + deletion support)
    event_commitments: Tuple[str, ...] = ()
    #: (node_id, leaf state hash) contributions folded into this state
    lineage: Tuple[Tuple[str, str], ...] = ()
    #: declared approximation info for non-exact guarantee classes
    approximation: Dict[str, float] = field(default_factory=dict)
    signature: str = ""

    # ------------------------------------------------------------ identity
    def _meta_for_hash(self) -> Dict:
        return {
            "node_id": self.node_id,
            "base_model_id": self.base_model_id,
            "representation_version": self.representation_version,
            "basis_id": self.basis_id,
            "objective_id": self.objective_id,
            "task_family": self.task_family,
            "guarantee": self.guarantee.value,
            "sample_count": self.sample_count,
            "effective_weight": self.effective_weight,
            "tenant_id": self.tenant_id,
            "time_range": list(self.time_range),
            "event_commitments": sorted(self.event_commitments),
            "lineage": sorted([list(t) for t in self.lineage]),
            "approximation": {k: self.approximation[k] for k in sorted(self.approximation)},
        }

    def _flat_arrays(self) -> Dict[str, np.ndarray]:
        return {
            f"{s}/{m}": self.accumulators[s][m]
            for s in sorted(self.accumulators)
            for m in sorted(self.accumulators[s])
        }

    @property
    def state_hash(self) -> str:
        return state_hash(self._meta_for_hash(), self._flat_arrays())

    def compat_key(self) -> Tuple:
        shapes = tuple(
            (s, m, self.accumulators[s][m].shape)
            for s in sorted(self.accumulators)
            for m in sorted(self.accumulators[s])
        )
        return (
            self.base_model_id,
            self.representation_version,
            self.basis_id,
            self.objective_id,
            self.task_family,
            shapes,
        )

    # ----------------------------------------------------------- operations
    def merge(self, other: "NeuralIntelligenceState") -> "NeuralIntelligenceState":
        if self.compat_key() != other.compat_key():
            raise StateError(
                "incompatible states: "
                f"{self.compat_key()!r} vs {other.compat_key()!r} - refusing to merge"
            )
        overlap = set(self.lineage) & set(other.lineage)
        if overlap:
            raise StateError(f"duplicate contributions detected: {sorted(overlap)!r}")
        acc = {
            s: {m: self.accumulators[s][m] + other.accumulators[s][m] for m in self.accumulators[s]}
            for s in self.accumulators
        }
        return replace(
            self,
            node_id="merged",
            accumulators=acc,
            sample_count=self.sample_count + other.sample_count,
            effective_weight=self.effective_weight + other.effective_weight,
            time_range=(
                min(self.time_range[0], other.time_range[0]),
                max(self.time_range[1], other.time_range[1]),
            ),
            event_commitments=tuple(sorted(set(self.event_commitments) | set(other.event_commitments))),
            lineage=tuple(sorted(set(self.lineage) | set(other.lineage))),
            guarantee=weaker(self.guarantee, other.guarantee),
            approximation={
                k: self.approximation.get(k, 0.0) + other.approximation.get(k, 0.0)
                for k in set(self.approximation) | set(other.approximation)
            },
            signature="",
        )

    def remove(self, contribution: "NeuralIntelligenceState") -> "NeuralIntelligenceState":
        """Exact deletion of a previously merged contribution."""
        if self.compat_key() != contribution.compat_key():
            raise StateError("cannot remove an incompatible contribution")
        if not set(contribution.lineage) <= set(self.lineage):
            raise StateError(
                f"contribution {contribution.state_hash} is not part of this state's lineage"
            )
        acc = {
            s: {
                m: self.accumulators[s][m] - contribution.accumulators[s][m]
                for m in self.accumulators[s]
            }
            for s in self.accumulators
        }
        return replace(
            self,
            accumulators=acc,
            sample_count=self.sample_count - contribution.sample_count,
            effective_weight=self.effective_weight - contribution.effective_weight,
            event_commitments=tuple(
                sorted(set(self.event_commitments) - set(contribution.event_commitments))
            ),
            lineage=tuple(sorted(set(self.lineage) - set(contribution.lineage))),
            signature="",
        )

    def scale(self, w: float) -> "NeuralIntelligenceState":
        if w < 0:
            raise StateError("scale weight must be non-negative")
        acc = {
            s: {m: self.accumulators[s][m] * w for m in self.accumulators[s]}
            for s in self.accumulators
        }
        return replace(self, accumulators=acc, effective_weight=self.effective_weight * w, signature="")

    def project(self, surfaces: Tuple[str, ...]) -> "NeuralIntelligenceState":
        """Restrict the state to a subset of adaptation surfaces."""
        missing = [s for s in surfaces if s not in self.accumulators]
        if missing:
            raise StateError(f"unknown surfaces {missing!r}")
        acc = {s: dict(self.accumulators[s]) for s in surfaces}
        return replace(self, accumulators=acc, signature="")

    def sketch(self, max_rank: int) -> "NeuralIntelligenceState":
        """Spectral sketch: truncate each Gram accumulator to its top
        eigenpairs. Declared error (dropped eigenvalue mass fraction) is
        recorded and the guarantee class weakens to SKETCHED-BOUNDED."""
        acc: Dict[str, Dict[str, np.ndarray]] = {}
        approx = dict(self.approximation)
        for s, mats in self.accumulators.items():
            acc[s] = {}
            for m, A in mats.items():
                if m in ("H", "G") and A.shape[0] == A.shape[1] and A.shape[0] > max_rank:
                    eigvals, eigvecs = np.linalg.eigh(A)
                    keep = eigvals[-max_rank:]
                    Vk = eigvecs[:, -max_rank:]
                    acc[s][m] = (Vk * keep) @ Vk.T
                    total = float(np.abs(eigvals).sum())
                    dropped = float(np.abs(eigvals[:-max_rank]).sum())
                    approx[f"sketch:{s}/{m}"] = dropped / total if total > 0 else 0.0
                else:
                    acc[s][m] = A
        return replace(
            self,
            accumulators=acc,
            guarantee=weaker(self.guarantee, GuaranteeClass.SKETCHED_BOUNDED),
            approximation=approx,
            signature="",
        )

    def verify(self) -> bool:
        """Numerical safety: finite accumulators, PSD Gram matrices,
        consistent counts. Unsafe states must be treated as UNSUPPORTED."""
        if self.sample_count < 0 or self.effective_weight < 0:
            return False
        for s, mats in self.accumulators.items():
            for m, A in mats.items():
                if not np.all(np.isfinite(A)):
                    return False
                if m in ("H", "G") and not is_finite_psd(A):
                    return False
        return True

    # -------------------------------------------------------- serialization
    def to_bytes(self) -> bytes:
        meta = self._meta_for_hash()
        meta["signature"] = self.signature
        meta["arrays"] = {}
        buf = io.BytesIO()
        flat = self._flat_arrays()
        arrs = {}
        for i, (name, a) in enumerate(sorted(flat.items())):
            key = f"a{i}"
            meta["arrays"][name] = {"key": key, "shape": list(a.shape)}
            arrs[key] = np.ascontiguousarray(a, dtype=np.float64)
        np.savez(buf, **arrs)
        blob = buf.getvalue()
        header = json.dumps(meta, sort_keys=True).encode()
        return len(header).to_bytes(8, "big") + header + blob

    @staticmethod
    def from_bytes(data: bytes) -> "NeuralIntelligenceState":
        hlen = int.from_bytes(data[:8], "big")
        meta = json.loads(data[8 : 8 + hlen].decode())
        arrs = np.load(io.BytesIO(data[8 + hlen :]))
        accumulators: Dict[str, Dict[str, np.ndarray]] = {}
        for name, info in meta["arrays"].items():
            s, m = name.split("/", 1)
            accumulators.setdefault(s, {})[m] = arrs[info["key"]].reshape(info["shape"])
        return NeuralIntelligenceState(
            node_id=meta["node_id"],
            base_model_id=meta["base_model_id"],
            representation_version=meta["representation_version"],
            basis_id=meta["basis_id"],
            objective_id=meta["objective_id"],
            task_family=meta["task_family"],
            guarantee=GuaranteeClass(meta["guarantee"]),
            accumulators=accumulators,
            sample_count=meta["sample_count"],
            effective_weight=meta["effective_weight"],
            tenant_id=meta["tenant_id"],
            time_range=tuple(meta["time_range"]),
            event_commitments=tuple(meta["event_commitments"]),
            lineage=tuple(tuple(t) for t in meta["lineage"]),
            approximation=meta["approximation"],
            signature=meta.get("signature", ""),
        )

    def nbytes(self) -> int:
        return len(self.to_bytes())

    def signed(self, signature: str) -> "NeuralIntelligenceState":
        return replace(self, signature=signature)


def zero_like(template: NeuralIntelligenceState) -> NeuralIntelligenceState:
    acc = {
        s: {m: np.zeros_like(a) for m, a in mats.items()}
        for s, mats in template.accumulators.items()
    }
    return replace(
        template,
        accumulators=acc,
        sample_count=0,
        effective_weight=0.0,
        event_commitments=(),
        lineage=(),
        signature="",
    )
