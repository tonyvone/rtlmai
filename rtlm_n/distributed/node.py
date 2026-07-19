"""Edge nodes: local experience -> compact state.

A node sees raw examples exactly once, extracts frozen representations,
folds them into per-batch leaf states, and keeps only the compact
statistics. Raw records never leave the node; what travels is the signed
state. Per-batch leaf retention is what makes event-level deletion exact:
deleting a batch subtracts precisely its contribution.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np

from rtlm_n.accumulators.state import NeuralIntelligenceState
from rtlm_n.accumulators.surfaces import (
    SURFACE_GUARANTEES,
    batch_commitment,
    head_stats,
    jacobian_stats,
    lowrank_head_stats,
    projected_jacobian,
)
from rtlm_n.bases.basis import CanonicalBasis
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.metering.meter import EDGE, DeviceProfile, EnergyMeter
from rtlm_n.models.backbone import TinyTransformer


class EdgeNode:
    def __init__(
        self,
        node_id: str,
        model: TinyTransformer,
        basis: CanonicalBasis,
        keys: KeyRegistry,
        secret: bytes,
        tenant_id: str = "default",
        representation_version: int = 1,
        device: DeviceProfile = EDGE,
        meter: Optional[EnergyMeter] = None,
    ) -> None:
        if basis.base_model_id != model.base_model_id:
            raise ValueError("basis was built for a different backbone")
        self.node_id = node_id
        self.model = model
        self.basis = basis
        self.keys = keys
        self.tenant_id = tenant_id
        self.representation_version = representation_version
        self.device = device
        self.meter = meter if meter is not None else EnergyMeter()
        keys.register(node_id, secret)
        #: leaf states per (objective_id, task_family), one per absorbed batch
        self._leaves: Dict[tuple, List[NeuralIntelligenceState]] = {}

    # ------------------------------------------------------------ features
    def extract_features(self, tokens: np.ndarray, category: str = "representation") -> np.ndarray:
        h = self.model.forward(tokens)
        self.meter.charge_flops(category, self.model.flops_per_example * len(tokens), self.device)
        return h

    # ------------------------------------------------------------- absorb
    def _leaf(
        self,
        surface_name: str,
        objective_id: str,
        task_family: str,
        acc: Dict[str, np.ndarray],
        n: int,
        commitment: str,
        guarantee,
        residual_norm: float,
        activation_norm: float,
        t: float,
        approximation: Optional[Dict[str, float]] = None,
    ) -> NeuralIntelligenceState:
        # compatibility gates on true dependencies: plain head surfaces do
        # not depend on the canonical basis, so they stay mergeable across
        # basis versions (and after backbone migration re-anchoring)
        basis_dependent = surface_name not in ("head", "residual-head")
        leaf = NeuralIntelligenceState(
            node_id=self.node_id,
            base_model_id=self.model.base_model_id,
            representation_version=self.representation_version,
            basis_id=self.basis.basis_id if basis_dependent else "",
            objective_id=objective_id,
            task_family=task_family,
            guarantee=guarantee,
            accumulators={surface_name: acc},
            sample_count=n,
            effective_weight=float(n),
            tenant_id=self.tenant_id,
            time_range=(t, t),
            event_commitments=(commitment,),
            lineage=((self.node_id, commitment),),
            approximation={
                "residual_norm_sq": residual_norm,
                "activation_norm_sq": activation_norm,
                **(approximation or {}),
            },
        )
        key = (objective_id, task_family)
        self._leaves.setdefault(key, []).append(leaf)
        return leaf

    def absorb_head_batch(
        self,
        tokens: np.ndarray,
        targets: np.ndarray,
        objective_id: str,
        task_family: str = "default",
        t: float = 0.0,
    ) -> NeuralIntelligenceState:
        """Stage A: exact output-head evidence (classification/regression/
        reward/confidence/routing heads all take one-hot / scalar targets)."""
        h = self.extract_features(tokens)
        acc = head_stats(h, targets)
        self.meter.charge_flops("state_build", 2.0 * h.size * (h.shape[1] + targets.shape[1]), self.device)
        return self._leaf(
            "head",
            objective_id,
            task_family,
            acc,
            len(tokens),
            batch_commitment(tokens, targets),
            SURFACE_GUARANTEES["head"],
            float((targets**2).sum()),
            float((h**2).sum()),
            t,
        )

    def absorb_residual_batch(
        self,
        tokens: np.ndarray,
        teacher_logits: np.ndarray,
        student_logits: np.ndarray,
        objective_id: str,
        task_family: str = "default",
        t: float = 0.0,
        h: Optional[np.ndarray] = None,
    ) -> NeuralIntelligenceState:
        """Stage B: teacher-residual evidence. The state captures what the
        small model was missing relative to verified teacher behavior."""
        residual = teacher_logits - student_logits
        if h is None:
            h = self.extract_features(tokens)
        acc = head_stats(h, residual)
        self.meter.charge_flops("state_build", 2.0 * h.size * (h.shape[1] + residual.shape[1]), self.device)
        return self._leaf(
            "residual-head",
            objective_id,
            task_family,
            acc,
            len(tokens),
            batch_commitment(tokens, residual),
            SURFACE_GUARANTEES["residual-head"],
            float((residual**2).sum()),
            float((h**2).sum()),
            t,
        )

    def absorb_lowrank_batch(
        self,
        tokens: np.ndarray,
        targets: np.ndarray,
        W0: np.ndarray,
        objective_id: str,
        task_family: str = "default",
        t: float = 0.0,
        h: Optional[np.ndarray] = None,
    ) -> NeuralIntelligenceState:
        """Stage C: evidence for C in the canonical low-rank head update
        W = W0 + U C V^T. All nodes share U, V; only C-coordinates travel."""
        if h is None:
            h = self.extract_features(tokens)
        acc = lowrank_head_stats(self.basis, h, targets, W0)
        self.meter.charge_flops("state_build", 4.0 * h.size * self.basis.rv, self.device)
        return self._leaf(
            "lowrank-head",
            objective_id,
            task_family,
            acc,
            len(tokens),
            batch_commitment(tokens, targets),
            SURFACE_GUARANTEES["lowrank-head"],
            float((targets - h @ W0.T).__pow__(2).sum()),
            float((h**2).sum()),
            t,
        )

    def absorb_jacobian_batch(
        self,
        tokens: np.ndarray,
        targets: np.ndarray,
        head_W: np.ndarray,
        surface: str,
        objective_id: str,
        task_family: str = "default",
        t: float = 0.0,
        c0: Optional[np.ndarray] = None,
    ) -> NeuralIntelligenceState:
        """Stage D: projected Gauss-Newton evidence for an internal adapter.

        residual r = target - f(theta + P c0, x); J = d logits / d c at c0
        in the shared canonical c-space. LINEARIZED-BOUNDED by construction.
        Iterated Gauss-Newton passes the current linearization point c0 -
        callers MUST scope objective_id per round, because states linearized
        at different points are not mergeable evidence.
        """
        logits0, J = projected_jacobian(self.model, self.basis, surface, tokens, head_W, c0=c0)
        r_dim = J.shape[2]
        # (r+1) forward passes for the differencing
        self.meter.charge_flops(
            "representation", self.model.flops_per_example * len(tokens) * (r_dim + 1), self.device
        )
        residual = targets - logits0
        acc = jacobian_stats(J, residual)
        self.meter.charge_flops("state_build", 2.0 * J.size * r_dim, self.device)
        return self._leaf(
            f"jacobian:{surface}",
            objective_id,
            task_family,
            acc,
            len(tokens),
            batch_commitment(tokens, targets),
            SURFACE_GUARANTEES["projected-jacobian"],
            float((residual**2).sum()),
            float((logits0**2).sum()),
            t,
        )

    # ------------------------------------------------------ state assembly
    def state(self, objective_id: str, task_family: str = "default") -> NeuralIntelligenceState:
        """Fold all retained leaf states into one signed node state."""
        leaves = self._leaves.get((objective_id, task_family), [])
        if not leaves:
            raise ValueError(f"node {self.node_id} has no state for {objective_id!r}")
        folded = leaves[0]
        for leaf in leaves[1:]:
            folded = folded.merge(leaf)
        folded = dataclasses.replace(folded, node_id=self.node_id, signature="")
        signed = folded.signed(self.keys.sign(self.node_id, folded.state_hash))
        self.meter.charge_flops(
            "merge", sum(a.size for s in folded.accumulators.values() for a in s.values()), self.device
        )
        return signed

    def delete_events(self, commitment: str, objective_id: str, task_family: str = "default") -> int:
        """Drop every leaf carrying this event commitment. Returns count.

        After deletion, `state()` is exactly the state that would have been
        built had the deleted events never existed.
        """
        key = (objective_id, task_family)
        before = len(self._leaves.get(key, []))
        self._leaves[key] = [
            l for l in self._leaves.get(key, []) if commitment not in l.event_commitments
        ]
        return before - len(self._leaves[key])
