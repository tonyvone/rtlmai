"""Merge and lineage engine.

Accepts signed node states, verifies signatures and numerical safety,
refuses incompatible coordinate systems and duplicate contributions, and
maintains the merged global state plus per-contributor copies so that
REMOVE (contributor deletion) is exact.

Merging is a sum, so the engine's result is invariant to arrival order and
grouping - the property the equivalence tests exercise directly.
"""

from __future__ import annotations

from typing import Dict, Optional

from rtlm_n.accumulators.state import NeuralIntelligenceState, StateError
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.metering.meter import CENTRAL, DeviceProfile, EnergyMeter


class IncompatibleStateError(StateError):
    """Different backbone, basis, objective, or shapes - refused."""


class DuplicateStateError(StateError):
    """A contribution with overlapping lineage was already merged."""


class TamperedStateError(StateError):
    """Signature verification failed or numerics are unsafe."""


class MergeEngine:
    def __init__(
        self,
        keys: KeyRegistry,
        device: DeviceProfile = CENTRAL,
        meter: Optional[EnergyMeter] = None,
    ) -> None:
        self.keys = keys
        self.device = device
        self.meter = meter if meter is not None else EnergyMeter()
        self.global_state: Optional[NeuralIntelligenceState] = None
        self.contributions: Dict[str, NeuralIntelligenceState] = {}

    def _validate(self, state: NeuralIntelligenceState) -> None:
        if not self.keys.verify(state.node_id, state.state_hash, state.signature):
            raise TamperedStateError(
                f"state from {state.node_id!r} failed signature verification"
            )
        if not state.verify():
            raise TamperedStateError(
                f"state from {state.node_id!r} failed numerical safety checks (UNSUPPORTED)"
            )

    def submit(self, state: NeuralIntelligenceState) -> NeuralIntelligenceState:
        """Verify and fold one node state into the global state."""
        self._validate(state)
        payload = state.nbytes()
        self.meter.charge_network("merge", payload, self.device)
        if self.global_state is None:
            self.global_state = state
        else:
            try:
                merged = self.global_state.merge(state)
            except StateError as e:
                msg = str(e)
                if "duplicate" in msg:
                    raise DuplicateStateError(msg) from e
                raise IncompatibleStateError(msg) from e
            self.meter.charge_flops(
                "merge",
                sum(a.size for s in state.accumulators.values() for a in s.values()),
                self.device,
            )
            self.global_state = merged
        self.contributions[state.node_id] = state
        return self.global_state

    def remove_contributor(self, node_id: str) -> NeuralIntelligenceState:
        """Exactly delete one contributor from the global state."""
        if self.global_state is None or node_id not in self.contributions:
            raise StateError(f"no contribution recorded for {node_id!r}")
        self.global_state = self.global_state.remove(self.contributions.pop(node_id))
        return self.global_state
