"""Compute and energy accounting.

RTLM-N's primary metric is Verified Outcomes per Joule, not accuracy alone.
Every FLOP spent on representation extraction, state building, merging,
materialization, and inference is charged to a category and a device
profile, so experiments can report:

- teacher calls avoided,
- central FLOPs avoided,
- energy break-even request counts,
- cumulative net energy saved.

Energy figures are analytic estimates (FLOPs x J/FLOP + fixed per-call
overheads) with device constants that are explicit and configurable; the
point is faithful *relative* accounting between RTLM-N and its baselines
under one consistent model, not absolute wattage claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class DeviceProfile:
    """Energy model for one class of hardware."""

    name: str
    joules_per_flop: float
    #: Fixed energy overhead per remote call (network + serialization + idle).
    joules_per_call: float = 0.0
    #: Energy per byte moved over the network to/from this device.
    joules_per_network_byte: float = 0.0


#: An efficient edge NPU/CPU: ~0.5 TFLOP/s/W effective -> 2 pJ/FLOP.
EDGE = DeviceProfile("edge", joules_per_flop=2e-12)

#: A datacenter GPU including PUE and host overhead: ~1 pJ/FLOP effective,
#: plus a fixed per-request cost and network transfer cost.
CENTRAL = DeviceProfile(
    "central",
    joules_per_flop=1e-12,
    joules_per_call=0.05,
    joules_per_network_byte=1e-8,
)


@dataclass
class EnergyMeter:
    """Accumulates FLOPs, bytes, calls, and joules by category."""

    flops: Dict[str, float] = field(default_factory=dict)
    joules: Dict[str, float] = field(default_factory=dict)
    network_bytes: Dict[str, float] = field(default_factory=dict)
    calls: Dict[str, int] = field(default_factory=dict)

    def charge_flops(self, category: str, flops: float, device: DeviceProfile) -> None:
        self.flops[category] = self.flops.get(category, 0.0) + flops
        self.joules[category] = self.joules.get(category, 0.0) + flops * device.joules_per_flop

    def charge_call(
        self, category: str, device: DeviceProfile, payload_bytes: float = 0.0, count: int = 1
    ) -> None:
        self.calls[category] = self.calls.get(category, 0) + count
        self.network_bytes[category] = self.network_bytes.get(category, 0.0) + payload_bytes
        self.joules[category] = (
            self.joules.get(category, 0.0)
            + count * device.joules_per_call
            + payload_bytes * device.joules_per_network_byte
        )

    def charge_network(self, category: str, payload_bytes: float, device: DeviceProfile) -> None:
        self.network_bytes[category] = self.network_bytes.get(category, 0.0) + payload_bytes
        self.joules[category] = (
            self.joules.get(category, 0.0) + payload_bytes * device.joules_per_network_byte
        )

    def total_joules(self) -> float:
        return sum(self.joules.values())

    def total_flops(self) -> float:
        return sum(self.flops.values())

    def merged_with(self, other: "EnergyMeter") -> "EnergyMeter":
        out = EnergyMeter()
        for src in (self, other):
            for k, v in src.flops.items():
                out.flops[k] = out.flops.get(k, 0.0) + v
            for k, v in src.joules.items():
                out.joules[k] = out.joules.get(k, 0.0) + v
            for k, v in src.network_bytes.items():
                out.network_bytes[k] = out.network_bytes.get(k, 0.0) + v
            for k, v in src.calls.items():
                out.calls[k] = out.calls.get(k, 0) + v
        return out

    def summary(self) -> Dict[str, Dict[str, float]]:
        cats = set(self.flops) | set(self.joules) | set(self.network_bytes) | set(self.calls)
        return {
            c: {
                "flops": self.flops.get(c, 0.0),
                "joules": self.joules.get(c, 0.0),
                "network_bytes": self.network_bytes.get(c, 0.0),
                "calls": float(self.calls.get(c, 0)),
            }
            for c in sorted(cats)
        }


def verified_outcomes_per_joule(accepted_tasks: int, meter: EnergyMeter) -> float:
    j = meter.total_joules()
    return accepted_tasks / j if j > 0 else float("inf")


def break_even_requests(
    e_large_per_task: float,
    e_small_per_task: float,
    e_learning_total: float,
) -> float:
    """Number of avoided-escalation tasks needed to repay the learning cost.

    Break-even: N_saved * (E_large - E_small) > E_repr + E_state + E_merge + E_mat.
    """
    saving = e_large_per_task - e_small_per_task
    if saving <= 0:
        return float("inf")
    return e_learning_total / saving
