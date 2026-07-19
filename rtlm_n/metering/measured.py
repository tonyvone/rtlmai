"""Measured compute/energy accounting (upgrade from the analytic model).

Measures actual process CPU-seconds (user+system, all threads) per phase
with `os.times`, and reads Intel RAPL energy counters when the platform
exposes them (`/sys/class/powercap/intel-rapl*`). Where RAPL is absent
(most containers), joules are derived as

    joules = cpu_seconds x RTLMN_CPU_WATTS   (default 15 W/busy core-set)

with the wattage explicit and configurable - measured time, assumed
power. Reports always state which of the two modes produced them.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

DEFAULT_CPU_WATTS = float(os.environ.get("RTLMN_CPU_WATTS", "15.0"))

_RAPL_GLOB = "/sys/class/powercap/intel-rapl:*/energy_uj"


def _rapl_uj() -> Optional[float]:
    total = 0.0
    found = False
    for path in glob.glob(_RAPL_GLOB):
        try:
            with open(path) as f:
                total += float(f.read().strip())
            found = True
        except OSError:
            continue
    return total if found else None


@dataclass
class MeasuredPhase:
    name: str
    wall_s: float = 0.0
    cpu_s: float = 0.0
    rapl_j: Optional[float] = None

    @property
    def joules(self) -> float:
        if self.rapl_j is not None:
            return self.rapl_j
        return self.cpu_s * DEFAULT_CPU_WATTS

    @property
    def mode(self) -> str:
        return "rapl" if self.rapl_j is not None else f"cpu_s x {DEFAULT_CPU_WATTS:g}W"


@dataclass
class MeasuredMeter:
    phases: Dict[str, MeasuredPhase] = field(default_factory=dict)

    class _Ctx:
        def __init__(self, meter: "MeasuredMeter", name: str) -> None:
            self.meter, self.name = meter, name

        def __enter__(self):
            t = os.times()
            self.cpu0 = t.user + t.system + t.children_user + t.children_system
            self.wall0 = time.perf_counter()
            self.rapl0 = _rapl_uj()
            return self

        def __exit__(self, *exc):
            t = os.times()
            cpu = t.user + t.system + t.children_user + t.children_system - self.cpu0
            wall = time.perf_counter() - self.wall0
            rapl = None
            if self.rapl0 is not None:
                r1 = _rapl_uj()
                if r1 is not None and r1 >= self.rapl0:
                    rapl = (r1 - self.rapl0) / 1e6
            p = self.meter.phases.setdefault(self.name, MeasuredPhase(self.name))
            p.wall_s += wall
            p.cpu_s += cpu
            if rapl is not None:
                p.rapl_j = (p.rapl_j or 0.0) + rapl
            return False

    def phase(self, name: str) -> "MeasuredMeter._Ctx":
        return MeasuredMeter._Ctx(self, name)

    def total_joules(self) -> float:
        return sum(p.joules for p in self.phases.values())

    def total_cpu_s(self) -> float:
        return sum(p.cpu_s for p in self.phases.values())

    def summary(self) -> str:
        lines = [f"{'phase':<28}{'wall s':>10}{'cpu s':>10}{'joules':>12}  mode"]
        for name in sorted(self.phases):
            p = self.phases[name]
            lines.append(
                f"{name:<28}{p.wall_s:>10.2f}{p.cpu_s:>10.2f}{p.joules:>12.2f}  {p.mode}"
            )
        return "\n".join(lines)
