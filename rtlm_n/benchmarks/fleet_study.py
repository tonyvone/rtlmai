"""Fleet lifetime energy study: what the paradigm is worth at scale.

Converts the measured micro-results of this repository into a fleet-scale
energy projection, using the standard systems methodology (measured
per-operation costs x a declared workload model). This is a SIMULATION -
every parameter is printed with its provenance (measured here vs industry
estimate), and the output is labeled EMPIRICAL-NEURAL. Its purpose is to
show which regime dominates and why, not to claim a precise wattage.

Three inference regimes over a 24-month fleet:
- teacher-always: every task goes to the datacenter model;
- static cascade: a tuned confidence router (no learning) keeps a fixed
  fraction local;
- RTLM-N accretion: escalations become state, escalation decays toward a
  floor (decay shape measured in exp5/R3), backbone upgrades partially
  reset it (migration recovery measured in R5), and per-tenant model
  refreshes are materializations instead of SGD runs (ratio measured in
  R2).

    python -m rtlm_n.benchmarks.fleet_study
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class FleetParams:
    # ---- workload ------------------------------------------------------
    n_nodes: int = 1000
    tasks_per_node_per_day: int = 200
    days: int = 730
    # ---- energy per operation (joules) ---------------------------------
    #: datacenter-class teacher task (7B model, few-hundred-token response,
    #: incl. serving overhead + PUE). Industry estimates span 10-300 J for
    #: this class; 50 J is a deliberately mid-range choice.
    e_teacher_task: float = 50.0            # source: industry estimate
    #: edge-class model task (~100M params on an efficient NPU)
    e_edge_task: float = 0.5                # source: industry estimate
    #: state fold per absorbed escalation (measured: reuses the edge
    #: forward pass; the fold itself is dwarfed by it)
    e_state_fold: float = 0.02              # source: measured (state_build)
    #: daily merge + rematerialization per node (measured R2: 0.0078
    #: cpu-s/solve at 15 W)
    e_rematerialize: float = 0.12           # source: measured
    #: per-tenant model refresh via SGD vs via materialization
    e_refresh_sgd: float = 25.2             # source: measured (1.68 cpu-s x 15 W)
    e_refresh_materialize: float = 0.117    # source: measured (215x ratio, R2)
    # ---- dynamics ------------------------------------------------------
    esc_initial: float = 0.90               # cold-start escalation
    esc_floor: float = 0.12                 # measured floor (exp5: 0.37->0.13..0.07)
    esc_tau_days: float = 30.0              # decay shape: fitted to exp5/R3 curves
    esc_static: float = 0.37                # a well-tuned static router
    #: backbone upgrade cadence and how much accreted intelligence the
    #: representation bridge preserves (measured R5: 78% of gap recovered)
    upgrade_every_days: int = 180
    migration_recovery: float = 0.78        # source: measured (R5)
    # ---- training side -------------------------------------------------
    tenant_models: int = 200
    tenant_refreshes_per_month: int = 1


@dataclass
class RegimeResult:
    name: str
    total_joules: float
    total_kwh: float
    daily_joules: List[float] = field(default_factory=list)


def simulate(p: FleetParams) -> Dict[str, RegimeResult]:
    tasks_day = p.n_nodes * p.tasks_per_node_per_day
    regimes: Dict[str, RegimeResult] = {}

    # ---- teacher-always -------------------------------------------------
    daily = [tasks_day * p.e_teacher_task] * p.days
    regimes["teacher_always"] = RegimeResult("teacher-always", sum(daily), sum(daily) / 3.6e6, daily)

    # ---- static cascade -------------------------------------------------
    daily = [
        tasks_day * (p.esc_static * p.e_teacher_task + p.e_edge_task)
    ] * p.days
    # static regime still refreshes tenant models by SGD
    refresh_j = p.tenant_models * p.tenant_refreshes_per_month * p.e_refresh_sgd
    daily = [d + refresh_j / 30.0 for d in daily]
    regimes["static_cascade"] = RegimeResult("static cascade", sum(daily), sum(daily) / 3.6e6, daily)

    # ---- RTLM-N accretion ----------------------------------------------
    daily = []
    t_since_reset = 0.0
    esc_start = p.esc_initial
    for day in range(p.days):
        if day > 0 and day % p.upgrade_every_days == 0:
            # backbone upgrade: migration preserves most accreted skill
            esc_now = p.esc_floor + (esc_start - p.esc_floor) * math.exp(
                -t_since_reset / p.esc_tau_days
            )
            gap_recovered = p.migration_recovery * (p.esc_initial - esc_now)
            esc_start = p.esc_initial - gap_recovered
            t_since_reset = 0.0
        esc = p.esc_floor + (esc_start - p.esc_floor) * math.exp(-t_since_reset / p.esc_tau_days)
        t_since_reset += 1.0
        j = tasks_day * (
            esc * (p.e_teacher_task + p.e_state_fold) + p.e_edge_task
        )
        j += p.n_nodes * p.e_rematerialize
        j += p.tenant_models * p.tenant_refreshes_per_month * p.e_refresh_materialize / 30.0
        daily.append(j)
    regimes["rtlmn"] = RegimeResult("RTLM-N accretion", sum(daily), sum(daily) / 3.6e6, daily)
    return regimes


def run(verbose: bool = True, params: FleetParams | None = None) -> Dict:
    p = params or FleetParams()
    regimes = simulate(p)
    ta, sc, rn = regimes["teacher_always"], regimes["static_cascade"], regimes["rtlmn"]
    out = {
        "guarantee": "EMPIRICAL-NEURAL (simulation over measured micro-costs + declared estimates)",
        "params": asdict(p),
        "totals_kwh": {r.name: round(r.total_kwh, 1) for r in regimes.values()},
        "reduction_vs_teacher_always": round(1 - rn.total_joules / ta.total_joules, 4),
        "reduction_vs_static_cascade": round(1 - rn.total_joules / sc.total_joules, 4),
        "monthly_kwh": {
            name: [round(sum(r.daily_joules[m * 30 : (m + 1) * 30]) / 3.6e6, 1) for m in range(p.days // 30)]
            for name, r in (("teacher-always", ta), ("static cascade", sc), ("RTLM-N", rn))
        },
    }
    if verbose:
        print("Fleet lifetime energy study "
              f"({p.n_nodes} nodes x {p.tasks_per_node_per_day} tasks/day x {p.days} days)")
        print(f"  guarantee: {out['guarantee']}")
        for r in regimes.values():
            print(f"  {r.name:<18} {r.total_kwh:>10.1f} kWh")
        print(f"  reduction vs teacher-always:  {out['reduction_vs_teacher_always']:.1%}")
        print(f"  reduction vs static cascade:  {out['reduction_vs_static_cascade']:.1%}")
        print("  parameter provenance: e_refresh_* and e_rematerialize measured in-repo;")
        print("  esc floor/decay shape from exp5/R3; migration recovery from R5;")
        print("  per-task teacher/edge joules are declared industry estimates.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, default="rtlmn_fleet_report.json")
    args = parser.parse_args()
    out = run()
    with open(args.report, "w") as f:
        json.dump(out, f, indent=2)
    print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
