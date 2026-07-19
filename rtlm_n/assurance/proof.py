"""Proof package engine: machine-checkable evidence for every claim.

Each experiment emits a proof package - a JSON-serializable record of the
claim, the guarantee class, the measured evidence, and a pass/fail verdict
against an explicit tolerance. Pass gates are asserted from these packages,
never from prose.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np

#: Tolerance for "exact" (floating-point) equivalence claims.
EXACT_TOL = 1e-8


@dataclass
class Claim:
    name: str
    guarantee: str
    passed: bool
    evidence: Dict[str, float] = field(default_factory=dict)
    note: str = ""


@dataclass
class ProofPackage:
    experiment: str
    claims: List[Claim] = field(default_factory=list)

    def add_equivalence(
        self, name: str, A: np.ndarray, B: np.ndarray, guarantee: str = "EXACT-FROZEN",
        tol: float = EXACT_TOL,
    ) -> Claim:
        diff = float(np.max(np.abs(A - B)))
        scale = float(np.max(np.abs(A)) + 1e-30)
        rel = diff / scale
        c = Claim(
            name=name,
            guarantee=guarantee,
            passed=bool(rel < tol),
            evidence={"max_abs_diff": diff, "rel_diff": rel, "tolerance": tol},
        )
        self.claims.append(c)
        return c

    def add_metric(
        self, name: str, value: float, threshold: float, higher_is_better: bool,
        guarantee: str = "EMPIRICAL-NEURAL", note: str = "",
    ) -> Claim:
        passed = value >= threshold if higher_is_better else value <= threshold
        c = Claim(
            name=name,
            guarantee=guarantee,
            passed=bool(passed),
            evidence={"value": float(value), "threshold": float(threshold)},
            note=note,
        )
        self.claims.append(c)
        return c

    def add_refusal(self, name: str, refused: bool, note: str = "") -> Claim:
        c = Claim(
            name=name, guarantee="UNSUPPORTED", passed=bool(refused),
            evidence={"refused": float(refused)}, note=note,
        )
        self.claims.append(c)
        return c

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.claims)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def summary(self) -> str:
        lines = [f"[{'PASS' if self.passed else 'FAIL'}] {self.experiment}"]
        for c in self.claims:
            mark = "ok " if c.passed else "FAIL"
            ev = ", ".join(f"{k}={v:.3g}" for k, v in c.evidence.items())
            lines.append(f"  [{mark}] ({c.guarantee}) {c.name}: {ev}{'  # ' + c.note if c.note else ''}")
        return "\n".join(lines)
