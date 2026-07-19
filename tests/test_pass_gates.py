"""End-to-end: every foundational experiment's proof package passes.

This is the executable form of the spec's non-negotiable pass gates - the
experiments emit machine-checked claims (equivalences with tolerances,
metric thresholds, refusal checks), and this test asserts all of them.
"""

import pytest

from rtlm_n.benchmarks import (
    exp0_exact_head,
    exp1_residual,
    exp2_canonical_adapter,
    exp3_jacobian,
    exp4_thousand_models,
    exp5_cascade,
    exp6_heterogeneous,
    exp7_migration,
)

EXPERIMENTS = [
    exp0_exact_head,
    exp1_residual,
    exp2_canonical_adapter,
    exp3_jacobian,
    exp4_thousand_models,
    exp5_cascade,
    exp6_heterogeneous,
    exp7_migration,
]


@pytest.mark.parametrize("mod", EXPERIMENTS, ids=[m.__name__.split(".")[-1] for m in EXPERIMENTS])
def test_experiment_proof_package_passes(mod):
    proof = mod.run(seed=0, verbose=False)
    failed = [c for c in proof.claims if not c.passed]
    assert proof.passed, f"{proof.experiment} failed claims: {[c.name for c in failed]}"
