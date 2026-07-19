"""Experiment 4: one state, 1,000 models.

The headline systems claim: process distributed experience ONCE into a
merged state, then materialize 1,000 model variants (regularization sweeps
x tenant exclusions x quantization) without re-running representation
extraction - and at a fraction of the compute of 1,000 independent
fine-tuning runs.
"""

from __future__ import annotations

import numpy as np

from rtlm_n.accumulators.surfaces import objective_id_for
from rtlm_n.assurance.proof import ProofPackage
from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.common import absorb_partitioned_head, make_nodes, merge_all
from rtlm_n.benchmarks.data import gen_tokens, make_world, partition_iid
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.metering.meter import CENTRAL, EDGE, EnergyMeter
from rtlm_n.materialization.factory import ModelFactory

N_VARIANTS = 1000


def run(seed: int = 0, n: int = 2000, n_nodes: int = 5, verbose: bool = True) -> ProofPackage:
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    basis = build_basis(world.small, "random-orthogonal", seed=seed)
    keys = KeyRegistry()
    objective = objective_id_for("classification")
    proof = ProofPackage("exp4-one-state-1000-models")

    tokens = gen_tokens(rng, n, world)
    targets = world.one_hot(world.labels(tokens))
    parts = partition_iid(n, n_nodes, rng)
    nodes = make_nodes(world, basis, keys, n_nodes)
    absorb_partitioned_head(nodes, parts, tokens, targets, objective)
    build_meter = EnergyMeter()
    for node in nodes:
        build_meter = build_meter.merged_with(node.meter)
    merged, engine = merge_all(nodes, keys, objective, meter=EnergyMeter())
    build_meter = build_meter.merged_with(engine.meter)

    # ---- materialize 1000 variants from ONE state ----------------------
    factory = ModelFactory(basis, meter=EnergyMeter())
    lams = np.logspace(-6, 2, 100)
    variants = []
    # 100 lambdas x (global + 5 tenant-exclusion slices) = 600
    for excl in [None] + [f"node-{i}" for i in range(n_nodes)]:
        state = merged if excl is None else merged.remove(engine.contributions[excl])
        for lam in lams:
            variants.append(
                factory.materialize_head(state, lam=float(lam), variant=f"lam={lam:g};excl={excl}")
            )
    # + 400 quantized power-budget variants
    for i, v in enumerate(variants[:400]):
        variants.append(v.quantized(bits=8 if i % 2 else 4))
    assert len(variants) >= N_VARIANTS

    flops_materialize = factory.meter.total_flops()
    joules_materialize = factory.meter.total_joules()

    # ---- baseline: 1000 independent fine-tuning runs -------------------
    # each run must re-extract representations for its data slice and solve
    flops_one_run = world.small.flops_per_example * n + factory.meter.total_flops() / len(variants)
    flops_independent = flops_one_run * N_VARIANTS
    joules_independent = flops_independent * CENTRAL.joules_per_flop

    flops_build = build_meter.total_flops()
    total_rtlmn = flops_build + flops_materialize
    ratio = flops_independent / total_rtlmn

    proof.add_metric(
        "models materialized from one state",
        len(variants),
        threshold=N_VARIANTS,
        higher_is_better=True,
        guarantee="EXACT-FROZEN",
    )
    proof.add_metric(
        "representation passes during materialization",
        factory.meter.flops.get("representation", 0.0),
        threshold=0.0,
        higher_is_better=False,
        guarantee="EXACT-FROZEN",
        note="materialization touches only the state, never raw data",
    )
    proof.add_metric(
        "compute advantage vs 1000 independent runs (x)",
        ratio,
        threshold=10.0,
        higher_is_better=True,
        guarantee="EXACT-FROZEN",
        note=(
            f"build={flops_build:.3g}F + materialize={flops_materialize:.3g}F "
            f"vs independent={flops_independent:.3g}F"
        ),
    )

    if verbose:
        print(proof.summary())
        print(
            f"  {len(variants)} variants; RTLM-N total {total_rtlmn:.3g} FLOPs "
            f"vs {flops_independent:.3g} FLOPs independent ({ratio:.1f}x); "
            f"materialization energy {joules_materialize:.3g} J vs {joules_independent:.3g} J"
        )
    return proof


if __name__ == "__main__":
    run()
