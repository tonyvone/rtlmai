"""Experiment 7: backbone migration - can intelligence survive its model?

The highest-risk, highest-value question. A state accumulated on backbone
v1 is transported to backbone v2 (different weights AND different width)
through a representation bridge fit on a small public anchor set. Compared
against: cold reset (lower bound), raw-data retraining on v2 (upper bound,
requires the raw data RTLM-N no longer holds).

Per the claim discipline: migration is labeled EMPIRICAL-NEURAL, its
bridge residual is reported, and it is NEVER called exact. The pass gate is
that transported intelligence recovers most of the gap between reset and
full retraining.
"""

from __future__ import annotations

import numpy as np

from rtlm_n.accumulators.surfaces import objective_id_for
from rtlm_n.assurance.proof import ProofPackage
from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.common import absorb_partitioned_head, make_nodes, merge_all
from rtlm_n.benchmarks.data import accuracy, gen_tokens, make_world, partition_iid
from rtlm_n.core.guarantees import GuaranteeClass
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.materialization.factory import ModelFactory
from rtlm_n.migration.transport import fit_bridge, transport_head, transport_state
from rtlm_n.models.backbone import TinyTransformer

LAM = 1e-3


def run(seed: int = 0, n: int = 1500, verbose: bool = True) -> ProofPackage:
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    keys = KeyRegistry()
    objective = objective_id_for("classification")
    proof = ProofPackage("exp7-backbone-migration")

    # backbone v2: different weights AND different width
    v2 = TinyTransformer(seed=seed + 5000, d_model=48, n_heads=4, n_layers=2, d_ff=96, name="edge-v2")

    basis = build_basis(world.small, "random-orthogonal", seed=seed)
    tokens = gen_tokens(rng, n, world)
    targets = world.one_hot(world.labels(tokens))
    parts = partition_iid(n, 4, rng)
    nodes = make_nodes(world, basis, keys, 4)
    absorb_partitioned_head(nodes, parts, tokens, targets, objective)
    merged, _ = merge_all(nodes, keys, objective)
    factory = ModelFactory(basis)
    W_v1 = factory.materialize_head(merged, lam=LAM)

    test = gen_tokens(rng, 600, world)
    test_labels = world.labels(test)
    h2_test = v2.forward(test)

    # ---- representation bridge on a small PUBLIC anchor set ------------
    anchors = gen_tokens(rng, 400, world)
    bridge = fit_bridge(world.small, v2, anchors, lam=LAM)

    # A) cold reset: random head on v2
    W_reset = np.random.default_rng(seed + 9).standard_normal((v2.d_model + 1, world.n_classes))
    acc_reset = accuracy(augment_bias(h2_test) @ W_reset, test_labels)

    # B) transported HEAD through the bridge
    m_head = transport_head(W_v1, bridge)
    acc_head = accuracy(m_head.predict(h2_test), test_labels)

    # C) transported STATE, then materialized natively on v2
    s2 = transport_state(merged, bridge)
    m_state = ModelFactory().materialize_head(s2, lam=LAM)
    acc_state = accuracy(m_state.predict(h2_test), test_labels)

    # D) upper bound: raw-data retraining on v2 (data RTLM-N no longer has)
    h2_all = augment_bias(v2.forward(tokens))
    W_retrain = ridge_solve(h2_all.T @ h2_all, h2_all.T @ targets, LAM)
    acc_retrain = accuracy(augment_bias(h2_test) @ W_retrain, test_labels)

    recovered = (max(acc_head, acc_state) - acc_reset) / max(acc_retrain - acc_reset, 1e-9)
    proof.add_metric(
        "transported intelligence beats cold reset",
        max(acc_head, acc_state) - acc_reset,
        threshold=0.05,
        higher_is_better=True,
        guarantee="EMPIRICAL-NEURAL",
        note=f"reset={acc_reset:.3f} head={acc_head:.3f} state={acc_state:.3f}",
    )
    proof.add_metric(
        "fraction of reset->retrain gap recovered",
        recovered,
        threshold=0.5,
        higher_is_better=True,
        guarantee="EMPIRICAL-NEURAL",
        note=f"retrain-upper-bound={acc_retrain:.3f}, bridge residual={bridge.residual:.3f}",
    )
    proof.add_metric(
        "migration is never labeled exact",
        1.0 if s2.guarantee == GuaranteeClass.EMPIRICAL_NEURAL else 0.0,
        threshold=1.0,
        higher_is_better=True,
        guarantee="EMPIRICAL-NEURAL",
        note=f"transported state guarantee={s2.guarantee.value}",
    )

    if verbose:
        print(proof.summary())
        print(
            f"  reset={acc_reset:.3f} transported-head={acc_head:.3f} "
            f"transported-state={acc_state:.3f} retrain={acc_retrain:.3f} "
            f"(recovered {recovered:.0%} of the gap)"
        )
    return proof


if __name__ == "__main__":
    run()
