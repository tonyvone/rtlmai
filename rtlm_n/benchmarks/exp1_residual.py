"""Experiment 1: teacher-residual intelligence (distributed distillation).

A small frozen model escalates to a verified large teacher; nodes capture
what the small model was missing as compact residual states. The merged
residual state materializes a correction head.

Baselines: student-only, teacher-always, centralized distillation,
FedAvg-style local-head averaging (non-IID), RTLM-N residual state.
"""

from __future__ import annotations

import numpy as np

from rtlm_n.accumulators.surfaces import objective_id_for
from rtlm_n.assurance.proof import ProofPackage
from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.common import make_nodes, merge_all
from rtlm_n.benchmarks.data import (
    accuracy,
    fit_small_head,
    gen_tokens,
    make_world,
    partition_dirichlet,
)
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.materialization.factory import ModelFactory

LAM = 1e-3


def run(seed: int = 0, n: int = 1500, n_nodes: int = 4, verbose: bool = True) -> ProofPackage:
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    basis = build_basis(world.small, "random-orthogonal", seed=seed)
    keys = KeyRegistry()
    objective = objective_id_for("cascade-residual")
    proof = ProofPackage("exp1-teacher-residual")

    # weak student head fit on a small bootstrap set
    boot = gen_tokens(rng, 150, world)
    W0 = fit_small_head(world, boot, world.one_hot(world.labels(boot)), LAM)

    tokens = gen_tokens(rng, n, world)
    labels = world.labels(tokens)
    teacher_logits = world.true_logits(tokens)  # verified teacher behavior

    test = gen_tokens(rng, 600, world)
    test_labels = world.labels(test)
    h_test = augment_bias(world.small.forward(test))

    # ---- RTLM-N: distributed residual states ---------------------------
    parts = partition_dirichlet(labels, n_nodes, alpha=0.15, rng=rng)
    nodes = make_nodes(world, basis, keys, n_nodes)
    for node, idx in zip(nodes, parts):
        if len(idx) == 0:
            continue
        for chunk in np.array_split(idx, 3):
            if not len(chunk):
                continue
            h = node.extract_features(tokens[chunk])
            student = augment_bias(h) @ W0
            node.absorb_residual_batch(
                tokens[chunk], teacher_logits[chunk], student, objective_id=objective, h=h
            )
    merged, _ = merge_all(nodes, keys, objective)
    factory = ModelFactory(basis)
    correction = factory.materialize_head(merged, lam=LAM, surface="residual-head")
    rtlmn_logits = h_test @ W0 + correction.predict(h_test[:, :-1])
    acc_rtlmn = accuracy(rtlmn_logits, test_labels)

    # ---- baselines -----------------------------------------------------
    acc_student = accuracy(h_test @ W0, test_labels)
    acc_teacher = accuracy(augment_bias(world.large.forward(test)) @ world.W_true, test_labels)

    h_all = augment_bias(world.small.forward(tokens))
    W_dist = ridge_solve(h_all.T @ h_all, h_all.T @ teacher_logits, LAM)
    acc_central_distill = accuracy(h_test @ W_dist, test_labels)

    # FedAvg-style: each node fits a full head locally, then average weights
    W_locals = []
    for idx in parts:
        if len(idx) < 10:
            continue
        hi = h_all[idx]
        W_locals.append(ridge_solve(hi.T @ hi, hi.T @ teacher_logits[idx], LAM))
    W_fedavg = np.mean(W_locals, axis=0)
    acc_fedavg = accuracy(h_test @ W_fedavg, test_labels)

    proof.add_metric(
        "RTLM-N matches centralized distillation",
        acc_rtlmn - acc_central_distill,
        threshold=-0.005,
        higher_is_better=True,
        guarantee="EXACT-FROZEN",
        note=f"rtlmn={acc_rtlmn:.3f} central={acc_central_distill:.3f}",
    )
    proof.add_metric(
        "RTLM-N beats student-only",
        acc_rtlmn - acc_student,
        threshold=0.02,
        higher_is_better=True,
        note=f"student={acc_student:.3f}",
    )
    proof.add_metric(
        "RTLM-N beats FedAvg head averaging (non-IID)",
        acc_rtlmn - acc_fedavg,
        threshold=0.0,
        higher_is_better=True,
        note=f"fedavg={acc_fedavg:.3f}",
    )
    proof.add_metric(
        "teacher ceiling sanity",
        acc_teacher,
        threshold=0.99,
        higher_is_better=True,
        note="teacher is well-specified by construction",
    )

    if verbose:
        print(proof.summary())
        print(
            f"  accuracies: student={acc_student:.3f} fedavg={acc_fedavg:.3f} "
            f"rtlmn={acc_rtlmn:.3f} central-distill={acc_central_distill:.3f} teacher={acc_teacher:.3f}"
        )
    return proof


if __name__ == "__main__":
    run()
