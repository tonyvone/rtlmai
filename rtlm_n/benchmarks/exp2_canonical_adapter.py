"""Experiment 2: canonical low-rank adapter state.

All nodes share globally fixed U, V surfaces; each accumulates evidence
only for the coordinates C. The global system SOLVES an adapter from the
merged state instead of averaging locally trained adapters.

The world is constructed so the true correction lives exactly in the
canonical space (W_gen = W_base + U C_gen V^T): the declared constrained
optimization problem is well-specified, so the experiment separates the
merging question from representation-capacity questions.

Claims that separate RTLM-N from federated LoRA:
1. merged node statistics reproduce centralized optimization in the same
   constrained space (EXACT-FROZEN equivalence);
2. canonical-state merging beats averaging locally-solved adapters under
   non-IID, uneven-size nodes (shared basis);
3. both beat averaging adapters trained in per-node MISALIGNED bases -
   the geometric-interference failure of naive LoRA merging, reproduced
   and then fixed by the canonical coordinate system.
"""

from __future__ import annotations

import numpy as np

from rtlm_n.accumulators.surfaces import lowrank_head_stats, objective_id_for
from rtlm_n.assurance.proof import ProofPackage
from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.common import make_nodes, merge_all
from rtlm_n.benchmarks.data import accuracy, gen_tokens, make_world, partition_dirichlet
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.materialization.factory import ModelFactory

LAM = 1e-3


def run(seed: int = 0, n: int = 1200, n_nodes: int = 4, verbose: bool = True) -> ProofPackage:
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    keys = KeyRegistry()
    objective = objective_id_for("lowrank-classification")
    proof = ProofPackage("exp2-canonical-lowrank-adapter")

    calib = gen_tokens(rng, 300, world)  # public calibration set
    basis = build_basis(
        world.small, "activation-svd", seed=seed, ru=4, rv=8, calibration_tokens=calib
    )
    m = basis.matrices("head")

    # ---- world: the true correction lies in the canonical space --------
    W_base = rng.standard_normal((world.n_classes, world.small.d_model)) * 0.6
    C_gen = 1.2 * rng.standard_normal((m["U"].shape[1], m["V"].shape[1]))
    W_gen = W_base + m["U"] @ C_gen @ m["V"].T

    def logits_of(toks: np.ndarray) -> np.ndarray:
        return world.small.forward(toks) @ W_gen.T

    tokens = gen_tokens(rng, n, world)
    clean = logits_of(tokens)
    labels = clean.argmax(axis=1)
    # observation noise: local solves on small skewed nodes become noisy,
    # so unweighted adapter averaging pays a real variance/bias price
    targets = clean + 0.8 * rng.standard_normal(clean.shape)
    test = gen_tokens(rng, 600, world)
    test_labels = logits_of(test).argmax(axis=1)
    h_test = world.small.forward(test)

    def acc_of(W: np.ndarray) -> float:
        return accuracy(h_test @ W.T, test_labels)

    # ---- distributed canonical states (non-IID, uneven sizes) ----------
    parts = partition_dirichlet(labels, n_nodes, alpha=0.1, rng=rng)
    # make node sizes very uneven: tiny nodes have high-variance local solves
    parts = [p[: max(20, len(p) // (1 + 2 * i))] for i, p in enumerate(parts)]
    nodes = make_nodes(world, basis, keys, n_nodes)
    for node, idx in zip(nodes, parts):
        if len(idx):
            node.absorb_lowrank_batch(tokens[idx], targets[idx], W_base, objective_id=objective)
    merged, _ = merge_all(nodes, keys, objective)
    factory = ModelFactory(basis)
    model = factory.materialize_lowrank_head(merged, W_base, lam=LAM)
    acc_rtlmn = accuracy(model.predict(h_test), test_labels)
    C_star = model.params["C"]

    # ---- centralized solve in the same constrained space ---------------
    h_all = world.small.forward(tokens)
    used = np.concatenate(parts)
    stats_c = lowrank_head_stats(basis, h_all[used], targets[used], W_base)
    C_central = ridge_solve(stats_c["H"], stats_c["R"], LAM).T
    proof.add_equivalence("merged C == centralized constrained solve", C_central, C_star)

    # ---- naive averaging of locally-solved adapters (shared basis) -----
    C_locals = []
    for idx in parts:
        if len(idx) < 10:
            continue
        s = lowrank_head_stats(basis, h_all[idx], targets[idx], W_base)
        C_locals.append(ridge_solve(s["H"], s["R"], LAM).T)
    C_avg = np.mean(C_locals, axis=0)
    acc_avg = acc_of(W_base + m["U"] @ C_avg @ m["V"].T)

    # ---- averaging adapters trained in MISALIGNED per-node bases -------
    deltas = []
    for i, idx in enumerate(parts):
        if len(idx) < 10:
            continue
        own = build_basis(world.small, "random-orthogonal", seed=1000 + i, ru=4, rv=8)
        s = lowrank_head_stats(own, h_all[idx], targets[idx], W_base)
        C_i = ridge_solve(s["H"], s["R"], LAM).T
        mi = own.matrices("head")
        deltas.append(mi["U"] @ C_i @ mi["V"].T)
    acc_misaligned = acc_of(W_base + np.mean(deltas, axis=0))

    acc_base = acc_of(W_base)
    err_merged = float(np.linalg.norm(C_star - C_gen) / np.linalg.norm(C_gen))
    err_avg = float(np.linalg.norm(C_avg - C_gen) / np.linalg.norm(C_gen))

    proof.add_metric(
        "canonical merge beats shared-basis adapter averaging",
        acc_rtlmn - acc_avg,
        threshold=0.0,
        higher_is_better=True,
        guarantee="EXACT-FROZEN",
        note=f"rtlmn={acc_rtlmn:.3f} avg={acc_avg:.3f}; C-error merged={err_merged:.3f} avg={err_avg:.3f}",
    )
    proof.add_metric(
        "canonical merge beats misaligned-basis adapter averaging",
        acc_rtlmn - acc_misaligned,
        threshold=0.02,
        higher_is_better=True,
        note=f"misaligned={acc_misaligned:.3f} (the federated-LoRA failure mode)",
    )
    proof.add_metric(
        "adapter improves over base head",
        acc_rtlmn - acc_base,
        threshold=0.05,
        higher_is_better=True,
        note=f"base={acc_base:.3f}",
    )

    if verbose:
        print(proof.summary())
        print(
            f"  accuracies: base={acc_base:.3f} misaligned-avg={acc_misaligned:.3f} "
            f"shared-avg={acc_avg:.3f} rtlmn={acc_rtlmn:.3f}"
        )
    return proof


if __name__ == "__main__":
    run()
