"""Experiment 6: heterogeneous edges.

Nodes differ in data volume (60..1200 samples), input distribution
(per-node vocabulary domains = covariate shift), label mix (Dirichlet
skew), contribution frequency (1..6 event batches), and numeric precision
(one node accumulates in float32). The gates: heterogeneity must not break
merging, full-precision subsets stay exactly equivalent to centralized
training, and the mixed-precision deviation is measured and small.
"""

from __future__ import annotations

import numpy as np

from rtlm_n.accumulators.surfaces import objective_id_for
from rtlm_n.assurance.proof import ProofPackage
from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.data import accuracy, gen_tokens, make_world
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.distributed.node import EdgeNode
from rtlm_n.distributed.merge import MergeEngine
from rtlm_n.materialization.factory import ModelFactory

LAM = 1e-3


def run(seed: int = 0, verbose: bool = True) -> ProofPackage:
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    basis = build_basis(world.small, "random-orthogonal", seed=seed)
    keys = KeyRegistry()
    objective = objective_id_for("classification")
    proof = ProofPackage("exp6-heterogeneous-edges")

    # node profiles: (samples, vocab domain, event batches, dtype)
    profiles = [
        (60, (0, 24), 1, np.float64),
        (300, (12, 40), 2, np.float64),
        (700, (20, 64), 4, np.float64),
        (1200, (0, 64), 6, np.float64),
        (250, (30, 64), 2, np.float32),  # low-precision edge device
    ]
    nodes, all_tokens, all_targets = [], [], []
    f32_node_id = None
    for i, (n_i, domain, n_batches, dtype) in enumerate(profiles):
        node = EdgeNode(f"node-{i}", world.small, basis, keys, f"s{i}".encode())
        toks = gen_tokens(rng, n_i, world, domain=domain)
        targets = world.one_hot(world.labels(toks))
        for chunk in np.array_split(np.arange(n_i), n_batches):
            leaf = node.absorb_head_batch(toks[chunk], targets[chunk], objective_id=objective)
            if dtype is np.float32:
                # simulate a device that stores its accumulators in fp32
                for m in leaf.accumulators["head"]:
                    leaf.accumulators["head"][m][...] = leaf.accumulators["head"][m].astype(
                        np.float32
                    )
                f32_node_id = node.node_id
        nodes.append(node)
        all_tokens.append(toks)
        all_targets.append(targets)

    engine = MergeEngine(keys)
    for node in nodes:
        engine.submit(node.state(objective))
    factory = ModelFactory(basis)
    W_hetero = factory.materialize_head(engine.global_state, lam=LAM).params["W"]

    # centralized reference over the identical pooled data
    tokens = np.concatenate(all_tokens)
    targets = np.concatenate(all_targets)
    h_all = augment_bias(world.small.forward(tokens))
    W_central = ridge_solve(h_all.T @ h_all, h_all.T @ targets, LAM)

    # full-precision subset must stay exactly equivalent
    sub = engine.global_state.remove(engine.contributions[f32_node_id])
    W_sub = factory.materialize_head(sub, lam=LAM).params["W"]
    n64 = sum(p[0] for p in profiles[:-1])
    h64 = h_all[:n64]
    W_central64 = ridge_solve(h64.T @ h64, h64.T @ targets[:n64], LAM)
    proof.add_equivalence(
        "float64 subset stays exactly centralized-equivalent", W_central64, W_sub
    )

    rel = float(np.max(np.abs(W_hetero - W_central)) / (np.max(np.abs(W_central)) + 1e-30))
    proof.add_metric(
        "mixed-precision deviation (measured, small)",
        rel,
        threshold=1e-3,
        higher_is_better=False,
        guarantee="SKETCHED-BOUNDED",
        note="fp32 accumulators weaken the guarantee; the error is declared, not hidden",
    )

    test = gen_tokens(rng, 600, world)
    test_labels = world.labels(test)
    h_test = augment_bias(world.small.forward(test))
    acc_merged = accuracy(h_test @ W_hetero, test_labels)
    acc_central = accuracy(h_test @ W_central, test_labels)
    local_accs = []
    for node in nodes:
        s = node.state(objective)
        W_i = factory.materialize_head(s, lam=LAM).params["W"]
        local_accs.append(accuracy(h_test @ W_i, test_labels))
    proof.add_metric(
        "non-IID merging matches centralized quality",
        abs(acc_merged - acc_central),
        threshold=0.005,
        higher_is_better=False,
        guarantee="EXACT-FROZEN",
        note=f"merged={acc_merged:.3f} centralized={acc_central:.3f}",
    )
    proof.add_metric(
        "merged model beats the typical heterogeneous node",
        acc_merged - float(np.mean(local_accs)),
        threshold=0.0,
        higher_is_better=True,
        note=(
            f"merged={acc_merged:.3f} mean-local={np.mean(local_accs):.3f} "
            f"best-local={max(local_accs):.3f} worst-local={min(local_accs):.3f}"
        ),
    )

    if verbose:
        print(proof.summary())
    return proof


if __name__ == "__main__":
    run()
