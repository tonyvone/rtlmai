"""Experiment 0: exact frozen-head state (EXACT-FROZEN).

Proves the original-RTLM property over deep representations: distributed
merged-state materialization is floating-point-identical to centralized
training under arbitrary partitions, merge orders, node removal, and
event deletion - and the transmitted state is far smaller than raw data.
"""

from __future__ import annotations

import numpy as np

from rtlm_n.accumulators.surfaces import head_stats, objective_id_for
from rtlm_n.assurance.proof import ProofPackage
from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.common import absorb_partitioned_head, make_nodes, merge_all
from rtlm_n.benchmarks.data import make_world, gen_tokens, partition_dirichlet
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.materialization.factory import ModelFactory

LAM = 1e-3


def run(seed: int = 0, n: int = 800, n_nodes: int = 5, verbose: bool = True) -> ProofPackage:
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    basis = build_basis(world.small, "random-orthogonal", seed=seed)
    keys = KeyRegistry()
    objective = objective_id_for("classification")
    proof = ProofPackage("exp0-exact-frozen-head")

    tokens = gen_tokens(rng, n, world)
    labels = world.labels(tokens)
    targets = world.one_hot(labels)

    # ---- centralized reference (raw data pooled in one place) ----------
    h_all = augment_bias(world.small.forward(tokens))
    W_central = ridge_solve(h_all.T @ h_all, h_all.T @ targets, LAM)

    # ---- distributed, severely non-IID, multiple event batches ---------
    parts = partition_dirichlet(labels, n_nodes, alpha=0.1, rng=rng)
    nodes = make_nodes(world, basis, keys, n_nodes)
    absorb_partitioned_head(nodes, parts, tokens, targets, objective, batches_per_node=3)

    factory = ModelFactory(basis)
    merged, engine = merge_all(nodes, keys, objective)
    W_merged = factory.materialize_head(merged, lam=LAM).params["W"]
    proof.add_equivalence("distributed == centralized (non-IID partition)", W_central, W_merged)

    # ---- merge-order invariance ----------------------------------------
    order = list(rng.permutation(n_nodes))
    merged_perm, _ = merge_all(nodes, keys, objective, order=order)
    W_perm = factory.materialize_head(merged_perm, lam=LAM).params["W"]
    proof.add_equivalence("merge-order invariance", W_merged, W_perm)

    # ---- node removal == retraining without that node ------------------
    removed = engine.remove_contributor("node-2")
    W_removed = factory.materialize_head(removed, lam=LAM).params["W"]
    keep = np.concatenate([p for i, p in enumerate(parts) if i != 2])
    hk = augment_bias(world.small.forward(tokens[keep]))
    W_retrain = ridge_solve(hk.T @ hk, hk.T @ targets[keep], LAM)
    proof.add_equivalence("node removal == retained-data retraining", W_retrain, W_removed)

    # ---- event deletion == retraining without those events -------------
    victim = nodes[0]
    key = (objective, "default")
    del_leaf = victim._leaves[key][0]
    del_commit = del_leaf.event_commitments[0]
    victim.delete_events(del_commit, objective)
    merged_del, _ = merge_all(nodes, keys, objective)
    W_del = factory.materialize_head(merged_del, lam=LAM).params["W"]
    # centralized retrain without the deleted events
    H_ref = h_all.T @ h_all - del_leaf.accumulators["head"]["H"]
    R_ref = h_all.T @ targets - del_leaf.accumulators["head"]["R"]
    W_ref = ridge_solve(H_ref, R_ref, LAM)
    proof.add_equivalence("event deletion == retained-event retraining", W_ref, W_del)

    # ---- compactness: state bytes << raw data bytes --------------------
    state_bytes = merged.nbytes()
    raw_bytes = tokens.nbytes + targets.nbytes + h_all.nbytes
    proof.add_metric(
        "state compression ratio (raw/state)",
        raw_bytes / state_bytes,
        threshold=3.0,
        higher_is_better=True,
        guarantee="EXACT-FROZEN",
        note=f"state={state_bytes}B raw={raw_bytes}B; state size is O(d^2), independent of n",
    )

    if verbose:
        print(proof.summary())
    return proof


if __name__ == "__main__":
    run()
