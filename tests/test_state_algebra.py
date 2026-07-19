"""Pass gate 1 (exact frozen-feature equivalence), gate 7 (deletion),
plus the required state-operation properties: associativity, commutativity,
merge-order stability, deterministic serialization."""

import numpy as np
import pytest

from rtlm_n.accumulators.state import NeuralIntelligenceState
from rtlm_n.accumulators.surfaces import objective_id_for
from rtlm_n.benchmarks.common import absorb_partitioned_head, make_nodes, merge_all
from rtlm_n.benchmarks.data import gen_tokens, partition_dirichlet
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.materialization.factory import ModelFactory

LAM = 1e-3
OBJ = objective_id_for("classification")


@pytest.fixture()
def setup(world, basis, keys, rng):
    n, n_nodes = 300, 3
    tokens = gen_tokens(rng, n, world)
    labels = world.labels(tokens)
    targets = world.one_hot(labels)
    parts = partition_dirichlet(labels, n_nodes, alpha=0.2, rng=rng)
    nodes = make_nodes(world, basis, keys, n_nodes)
    absorb_partitioned_head(nodes, parts, tokens, targets, OBJ, batches_per_node=2)
    return tokens, targets, parts, nodes


def _centralized(world, tokens, targets):
    h = augment_bias(world.small.forward(tokens))
    return ridge_solve(h.T @ h, h.T @ targets, LAM)


def test_distributed_equals_centralized(world, basis, keys, setup):
    tokens, targets, parts, nodes = setup
    merged, _ = merge_all(nodes, keys, OBJ)
    W = ModelFactory(basis).materialize_head(merged, lam=LAM).params["W"]
    np.testing.assert_allclose(W, _centralized(world, tokens, targets), atol=1e-8)


def test_merge_order_and_associativity(keys, setup, basis):
    _, _, _, nodes = setup
    s = [n.state(OBJ) for n in nodes]
    ab_c = s[0].merge(s[1]).merge(s[2])
    a_bc = s[2].merge(s[1]).merge(s[0])
    for surf in ab_c.accumulators:
        for m in ab_c.accumulators[surf]:
            np.testing.assert_allclose(
                ab_c.accumulators[surf][m], a_bc.accumulators[surf][m], atol=1e-9
            )
    assert ab_c.sample_count == a_bc.sample_count
    assert set(ab_c.lineage) == set(a_bc.lineage)


def test_contributor_removal_equals_retraining(world, basis, keys, setup):
    tokens, targets, parts, nodes = setup
    merged, engine = merge_all(nodes, keys, OBJ)
    removed = engine.remove_contributor("node-1")
    W_removed = ModelFactory(basis).materialize_head(removed, lam=LAM).params["W"]
    keep = np.concatenate([p for i, p in enumerate(parts) if i != 1])
    W_retrain = _centralized(world, tokens[keep], targets[keep])
    np.testing.assert_allclose(W_removed, W_retrain, atol=1e-8)


def test_event_deletion_equals_retraining(world, basis, keys, setup):
    tokens, targets, parts, nodes = setup
    leaf = nodes[0]._leaves[(OBJ, "default")][0]
    commit = leaf.event_commitments[0]
    assert nodes[0].delete_events(commit, OBJ) == 1
    merged, _ = merge_all(nodes, keys, OBJ)
    W = ModelFactory(basis).materialize_head(merged, lam=LAM).params["W"]
    h = augment_bias(world.small.forward(tokens))
    H = h.T @ h - leaf.accumulators["head"]["H"]
    R = h.T @ targets - leaf.accumulators["head"]["R"]
    np.testing.assert_allclose(W, ridge_solve(H, R, LAM), atol=1e-8)


def test_serialization_roundtrip_preserves_hash(keys, setup):
    _, _, _, nodes = setup
    s = nodes[0].state(OBJ)
    restored = NeuralIntelligenceState.from_bytes(s.to_bytes())
    assert restored.state_hash == s.state_hash
    assert restored.signature == s.signature
    assert restored.guarantee == s.guarantee


def test_scale_and_project(keys, setup):
    _, _, _, nodes = setup
    s = nodes[0].state(OBJ)
    doubled = s.scale(2.0)
    np.testing.assert_allclose(
        doubled.accumulators["head"]["H"], 2.0 * s.accumulators["head"]["H"]
    )
    proj = s.project(("head",))
    assert set(proj.accumulators) == {"head"}
    with pytest.raises(Exception):
        s.project(("nonexistent",))


def test_sketch_reports_declared_error(keys, setup):
    _, _, _, nodes = setup
    s = nodes[0].state(OBJ)
    sk = s.sketch(max_rank=8)
    assert sk.guarantee.value == "SKETCHED-BOUNDED"
    err_keys = [k for k in sk.approximation if k.startswith("sketch:")]
    assert err_keys, "sketch must declare its approximation error"
    assert all(0.0 <= sk.approximation[k] <= 1.0 for k in err_keys)
