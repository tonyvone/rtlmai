"""Pass gates 5/13 and mutation-defect detection (gate 15): incompatible
bases refused, duplicates refused, tampering refused, unsafe numerics
labeled UNSUPPORTED and refused at materialization."""

import numpy as np
import pytest

from rtlm_n.accumulators.state import StateError
from rtlm_n.accumulators.surfaces import objective_id_for
from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.data import gen_tokens, make_world
from rtlm_n.core.guarantees import GuaranteeClass
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.distributed.merge import (
    DuplicateStateError,
    IncompatibleStateError,
    MergeEngine,
    TamperedStateError,
)
from rtlm_n.distributed.node import EdgeNode
from rtlm_n.materialization.factory import ModelFactory, UnsupportedMaterializationError

OBJ = objective_id_for("lowrank-classification")


def _node_with_state(world, basis, keys, node_id, seed, surface="lowrank"):
    node = EdgeNode(node_id, world.small, basis, keys, f"k-{node_id}".encode())
    rng = np.random.default_rng(seed)
    toks = gen_tokens(rng, 60, world)
    targets = world.one_hot(world.labels(toks))
    W0 = np.zeros((world.n_classes, world.small.d_model))
    node.absorb_lowrank_batch(toks, targets, W0, objective_id=OBJ)
    return node


def test_incompatible_basis_versions_refused(world, keys):
    basis_a = build_basis(world.small, "random-orthogonal", seed=1)
    basis_b = build_basis(world.small, "random-orthogonal", seed=2)
    assert basis_a.basis_id != basis_b.basis_id
    node_a = _node_with_state(world, basis_a, keys, "a", 1)
    node_b = _node_with_state(world, basis_b, keys, "b", 2)
    engine = MergeEngine(keys)
    engine.submit(node_a.state(OBJ))
    with pytest.raises(IncompatibleStateError):
        engine.submit(node_b.state(OBJ))


def test_different_backbone_refused(world, keys):
    other_world = make_world(seed=99)
    basis_a = build_basis(world.small, "random-orthogonal", seed=1)
    basis_b = build_basis(other_world.small, "random-orthogonal", seed=1)
    node_a = _node_with_state(world, basis_a, keys, "a", 1)
    node_b = _node_with_state(other_world, basis_b, keys, "b", 2)
    engine = MergeEngine(keys)
    engine.submit(node_a.state(OBJ))
    with pytest.raises(IncompatibleStateError):
        engine.submit(node_b.state(OBJ))


def test_duplicate_contribution_refused(world, basis, keys):
    node = _node_with_state(world, basis, keys, "a", 1)
    engine = MergeEngine(keys)
    engine.submit(node.state(OBJ))
    with pytest.raises(DuplicateStateError):
        engine.submit(node.state(OBJ))


def test_tampered_state_refused(world, basis, keys):
    node = _node_with_state(world, basis, keys, "a", 1)
    state = node.state(OBJ)
    # adversary inflates its own evidence after signing
    state.accumulators["lowrank-head"]["R"][0, 0] += 100.0
    engine = MergeEngine(keys)
    with pytest.raises(TamperedStateError):
        engine.submit(state)


def test_unknown_signer_refused(world, basis):
    keys_a, keys_b = KeyRegistry(), KeyRegistry()
    node = _node_with_state(world, basis, keys_a, "a", 1)
    engine = MergeEngine(keys_b)  # engine has no key for node "a"
    with pytest.raises(TamperedStateError):
        engine.submit(node.state(OBJ))


def test_nonfinite_state_is_unsupported(world, basis, keys):
    node = _node_with_state(world, basis, keys, "a", 1)
    state = node.state(OBJ)
    state.accumulators["lowrank-head"]["H"][0, 0] = np.nan
    assert not state.verify()
    with pytest.raises(UnsupportedMaterializationError):
        ModelFactory(basis).materialize_lowrank_head(
            state, np.zeros((world.n_classes, world.small.d_model))
        )


def test_ill_conditioned_state_is_unsupported(world, basis, keys):
    node = EdgeNode("a", world.small, basis, keys, b"k")
    rng = np.random.default_rng(0)
    toks = gen_tokens(rng, 2, world)  # rank-deficient evidence
    targets = world.one_hot(world.labels(toks))
    node.absorb_head_batch(toks, targets, objective_id=OBJ)
    state = node.state(OBJ)
    with pytest.raises(UnsupportedMaterializationError):
        ModelFactory(basis).materialize_head(state, lam=0.0)  # unregularized


def test_unsupported_label_blocks_materialization(world, basis, keys):
    import dataclasses

    node = _node_with_state(world, basis, keys, "a", 1)
    state = dataclasses.replace(node.state(OBJ), guarantee=GuaranteeClass.UNSUPPORTED)
    with pytest.raises(UnsupportedMaterializationError):
        ModelFactory(basis).materialize_lowrank_head(
            state, np.zeros((world.n_classes, world.small.d_model))
        )


def test_direct_merge_of_incompatible_objectives_refused(world, basis, keys):
    node = EdgeNode("a", world.small, basis, keys, b"k")
    rng = np.random.default_rng(0)
    toks = gen_tokens(rng, 40, world)
    targets = world.one_hot(world.labels(toks))
    node.absorb_head_batch(toks, targets, objective_id="obj-one")
    node.absorb_head_batch(toks, targets, objective_id="obj-two")
    with pytest.raises(StateError):
        node.state("obj-one").merge(node.state("obj-two"))
