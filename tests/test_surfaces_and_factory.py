"""Pass gates 2, 3, 6, 8, 9: compact transmission, one-state-many-models,
approximation reporting, internal surfaces beyond the head, and
materialization compute below independent runs."""

import numpy as np

from rtlm_n.accumulators.surfaces import (
    linearization_error,
    lowrank_head_stats,
    objective_id_for,
)
from rtlm_n.bases.basis import adapter_delta, build_basis
from rtlm_n.benchmarks.common import absorb_partitioned_head, make_nodes, merge_all
from rtlm_n.benchmarks.data import gen_tokens, partition_iid
from rtlm_n.core.guarantees import GuaranteeClass
from rtlm_n.core.linalg import ridge_solve
from rtlm_n.materialization.factory import ModelFactory
from rtlm_n.metering.meter import EnergyMeter

OBJ = objective_id_for("classification")


def test_state_is_compact_not_raw_data(world, basis, keys, rng):
    n = 500
    tokens = gen_tokens(rng, n, world)
    targets = world.one_hot(world.labels(tokens))
    nodes = make_nodes(world, basis, keys, 2)
    absorb_partitioned_head(nodes, partition_iid(n, 2, rng), tokens, targets, OBJ)
    state = nodes[0].state(OBJ)
    raw_bytes = tokens.nbytes + targets.nbytes
    assert state.nbytes() < raw_bytes / 3
    # growing the data does not grow the state
    more = gen_tokens(rng, n, world)
    nodes[0].absorb_head_batch(more, world.one_hot(world.labels(more)), objective_id=OBJ)
    assert abs(nodes[0].state(OBJ).nbytes() - state.nbytes()) < 2048


def test_many_models_from_one_state_without_data(world, basis, keys, rng):
    n = 400
    tokens = gen_tokens(rng, n, world)
    targets = world.one_hot(world.labels(tokens))
    nodes = make_nodes(world, basis, keys, 2)
    absorb_partitioned_head(nodes, partition_iid(n, 2, rng), tokens, targets, OBJ)
    merged, engine = merge_all(nodes, keys, OBJ)

    factory = ModelFactory(basis, meter=EnergyMeter())
    models = factory.materialize_many(merged, lams=np.logspace(-4, 1, 50))
    assert len(models) == 50
    assert factory.meter.flops.get("representation", 0.0) == 0.0
    # distinct variants are genuinely different models
    assert not np.allclose(models[0].params["W"], models[-1].params["W"])
    # tenant-exclusion variant == retraining without that tenant
    excl = merged.remove(engine.contributions["node-1"])
    W_excl = factory.materialize_head(excl, lam=1e-3).params["W"]
    W_only0 = factory.materialize_head(nodes[0].state(OBJ), lam=1e-3).params["W"]
    np.testing.assert_allclose(W_excl, W_only0, atol=1e-8)
    # materialization compute is far below one representation pass
    assert factory.meter.total_flops() < world.small.flops_per_example * n / 10


def test_lowrank_rank_truncation_is_exact_subsolve(world, keys, rng):
    basis = build_basis(world.small, "random-orthogonal", seed=3, ru=4, rv=8)
    tokens = gen_tokens(rng, 300, world)
    targets = world.one_hot(world.labels(tokens))
    W0 = np.zeros((world.n_classes, world.small.d_model))
    node = make_nodes(world, basis, keys, 1)[0]
    node.absorb_lowrank_batch(tokens, targets, W0, objective_id=OBJ)
    merged, _ = merge_all([node], keys, OBJ)
    factory = ModelFactory(basis)
    m = factory.materialize_lowrank_head(merged, W0, lam=1e-3, rank=3)
    Hz = merged.accumulators["lowrank-head"]["H"][:3, :3]
    Rz = merged.accumulators["lowrank-head"]["R"][:3, :]
    C_direct = ridge_solve(Hz, Rz, 1e-3).T
    np.testing.assert_allclose(m.params["C"][:, :3], C_direct, atol=1e-9)
    assert np.all(m.params["C"][:, 3:] == 0.0)


def test_quantization_weakens_guarantee(world, basis, keys, rng):
    tokens = gen_tokens(rng, 200, world)
    targets = world.one_hot(world.labels(tokens))
    node = make_nodes(world, basis, keys, 1)[0]
    node.absorb_head_batch(tokens, targets, objective_id=OBJ)
    merged, _ = merge_all([node], keys, OBJ)
    m = ModelFactory(basis).materialize_head(merged)
    assert m.guarantee == GuaranteeClass.EXACT_FROZEN
    q = m.quantized(bits=8)
    assert q.guarantee == GuaranteeClass.EMPIRICAL_NEURAL
    assert q.meta["quantized_bits"] == 8


def test_jacobian_surface_reports_linearization_error(world, keys, rng):
    basis = build_basis(world.small, "random-orthogonal", seed=5, ru=2, rv=3, adapter_layers=(1,))
    C_true = 0.4 * rng.standard_normal((2, 3))
    delta = adapter_delta(basis, "mlp:1", C_true)
    W_head = rng.standard_normal((world.n_classes, world.small.d_model))
    tokens = gen_tokens(rng, 120, world)
    targets = world.small.forward(tokens, mlp_deltas={1: delta}) @ W_head.T

    node = make_nodes(world, basis, keys, 1)[0]
    node.absorb_jacobian_batch(tokens, targets, W_head, surface="mlp:1", objective_id=OBJ)
    merged, _ = merge_all([node], keys, OBJ)
    assert merged.guarantee == GuaranteeClass.LINEARIZED_BOUNDED

    factory = ModelFactory(basis)
    c_star = ridge_solve(
        merged.accumulators["jacobian:mlp:1"]["G"], merged.accumulators["jacobian:mlp:1"]["g"], 1e-4
    ).ravel()
    err = linearization_error(world.small, basis, "mlp:1", tokens[:40], W_head, c_star)
    model = factory.materialize_adapter(merged, "mlp:1", lam=1e-4, linearization_error=err)
    assert model.meta["linearization_error"] is not None
    assert model.guarantee == GuaranteeClass.LINEARIZED_BOUNDED
    # the internal surface recovers the internal shift an output head cannot
    rel = np.linalg.norm(model.params["c"] - C_true.ravel()) / np.linalg.norm(C_true)
    assert rel < 0.5


def test_residual_surface_improves_student(world, basis, keys, rng):
    from rtlm_n.core.linalg import augment_bias

    tokens = gen_tokens(rng, 500, world)
    teacher_logits = world.true_logits(tokens)
    W0 = rng.standard_normal((world.small.d_model + 1, world.n_classes)) * 0.1
    node = make_nodes(world, basis, keys, 1)[0]
    h = node.extract_features(tokens)
    student = augment_bias(h) @ W0
    node.absorb_residual_batch(tokens, teacher_logits, student, objective_id=OBJ, h=h)
    merged, _ = merge_all([node], keys, OBJ)
    corr = ModelFactory(basis).materialize_head(merged, surface="residual-head")

    test = gen_tokens(rng, 300, world)
    ht = world.small.forward(test)
    y = world.labels(test)
    base_acc = float(((augment_bias(ht) @ W0).argmax(1) == y).mean())
    corrected = augment_bias(ht) @ W0 + corr.predict(ht)
    assert float((corrected.argmax(1) == y).mean()) > base_acc
