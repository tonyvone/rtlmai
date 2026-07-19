"""Pass gates 10-12 (escalation declines, energy accounting) and gate 14
(backbone migration is never called exact)."""

import numpy as np

from rtlm_n.accumulators.surfaces import objective_id_for
from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.common import make_nodes, merge_all
from rtlm_n.benchmarks.data import fit_small_head, gen_tokens
from rtlm_n.core.guarantees import GuaranteeClass
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import augment_bias
from rtlm_n.distributed.merge import MergeEngine
from rtlm_n.distributed.node import EdgeNode
from rtlm_n.inference.cascade import CascadeRuntime
from rtlm_n.materialization.factory import ModelFactory
from rtlm_n.metering.meter import EnergyMeter, break_even_requests, verified_outcomes_per_joule
from rtlm_n.migration.transport import fit_bridge, transport_head, transport_state
from rtlm_n.models.backbone import TinyTransformer

OBJ = objective_id_for("classification")


def test_cascade_escalation_declines_and_state_grows(world, basis, keys):
    rng = np.random.default_rng(7)
    W_task = rng.standard_normal((world.small.d_model + 1, world.n_classes)) * 2.0

    def true_logits_of(toks):
        return augment_bias(world.small.forward(toks)) @ W_task

    boot = gen_tokens(rng, 50, world)
    W0 = fit_small_head(world, boot, world.one_hot(true_logits_of(boot).argmax(1)))
    node = EdgeNode("edge", world.small, basis, keys, b"s")
    runtime = CascadeRuntime(
        node=node,
        engine=MergeEngine(keys),
        factory=ModelFactory(basis),
        small_head_W=W0,
        teacher_model=world.large,
        teacher_fn=lambda toks: 2.0 * true_logits_of(toks),
        confidence_threshold=0.55,
        rematerialize_every=24,
    )
    batches = [gen_tokens(rng, 16, world) for _ in range(60)]
    labels = [true_logits_of(b).argmax(1) for b in batches]
    history = runtime.run(batches, labels, window=10)

    assert history[-1].escalation_rate < history[0].escalation_rate - 0.05
    assert history[-1].state_samples > history[0].state_samples
    assert history[-1].joules_per_task < history[0].joules_per_task
    late_acc = np.mean([h.accuracy for h in history[-2:]])
    early_acc = np.mean([h.accuracy for h in history[:2]])
    assert late_acc >= early_acc - 0.05


def test_energy_meter_and_break_even_math():
    m = EnergyMeter()
    from rtlm_n.metering.meter import CENTRAL, EDGE

    m.charge_flops("inference_small", 1e9, EDGE)
    m.charge_call("teacher_call", CENTRAL, payload_bytes=1000.0, count=3)
    assert m.total_flops() == 1e9
    assert m.calls["teacher_call"] == 3
    assert m.total_joules() > 3 * CENTRAL.joules_per_call
    assert verified_outcomes_per_joule(10, m) == 10 / m.total_joules()
    assert break_even_requests(1.0, 0.1, 9.0) == 10.0
    assert break_even_requests(0.1, 1.0, 9.0) == float("inf")


def test_migration_is_empirical_never_exact(world, basis, keys, rng):
    v2 = TinyTransformer(seed=777, d_model=48, n_heads=4, n_layers=2, d_ff=96, name="v2")
    tokens = gen_tokens(rng, 500, world)
    targets = world.one_hot(world.labels(tokens))
    node = make_nodes(world, basis, keys, 1)[0]
    node.absorb_head_batch(tokens, targets, objective_id=OBJ)
    merged, _ = merge_all([node], keys, OBJ)
    m_v1 = ModelFactory(basis).materialize_head(merged)

    anchors = gen_tokens(rng, 300, world)
    bridge = fit_bridge(world.small, v2, anchors)
    assert bridge.residual > 0  # bridges are lossy and say so

    m_t = transport_head(m_v1, bridge)
    assert m_t.guarantee == GuaranteeClass.EMPIRICAL_NEURAL
    assert m_t.meta["bridge_residual"] == bridge.residual

    s_t = transport_state(merged, bridge)
    assert s_t.guarantee == GuaranteeClass.EMPIRICAL_NEURAL
    assert s_t.base_model_id == v2.base_model_id
    assert "bridge_residual" in s_t.approximation

    # transported intelligence beats a cold reset on the new backbone
    test = gen_tokens(rng, 300, world)
    y = world.labels(test)
    h2 = v2.forward(test)
    acc_t = float((m_t.predict(h2).argmax(1) == y).mean())
    W_rand = np.random.default_rng(1).standard_normal((v2.d_model + 1, world.n_classes))
    acc_reset = float(((augment_bias(h2) @ W_rand).argmax(1) == y).mean())
    assert acc_t > acc_reset


def test_transported_state_merges_with_native_states(world, basis, keys, rng):
    """After migration, the transported head state keeps accumulating with
    states built natively on the new backbone."""
    import dataclasses

    v2 = TinyTransformer(seed=777, d_model=48, n_heads=4, n_layers=2, d_ff=96, name="v2")
    tokens = gen_tokens(rng, 300, world)
    targets = world.one_hot(world.labels(tokens))
    node_v1 = make_nodes(world, basis, keys, 1)[0]
    node_v1.absorb_head_batch(tokens, targets, objective_id=OBJ)
    merged_v1, _ = merge_all([node_v1], keys, OBJ)
    bridge = fit_bridge(world.small, v2, gen_tokens(rng, 300, world))
    s_t = transport_state(merged_v1, bridge)

    basis_v2 = build_basis(v2, "random-orthogonal", seed=0)
    node_v2 = EdgeNode("native-v2", v2, basis_v2, keys, b"k2")
    fresh = gen_tokens(rng, 200, world)
    node_v2.absorb_head_batch(fresh, world.one_hot(world.labels(fresh)), objective_id=OBJ)
    native = node_v2.state(OBJ)

    combined = s_t.merge(native)
    assert combined.sample_count == s_t.sample_count + native.sample_count
    assert combined.guarantee == GuaranteeClass.EMPIRICAL_NEURAL  # weaker label wins
    W = ModelFactory().materialize_head(combined).params["W"]
    assert np.all(np.isfinite(W))
