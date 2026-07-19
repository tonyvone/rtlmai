"""Experiment 5: continual inference accretion and energy break-even.

A repeated enterprise workload flows through a small/large cascade. Every
escalation is absorbed into the intelligence state; the correction head is
periodically rematerialized. The core curves:

    teacher calls          declining
    energy per task        declining
    quality                stable or improving
    accumulated state      increasing

plus the measured break-even: lifetime energy saved vs learning cost.
"""

from __future__ import annotations

import numpy as np

from rtlm_n.assurance.proof import ProofPackage
from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.data import fit_small_head, gen_tokens, make_world
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.distributed.merge import MergeEngine
from rtlm_n.distributed.node import EdgeNode
from rtlm_n.inference.cascade import CascadeRuntime
from rtlm_n.materialization.factory import ModelFactory
from rtlm_n.metering.meter import CENTRAL, EDGE, break_even_requests, verified_outcomes_per_joule


def run(seed: int = 0, n_batches: int = 120, batch: int = 16, verbose: bool = True) -> ProofPackage:
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    basis = build_basis(world.small, "random-orthogonal", seed=seed)
    keys = KeyRegistry()
    proof = ProofPackage("exp5-continual-inference-accretion")

    # The workload's ground truth is learnable by the edge model (a hidden
    # linear rule over its own frozen representation): what the edge lacks
    # at t=0 is EVIDENCE, not capacity. The teacher is a verified workflow
    # (large model + verification) that returns confirmed answers at
    # central-datacenter cost.
    W_task = rng.standard_normal((world.small.d_model + 1, world.n_classes)) * 2.0

    from rtlm_n.core.linalg import augment_bias

    def true_logits_of(toks: np.ndarray) -> np.ndarray:
        return augment_bias(world.small.forward(toks)) @ W_task

    def true_labels_of(toks: np.ndarray) -> np.ndarray:
        return true_logits_of(toks).argmax(axis=1)

    def verified_teacher(toks: np.ndarray) -> np.ndarray:
        # the verified workflow returns confirmed graded scores
        return 2.0 * true_logits_of(toks)

    # deliberately weak starting head: tiny bootstrap set
    boot = gen_tokens(rng, 60, world)
    W0 = fit_small_head(world, boot, world.one_hot(true_labels_of(boot)))

    node = EdgeNode("edge-0", world.small, basis, keys, b"edge-secret")
    engine = MergeEngine(keys)
    factory = ModelFactory(basis)
    runtime = CascadeRuntime(
        node=node,
        engine=engine,
        factory=factory,
        small_head_W=W0,
        teacher_model=world.large,
        teacher_fn=verified_teacher,
        confidence_threshold=0.55,
        rematerialize_every=32,
    )

    batches = [gen_tokens(rng, batch, world) for _ in range(n_batches)]
    labels = [true_labels_of(b) for b in batches]
    history = runtime.run(batches, labels, window=10)

    first, last = history[0], history[-1]
    early_acc = np.mean([h.accuracy for h in history[:3]])
    late_acc = np.mean([h.accuracy for h in history[-3:]])

    proof.add_metric(
        "teacher escalation declines",
        first.escalation_rate - last.escalation_rate,
        threshold=0.1,
        higher_is_better=True,
        note=f"first-window={first.escalation_rate:.2f} last-window={last.escalation_rate:.2f}",
    )
    proof.add_metric(
        "quality stable or improving",
        late_acc - early_acc,
        threshold=-0.03,
        higher_is_better=True,
        note=f"early={early_acc:.3f} late={late_acc:.3f}",
    )
    proof.add_metric(
        "energy per task declines",
        first.joules_per_task - last.joules_per_task,
        threshold=0.0,
        higher_is_better=True,
        note=f"first={first.joules_per_task:.3g}J last={last.joules_per_task:.3g}J",
    )
    proof.add_metric(
        "intelligence state accumulates",
        last.state_samples - first.state_samples,
        threshold=1,
        higher_is_better=True,
        note=f"{first.state_samples} -> {last.state_samples} absorbed residuals",
    )

    # ---- energy break-even ---------------------------------------------
    e_large = (
        world.large.flops_per_example * CENTRAL.joules_per_flop + CENTRAL.joules_per_call
    )
    e_small = world.small.flops_per_example * EDGE.joules_per_flop
    e_learning = (
        node.meter.total_joules() + factory.meter.total_joules() + engine.meter.total_joules()
    )
    n_be = break_even_requests(e_large, e_small, e_learning)
    total_tasks = sum(h.tasks for h in history)
    # escalations avoided vs a static cascade pinned at the initial rate
    avoided = sum(
        max(0.0, first.escalation_rate - h.escalation_rate) * h.tasks for h in history
    )
    net_saved = avoided * (e_large - e_small) - e_learning
    all_meters = runtime.meter.merged_with(node.meter).merged_with(factory.meter)
    vopj = verified_outcomes_per_joule(
        int(sum(h.accuracy * h.tasks for h in history)), all_meters
    )
    # baseline: teacher-always for the same workload
    vopj_teacher_always = 1.0 / e_large

    proof.add_metric(
        "break-even repaid within workload (avoided/needed)",
        avoided / n_be if np.isfinite(n_be) else 0.0,
        threshold=1.0,
        higher_is_better=True,
        note=f"break-even N={n_be:.0f} tasks, avoided={avoided:.0f}, net saved={net_saved:.3g}J",
    )
    proof.add_metric(
        "verified outcomes per joule vs teacher-always (x)",
        vopj / vopj_teacher_always,
        threshold=1.0,
        higher_is_better=True,
        note=f"rtlmn={vopj:.3g}/J teacher-always={vopj_teacher_always:.3g}/J",
    )

    if verbose:
        print(proof.summary())
        print("  window  esc-rate  accuracy  J/task    teacher-calls  state-samples")
        for h in history:
            print(
                f"  {h.window:>6}  {h.escalation_rate:>8.2f}  {h.accuracy:>8.3f}  "
                f"{h.joules_per_task:>8.3g}  {h.cumulative_teacher_calls:>13}  {h.state_samples:>13}"
            )
    return proof


if __name__ == "__main__":
    run()
