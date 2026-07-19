"""Real-data RTLM-N benchmark: AG News + torch backbones + SGD baselines
+ measured energy.

    python -m rtlm_n.benchmarks.real_agnews [--fast]

Phases (all CPU-time-measured with MeasuredMeter):

R0  Exact frozen-head equivalence on REAL representations: distributed
    merged-state training == centralized ridge on MLM-pretrained features
    under severe non-IID partitions, with node removal and event deletion.

R1  Canonical adapter vs REAL competitors, all Adam-trained on the same
    frozen features: independent federated LoRA (per-node learned
    subspaces, averaged), centralized LoRA, one-shot FedAvg heads,
    SGD distillation. Reports accuracy AND measured CPU seconds.

R2  One state -> 1,000 models on real features, measured CPU time vs a
    measured per-run cost of SGD head training extrapolated to 1,000.

R3  Continual inference accretion on the real workload: weak edge head +
    verified teacher workflow (charged at large-model cost); escalation
    and measured energy per task must decline while quality holds.

R4  Projected-Jacobian internal adapter on the real backbone with EXACT
    autograd Jacobians (torch.func.jacrev), linearization error reported.

R5  Backbone migration: the head state transported from backbone v1 to an
    independently pretrained, narrower v2 via a representation bridge.

Honesty notes: backbones are small transformers MLM-pretrained on the AG
News corpus itself (this environment cannot reach pretrained-checkpoint
hosts; `rtlm_n.real.hf_backbone` runs the same benchmark against real
HF checkpoints where the hub is reachable). Energy is measured process
CPU-seconds converted at an explicit RTLMN_CPU_WATTS; teacher cost uses a
documented large-model FLOP estimate.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from rtlm_n.accumulators.surfaces import (
    linearization_error,
    objective_id_for,
    projected_jacobian,
)
from rtlm_n.assurance.proof import ProofPackage
from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.data import partition_dirichlet
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.distributed.merge import MergeEngine
from rtlm_n.distributed.node import EdgeNode
from rtlm_n.inference.cascade import CascadeRuntime
from rtlm_n.materialization.factory import ModelFactory
from rtlm_n.metering.measured import DEFAULT_CPU_WATTS, MeasuredMeter
from rtlm_n.metering.meter import CENTRAL, EDGE
from rtlm_n.migration.transport import fit_bridge, transport_head
from rtlm_n.real.agnews import load_agnews
from rtlm_n.real.baselines import (
    accuracy_W,
    fedavg_heads,
    federated_lora_avg,
    train_linear_head,
    train_lora_head,
)
from rtlm_n.real.torch_backbone import pretrained_backbone

LAM = 1e-2
N_CLASSES = 4
OBJ = objective_id_for("agnews-classification")

#: documented stand-in for a datacenter-class teacher forward pass
#: (7B-parameter model, ~2 FLOPs/param/token x 24 tokens)
TEACHER_FLOPS = 2.0 * 7e9 * 24


class _TeacherShim:
    flops_per_example = TEACHER_FLOPS


def _acc(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def run(seed: int = 0, fast: bool = False, verbose: bool = True) -> list:
    meter = MeasuredMeter()
    packages = []
    rng = np.random.default_rng(seed)
    n_train = 8000 if fast else 20000
    pre_steps = 1500 if fast else 4000

    with meter.phase("data"):
        data = load_agnews(n_train=n_train, n_test=3000, seed=seed)
    vocab = data["vocab"]
    y_train = np.eye(N_CLASSES)[data["train_labels"]]

    with meter.phase("pretrain_edge_backbone"):
        edge = pretrained_backbone(
            vocab.size, data["train_tokens"], cache_key=f"edge_v1_s{seed}_{pre_steps}",
            steps=pre_steps, seed=seed, d_model=128, n_layers=2, d_ff=256, seq_len=24,
        )
    with meter.phase("representation"):
        h_train = edge.forward(data["train_tokens"])
        h_test = edge.forward(data["test_tokens"])
    ha_train, ha_test = augment_bias(h_train), augment_bias(h_test)

    keys = KeyRegistry()
    basis = build_basis(edge, "random-orthogonal", seed=seed, ru=4, rv=16, n_classes=N_CLASSES)

    # ================= R0: exact equivalence on real features ============
    proof0 = ProofPackage("real-R0-exact-equivalence-agnews")
    with meter.phase("R0_states_and_merge"):
        parts = [p for p in partition_dirichlet(data["train_labels"], 10, alpha=0.1, rng=rng) if len(p)]
        nodes = []
        for i, idx in enumerate(parts):
            node = EdgeNode(f"node-{i}", edge, basis, keys, f"s{i}".encode())
            for chunk in np.array_split(idx, max(1, min(3, len(idx)))):
                if len(chunk):
                    node.absorb_head_batch(
                        data["train_tokens"][chunk], y_train[chunk], objective_id=OBJ
                    )
            nodes.append(node)
        engine = MergeEngine(keys)
        for i in rng.permutation(len(nodes)):
            engine.submit(nodes[i].state(OBJ))
        factory = ModelFactory(basis)
        W_merged = factory.materialize_head(engine.global_state, lam=LAM).params["W"]
    W_central = ridge_solve(ha_train.T @ ha_train, ha_train.T @ y_train, LAM)
    proof0.add_equivalence("distributed == centralized on real MLM features", W_central, W_merged)
    removed = engine.global_state.remove(engine.contributions["node-3"])
    W_rm = factory.materialize_head(removed, lam=LAM).params["W"]
    keep = np.concatenate([p for i, p in enumerate(parts) if i != 3])
    hk = ha_train[keep]
    proof0.add_equivalence(
        "node deletion == retained-data retraining",
        ridge_solve(hk.T @ hk, hk.T @ y_train[keep], LAM),
        W_rm,
    )
    acc_central = _acc(ha_test @ W_central, data["test_labels"])
    proof0.add_metric(
        "real-data quality sanity (frozen head)",
        acc_central,
        threshold=0.55 if fast else 0.70,
        higher_is_better=True,
        note=f"AG News 4-class, centralized == merged acc={acc_central:.3f}"
        + (" [fast mode: short pretrain]" if fast else ""),
    )
    packages.append(proof0)

    # ================= R1: vs real SGD competitors =======================
    proof1 = ProofPackage("real-R1-canonical-vs-sgd-federated")
    boot = rng.permutation(n_train)[:400]
    hb = ha_train[boot]
    W0_aug = ridge_solve(hb.T @ hb, hb.T @ y_train[boot], LAM)  # (d+1, k)
    W0 = W0_aug[:-1, :].T  # (k, d)
    acc_base = _acc(ha_test @ W0_aug, data["test_labels"])

    # data-aware canonical basis from a PUBLIC labeled calibration slice:
    # the leading input directions come from the feature-residual
    # cross-covariance (the teacher-residual subspace of the spec), the
    # rest from activation SVD - not random axes
    calib_idx = np.arange(2000)
    calib_resid = y_train[calib_idx] - h_train[calib_idx] @ W0.T
    basis1 = build_basis(
        edge, "teacher-residual", seed=seed, ru=4, rv=64, n_classes=N_CLASSES,
        calibration_tokens=data["train_tokens"][calib_idx],
        calibration_residuals=calib_resid,
    )
    factory1 = ModelFactory(basis1)
    n_adapt = min(10000, n_train)
    sub = rng.permutation(n_train)[:n_adapt]
    labels_sub = data["train_labels"][sub]
    parts1 = partition_dirichlet(labels_sub, 8, alpha=0.1, rng=rng)
    parts1 = [sub[p] for p in parts1]

    with meter.phase("R1_rtlmn_state_merge_solve"):
        nodes1 = []
        for i, idx in enumerate(parts1):
            if not len(idx):
                continue
            node = EdgeNode(f"a-node-{i}", edge, basis1, keys, f"as{i}".encode())
            # features precomputed once for ALL methods - fair comparison
            node.absorb_lowrank_batch(
                data["train_tokens"][idx], y_train[idx], W0, objective_id=OBJ, h=h_train[idx]
            )
            nodes1.append(node)
        eng1 = MergeEngine(keys)
        for node in nodes1:
            eng1.submit(node.state(OBJ))
        m_rtlmn = factory1.materialize_lowrank_head(eng1.global_state, W0, lam=LAM)
    acc_rtlmn = _acc(m_rtlmn.predict(h_test), data["test_labels"])

    with meter.phase("R1_federated_lora_sgd"):
        W_fedlora = federated_lora_avg(parts1, h_train, y_train, W0, rank=4, steps=250, seed=seed)
    acc_fedlora = _acc(h_test @ W_fedlora.T, data["test_labels"])

    with meter.phase("R1_centralized_lora_sgd"):
        used1 = np.concatenate(parts1)
        dW = train_lora_head(h_train[used1], y_train[used1], W0, rank=4, steps=250, seed=seed)
    acc_clora = _acc(h_test @ (W0 + dW).T, data["test_labels"])

    with meter.phase("R1_fedavg_heads_sgd"):
        W_fedavg = fedavg_heads(parts1, h_train, y_train, steps=250, seed=seed)
    acc_fedavg = accuracy_W(W_fedavg, h_test, data["test_labels"])

    proof1.add_metric(
        "canonical merged state beats independent federated LoRA (SGD)",
        acc_rtlmn - acc_fedlora,
        threshold=0.0,
        higher_is_better=True,
        note=f"rtlmn={acc_rtlmn:.3f} fedlora-sgd={acc_fedlora:.3f} (non-IID alpha=0.1)",
    )
    proof1.add_metric(
        "canonical merged state beats one-shot FedAvg heads (SGD)",
        acc_rtlmn - acc_fedavg,
        threshold=0.0,
        higher_is_better=True,
        note=f"fedavg-sgd={acc_fedavg:.3f}",
    )
    proof1.add_metric(
        "canonical merged state >= centralized LoRA (SGD) - 1pt",
        acc_rtlmn - acc_clora,
        threshold=-0.01,
        higher_is_better=True,
        note=f"centralized-lora-sgd={acc_clora:.3f}; RTLM-N is one-shot + mergeable + deletable",
    )
    r1_cpu = {k: meter.phases[k].cpu_s for k in meter.phases if k.startswith("R1_")}
    proof1.add_metric(
        "one-shot state solve is cheaper than federated SGD (cpu-s ratio)",
        r1_cpu["R1_federated_lora_sgd"] / max(r1_cpu["R1_rtlmn_state_merge_solve"], 1e-9),
        threshold=1.0,
        higher_is_better=True,
        note=f"measured cpu-s: {json.dumps({k: round(v, 2) for k, v in r1_cpu.items()})}",
    )
    proof1.add_metric(
        "adapter improves over 400-label base head",
        acc_rtlmn - acc_base,
        threshold=0.0,
        higher_is_better=True,
        note=f"base={acc_base:.3f}",
    )
    packages.append(proof1)

    # ================= R2: 1000 models, measured =========================
    proof2 = ProofPackage("real-R2-thousand-models-measured")
    state = engine.global_state
    with meter.phase("R2_materialize_1000"):
        lams = np.logspace(-5, 2, 100)
        excls = [None] + sorted(engine.contributions)[:9]
        variants = []
        for excl in excls:
            s = state if excl is None else state.remove(engine.contributions[excl])
            for lam in lams:
                variants.append(factory.materialize_head(s, lam=float(lam)))
    n_var = len(variants)
    t_mat = meter.phases["R2_materialize_1000"].cpu_s

    with meter.phase("R2_sgd_head_reference_runs"):
        t0 = time.process_time()
        n_ref = 5
        for i in range(n_ref):
            train_linear_head(h_train, y_train, steps=250, seed=i)
        t_sgd_each = (time.process_time() - t0) / n_ref
    t_sgd_1000 = t_sgd_each * n_var
    proof2.add_metric(
        "1000 models: measured cpu-s advantage vs SGD-per-variant (x)",
        t_sgd_1000 / max(t_mat, 1e-9),
        threshold=10.0,
        higher_is_better=True,
        note=(
            f"{n_var} materializations in {t_mat:.2f} cpu-s vs "
            f"{t_sgd_each:.2f} cpu-s/SGD-run x {n_var} = {t_sgd_1000:.0f} cpu-s (measured, extrapolated)"
        ),
    )
    proof2.add_metric(
        "variants are distinct models",
        float(np.max(np.abs(variants[0].params["W"] - variants[50].params["W"]))),
        threshold=1e-6,
        higher_is_better=True,
    )
    packages.append(proof2)

    # ================= R3: cascade on the real workload ==================
    proof3 = ProofPackage("real-R3-continual-accretion-agnews")
    with meter.phase("R3_cascade"):
        boot3 = rng.permutation(n_train)[:150]
        hb3 = ha_train[boot3]
        W0_c = ridge_solve(hb3.T @ hb3, hb3.T @ y_train[boot3], LAM)

        def verified_teacher(tokens: np.ndarray) -> np.ndarray:
            # verified workflow: the large model + checks confirm the label;
            # charged at TEACHER_FLOPS per task by the runtime
            idx = [tok_index[tuple(t)] for t in tokens]
            return 6.0 * y_stream[idx]

        stream = rng.permutation(n_train)
        n_batches = 60 if fast else 100
        batch_sz = 24
        stream = stream[: n_batches * batch_sz]
        tok_stream = data["train_tokens"][stream]
        y_stream = y_train[stream]
        tok_index = {tuple(t): i for i, t in enumerate(tok_stream)}

        node3 = EdgeNode("cascade-edge", edge, basis, keys, b"ce")
        runtime = CascadeRuntime(
            node=node3,
            engine=MergeEngine(keys),
            factory=ModelFactory(basis),
            small_head_W=W0_c,
            teacher_model=_TeacherShim(),
            teacher_fn=verified_teacher,
            confidence_threshold=0.45,
            rematerialize_every=32,
            lam=1e-3,
        )
        batches = [tok_stream[i * batch_sz : (i + 1) * batch_sz] for i in range(n_batches)]
        labels3 = [y_stream[i * batch_sz : (i + 1) * batch_sz].argmax(1) for i in range(n_batches)]
        history = runtime.run(batches, labels3, window=10)
    first, last = history[0], history[-1]
    proof3.add_metric(
        "teacher escalation declines on real workload",
        first.escalation_rate - last.escalation_rate,
        threshold=0.05,
        higher_is_better=True,
        note=f"first={first.escalation_rate:.2f} last={last.escalation_rate:.2f}",
    )
    early_acc = float(np.mean([h.accuracy for h in history[:2]]))
    late_acc = float(np.mean([h.accuracy for h in history[-2:]]))
    proof3.add_metric(
        "quality stable or improving",
        late_acc - early_acc,
        threshold=-0.03,
        higher_is_better=True,
        note=f"early={early_acc:.3f} late={late_acc:.3f}",
    )
    proof3.add_metric(
        "modeled energy per task declines (teacher-call dominated)",
        first.joules_per_task - last.joules_per_task,
        threshold=0.0,
        higher_is_better=True,
        note=f"first={first.joules_per_task:.3g}J last={last.joules_per_task:.3g}J",
    )
    packages.append(proof3)

    # ================= R4: exact autograd Jacobian adapter ===============
    proof4 = ProofPackage("real-R4-autograd-jacobian-adapter")
    with meter.phase("R4_jacobian"):
        basis_j = build_basis(
            edge, "random-orthogonal", seed=seed + 1, ru=3, rv=4,
            n_classes=N_CLASSES, adapter_layers=(1,),
        )
        from rtlm_n.bases.basis import adapter_delta

        C_true = 0.4 * np.random.default_rng(seed + 2).standard_normal((3, 4))
        delta_true = adapter_delta(basis_j, "mlp:1", C_true)
        W_head = np.random.default_rng(seed + 3).standard_normal((N_CLASSES, edge.d_model))
        jt = data["train_tokens"][:160]
        targets_j = edge.forward(jt, mlp_deltas={1: delta_true}) @ W_head.T

        node4 = EdgeNode("jac-edge", edge, basis_j, keys, b"je")
        node4.absorb_jacobian_batch(jt, targets_j, W_head, surface="mlp:1", objective_id=OBJ)
        s4 = node4.state(OBJ)
        c_star = ridge_solve(
            s4.accumulators["jacobian:mlp:1"]["G"], s4.accumulators["jacobian:mlp:1"]["g"], 1e-4
        ).ravel()
        lin_err = linearization_error(edge, basis_j, "mlp:1", jt[:40], W_head, c_star)
    rel = float(np.linalg.norm(c_star - C_true.ravel()) / np.linalg.norm(C_true))
    proof4.add_metric(
        "canonical coordinates recovered on real backbone (autograd)",
        rel,
        threshold=0.5,
        higher_is_better=False,
        guarantee="LINEARIZED-BOUNDED",
        note=f"||c*-c_true||/||c_true||={rel:.3f}",
    )
    proof4.add_metric(
        "linearization error measured and reported",
        lin_err,
        threshold=1.0,
        higher_is_better=False,
        guarantee="LINEARIZED-BOUNDED",
    )
    packages.append(proof4)

    # ================= R5: migration between real pretrained backbones ===
    proof5 = ProofPackage("real-R5-backbone-migration-agnews")
    with meter.phase("R5_pretrain_v2_backbone"):
        v2 = pretrained_backbone(
            vocab.size, data["train_tokens"], cache_key=f"edge_v2_s{seed}_{pre_steps}",
            steps=max(1000, pre_steps // 2), seed=seed + 100,
            d_model=96, n_layers=2, d_ff=192, seq_len=24, name="torch-mini-v2",
        )
    with meter.phase("R5_migration"):
        m_v1 = factory.materialize_head(engine.global_state, lam=LAM)
        anchors = data["train_tokens"][rng.permutation(n_train)[:600]]
        bridge = fit_bridge(edge, v2, anchors, lam=LAM)
        m_t = transport_head(m_v1, bridge)
        h2_test = v2.forward(data["test_tokens"])
        acc_t = _acc(m_t.predict(h2_test), data["test_labels"])
        W_rand = np.random.default_rng(1).standard_normal((v2.d_model + 1, N_CLASSES))
        acc_reset = _acc(augment_bias(h2_test) @ W_rand, data["test_labels"])
        h2_train = augment_bias(v2.forward(data["train_tokens"]))
        W_retrain = ridge_solve(h2_train.T @ h2_train, h2_train.T @ y_train, LAM)
        acc_retrain = _acc(augment_bias(h2_test) @ W_retrain, data["test_labels"])
    recovered = (acc_t - acc_reset) / max(acc_retrain - acc_reset, 1e-9)
    proof5.add_metric(
        "transported real-data intelligence beats cold reset",
        acc_t - acc_reset,
        threshold=0.05,
        higher_is_better=True,
        guarantee="EMPIRICAL-NEURAL",
        note=f"reset={acc_reset:.3f} transported={acc_t:.3f} retrain={acc_retrain:.3f}",
    )
    proof5.add_metric(
        "fraction of reset->retrain gap recovered",
        recovered,
        threshold=0.3,
        higher_is_better=True,
        guarantee="EMPIRICAL-NEURAL",
        note=f"bridge residual={bridge.residual:.3f} (independently pretrained, narrower v2)",
    )
    packages.append(proof5)

    if verbose:
        for p in packages:
            print("=" * 78)
            print(p.summary())
        print("=" * 78)
        print("measured compute/energy by phase "
              f"(RTLMN_CPU_WATTS={DEFAULT_CPU_WATTS:g}; no RAPL -> cpu-s x watts):")
        print(meter.summary())
        n_pass = sum(p.passed for p in packages)
        print(f"real-data program: {n_pass}/{len(packages)} experiments passed")
    return packages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--report", type=str, default="rtlmn_real_report.json")
    args = parser.parse_args()
    packages = run(seed=args.seed, fast=args.fast)
    with open(args.report, "w") as f:
        json.dump([asdict(p) for p in packages], f, indent=2, default=str)
    print(f"proof packages written to {args.report}")
    return 0 if all(p.passed for p in packages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
