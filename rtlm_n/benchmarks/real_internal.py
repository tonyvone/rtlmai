"""Experiment R6: iterated Gauss-Newton vs TRUE internal LoRA (real data).

The hardest fair fight in the program. Adaptation budget: ONLY an internal
MLP adapter at layer 1 of the real backbone (the head is fixed to a weak
base head), so every method must move representations, not just re-read
them. Methods:

- base head, no adaptation (floor);
- TRUE centralized internal LoRA: Delta W1 = B A, Adam, backprop through
  the frozen backbone (the strong competitor);
- TRUE federated internal LoRA: per-node internal LoRA in per-node random
  subspaces, averaged (the crowded field's approach, non-IID);
- RTLM-N one-shot Gauss-Newton state (the linearization band);
- RTLM-N ITERATED Gauss-Newton: re-linearize -> accumulate -> merge ->
  step, per-round exact merge + per-round measured linearization error.

Gates: iteration must escape the one-shot band; the canonical merged state
must beat federated internal-LoRA averaging under non-IID; costs are
measured cpu-seconds. Centralized SGD LoRA's number is reported unfiltered
- whatever it is.

    python -m rtlm_n.benchmarks.real_internal [--fast]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from rtlm_n.assurance.proof import ProofPackage
from rtlm_n.bases.basis import build_basis
from rtlm_n.benchmarks.data import partition_dirichlet
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.materialization.iterated import iterated_gauss_newton
from rtlm_n.metering.measured import DEFAULT_CPU_WATTS, MeasuredMeter
from rtlm_n.real.agnews import load_agnews
from rtlm_n.real.baselines import train_internal_lora
from rtlm_n.real.torch_backbone import pretrained_backbone

LAM = 1e-2
N_CLASSES = 4
LAYER = 1
SURFACE = f"mlp:{LAYER}"


def _acc(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def run(seed: int = 0, fast: bool = False, verbose: bool = True):
    meter = MeasuredMeter()
    rng = np.random.default_rng(seed)
    proof = ProofPackage("real-R6-iterated-gn-vs-internal-lora")
    n_adapt = 600 if fast else 1200
    n_nodes = 4 if fast else 6
    rounds = 2 if fast else 3
    sgd_steps = 120 if fast else 250
    pre_steps = 1500 if fast else 4000

    with meter.phase("data+backbone"):
        data = load_agnews(n_train=20000 if not fast else 8000, n_test=2000, seed=seed)
        edge = pretrained_backbone(
            data["vocab"].size, data["train_tokens"], cache_key=f"edge_v1_s{seed}_{pre_steps}",
            steps=pre_steps, seed=seed, d_model=128, n_layers=2, d_ff=256, seq_len=24,
        )
    y_all = np.eye(N_CLASSES)[data["train_labels"]]
    test_tokens = data["test_tokens"][:1500]
    test_labels = data["test_labels"][:1500]

    # weak FIXED head from a small bootstrap: internal adaptation must do
    # the work (scale targets so the fixed head has signal to chase)
    boot = rng.permutation(len(y_all))[:400]
    hb = augment_bias(edge.forward(data["train_tokens"][boot]))
    W0_aug = ridge_solve(hb.T @ hb, hb.T @ y_all[boot], LAM)
    W_head = W0_aug[:-1, :].T  # (k, d) fixed for every method

    sub = rng.permutation(len(y_all))[:n_adapt]
    tokens = data["train_tokens"][sub]
    targets = 2.0 * y_all[sub]
    labels_sub = data["train_labels"][sub]

    def acc_with_delta(delta) -> float:
        h = edge.forward(test_tokens, mlp_deltas=None if delta is None else {LAYER: delta})
        return _acc(h @ W_head.T, test_labels)

    acc_base = acc_with_delta(None)

    # ---- canonical basis for the internal surface (activation-SVD) ------
    basis = build_basis(
        edge, "activation-svd", seed=seed, ru=6, rv=8, n_classes=N_CLASSES,
        adapter_layers=(LAYER,), calibration_tokens=data["train_tokens"][:600],
    )
    keys = KeyRegistry()
    parts = partition_dirichlet(labels_sub, n_nodes, alpha=0.1, rng=rng)
    node_data = [
        (f"n{i}", tokens[idx], targets[idx]) for i, idx in enumerate(parts) if len(idx) >= 8
    ]
    probe = tokens[:60]

    # ---- RTLM-N: one-shot and iterated Gauss-Newton ---------------------
    with meter.phase("R6_rtlmn_oneshot_gn"):
        one = iterated_gauss_newton(
            edge, basis, SURFACE, node_data, W_head, keys,
            rounds=1, probe_tokens=probe, objective_prefix="one",
        )
    acc_oneshot = acc_with_delta(one.delta)

    with meter.phase("R6_rtlmn_iterated_gn"):
        it = iterated_gauss_newton(
            edge, basis, SURFACE, node_data, W_head, keys,
            rounds=rounds, probe_tokens=probe, objective_prefix="ign",
        )
    acc_iterated = acc_with_delta(it.delta)

    # ---- TRUE internal LoRA: centralized and federated ------------------
    used = np.concatenate([idx for idx in parts if len(idx) >= 8])
    with meter.phase("R6_internal_lora_central"):
        d_central = train_internal_lora(
            edge, tokens[used], targets[used], W_head, LAYER,
            rank=4, steps=sgd_steps, seed=seed,
        )
    acc_central_lora = acc_with_delta(d_central)

    with meter.phase("R6_internal_lora_federated"):
        deltas = []
        for i, idx in enumerate(parts):
            if len(idx) < 8:
                continue
            deltas.append(
                train_internal_lora(
                    edge, tokens[idx], targets[idx], W_head, LAYER,
                    rank=4, steps=sgd_steps, seed=1000 + i,
                )
            )
        d_fed = np.mean(deltas, axis=0)
    acc_fed_lora = acc_with_delta(d_fed)

    # ---- claims ---------------------------------------------------------
    proof.add_metric(
        "iteration escapes the one-shot linearization band",
        acc_iterated - acc_oneshot,
        threshold=0.0,
        higher_is_better=True,
        guarantee="LINEARIZED-BOUNDED",
        note=f"one-shot={acc_oneshot:.3f} iterated({rounds})={acc_iterated:.3f}",
    )
    proof.add_metric(
        "per-round linearization error is measured and shrinks",
        it.history[-1].linearization_error - it.history[0].linearization_error,
        threshold=0.0,
        higher_is_better=False,
        guarantee="LINEARIZED-BOUNDED",
        note="errors/round: " + ", ".join(f"{h.linearization_error:.3f}" for h in it.history),
    )
    proof.add_metric(
        "canonical GN state beats TRUE federated internal LoRA (non-IID)",
        acc_iterated - acc_fed_lora,
        threshold=0.0,
        higher_is_better=True,
        note=f"iterated-gn={acc_iterated:.3f} federated-internal-lora={acc_fed_lora:.3f}",
    )
    proof.add_metric(
        "internal surface improves over fixed head",
        acc_iterated - acc_base,
        threshold=0.0,
        higher_is_better=True,
        note=f"base={acc_base:.3f}",
    )
    gap = acc_iterated - acc_central_lora
    proof.add_metric(
        "gap to TRUE centralized internal LoRA (reported, not gated)",
        gap,
        threshold=-1.0,  # informational: never fails, never hidden
        higher_is_better=True,
        note=f"centralized-internal-lora={acc_central_lora:.3f} (Adam, backprop, adaptive subspace)",
    )
    cpu = {k: round(meter.phases[k].cpu_s, 2) for k in meter.phases if k.startswith("R6_")}
    proof.add_metric(
        "measured cpu-s: iterated GN vs federated internal LoRA (ratio)",
        cpu["R6_internal_lora_federated"] / max(cpu["R6_rtlmn_iterated_gn"], 1e-9),
        threshold=0.0,
        higher_is_better=True,
        note=json.dumps(cpu),
    )

    if verbose:
        print(proof.summary())
        print(
            f"  base={acc_base:.3f} one-shot-GN={acc_oneshot:.3f} iterated-GN={acc_iterated:.3f} "
            f"fed-internal-lora={acc_fed_lora:.3f} central-internal-lora={acc_central_lora:.3f}"
        )
        print(f"  measured (RTLMN_CPU_WATTS={DEFAULT_CPU_WATTS:g}):")
        print(meter.summary())
    return proof


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--report", type=str, default="rtlmn_r6_report.json")
    args = parser.parse_args()
    proof = run(seed=args.seed, fast=args.fast)
    with open(args.report, "w") as f:
        json.dump(asdict(proof), f, indent=2, default=str)
    print(f"proof package written to {args.report}")
    return 0 if proof.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
