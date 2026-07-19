"""Experiment R7: the incrementality proof - O(new) learning, zero
forgetting, exact deletion.

The paradigm claim in one experiment. Data arrives in periods with shifted
distributions (each period's class mix is heavily skewed, as real
workloads drift). A deployed model must stay current across ALL history.
Four regimes:

- RTLM-N incremental: fold ONLY the new period into the state, resolve.
  Each raw example is touched exactly once, ever. Exact equivalence to
  full retraining at every period, by algebra.
- Full replay retrain: recompute representations over ALL accumulated raw
  data each period (what pipelines without a sufficient state must do).
- SGD refresh on all data (features generously pre-cached): optimizer
  cost still grows with total history each period.
- SGD warm-start on new data only: the cheap incremental baseline -
  O(new) cost, but catastrophic forgetting under drift.

Then a deletion event: a contributor from period 2 must be removed.
RTLM-N subtracts its state and re-solves; replay-based regimes must
reprocess everything that remains.

Pass gates: exact equivalence of the incremental state to full retraining
at every period; the warm-start baseline measurably forgets; measured
cumulative compute for RTLM-N stays ~flat per period while replay grows;
deletion is orders of magnitude cheaper than replay.

    python -m rtlm_n.benchmarks.real_continual [--fast]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from rtlm_n.accumulators.surfaces import objective_id_for
from rtlm_n.assurance.proof import ProofPackage
from rtlm_n.bases.basis import build_basis
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.distributed.merge import MergeEngine
from rtlm_n.distributed.node import EdgeNode
from rtlm_n.materialization.factory import ModelFactory
from rtlm_n.metering.measured import DEFAULT_CPU_WATTS, MeasuredMeter
from rtlm_n.real.agnews import load_agnews
from rtlm_n.real.baselines import train_linear_head
from rtlm_n.real.torch_backbone import pretrained_backbone

LAM = 1e-2
N_CLASSES = 4
OBJ = objective_id_for("agnews-continual")


def _acc(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def run(seed: int = 0, fast: bool = False, verbose: bool = True):
    meter = MeasuredMeter()
    rng = np.random.default_rng(seed)
    proof = ProofPackage("real-R7-continual-incrementality")
    n_periods = 4 if fast else 6
    per_period = 800 if fast else 1500
    sgd_steps_per_example = 0.2  # SGD budget scales with data size, honestly
    pre_steps = 1500 if fast else 4000

    with meter.phase("data+backbone"):
        data = load_agnews(n_train=20000, n_test=3000, seed=seed)
        edge = pretrained_backbone(
            data["vocab"].size, data["train_tokens"], cache_key=f"edge_v1_s{seed}_{pre_steps}",
            steps=pre_steps, seed=seed, d_model=128, n_layers=2, d_ff=256, seq_len=24,
        )
    y_all = np.eye(N_CLASSES)[data["train_labels"]]
    h_test = augment_bias(edge.forward(data["test_tokens"]))
    test_labels = data["test_labels"]

    # ---- drifting periods: each period is 70% one class ------------------
    remaining = list(rng.permutation(len(y_all)))
    periods = []
    for p in range(n_periods):
        major = p % N_CLASSES
        maj_idx = [i for i in remaining if data["train_labels"][i] == major]
        min_idx = [i for i in remaining if data["train_labels"][i] != major]
        take_maj = int(per_period * 0.7)
        chosen = maj_idx[:take_maj] + min_idx[: per_period - take_maj]
        periods.append(np.array(chosen))
        used = set(chosen)
        remaining = [i for i in remaining if i not in used]
    seen = [i for p in periods for i in p]
    assert len(set(seen)) == len(seen)

    basis = build_basis(edge, "random-orthogonal", seed=seed, n_classes=N_CLASSES)
    keys = KeyRegistry()
    engine = MergeEngine(keys)
    factory = ModelFactory(basis)

    curves = {
        "rtlmn_acc": [], "replay_acc": [], "sgd_all_acc": [], "sgd_new_acc": [],
        "rtlmn_cpu": [], "replay_cpu": [], "sgd_all_cpu": [], "sgd_new_cpu": [],
        "equivalence": [],
    }
    W_sgd_new = None
    features_cache = {}  # generous to SGD baselines: features computed once

    def cached_features(idx_key, tokens):
        if idx_key not in features_cache:
            features_cache[idx_key] = edge.forward(tokens)
        return features_cache[idx_key]

    for p, idx in enumerate(periods):
        toks_new = data["train_tokens"][idx]
        y_new = y_all[idx]

        # --- RTLM-N: touch ONLY the new period -------------------------
        with meter.phase(f"rtlmn_p{p}"):
            node = EdgeNode(f"period-{p}", edge, basis, keys, f"p{p}".encode())
            node.absorb_head_batch(toks_new, y_new, objective_id=OBJ)
            engine.submit(node.state(OBJ))
            W_rtlmn = factory.materialize_head(engine.global_state, lam=LAM).params["W"]
        curves["rtlmn_cpu"].append(meter.phases[f"rtlmn_p{p}"].cpu_s)
        curves["rtlmn_acc"].append(_acc(h_test @ W_rtlmn, test_labels))

        # --- full replay: recompute representations over ALL history ---
        all_idx = np.concatenate(periods[: p + 1])
        with meter.phase(f"replay_p{p}"):
            h_replay = augment_bias(edge.forward(data["train_tokens"][all_idx]))
            W_replay = ridge_solve(h_replay.T @ h_replay, h_replay.T @ y_all[all_idx], LAM)
        curves["replay_cpu"].append(meter.phases[f"replay_p{p}"].cpu_s)
        curves["replay_acc"].append(_acc(h_test @ W_replay, test_labels))
        diff = float(np.max(np.abs(W_rtlmn - W_replay)))
        curves["equivalence"].append(diff)

        # --- SGD on all history (features cached for free) --------------
        with meter.phase(f"sgd_all_p{p}"):
            h_all_cached = np.concatenate(
                [cached_features(pi, data["train_tokens"][periods[pi]]) for pi in range(p + 1)]
            )
            steps = max(40, int(len(all_idx) * sgd_steps_per_example))
            W_sgd_all = train_linear_head(h_all_cached, y_all[all_idx], steps=steps, seed=seed)
        curves["sgd_all_cpu"].append(meter.phases[f"sgd_all_p{p}"].cpu_s)
        curves["sgd_all_acc"].append(
            _acc(h_test[:, :-1] @ W_sgd_all[:, :-1].T + W_sgd_all[:, -1], test_labels)
        )

        # --- SGD warm-start on NEW data only (the forgetting baseline) --
        with meter.phase(f"sgd_new_p{p}"):
            h_new_cached = cached_features(p, toks_new)
            steps = max(40, int(len(idx) * sgd_steps_per_example))
            W_sgd_new = train_linear_head(
                h_new_cached, y_new, steps=steps, seed=seed, init_W=W_sgd_new
            )
        curves["sgd_new_cpu"].append(meter.phases[f"sgd_new_p{p}"].cpu_s)
        curves["sgd_new_acc"].append(
            _acc(h_test[:, :-1] @ W_sgd_new[:, :-1].T + W_sgd_new[:, -1], test_labels)
        )

    # ---- deletion event: remove period 1's contributor entirely ---------
    with meter.phase("deletion_rtlmn"):
        removed = engine.remove_contributor("period-1")
        W_del = factory.materialize_head(removed, lam=LAM).params["W"]
    with meter.phase("deletion_replay"):
        keep_idx = np.concatenate([periods[i] for i in range(n_periods) if i != 1])
        h_keep = augment_bias(edge.forward(data["train_tokens"][keep_idx]))
        W_del_ref = ridge_solve(h_keep.T @ h_keep, h_keep.T @ y_all[keep_idx], LAM)
    del_diff = float(np.max(np.abs(W_del - W_del_ref)))
    del_cpu_rtlmn = meter.phases["deletion_rtlmn"].cpu_s
    del_cpu_replay = meter.phases["deletion_replay"].cpu_s

    # ---- claims ---------------------------------------------------------
    proof.add_metric(
        "incremental state == full retraining at EVERY period",
        max(curves["equivalence"]),
        threshold=1e-8,
        higher_is_better=False,
        guarantee="EXACT-FROZEN",
        note=f"max |dW| across {n_periods} periods; zero forgetting by algebra",
    )
    forget_gap = curves["rtlmn_acc"][-1] - curves["sgd_new_acc"][-1]
    proof.add_metric(
        "cheap SGD warm-start forgets under drift; the state does not",
        forget_gap,
        threshold=0.05,
        higher_is_better=True,
        note=(
            f"final acc: rtlmn={curves['rtlmn_acc'][-1]:.3f} "
            f"sgd-new-only={curves['sgd_new_acc'][-1]:.3f} (70%-skewed periods)"
        ),
    )
    # per-period cost: RTLM-N flat (O(new)); replay grows (O(total))
    rtlmn_growth = curves["rtlmn_cpu"][-1] / max(curves["rtlmn_cpu"][0], 1e-9)
    replay_growth = curves["replay_cpu"][-1] / max(curves["replay_cpu"][0], 1e-9)
    proof.add_metric(
        "per-period cost: replay grows with history, RTLM-N does not",
        replay_growth / max(rtlmn_growth, 1e-9),
        threshold=1.5,
        higher_is_better=True,
        note=(
            f"period-cost growth over {n_periods} periods: "
            f"rtlmn x{rtlmn_growth:.2f} vs replay x{replay_growth:.2f}"
        ),
    )
    cum_rtlmn = sum(curves["rtlmn_cpu"])
    cum_replay = sum(curves["replay_cpu"])
    cum_sgd_all = sum(curves["sgd_all_cpu"])
    proof.add_metric(
        "cumulative measured compute vs replay retraining (x cheaper)",
        cum_replay / max(cum_rtlmn, 1e-9),
        threshold=1.5,
        higher_is_better=True,
        note=(
            f"cumulative cpu-s: rtlmn={cum_rtlmn:.1f} replay={cum_replay:.1f} "
            f"sgd-all(features free)={cum_sgd_all:.1f}"
        ),
    )
    proof.add_metric(
        "deletion: exact and cheap (vs replay-without-the-data)",
        del_cpu_replay / max(del_cpu_rtlmn, 1e-9),
        threshold=10.0,
        higher_is_better=True,
        guarantee="EXACT-FROZEN",
        note=(
            f"|dW|={del_diff:.2e}; cpu-s: subtract+solve={del_cpu_rtlmn:.3f} "
            f"vs replay={del_cpu_replay:.1f}"
        ),
    )

    if verbose:
        print(proof.summary())
        print("  period  rtlmn-acc  sgd-new-acc  rtlmn-cpu  replay-cpu  sgd-all-cpu")
        for p in range(n_periods):
            print(
                f"  {p:>6}  {curves['rtlmn_acc'][p]:>9.3f}  {curves['sgd_new_acc'][p]:>11.3f}"
                f"  {curves['rtlmn_cpu'][p]:>9.2f}  {curves['replay_cpu'][p]:>10.2f}"
                f"  {curves['sgd_all_cpu'][p]:>11.2f}"
            )
        print(f"  (RTLMN_CPU_WATTS={DEFAULT_CPU_WATTS:g})")
    proof_dict = asdict(proof)
    proof_dict["curves"] = curves
    return proof, proof_dict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--report", type=str, default="rtlmn_r7_report.json")
    args = parser.parse_args()
    proof, proof_dict = run(seed=args.seed, fast=args.fast)
    with open(args.report, "w") as f:
        json.dump(proof_dict, f, indent=2, default=str)
    print(f"proof package written to {args.report}")
    return 0 if proof.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
