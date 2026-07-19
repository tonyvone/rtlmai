"""Experiment 3: projected Jacobian (Gauss-Newton) internal adaptation.

Moves adaptation INSIDE the transformer: nodes accumulate G = sum J^T J,
g = sum J^T r for a canonical MLP adapter surface, where J is the network
Jacobian projected into the shared basis. The world's labels come from a
genuinely perturbed backbone (an internal shift no output head can fully
express), so this experiment tests the claim that at least one internal
surface improves beyond an output head - with the linearization error
measured and reported, per the LINEARIZED-BOUNDED discipline.
"""

from __future__ import annotations

import numpy as np

from rtlm_n.accumulators.surfaces import (
    linearization_error,
    objective_id_for,
    projected_jacobian,
)
from rtlm_n.assurance.proof import ProofPackage
from rtlm_n.bases.basis import adapter_delta, build_basis
from rtlm_n.benchmarks.common import make_nodes, merge_all
from rtlm_n.benchmarks.data import accuracy, gen_tokens, make_world
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.materialization.factory import ModelFactory

LAM = 1e-4
SURFACE = "mlp:1"
LAYER = 1


def run(seed: int = 0, n: int = 400, n_nodes: int = 3, verbose: bool = True) -> ProofPackage:
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    keys = KeyRegistry()
    objective = objective_id_for("jacobian-adaptation")
    proof = ProofPackage("exp3-projected-jacobian")

    calib = gen_tokens(rng, 200, world)
    basis = build_basis(
        world.small, "activation-svd", seed=seed, ru=3, rv=4,
        adapter_layers=(LAYER,), calibration_tokens=calib,
    )

    # ---- a ground-truth INTERNAL perturbation in the canonical space ----
    C_true = 0.5 * rng.standard_normal((basis.ru, basis.rv))
    delta_true = adapter_delta(basis, SURFACE, C_true)
    head_rng = np.random.default_rng(seed + 3)
    W_head = head_rng.standard_normal((world.n_classes, world.small.d_model)) * 1.5

    def true_logits(toks: np.ndarray) -> np.ndarray:
        h = world.small.forward(toks, mlp_deltas={LAYER: delta_true})
        return h @ W_head.T

    tokens = gen_tokens(rng, n, world)
    targets = true_logits(tokens)
    test = gen_tokens(rng, 300, world)
    test_logits_true = true_logits(test)
    test_labels = test_logits_true.argmax(axis=1)

    # ---- baseline: the best OUTPUT HEAD on frozen features -------------
    h_frozen = augment_bias(world.small.forward(tokens))
    W_head_only = ridge_solve(h_frozen.T @ h_frozen, h_frozen.T @ targets, LAM)
    h_test_frozen = augment_bias(world.small.forward(test))
    acc_head = accuracy(h_test_frozen @ W_head_only, test_labels)

    # ---- RTLM-N: distributed projected-Jacobian states -----------------
    nodes = make_nodes(world, basis, keys, n_nodes)
    parts = np.array_split(rng.permutation(n), n_nodes)
    for node, idx in zip(nodes, parts):
        node.absorb_jacobian_batch(
            tokens[idx], targets[idx], W_head, surface=SURFACE, objective_id=objective
        )
    merged, _ = merge_all(nodes, keys, objective)

    # centralized reference in the same linearized space
    logits0, J = projected_jacobian(world.small, basis, SURFACE, tokens, W_head)
    r = targets - logits0
    G_c = np.einsum("bkr,bks->rs", J, J)
    g_c = np.einsum("bkr,bk->r", J, r).reshape(-1, 1)
    key = f"jacobian:{SURFACE}"
    proof.add_equivalence(
        "merged G == centralized G", G_c, merged.accumulators[key]["G"],
        guarantee="LINEARIZED-BOUNDED",
    )

    factory = ModelFactory(basis)
    c_star = ridge_solve(merged.accumulators[key]["G"], merged.accumulators[key]["g"], LAM).ravel()
    lin_err = linearization_error(world.small, basis, SURFACE, test[:100], W_head, c_star)
    model = factory.materialize_adapter(
        merged, SURFACE, lam=LAM, linearization_error=lin_err
    )

    h_adapted = world.small.forward(test, mlp_deltas=model.mlp_deltas())
    logits_adapter = h_adapted @ W_head.T
    acc_adapter = accuracy(logits_adapter, test_labels)
    mse_head = float(np.mean((h_test_frozen @ W_head_only - test_logits_true) ** 2))
    mse_adapter = float(np.mean((logits_adapter - test_logits_true) ** 2))

    c_recovery = float(
        np.linalg.norm(model.params["c"] - C_true.ravel()) / np.linalg.norm(C_true)
    )
    proof.add_metric(
        "canonical coordinate recovery error",
        c_recovery,
        threshold=0.5,
        higher_is_better=False,
        guarantee="LINEARIZED-BOUNDED",
        note="||c* - c_true|| / ||c_true||",
    )
    proof.add_metric(
        "linearization error reported",
        lin_err,
        threshold=1.0,
        higher_is_better=False,
        guarantee="LINEARIZED-BOUNDED",
        note="mandatory report for every non-exact surface",
    )
    proof.add_metric(
        "internal adapter beats best output head (accuracy)",
        acc_adapter - acc_head,
        threshold=0.0,
        higher_is_better=True,
        guarantee="LINEARIZED-BOUNDED",
        note=f"adapter={acc_adapter:.3f} head-only={acc_head:.3f}",
    )
    proof.add_metric(
        "internal adapter beats best output head (logit fidelity)",
        mse_head / max(mse_adapter, 1e-12),
        threshold=1.5,
        higher_is_better=True,
        guarantee="LINEARIZED-BOUNDED",
        note=f"MSE head-only={mse_head:.4f} adapter={mse_adapter:.4f}; the internal "
        "shift is not expressible by any output head",
    )

    if verbose:
        print(proof.summary())
        print(
            f"  head-only={acc_head:.3f} adapter={acc_adapter:.3f} "
            f"linearization-error={lin_err:.4f} c-recovery={c_recovery:.4f}"
        )
    return proof


if __name__ == "__main__":
    run()
