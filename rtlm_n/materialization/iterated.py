"""Iterated Gauss-Newton materialization: escaping the linearization band.

One-shot projected-Jacobian adaptation is exact only inside a band around
the reference model. This module iterates the whole RTLM-N loop:

    round r:  linearize every node at the CURRENT coordinates c_r
              -> nodes accumulate G_r = sum J^T J, g_r = sum J^T res   (local)
              -> states merge exactly (within the round)               (global)
              -> solve the damped increment  dc = (G_r + lam I)^{-1} g_r
              -> c_{r+1} = c_r + damping * dc
              -> measure the linearization error of the step, always

This is distributed Gauss-Newton with sufficient-statistic communication:
each round transmits one compact state per node (never gradients streams,
never raw data), and within every round the merge keeps the full exact
algebra - order-free, deletable, duplicate-safe. Across rounds the claim
discipline is explicit: states from different linearization points are
DIFFERENT objectives and are never merged together (objective_id is scoped
per round); the overall result is LINEARIZED-BOUNDED with a per-round
error report, not exact.

Trust-region behavior comes from the ridge term (Levenberg-Marquardt
flavor): lam both regularizes the solve and bounds the step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rtlm_n.accumulators.surfaces import linearization_error
from rtlm_n.bases.basis import CanonicalBasis, adapter_delta
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import ridge_solve
from rtlm_n.distributed.merge import MergeEngine
from rtlm_n.distributed.node import EdgeNode


@dataclass
class GaussNewtonRound:
    round: int
    step_norm: float
    residual_sse: float
    linearization_error: float
    sample_count: int


@dataclass
class IteratedResult:
    c: np.ndarray
    delta: np.ndarray
    layer: int
    history: List[GaussNewtonRound] = field(default_factory=list)

    def mlp_deltas(self) -> Dict[int, np.ndarray]:
        return {self.layer: self.delta}


def iterated_gauss_newton(
    model,
    basis: CanonicalBasis,
    surface: str,
    node_data: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    head_W: np.ndarray,
    keys: KeyRegistry,
    rounds: int = 3,
    lam: float = 1e-6,
    mu: float = 1.0,
    damping: float = 0.7,
    probe_tokens: Optional[np.ndarray] = None,
    objective_prefix: str = "ign",
    on_round: Optional[Callable[[GaussNewtonRound], None]] = None,
) -> IteratedResult:
    """Run distributed iterated Gauss-Newton over an internal surface.

    node_data: per-node (node_id, tokens, targets); raw data stays inside
    each node's absorb call, exactly as in the one-shot path.

    Trust region: the solve uses lam_eff = lam + mu * trace(G)/dim - the
    Levenberg-Marquardt convention of damping RELATIVE to local curvature,
    so step size adapts to the landscape instead of a fixed ridge. mu ~ 1
    is conservative; smaller mu takes bolder steps. `damping` additionally
    scales the accepted step.
    """
    m = basis.matrices(surface)
    ru, rv = m["U"].shape[1], m["V"].shape[1]
    layer = int(surface.split(":")[1])
    c = np.zeros(ru * rv)
    history: List[GaussNewtonRound] = []

    for r in range(rounds):
        objective = f"{objective_prefix}-round{r}"
        engine = MergeEngine(keys)
        n_total = 0
        for node_id, tokens, targets in node_data:
            node = EdgeNode(f"{node_id}-r{r}", model, basis, keys, f"{node_id}-r{r}".encode())
            node.absorb_jacobian_batch(
                tokens, targets, head_W, surface=surface, objective_id=objective, c0=c
            )
            engine.submit(node.state(objective))
            n_total += len(tokens)
        state = engine.global_state
        key = f"jacobian:{surface}"
        G = state.accumulators[key]["G"]
        g = state.accumulators[key]["g"]
        lam_eff = lam + mu * float(np.trace(G)) / G.shape[0]
        dc = damping * ridge_solve(G, g, lam_eff).ravel()

        lin_err = float("nan")
        if probe_tokens is not None:
            lin_err = linearization_error(
                model, basis, surface, probe_tokens, head_W, dc, c0=c
            )
        c = c + dc
        rec = GaussNewtonRound(
            round=r,
            step_norm=float(np.linalg.norm(dc)),
            residual_sse=float(state.approximation.get("residual_norm_sq", np.nan)),
            linearization_error=lin_err,
            sample_count=n_total,
        )
        history.append(rec)
        if on_round is not None:
            on_round(rec)

    delta = adapter_delta(basis, surface, c.reshape(ru, rv))
    return IteratedResult(c=c, delta=delta, layer=layer, history=history)
