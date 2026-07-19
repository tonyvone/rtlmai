"""Backbone migration: can intelligence survive replacement of its model?

When the foundation backbone changes, its hidden coordinates change, and a
state accumulated on backbone A is not directly meaningful on backbone B.
This module implements representation bridges fit on a small PUBLIC anchor
set:

    h_A(x) ~ h_B(x) @ B_ba      (ridge-fit linear bridge, both directions)

Two transport modes:
- transport_head: reuse a materialized head through the bridge
      y = aug(h_A) W  ->  y ~ aug(h_B) B_ba W
- transport_state: map the sufficient statistics themselves
      H_B ~ B_ab^T H_A B_ab,  R_B ~ B_ab^T R_A
  so the transported state can keep merging with native B states.

Both are labeled EMPIRICAL-NEURAL and carry the measured bridge residual.
Migration is NEVER exact and is never claimed to be: the pass gate is that
transported intelligence beats a cold reset, not that it equals retraining.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict

import numpy as np

from rtlm_n.accumulators.state import NeuralIntelligenceState
from rtlm_n.core.guarantees import GuaranteeClass
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.materialization.factory import MaterializedModel
from rtlm_n.models.backbone import TinyTransformer


@dataclass
class RepresentationBridge:
    source_model_id: str
    target_model_id: str
    #: maps aug(h_target) -> aug(h_source): (d_t+1, d_s+1)
    B_ts: np.ndarray
    #: maps aug(h_source) -> aug(h_target): (d_s+1, d_t+1)
    B_st: np.ndarray
    #: relative residual of the fit on held-out anchors
    residual: float


def fit_bridge(
    source: TinyTransformer,
    target: TinyTransformer,
    anchor_tokens: np.ndarray,
    lam: float = 1e-3,
    holdout_frac: float = 0.25,
) -> RepresentationBridge:
    """Fit linear bridges between backbone representation spaces on a public
    anchor set, reporting held-out relative residual."""
    n = len(anchor_tokens)
    n_fit = int(n * (1 - holdout_frac))
    hs = augment_bias(source.forward(anchor_tokens))
    ht = augment_bias(target.forward(anchor_tokens))
    hs_fit, hs_val = hs[:n_fit], hs[n_fit:]
    ht_fit, ht_val = ht[:n_fit], ht[n_fit:]
    B_ts = ridge_solve(ht_fit.T @ ht_fit, ht_fit.T @ hs_fit, lam)
    B_st = ridge_solve(hs_fit.T @ hs_fit, hs_fit.T @ ht_fit, lam)
    pred = ht_val @ B_ts
    residual = float(np.linalg.norm(pred - hs_val) / (np.linalg.norm(hs_val) + 1e-12))
    return RepresentationBridge(
        source_model_id=source.base_model_id,
        target_model_id=target.base_model_id,
        B_ts=B_ts,
        B_st=B_st,
        residual=residual,
    )


def transport_head(model: MaterializedModel, bridge: RepresentationBridge) -> MaterializedModel:
    """Reuse a source-backbone head on the target backbone via the bridge."""
    W_new = bridge.B_ts @ model.params["W"]
    return MaterializedModel(
        kind=model.kind,
        guarantee=GuaranteeClass.EMPIRICAL_NEURAL,
        lam=model.lam,
        params={"W": W_new},
        meta={
            **model.meta,
            "migrated_from": bridge.source_model_id,
            "migrated_to": bridge.target_model_id,
            "bridge_residual": bridge.residual,
        },
    )


def transport_state(
    state: NeuralIntelligenceState,
    bridge: RepresentationBridge,
    surface: str = "head",
) -> NeuralIntelligenceState:
    """Map H/R sufficient statistics into the target representation space.

    With aug(h_t) ~ aug(h_s) @ B_st:
        H_t = sum aug(h_t)^T aug(h_t) ~ B_st^T H_s B_st
        R_t = sum aug(h_t)^T r       ~ B_st^T R_s
    EMPIRICAL-NEURAL: the transported state may keep merging with native
    target-backbone states, but no equivalence is claimed.
    """
    if state.base_model_id != bridge.source_model_id:
        raise ValueError("state was not built on the bridge's source backbone")
    H = state.accumulators[surface]["H"]
    R = state.accumulators[surface]["R"]
    acc = dict(state.accumulators)
    acc[surface] = {"H": bridge.B_st.T @ H @ bridge.B_st, "R": bridge.B_st.T @ R}
    return dataclasses.replace(
        state,
        base_model_id=bridge.target_model_id,
        basis_id="",  # basis does not survive migration; head surface only
        accumulators=acc,
        guarantee=GuaranteeClass.EMPIRICAL_NEURAL,
        approximation={**state.approximation, "bridge_residual": bridge.residual},
        signature="",
    )
