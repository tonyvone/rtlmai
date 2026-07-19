"""Shared experiment scaffolding."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from rtlm_n.accumulators.state import NeuralIntelligenceState
from rtlm_n.bases.basis import CanonicalBasis
from rtlm_n.benchmarks.data import World
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.distributed.merge import MergeEngine
from rtlm_n.distributed.node import EdgeNode
from rtlm_n.metering.meter import EnergyMeter


def make_nodes(
    world: World,
    basis: CanonicalBasis,
    keys: KeyRegistry,
    n_nodes: int,
    tenant_prefix: str = "tenant",
) -> List[EdgeNode]:
    return [
        EdgeNode(
            node_id=f"node-{i}",
            model=world.small,
            basis=basis,
            keys=keys,
            secret=f"secret-{i}".encode(),
            tenant_id=f"{tenant_prefix}-{i}",
        )
        for i in range(n_nodes)
    ]


def absorb_partitioned_head(
    nodes: List[EdgeNode],
    parts: List[np.ndarray],
    tokens: np.ndarray,
    targets: np.ndarray,
    objective_id: str,
    batches_per_node: int = 2,
) -> None:
    """Each node absorbs its partition in several event batches."""
    for node, idx in zip(nodes, parts):
        if len(idx) == 0:
            continue
        for chunk in np.array_split(idx, min(batches_per_node, max(1, len(idx)))):
            if len(chunk):
                node.absorb_head_batch(tokens[chunk], targets[chunk], objective_id=objective_id)


def merge_all(
    nodes: List[EdgeNode],
    keys: KeyRegistry,
    objective_id: str,
    order: Optional[List[int]] = None,
    meter: Optional[EnergyMeter] = None,
) -> Tuple[NeuralIntelligenceState, MergeEngine]:
    engine = MergeEngine(keys, meter=meter)
    order = order if order is not None else list(range(len(nodes)))
    for i in order:
        engine.submit(nodes[i].state(objective_id))
    return engine.global_state, engine
