"""Continual inference accretion: the small/large cascade that learns.

    Small local model attempts task
    -> materialized RTLM-N correction assists
    -> verifier evaluates confidence
    -> failure escalates to the large teacher
    -> the teacher's verified answer becomes new residual state
    -> states merge, the correction rematerializes
    -> the small model improves, future escalation declines

The runtime does not merely route between models: every escalation is
permanently absorbed into the intelligence state, so the system's reliance
on the teacher falls over time. The metrics it emits are the core curve of
the thesis: teacher calls declining, energy per accepted task declining,
quality stable, accumulated intelligence increasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from rtlm_n.accumulators.surfaces import objective_id_for
from rtlm_n.core.linalg import softmax
from rtlm_n.distributed.merge import MergeEngine
from rtlm_n.distributed.node import EdgeNode
from rtlm_n.materialization.factory import MaterializedModel, ModelFactory
from rtlm_n.metering.meter import CENTRAL, EDGE, EnergyMeter
from rtlm_n.models.backbone import TinyTransformer


@dataclass
class CascadeMetrics:
    window: int
    tasks: int
    escalations: int
    accuracy: float
    escalation_rate: float
    joules: float
    joules_per_task: float
    cumulative_teacher_calls: int
    state_samples: int


class CascadeRuntime:
    def __init__(
        self,
        node: EdgeNode,
        engine: MergeEngine,
        factory: ModelFactory,
        small_head_W: np.ndarray,
        teacher_model: TinyTransformer,
        teacher_head_W: Optional[np.ndarray] = None,
        teacher_fn: Optional[object] = None,
        confidence_threshold: float = 0.7,
        rematerialize_every: int = 32,
        lam: float = 1e-2,
        task_family: str = "cascade",
    ) -> None:
        self.node = node
        self.engine = engine
        self.factory = factory
        self.small_head_W = small_head_W  # (d+1, k) augmented-row convention
        self.teacher_model = teacher_model
        self.teacher_head_W = teacher_head_W
        #: optional verified workflow: tokens -> verified logits. The large
        #: model still runs (and is charged) as part of verification.
        self.teacher_fn = teacher_fn
        if teacher_head_W is None and teacher_fn is None:
            raise ValueError("provide teacher_head_W or teacher_fn")
        self.threshold = confidence_threshold
        self.rematerialize_every = rematerialize_every
        self.lam = lam
        self.objective_id = objective_id_for("cascade-residual")
        self.task_family = task_family
        self.correction: Optional[MaterializedModel] = None
        self.meter = EnergyMeter()
        self.teacher_calls = 0
        self._pending_escalations = 0
        self._merged_once = False

    # ------------------------------------------------------------ helpers
    def _small_logits(self, h: np.ndarray) -> np.ndarray:
        ha = np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)
        logits = ha @ self.small_head_W
        if self.correction is not None:
            logits = logits + self.correction.predict(h)
        return logits

    def _teacher_logits(self, tokens: np.ndarray) -> np.ndarray:
        flops = self.teacher_model.flops_per_example * len(tokens)
        self.meter.charge_flops("inference_large", flops, CENTRAL)
        # every escalated task is its own remote request
        self.meter.charge_call(
            "teacher_call", CENTRAL, payload_bytes=float(tokens.nbytes), count=len(tokens)
        )
        self.teacher_calls += len(tokens)
        if self.teacher_fn is not None:
            return np.asarray(self.teacher_fn(tokens), dtype=np.float64)
        h = self.teacher_model.forward(tokens)
        return np.concatenate([h, np.ones((h.shape[0], 1))], axis=1) @ self.teacher_head_W

    def _rematerialize(self) -> None:
        state = self.node.state(self.objective_id, self.task_family)
        engine = MergeEngine(self.node.keys, meter=self.engine.meter)
        engine.submit(state)
        self.correction = self.factory.materialize_head(
            engine.global_state, lam=self.lam, surface="residual-head", variant="cascade-correction"
        )
        self._merged_once = True

    # -------------------------------------------------------------- run
    def process_batch(self, tokens: np.ndarray, true_labels: np.ndarray, t: float) -> Dict:
        """One batch of tasks through the cascade. Returns batch metrics."""
        h = self.node.model.forward(tokens)
        self.meter.charge_flops(
            "inference_small", self.node.model.flops_per_example * len(tokens), EDGE
        )
        logits = self._small_logits(h)
        probs = softmax(logits)
        top2 = np.sort(probs, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        escalate = margin < self.threshold

        final_pred = logits.argmax(axis=1)
        n_esc = int(escalate.sum())
        if n_esc > 0:
            esc_tokens = tokens[escalate]
            teacher_logits = self._teacher_logits(esc_tokens)
            final_pred[escalate] = teacher_logits.argmax(axis=1)
            # verified teacher behavior becomes new residual intelligence
            student_logits_base = (
                np.concatenate([h[escalate], np.ones((n_esc, 1))], axis=1) @ self.small_head_W
            )
            self.node.absorb_residual_batch(
                esc_tokens,
                teacher_logits,
                student_logits_base,
                objective_id=self.objective_id,
                task_family=self.task_family,
                t=t,
                h=h[escalate],
            )
            self._pending_escalations += n_esc
            if self._pending_escalations >= self.rematerialize_every:
                self._rematerialize()
                self._pending_escalations = 0

        acc = float((final_pred == true_labels).mean())
        return {"tasks": len(tokens), "escalations": n_esc, "accuracy": acc}

    def run(
        self,
        batches: List[np.ndarray],
        labels: List[np.ndarray],
        window: int = 8,
    ) -> List[CascadeMetrics]:
        """Feed a workload through the cascade; report per-window metrics."""
        history: List[CascadeMetrics] = []
        w_tasks = w_esc = 0
        w_correct = 0.0
        j_prev = 0.0
        widx = 0
        for i, (tokens, y) in enumerate(zip(batches, labels)):
            out = self.process_batch(tokens, y, t=float(i))
            w_tasks += out["tasks"]
            w_esc += out["escalations"]
            w_correct += out["accuracy"] * out["tasks"]
            if (i + 1) % window == 0:
                j_total = (
                    self.meter.total_joules()
                    + self.node.meter.total_joules()
                    + self.factory.meter.total_joules()
                )
                j_window = j_total - j_prev
                j_prev = j_total
                history.append(
                    CascadeMetrics(
                        window=widx,
                        tasks=w_tasks,
                        escalations=w_esc,
                        accuracy=w_correct / max(w_tasks, 1),
                        escalation_rate=w_esc / max(w_tasks, 1),
                        joules=j_window,
                        joules_per_task=j_window / max(w_tasks, 1),
                        cumulative_teacher_calls=self.teacher_calls,
                        state_samples=sum(
                            l.sample_count
                            for ls in self.node._leaves.values()
                            for l in ls
                        ),
                    )
                )
                widx += 1
                w_tasks = w_esc = 0
                w_correct = 0.0
        return history
