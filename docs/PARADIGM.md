# The RTLM-N Paradigm Case: Claims → Evidence → What Remains

**Thesis.** Intelligence should be a durable, mergeable, deletable
computational object — accumulated at the edge, merged globally, and
materialized into models on demand. Models become temporary executions of
an intelligence state, not the permanent containers of intelligence. The
payoff is a structural reduction in the compute and power AI consumes.

This document is the ledger: every claim, the measured evidence behind it,
its guarantee class, and — with equal weight — what has *not* been proven
yet. All numbers are reproducible from this repository
(`pytest`; `python -m rtlm_n.benchmarks.run_all`; `... real_agnews`;
`... real_internal`; `... real_continual`; `... fleet_study`).

---

## Pillar 1 — Amortization: pay per experience, not per model

*The old economics:* every model variant (tenant, domain, power budget,
policy, deletion request) is a training run over the data.

*The new economics:* experience is folded once into a sufficient state;
every variant is a regularized solve against that state.

| Claim | Evidence | Guarantee |
|---|---|---|
| One state materializes 1,000 variants with zero new representation passes | R2/exp4: 7.8 measured cpu-s vs ~1,675 cpu-s SGD-per-variant (**215×**); toy program ~980× in FLOPs | EXACT-FROZEN |
| Distributed state ≡ centralized training, any partition, any merge order | exp0/R0: max deviation ~1e-10 on real MLM features, severe non-IID | EXACT-FROZEN |
| Exact per-tenant slicing and contributor exclusion | exp0/R0 + factory: excluded-node variant ≡ retraining without that node | EXACT-FROZEN |

## Pillar 2 — Accretion: the edge permanently absorbs the teacher

*The old economics:* a router decides small-vs-large per request, forever
paying the large-model tax on hard traffic.

*The new economics:* every escalation becomes state; the small model's
competence — and the fleet's energy bill — improves monotonically.

| Claim | Evidence | Guarantee |
|---|---|---|
| Teacher escalation declines under accretion, quality holds | exp5: 0.37 → 0.13 at accuracy 1.000; R3 (real text): 0.36 → 0.30 with quality 0.62 → 0.86 | EMPIRICAL-NEURAL |
| Energy per accepted task declines; break-even repaid in-workload | exp5: 0.018 → 0.007 J/task; 7.7× verified outcomes/joule vs teacher-always | EMPIRICAL-NEURAL |
| Accreted intelligence survives backbone replacement | R5/exp7: transported state recovers 78–87% of the reset→retrain gap on an independently-pretrained, different-width backbone, bridge residual reported | EMPIRICAL-NEURAL (never claimed exact) |
| Canonical evidence beats trained-adapter averaging (the federated-LoRA failure) | R1: 0.750 vs 0.538 (true Adam-trained federated LoRA, non-IID), ≥ centralized SGD LoRA at 46× less measured compute | EXACT-FROZEN merge; comparison EMPIRICAL |
| Internal (nonlinear) surfaces reachable by iterated evidence | R6: distributed Gauss–Newton 0.566 → 0.710, within 4.5 pts of full backprop from a 48-coordinate mergeable subspace; per-round error 0.26 → 0.14 | LINEARIZED-BOUNDED, error measured |

## Pillar 3 — Incrementality: learning cost O(new), zero forgetting, exact deletion

*The old economics:* to stay current you either replay history (cost grows
with everything you have ever seen) or fine-tune incrementally (and
forget). Deletion means retraining.

*The new economics:* each example is touched once, ever. The state is
bit-identical to full retraining at every moment, cannot forget, and
deletion is a subtraction.

| Claim | Evidence | Guarantee |
|---|---|---|
| Incremental ≡ full retraining at every period | R7: max deviation 6e-11 across 6 drifting periods | EXACT-FROZEN |
| The cheap alternative forgets; the state cannot | R7: SGD warm-start on new data 0.482 vs state 0.770 under 70% class drift | EMPIRICAL-NEURAL |
| Replay cost grows with history; state cost does not | R7 measured: per-period cost ×6.0 (replay) vs ×1.6 (state); cumulative 3.4× and diverging | EXACT-FROZEN accounting |
| Deletion without retraining | R7: subtract + solve, 269× cheaper than replay, exact to 1e-11 | EXACT-FROZEN |

## Fleet-scale synthesis (simulation, labeled as such)

`fleet_study.py` projects the measured micro-costs onto a declared
workload (1,000 nodes × 200 tasks/day × 24 months, backbone upgrades every
6 months, per-tenant refreshes): **−81% lifetime energy vs
teacher-always, −51% vs a tuned static cascade.** Per-task teacher/edge
joules are declared industry estimates; every parameter is printed with
its provenance. The claim is the shape of the curves, not the wattage.

---

## What has NOT been proven (and what would prove it)

Claim discipline cuts both ways. The following are open, in order of
importance:

1. **Scale.** Every result here uses small transformers MLM-pretrained on
   the task corpus (this build environment cannot reach pretrained-model
   hosts). The machinery is dimension-independent and `rtlm_n/real/hf_backbone.py`
   wraps any HuggingFace encoder — the single highest-value next step is
   re-running R0–R7 on a MiniLM/BERT/7B-class backbone with standard
   benchmarks. Note the favorable asymmetry: the better the foundation
   model, the more of downstream adaptation is captured by corrections in
   its representation space — scale should help this thesis, not hurt it.
2. **Generative tasks.** All evidence is classification/regression-shaped.
   The cascade and residual machinery applies to scoring, routing, reward,
   and verification heads around generation, but no generative-quality
   result exists yet.
3. **Absolute energy.** Measured CPU-seconds at a declared wattage, plus
   RAPL where exposed. A GPU fleet pilot with wall-power measurement is
   needed for headline wattage claims.
4. **Migration at scale.** 78–87% gap recovery between small backbones is
   promising, not decisive. Transport between real model generations
   (e.g., MiniLM → BERT, or v1 → v2 of a production model) is the
   highest-risk, highest-value open experiment.
5. **Adversarial robustness.** Signatures and duplicate rejection are
   implemented and tested; Byzantine-robust aggregation (malicious but
   validly-signed statistics) is not addressed.

## Reproduce everything

```bash
pip install -e .[real,dev]
pytest                                        # 41 tests, the 15 pass gates
python -m rtlm_n.benchmarks.run_all           # reference program (8/8)
python -m rtlm_n.benchmarks.real_agnews       # real data R0-R5 (6/6)
python -m rtlm_n.benchmarks.real_internal     # R6 iterated Gauss-Newton
python -m rtlm_n.benchmarks.real_continual    # R7 incrementality
python -m rtlm_n.benchmarks.fleet_study       # fleet lifetime projection
```
