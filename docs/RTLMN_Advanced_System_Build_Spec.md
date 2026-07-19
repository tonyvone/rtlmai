# RTLM-N: Distributed Neural Intelligence State
## Advanced Build Specification

## Mission
Build the deep-learning analogue of original RTLM: a system where edge nodes convert local experience into compact neural intelligence states, merge those states without moving raw data, and materialize many task-, tenant-, domain-, and power-specific models without independently retraining each model.

## Core scientific question
Can a frozen or slowly changing foundation model expose a canonical adaptation space in which local examples are converted into compact, mergeable learning states?

Let a frozen backbone produce h = f_theta(x). Let adaptation parameters live in a shared basis P, so Delta theta = P c.

For supported surfaces, nodes accumulate:
- G_i = sum J_i^T W_i J_i
- g_i = sum J_i^T r_i
- H_i = sum h_i h_i^T
- R_i = sum h_i r_i^T

Global state:
- G = sum_i G_i
- g = sum_i g_i
- H = sum_i H_i
- R = sum_i R_i

Materialization:
- c* = solve(G + lambda I, g)
- W* = solve(H + lambda I, R)

This is the closest neural equivalent to the original RTLM sufficient-state construction.

## What RTLM-N is not
- not FedAvg;
- not averaging LoRA weights;
- not a cache;
- not an inference router;
- not a claim of exact end-to-end neural training;
- not privacy merely because raw records stay local.

## Guarantee classes
Every result must be labeled:
1. EXACT-FROZEN — centralized equivalence for a declared frozen representation and objective.
2. LINEARIZED-BOUNDED — exact solution to a local linear/quadratic approximation with error bounds.
3. SKETCHED-BOUNDED — compressed state with declared approximation probability.
4. EMPIRICAL-NEURAL — useful experimentally, without exact equivalence.
5. UNSUPPORTED — incompatible bases, unsafe numerics, representation drift, or invalid claim.

## Architecture
Local experience
-> frozen representation extraction
-> teacher/target generation
-> residual construction
-> projection into canonical adaptation basis
-> local neural intelligence state
-> signed distributed merge
-> materialization engine
-> many executable models
-> deployment and inference
-> new residual experience

Core services:
- Base Model Registry
- Representation Extractor
- Adaptation Surface Registry
- Canonical Basis Manager
- Residual Generator
- Local State Accumulator
- Merge and Lineage Engine
- Neural Materializer
- Multi-Model Factory
- Inference Runtime
- Compute/Energy Meter
- Proof Package Engine

## Adaptation surfaces
Stage A: exact output heads
- classification;
- regression;
- reward;
- retrieval;
- confidence;
- routing.

Stage B: residual logit correction
- learn a compact map from hidden state to teacher/student residual.

Stage C: canonical low-rank adapters
For layer l:
Delta W_l = U_l C_l V_l^T
U_l and V_l are fixed globally; only C_l is learned from mergeable states.

Stage D: projected Jacobian adaptation
f(theta + Pc, x) approximately f(theta, x) + J_P(x)c
Accumulate Gauss-Newton or ridge states in c-space.

Stage E: coupled multi-layer state
- block diagonal;
- Kronecker-factored;
- low-rank cross-layer curvature.

Stage F: representation-epoch migration
- test whether intelligence state can survive backbone changes.

## Canonical basis
All nodes must share an aligned coordinate system. Implement:
- random orthogonal basis;
- public-calibration SVD basis;
- Fisher/gradient principal subspace;
- teacher-residual subspace;
- reusable adaptation dictionary.

Each basis has a model hash, layer map, rank, construction commitment, normalization, version, and compatibility rules.

Never merge incompatible basis versions.

## Local state schema
NeuralIntelligenceState:
- node_id
- tenant_id
- base_model_id
- representation_version
- basis_id
- objective_id
- task_family
- sample_count
- effective_weight
- G/g or H/R accumulators
- optional sketch
- residual norms
- activation norms
- uncertainty
- time range
- event commitments
- deletion commitments
- numerical diagnostics
- guarantee class
- state hash
- signature

Target state size:
- initial: <100 MB/node/family
- strong: <10 MB
- stretch: <1 MB

## Required operations
MERGE
REMOVE
SCALE
PROJECT
SPLIT
VERIFY
CERTIFY
MATERIALIZE

Required properties:
- associativity;
- commutativity;
- deterministic serialization;
- merge-order stability;
- exact centralized equivalence in supported modes;
- deletion and rematerialization;
- duplicate rejection;
- incompatible-state refusal.

## Model factory
One merged state must generate many model views:
- multiple regularization values;
- multiple ranks;
- one model per tenant;
- one per domain;
- one per power budget;
- one excluding a deleted node;
- high-accuracy and low-power versions;
- quantized specialist models;
- adapter mixtures.

The state is the durable asset. Models are materialized views.

## Training loop
1. Freeze a base model for a representation epoch.
2. Each node processes its local data once.
3. Extract hidden states and teacher/verified targets.
4. Update compact state.
5. Merge compatible states.
6. Solve and compile adapters or heads.
7. Evaluate and deploy.
8. Incrementally update state with new data without replaying old records.

## Inference loop
1. Run the smallest eligible local model.
2. Apply RTLM-N materialized components.
3. Verify confidence, policy, or correctness.
4. Escalate only when necessary.
5. Convert the verified teacher result into a residual state update.
6. Merge and rematerialize periodically.
7. Measure whether teacher escalation and energy per task decline.

## Power accounting
Instrument:
- GPU seconds;
- CPU seconds;
- FLOPs;
- memory traffic;
- network bytes;
- tokens;
- energy;
- peak memory;
- latency.

Primary metric:
Verified Outcomes per Joule

Other metrics:
- teacher calls avoided;
- central FLOPs avoided;
- state bytes per quality point;
- model variants per representation pass;
- watt-hours per accepted task;
- energy break-even request count;
- cumulative net energy saved.

Break-even:
N_saved * (E_large - E_small)
>
E_representation + E_state + E_merge + E_materialization

## Foundational experiments

### Experiment 0: exact frozen head
Compare centralized raw-data training with distributed merged-state training under arbitrary partitions, merge orders, and deletions.

### Experiment 1: residual distillation
Teacher: large model.
Student: small frozen model plus RTLM residual head.
Compare with ordinary distillation, FedAvg, LoRA, adapter averaging, and teacher-always.

### Experiment 2: canonical low-rank adapter
Use fixed U/V bases across nodes and solve C from merged state.
Compare with centralized LoRA, FedLoRA, and naive adapter merging.

### Experiment 3: projected Jacobian state
Compare projected Gauss-Newton materialization with LoRA and full fine-tuning. Measure approximation radius and raw-data passes avoided.

### Experiment 4: one state, 1,000 models
Materialize at least 1,000 variants from one merged state. Compare total compute with independently fine-tuning 1,000 models.

### Experiment 5: continual inference accretion
Run a small/large model cascade. Every escalation updates edge intelligence. Demonstrate declining teacher usage and energy per request while preserving quality.

### Experiment 6: heterogeneous edges
Vary data distribution, hardware, precision, model size, connectivity, and node participation.

### Experiment 7: backbone migration
Compare reset, raw replay, representation bridge, anchor alignment, and latent-dictionary transport.

## Mandatory baselines
- full centralized fine-tuning;
- centralized LoRA;
- FedAvg;
- federated LoRA;
- local adapter averaging;
- task-vector merging;
- knowledge distillation;
- static model cascade;
- mixture of adapters;
- RTLM-N.

## Non-negotiable pass gates
1. Exact frozen-feature equivalence across arbitrary partitions.
2. Nodes transmit compact state, not raw examples or full checkpoints.
3. One state materializes many models without rerunning representation extraction.
4. Better than naive adapter averaging under non-IID data.
5. Incompatible basis versions are refused.
6. Every non-exact surface reports approximation error.
7. Supported contributor deletion rematerializes the retained model.
8. At least one nonlinear adaptation surface improves beyond an output head.
9. Multi-model materialization uses less compute than independent fine-tuning.
10. Continual inference reduces large-model escalation.
11. Measured lifetime energy savings exceed learning costs.
12. Verified outcomes per joule improves over baselines.
13. Unsafe numerical states return UNSUPPORTED.
14. Privacy is measured, not inferred from data locality.
15. Mutation tests catch merge, deletion, basis, and proof defects.

## Failure conditions
The project is not a breakthrough if:
- it only averages LoRA weights;
- state size approaches raw-data size;
- each new model requires replaying activations;
- compute savings do not exceed build cost;
- non-IID nodes destroy performance;
- one backbone change invalidates everything;
- quality does not beat standard distillation/FedLoRA;
- it cannot produce many models from one state;
- it measures training speed but not lifetime power savings.

## Repository
rtlm_n/
  core/
  models/
  bases/
  accumulators/
  distributed/
  materialization/
  inference/
  migration/
  metering/
  assurance/
  benchmarks/
  api/
  dashboard/
  tests/

## Milestones
M0: Exact transformer-head state.
M1: Distributed residual distillation.
M2: Canonical low-rank adapter state.
M3: Projected Jacobian/Gauss-Newton state.
M4: 1,000-model materialization benchmark.
M5: Continual inference and measured energy break-even.
M6: Backbone-independent intelligence transport.

## Publication thesis
RTLM-N separates a portion of neural learning from raw data and final model weights. Distributed nodes accumulate compact intelligence states in a shared neural coordinate system; those states can be merged, removed, and reused to materialize many executable models. For supported frozen and linearized adaptation surfaces, this avoids repeated data passes and amortizes learning across model variants. During inference, accumulated state progressively transfers work from large centralized models to smaller edge models.

## Long-term game-changing hypothesis
Intelligence can become a persistent, mergeable computational object independent of any single dataset copy, training run, or model materialization.
