# RTLM-N: Distributed Neural Intelligence State

RTLM-N is the deep-learning analogue of the original RTLM statistical
state. Edge nodes transform local experience into **compact neural
learning states** inside a shared canonical coordinate system. Those
states **merge exactly** without moving raw data, and a model factory
**materializes many models** — per tenant, domain, power budget, or
deletion request — from one durable intelligence object.

```
Local data and outcomes
        ↓
Frozen foundation model creates representations
        ↓
Node converts learning into compact statistics       H = Σ h hᵀ   R = Σ h rᵀ
        ↓                                            G = Σ JᵀJ    g = Σ Jᵀr
Statistics remain on small edge memory               (KB–MB, independent of n)
        ↓
Compatible states merge across nodes                 H = Σᵢ Hᵢ  (exact, order-free)
        ↓
Model factory materializes many neural models        W* = (H + λI)⁻¹R
        ↓
Models deploy back to edge
        ↓
New inference creates additional intelligence
```

The thesis: **models become temporary executions of intelligence state,
rather than permanent containers of intelligence.**

## What makes this different from federated LoRA

Independently trained low-rank adapters live in accidental, geometrically
misaligned subspaces — averaging them interferes destructively. RTLM-N
gives every node the **same** globally fixed adaptation basis
(ΔW = U C Vᵀ with U, V shared; Δθ = P c in general) and asks nodes to
accumulate **evidence** for the coordinates, not trained weights. Merged
evidence is then **solved**, not averaged. Experiment 2 reproduces the
misalignment failure (0.75 accuracy) and fixes it with canonical
coordinates (0.99, floating-point-equal to centralized training).

## Guarantee classes (claim discipline)

Every state and materialized model carries exactly one label:

| Class | Meaning |
|---|---|
| `EXACT-FROZEN` | Bitwise centralized equivalence for a declared frozen representation + convex objective |
| `LINEARIZED-BOUNDED` | Exact solve of a local linearization, with **measured** linearization error attached |
| `SKETCHED-BOUNDED` | Compressed state with declared spectral error |
| `EMPIRICAL-NEURAL` | Experimentally useful, no equivalence claim (e.g. backbone migration) |
| `UNSUPPORTED` | Incompatible bases / unsafe numerics — materialization **refuses** |

Merging weakens to the weaker label. Nothing is allowed to imply
exactness it does not have: migration is never called exact, quantization
demotes the label, non-finite or ill-conditioned states are refused.

## Adaptation surfaces

- **Stage A — exact output heads** (`head`): classification, regression,
  reward, confidence, routing. `EXACT-FROZEN`.
- **Stage B — teacher-residual heads** (`residual-head`): r = teacher −
  student; the merged state materializes a correction that permanently
  absorbs what the large model teaches. `EXACT-FROZEN`.
- **Stage C — canonical low-rank adapters** (`lowrank-head`):
  W = W₀ + U C Vᵀ, U/V globally fixed; nodes accumulate Hz = Σ z zᵀ,
  A = Σ (Uᵀe) zᵀ with z = Vᵀh, and C is solved from the merged state.
  `EXACT-FROZEN` within the constrained space.
- **Stage D — projected Jacobian adaptation** (`jacobian:mlp:<l>`):
  f(θ+Pc, x) ≈ f(θ, x) + J_P(x)c inside transformer MLP layers;
  Gauss-Newton states G = Σ JᵀJ, g = Σ Jᵀr solved in c-space.
  `LINEARIZED-BOUNDED` with measured error.

## State operations

`MERGE` (associative, commutative, order-stable), `REMOVE` (exact
contributor deletion), `SCALE`, `PROJECT`, `SKETCH` (declared spectral
error), `VERIFY` (finite/PSD/conditioning), deterministic serialization,
content hashing, HMAC signing, duplicate rejection via lineage,
incompatible-coordinate refusal.

## Quick start

```bash
pip install -e .[dev]

# full experimental program + pass-gate report (≈10 s, CPU only)
python -m rtlm_n.benchmarks.run_all

# test suite (34 tests mapping to the 15 pass gates)
pytest
```

Minimal usage:

```python
import numpy as np
from rtlm_n import EdgeNode, MergeEngine, ModelFactory
from rtlm_n.bases.basis import build_basis
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.models.backbone import TinyTransformer

model = TinyTransformer(seed=0)                 # frozen backbone
basis = build_basis(model, "random-orthogonal") # shared coordinate system
keys = KeyRegistry()

node = EdgeNode("node-0", model, basis, keys, b"secret")
node.absorb_head_batch(tokens, targets, objective_id="clf")  # data stays local

engine = MergeEngine(keys)
engine.submit(node.state("clf"))                # signed compact state travels

factory = ModelFactory(basis)
m = factory.materialize_head(engine.global_state, lam=1e-3)  # W* = (H+λI)⁻¹R
predictions = m.predict(model.forward(new_tokens))
```

## The foundational experiments (all passing)

| # | Experiment | Headline result |
|---|---|---|
| 0 | Exact frozen head | Distributed ≡ centralized to ~1e-11 under non-IID partitions, arbitrary merge order, node removal, event deletion; 29× state compression |
| 1 | Teacher-residual distillation | Distributed residual state ≡ centralized distillation; beats student-only (+0.09) and FedAvg averaging under non-IID |
| 2 | Canonical low-rank adapter | Merged C ≡ centralized constrained solve (1e-15); canonical merge 0.99 vs misaligned-adapter averaging 0.75 |
| 3 | Projected Jacobian | Internal shift recovered (11% coordinate error); 6× better logit fidelity than any output head; linearization error measured and attached |
| 4 | One state, 1,000 models | 1,000 variants (λ-sweeps × tenant exclusions × quantization) with **zero** new representation passes; ~980× less compute than independent runs |
| 5 | Continual inference accretion | Teacher escalation 0.37 → 0.13, quality 1.000 throughout, energy/task 0.018 J → 0.007 J, break-even repaid ~10⁴×, 7.7× verified outcomes/joule vs teacher-always |
| 6 | Heterogeneous edges | 20× size spread, covariate + label skew, fp32 node: full-precision subset stays exact, mixed-precision deviation measured (5e-4), merged beats typical node |
| 7 | Backbone migration | Transported state/head recovers 87% of the reset→retrain gap on a different-width backbone, labeled `EMPIRICAL-NEURAL`, never exact |

Primary metric: **Verified Outcomes per Joule**, from an explicit,
configurable energy model (FLOPs × J/FLOP + per-call and per-byte
overheads for edge vs central devices). Absolute joules are estimates;
relative comparisons between RTLM-N and its baselines use one consistent
accounting.

## Real-data benchmark (all passing): AG News + torch + SGD baselines

```bash
pip install -e .[real]
python -m rtlm_n.benchmarks.real_agnews          # full (~12 min cold, ~1 min cached)
python -m rtlm_n.benchmarks.real_agnews --fast   # smoke
```

The real program swaps the toy backbone for a torch transformer
MLM-pretrained on the real AG News corpus (120k news texts, 4 classes),
runs the identical state machinery on it, and competes against **actually
Adam-trained** baselines with **measured process CPU-seconds** (converted
to joules at an explicit `RTLMN_CPU_WATTS`; RAPL is read when the
platform exposes it):

| # | Result on real data |
|---|---|
| R0 | Distributed ≡ centralized to ~2e-10 on real MLM features; deletion exact; frozen-head accuracy 0.78 |
| R1 | Canonical merged state **0.750** vs independent federated LoRA (SGD, non-IID) **0.538**, one-shot FedAvg heads **0.557**, centralized SGD LoRA **0.742** — and the one-shot state solve measured **46× cheaper** than federated SGD (0.10 vs 4.67 cpu-s) |
| R2 | 1,000 model variants in 7.8 measured cpu-s vs ~1,675 cpu-s for SGD-per-variant: **215× measured** |
| R3 | Cascade on the real workload: escalation declines, quality rises 0.62 → 0.86, energy/task declines |
| R4 | Internal adapter with **exact autograd Jacobians** (`torch.func.jacrev`): 8.7% coordinate recovery error, linearization error 0.085 reported |
| R5 | Head state transported to an independently-pretrained, narrower backbone recovers **78%** of the reset→retrain gap (EMPIRICAL-NEURAL, bridge residual reported) |

The canonical basis that wins R1 is the spec's *teacher-residual
subspace*: leading input directions from the feature-residual
cross-covariance on a public calibration slice, filled out with
activation-SVD directions — data-aware shared coordinates, not random
axes and not per-node learned (misaligned) subspaces.

`rtlm_n/real/hf_backbone.py` wraps any HuggingFace encoder
(`pip install -e .[hf]`) behind the same interface for environments with
hub access; this container's network policy blocks checkpoint hosts, so
the shipped results use the self-pretrained backbone and say so.

## Repository layout

```
rtlm_n/
  core/            guarantee classes, ridge/linear algebra, hashing & signing
  models/          frozen NumPy transformer backbones + registry + FLOP model
  bases/           canonical adaptation bases (random-orthogonal, activation-SVD,
                   teacher-residual) with construction commitments
  accumulators/    NeuralIntelligenceState (merge/remove/scale/sketch/verify)
                   + surface statistics builders
  distributed/     edge nodes, signed merge & lineage engine
  materialization/ model factory: heads, low-rank adapters, Jacobian adapters,
                   rank truncation, quantization, variant sweeps
  inference/       small/large cascade runtime with continual accretion
  migration/       representation bridges, head & state transport
  metering/        FLOP/energy model + measured CPU/RAPL energy meter
  assurance/       proof packages: machine-checkable claims with tolerances
  benchmarks/      experiments 0–7 + run_all + real_agnews (real data)
  real/            torch backbones (MLM-pretrained + HF wrapper), AG News,
                   Adam-trained LoRA/FedAvg/distillation baselines
tests/             39 tests mapping to the 15 non-negotiable pass gates
                   (torch tests auto-skip where torch is absent)
```

## Scope and honesty

The backbone here is a deliberately small pure-NumPy transformer so that
every claim is verifiable end-to-end on CPU in seconds. The mathematics —
sufficient statistics over frozen representations, canonical constrained
subspaces, projected Gauss-Newton states — is dimension-independent; the
engineering path to large backbones is swapping the representation
extractor and Jacobian computation, not changing the state algebra.
What is exact here is exact only for the declared frozen or linearized
objective; unrestricted end-to-end training is out of scope by design,
and every non-exact extension reports its approximation error.
