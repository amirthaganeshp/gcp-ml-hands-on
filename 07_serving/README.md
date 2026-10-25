# Part 07 — Serving and scaling models

> **Blueprint section 4 (~20% of the exam)** · Success Guide, Part 4
> The single largest domain, and the one most repos skip entirely.

| example | covers | state |
|---|---|---|
| [`01_batch_vs_online/`](01_batch_vs_online/) | the decision, and the cost bug inside it | ✅ runnable |
| [`02_rollout_strategies/`](02_rollout_strategies/) | canary vs A/B vs blue-green vs shadow | ✅ runnable |
| [`03_model_optimization/`](03_model_optimization/) | quantization, pruning, batching | ✅ runnable |

Everything below this line is **📝 documented, not demonstrated** — it needs a real endpoint, and
nothing here pretends otherwise.

---

## Why this part exists

Parts 01–06 are about **building** a model. This part is about **running** one, which is where the
exam's weight has moved: sections 4 and 6 of the blueprint are a third of the paper, and both are
about what happens after training.

## Where to serve — 📝

| target | choose when |
|---|---|
| **Agent Platform Inference** *(formerly Vertex AI Endpoints)* | default managed serving; traffic splitting, autoscaling, monitoring and explanations built in |
| **Cloud Run** | scale-to-zero, spiky or low traffic, simple container, HTTP; GPU available |
| **GKE** | maximum control, multi-model packing, custom stacks (vLLM, Triton), existing Kubernetes estate |
| **BigQuery ML `ML.PREDICT`** | scoring where the data already lives, SQL consumers |
| **Edge / on-device** | offline operation, or latency/privacy at the device |

## The serving contract — 📝

A **custom serving container** must expose an HTTP server implementing:

| | |
|---|---|
| `AIP_PREDICT_ROUTE` | the predict route |
| `AIP_HEALTH_ROUTE` | health, returning 200 when ready |
| `AIP_HTTP_PORT` | the port to listen on (default 8080) |

**Custom Prediction Routines (CPR)** let you supply Python pre/post-processing around a model
without hand-writing a server — the answer to *"we need to transform the request before it reaches
the model and format the response after."*

**Prebuilt containers** exist for TensorFlow (SavedModel), PyTorch, scikit-learn (joblib/pickle)
and XGBoost, each with expected artifact filenames and a GCS `artifactUri`.

## Model Registry and versions — 📝

**Agent Platform Model Registry** *(formerly Vertex AI Model Registry)* holds model **versions** and
**aliases** (`default`, `champion`). Deployment puts one or more versions behind an **Endpoint**
with a **traffic split** — the mechanism behind canary, A/B and blue-green. Shadow is not a split:
it mirrors (or replays) requests to the new model and discards its responses.

## Endpoint types — 📝

| type | characteristics |
|---|---|
| **Shared public** | default public DNS, shared infrastructure, request/response size limits |
| **Dedicated public** | own network path — higher throughput, larger payloads, streaming |
| **Private (Private Service Connect)** | reachable only from your VPC; **never traverses the public internet** |

> **"Predictions must never traverse the public internet"** → a **PSC private endpoint**, combined
> with VPC Service Controls. **IAM alone does not satisfy a network-path requirement** — the same
> *who* vs *from where* distinction as VPC-SC.

## Autoscaling — 📝

`min_replica_count`, `max_replica_count`, and a target metric — CPU utilisation (default target
~60%) or **GPU duty cycle** for accelerator-backed deployments.

The two requirements that pull in opposite directions, and both are examinable:

| requirement | answer |
|---|---|
| "must not pay when idle" | scale to zero (Cloud Run) |
| "no cold starts, consistent p99" | `min_replica_count >= 1` |

[`01_batch_vs_online/`](01_batch_vs_online/) computes what each choice costs: **92% of the bill**
wasted on an endpoint used two hours a day.

## What is measured here, and what is not

| | |
|---|---|
| ✅ **Measured** | cost crossovers, blast radius, A/B sample sizes, quantization size/accuracy/latency, pruning, batching throughput |
| 🔬 **Modelled and labelled** | queue wait `(B−1)/(2λ)` in the batching example — the compute beside it is measured |
| 📝 **Documented** | endpoints, containers, Model Registry, PSC, autoscaling — all need a real project |

**Nothing here simulates a Google Cloud endpoint.** Where there is no honest local proxy, the
README says so rather than inventing one.
