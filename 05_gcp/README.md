# Google Cloud

**What will Google Cloud build for you, and what must you build yourself?**

Google Cloud sells machine learning at four levels of effort. The most valuable skill this section
teaches — and the one the exam tests hardest — is **choosing between them**, not implementing any
one of them.

| level | your data | your model code | to first result | use when |
|---|---|---|---|---|
| **Pre-trained API** | none | none | minutes | a common task on common data |
| **AutoML** | your labels | none | hours | your data, standard task, no ML staff |
| **BigQuery ML** | your table | SQL only | minutes | the data is already in BigQuery |
| **Custom training** | your data | you write it | days | a custom architecture or loss |

> **The question is not "which is most accurate" — it is "how unusual is my problem".** Pick the
> least custom option that fits. A question mentioning *"no ML expertise"* or *"as quickly as
> possible"* is steering you away from custom training; one mentioning a novel architecture or a
> custom loss is steering you towards it.

---

## What was actually run

This section is **write-only against the cloud**, deliberately. No project was touched, no job
submitted, no billing incurred. Being precise about that matters more than usual here, because
everywhere else in this repository the numbers are measured.

| file | executed? | what that means |
|---|---|---|
| `01_pretrained_apis/pretrained_apis.py` | ✅ runs | explains and decides; the five API clients are **defined, not called** |
| `02_bigquery_ml/bigquery_ml.py` | ✅ runs | explains and decides |
| `02_bigquery_ml/models.sql` | ❌ | 9 real `CREATE MODEL` statements, never submitted |
| `03_automl/automl.py` | ✅ runs | explains and decides; Vertex calls **defined, not made** |
| `04_custom_training/submit.py` | ✅ runs | explains and decides; jobs **defined, not submitted** |
| **`04_custom_training/trainer/task.py`** | ✅ **really trains** | **0.8000 test accuracy in 2.6 s, locally** |
| `04_custom_training/Dockerfile` | ✅ **builds** | `docker build` succeeds in **265 s**; the image is assembled locally and **never pushed** |
| `05_adk_agents/adk_agents.py` | ✅ runs | explains and decides; agents **defined, never invoked** — but §6 **really measures** the orchestration patterns |

**Every cloud import sits inside a function**, never at module level. That is why all six scripts
run with no credentials, no project and no `google-cloud-*` libraries installed — and it is a
reasonable pattern for real code too, keeping a heavy credential-dependent dependency off the
import path.

The ADK file is the one exception to "explains but does not measure": §6 runs a pure-`asyncio`
simulation of the sequential/parallel/loop semantics with stub coroutines, and reports real
wall-clock. **No model is called and ADK itself is not imported at runtime** — it measures the
orchestration pattern, not the framework. Every ADK class and field it names was verified by
introspecting **google-adk 1.32.0** rather than recalled.

The one thing that genuinely executes is the **trainer package**, and that is deliberate: it is
both the most exam-relevant executable artefact and the place where running it locally first saves
the most time.

---

## Contents

| folder | covers |
|---|---|
| [`01_pretrained_apis/`](01_pretrained_apis/) | Vision, Document AI, Natural Language, Speech, Translation — and cost/latency, which decides more architectures than accuracy |
| [`02_bigquery_ml/`](02_bigquery_ml/) | the `MODEL_TYPE` map, what BQML automates and hides, the limits, and [`models.sql`](02_bigquery_ml/models.sql) |
| [`03_automl/`](03_automl/) | coverage by data type, the workflow, data requirements, and when *not* to use it |
| [`04_custom_training/`](04_custom_training/) | a trainer package that **runs locally**, a Dockerfile, submission, hyperparameter tuning, and deploy-vs-batch |
| [`05_adk_agents/`](05_adk_agents/) | ADK agent types, tools, state, the four deployment doors — and a **measured** comparison of the orchestration patterns |
| [`06_hardware/`](06_hardware/) | CPU vs GPU vs TPU, the utilisation trap, distribution strategies, data vs model parallelism |

---

## The platform, service by service

> **Naming:** Vertex AI is now **Gemini Enterprise Agent Platform** in the June 2026 blueprint.
> Full rename table: [`99_reference/SERVICE_NAMES.md`](../99_reference/SERVICE_NAMES.md).

```
MODELS                                  AGENTS
  Model Garden ..... catalogue           Agent Garden .... prebuilt agents, templates
  Tuning ........... fine-tune / distil  ADK ............. code-first framework
  Evaluation ....... Gen AI Eval         MCP Servers ..... register tool servers
  Model Registry ... versions + aliases  RAG Engine ...... ingest->chunk->embed->retrieve
  Endpoints ........ online serving      Vector Search ... ANN index
  Batch Inference .. no endpoint needed  Deployments ..... Agent Runtime
  Provisioned                            Memory Bank ..... memory ACROSS sessions
    Throughput ..... reserved capacity   Sessions ........ per-conversation state
                                         Agent Registry .. catalogue of agents/tools
  Datasets ......... managed, labelling  Policies ........ rules on agent traffic
  Feature Store .... BigQuery-backed     Gateways ........ ingress/egress choke point
  Experiments ...... run tracking        Security ........ Model Armor, safety filters
  Training ......... CustomJob, HP tune  Topology ........ map of agents and tools
  Ray .............. distributed Python  Evaluation ...... eval cases, rubrics
  Pipelines ........ managed Kubeflow
  Metadata ......... lineage store      NOTEBOOKS
  Monitoring ....... skew / drift         Workbench ....... managed JupyterLab
                                          Colab Enterprise  zero-setup runtimes
```

**Note the deliberate symmetry — it is the fastest way to remember the console:**

| models | agents |
|---|---|
| Model **Garden** | Agent **Garden** |
| Model **Registry** | Agent **Registry** |
| **Endpoints** | **Deployments** |
| Evaluation | Evaluation |

Models and Agents are the same design applied to two different units of work. Once you see the
mirror, the rest follows.

### The services this repo does not have a folder for

Examinable, and covered here rather than pretended at:

| service | what it is | why it matters |
|---|---|---|
| **Feature Store** | BigQuery-backed feature storage with online serving | the standard answer to **training-serving skew** — one definition, read by both paths ([`08_monitoring`](../08_monitoring/)) |
| **Experiments** | run tracking and comparison | blueprint §2.3; the answer to "which hyperparameters produced that result" |
| **Metadata** | ML Metadata lineage store | artifact lineage — which data produced which model |
| **Workbench / Colab Enterprise** | managed notebooks | blueprint §2.2. Workbench = managed JupyterLab instances; Colab Enterprise = zero-setup on runtime templates. **Idle shutdown** is the cost control. |
| **Ray on Agent Platform** | managed Ray clusters | distributed Python, alongside Pipelines and Airflow as an orchestration option |
| **Model Garden** | catalogue of Google, partner, open and your own models | where "select the appropriate model for the task" is answered |
| **Tuning** | fine-tune or distil a foundation model | **LoRA/PEFT adapters are the default**; full fine-tuning needs deep domain adaptation and a large dataset |
| **Provisioned Throughput** | reserved generative capacity | the answer to "guaranteed capacity for a generative workload" |

> **Fine-tuning also cuts inference cost** when it lets you delete a huge few-shot prompt. *"Our
> prompt is 8,000 tokens of examples and latency is unacceptable"* is a legitimate fine-tuning
> answer — the tuned model internalises what the examples were teaching.

---

## The handful of facts that carry the most weight

**Vision OCR extracts text; Document AI understands layout** and returns typed entities. *"Extract
line items from invoices"* is Document AI every time — and it costs roughly an order of magnitude
more per page, which is the counter-signal when a question stresses volume.

**A pre-trained API costs nothing at rest.** A deployed Vertex endpoint bills per node-hour whether
or not anyone calls it, and `min_replica_count` cannot be 0. *"Predictions needed once a day"* is
**batch**, and an endpoint is the distractor.

**BQML's pitch is that the data never moves** — no export, no training job, no serving system. It
is priced on **bytes scanned**, so `SELECT *` in a `CREATE MODEL` is a real cost. `ML.PREDICT` is
batch; online serving means exporting to a Vertex endpoint.

**`ARIMA_PLUS` is the managed forecasting answer**, and `time_series_id_col` fits thousands of
independent series in one statement.

**Random splits are the default nearly everywhere** — BQML's `AUTO_SPLIT`, AutoML's default — and
they leak the future on time-ordered data. That is the failure measured in
[02_traditional_ml/06](../02_traditional_ml/06_time_series/), where it made the error look 2×
better than it was. Set `SEQ` / a timestamp split.

**AutoML cannot see leakage.** Hand it the bike-sharing table from
[01_foundations/05](../01_foundations/05_data_leakage/) with `casual` and `registered` still
present and it will build a superb model of `cnt = casual + registered`, report near-perfect
metrics, and be worthless. It removes the architecture search, not the thinking.

**A trainer package takes hyperparameters as command-line flags** and writes to `AIP_MODEL_DIR`.
Anything hardcoded cannot be tuned — and the tuning metric name must match what the trainer
reports, or the job runs, bills you, and reports nothing with no error.

### On the container, now that it has actually been built

**Both `pip install` layers were no-ops**, which is the most useful thing the build revealed:

```
#8 [4/6] RUN pip install -r requirements.txt
#8 Requirement already satisfied: keras ...          DONE 1.9s
#9 [5/6] RUN pip install cloudml-hypertune
#9 Requirement already satisfied: cloudml-hypertune  DONE 1.2s
```

The Vertex prebuilt base already ships Keras **and** `cloudml-hypertune` — the very package the
Dockerfile's comment says you must add so hyperparameter tuning can report back. Nothing was
downloaded. That is the argument for a prebuilt base made concrete: the platform-specific pieces
are in there already, and this `requirements.txt` is closer to documentation of intent than a
functional requirement.

**The base is Python 3.10 on conda**, not the 3.12 this repository runs locally. A good reminder
that the image — not your venv — is what the training job actually executes in.

**`.dockerignore` is not optional, and Docker does not read `.gitignore`.** Without one, the build
context carried `local_output/model.keras` and `submit.py` to the daemon on every build even though
the Dockerfile never `COPY`s them. They never reach the image, but they are uploaded, hashed and
folded into the cache key — a common cause of a mysteriously slow build.

**Built, never pushed.** Pushing needs a real project and an Artifact Registry repository; the only
project configured on this machine is a production one, so the build stops at the local image
deliberately.

### On agents

**ADK is the framework; Agent Runtime is one of four hosts.** Separate decisions. `vertexai.agent_engines`
ships wrappers for LangChain, LangGraph and AG2 as well as ADK, so the managed host is not
ADK-only — and an ADK agent deploys just as happily to Cloud Run. The rename from Agent Engine
leaked: the API resource is still `reasoningEngines`.

**"Workflow agents contain no LLM" is structural, not a simplification.** `SequentialAgent`,
`ParallelAgent` and `LoopAgent` declare only `name`, `description`, `sub_agents`, `parent_agent`
and two callbacks — plus `max_iterations` on the loop. **There is no `model` field to put one in.**
`LlmAgent` has `model`, `instruction`, `tools`, `planner` and `output_key`.

**The agent-type question is "who decides control flow — the model, or you?"** Every `LlmAgent`
decision is a model call, paid and waited for. Letting a model choose between two branches you
could select with an `if` is the agent-shaped version of reaching for custom training when BigQuery
ML would do.

**A parallel fan-out is only as fast as its slowest branch.** Sequential time is the **sum**;
parallel is the **max** — so the speedup is `sum/max`, governed by how *uneven* the branches are,
not how many there are. Measured at 0.93 s → 0.46 s for branches of 0.45/0.15/0.30 s.

**Parallel branches must not read each other's `output_key`.** They complete in finish order, not
declared order — measured above as `traffic → labour → sales` against a declared `sales, traffic,
labour`. A branch reading a key another is still writing is a race that returns empty or stale
data rather than raising.

**`InMemorySessionService` is correct locally and silently wrong deployed.** State lives in one
process, so it is lost on restart and not shared between replicas — on Cloud Run with more than one
instance, a user's second message can land somewhere that has never heard of them. Intermittent,
load-dependent, and nothing raises.

**Cloud Run scales to zero; a Vertex prediction endpoint cannot.** For an agent called rarely, that
single fact outweighs every capability comparison.

> **Scope note:** ADK is **not on the ML Engineer exam** as of the June 2026 guide — nor are Agent
> Garden, MCP, A2A, Memory Bank or Agent Registry. Despite the Vertex AI → Agent Platform rename,
> the stated objectives stayed model-centric. This section is here because agents are what the
> platform is being built around, and because §7's cost reasoning is ordinary engineering judgement
> that outlasts a guide revision.

---

## Running it

```bash
python 05_gcp/01_pretrained_apis/pretrained_apis.py
python 05_gcp/02_bigquery_ml/bigquery_ml.py
python 05_gcp/03_automl/automl.py
python 05_gcp/04_custom_training/submit.py
python 05_gcp/05_adk_agents/adk_agents.py

# the one that actually trains
python 05_gcp/04_custom_training/trainer/task.py --epochs 3
```

To run any of the cloud code for real you need `pip install -r requirements-dl.txt` (which
includes `google-cloud-aiplatform`, `google-cloud-bigquery` and `kfp`), a project, and
`gcloud auth application-default login`. **Replace every `your-project-id` placeholder** — nothing
in this repository contains a real project ID, and it should stay that way.

**Actually doing it: [`RUNNING_ON_YOUR_OWN_PROJECT.md`](RUNNING_ON_YOUR_OWN_PROJECT.md)** — the
runbook for pushing the image to Artifact Registry and submitting a real training job on a
**personal** project. It covers isolating `gcloud` configurations so a work default cannot leak,
what each step costs, and how to clean up. Read the warning at the top before running anything:
following it spends money, and a deployed endpoint bills whether or not you call it.

---

Next: [`06_pipelines`](../06_pipelines/) — Kubeflow Pipelines v2, components, and a conditional
deployment gate.
