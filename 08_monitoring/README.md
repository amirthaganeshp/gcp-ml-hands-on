# Part 08 — Monitoring AI solutions

> **Blueprint section 6 (~13% of the exam)** · Success Guide, Part 6
> Largely **new** in the June 2026 blueprint, and absent from most older material.

| example | covers | state |
|---|---|---|
| [`01_skew_and_drift/`](01_skew_and_drift/) | the four failures, and the one input monitoring cannot see | ✅ runnable |
| [`02_fairness_and_slices/`](02_fairness_and_slices/) | slices, proxies, the three fairness measures | ✅ runnable |

Everything below is **📝 documented, not demonstrated** — it needs a real endpoint or a live LLM.

---

## The headline from the runnable half

| failure | input divergence | | accuracy |
|---|---|---|---|
| *(baseline)* | — | | 0.8008 |
| training-serving skew | 0.6378 | caught | 0.7282 |
| data drift | 0.6060 | caught | 0.8024 |
| **concept drift** | **0.0000** | **MISSED** | **0.6500** |

**Input monitoring misses concept drift completely**, and concept drift is the failure that
quietly destroys a model looking healthy on every dashboard. Detecting it needs **ground-truth
labels**, not distribution comparison.

## Explainable AI — which attribution method — 📝

**The differentiability split is the whole question.**

| method | use for |
|---|---|
| **Sampled Shapley** | **non-differentiable** models — trees, ensembles, AutoML tabular, XGBoost |
| **Integrated Gradients** | **differentiable** models — neural networks on tabular, text, images |
| **XRAI** | images, where **region-level** explanations beat pixel-level ones |
| **Example-based** | "which training examples are most like this input?" |

> Integrated Gradients requires gradients and therefore **cannot** be used on a tree ensemble. If
> an option pairs Integrated Gradients with XGBoost or AutoML tabular, **eliminate it**.
>
> XRAI beats Integrated Gradients on images when a human needs to see *which region* mattered
> rather than a pixel heat map.

## Securing AI systems — Model Armor and friends — 📝

**New, and explicitly named in the blueprint.** The exam guide names three tools: *Regex, safety
filters, and Model Armor.* **They are a ladder, not alternatives.**

| threat | control |
|---|---|
| prompt injection / jailbreak | **Model Armor** prompt screening |
| sensitive data pasted into prompts | **Model Armor + Sensitive Data Protection** |
| model emitting PII or secrets | Model Armor **response** screening |
| malicious URLs in output | Model Armor URL detection |
| harmful content | **safety filters** (harm category × threshold) |
| off-topic / out-of-policy | system instructions + safety filters |
| deterministic patterns (account numbers, internal IDs) | **regex** filters |
| model or data exfiltration over the network | **VPC Service Controls** |
| unauthorised access to models/endpoints | **IAM** least privilege |
| keys under customer control | **CMEK** |

> **Regex** catches known deterministic patterns. **Safety filters** handle harm categories.
> **Model Armor** handles adversarial *intent* — injection and jailbreaks — which regex cannot.
>
> For *"employees might paste customer data into an LLM"*, the answer is **Model Armor (with
> Sensitive Data Protection)**, not a safety filter.

**Carry the IAM/VPC-SC distinction over intact:** IAM controls **who**; VPC Service Controls
controls **from where**. A legitimately-authorised insider exporting a model or its training data
is a **VPC-SC** answer, because their IAM permissions are valid.

## Monitoring gen AI and agents — 📝

Traditional model monitoring does not fit an agent. What you monitor instead:

| dimension | metric |
|---|---|
| **response quality** | model-based metrics — groundedness, coherence, instruction-following, safety |
| **trajectory** — did it call the right tools in the right order? | tool-use trajectory metrics: exact match, in-order match, any-order match, precision, recall |
| **latency and cost** | tokens per request, thinking tokens, tool-call count, end-to-end latency |
| **safety** | Model Armor hit rates, safety-filter blocks |
| **traces** | OpenTelemetry-style tracing of the agent's steps in Agent Runtime |

> **For an agent, a correct final answer reached by the wrong path is still a defect** — it will
> not generalise. **Trajectory evaluation is why agent evaluation differs from model evaluation**,
> and it is the concept the new blueprint is testing.

The repo's agent example is [`05_gcp/05_adk_agents/`](../05_gcp/05_adk_agents/).

## What is measured here, and what is not

| | |
|---|---|
| ✅ **Measured** | Jensen-Shannon and L-infinity divergence, all four drift types, sliced metrics, fairness gaps, proxy recoverability, threshold mitigation |
| 📝 **Documented** | Model Monitoring configuration, Explainable AI, Model Armor, safety filters, agent trajectory evaluation |

**Nothing here simulates Model Monitoring or Model Armor.** The *statistics* they compute are
real and computed locally; the services are not pretended.
