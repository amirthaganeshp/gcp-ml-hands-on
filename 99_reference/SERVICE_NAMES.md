# What things are called now

Google renamed most of the ML surface for the **June 1, 2026** exam blueprint. Nearly every
tutorial, course and practice exam still uses the old names, so you need both.

**The exam uses the new names. Everything else you will read uses the old ones.**

---

## The big one: Vertex AI → Gemini Enterprise Agent Platform

| you will read | the exam says |
|---|---|
| **Vertex AI** | **Gemini Enterprise Agent Platform** (often shortened to *Agent Platform*) |
| Vertex AI AutoML | **Agent Platform AutoML** |
| Vertex AI Model Registry | **Agent Platform Model Registry** |
| Vertex AI Feature Store | **Agent Platform Feature Store** |
| Vertex AI Pipelines | **Agent Platform Pipelines** |
| Vertex AI Workbench | **Agent Platform Workbench** |
| Vertex AI Prediction / Endpoints | **Agent Platform Inference** |
| Vertex AI Model Monitoring | **Model Monitoring on Agent Platform** |
| Vertex AI Experiments | **Agent Platform Experiments** |

**Model Garden** kept its name. So did **Colab Enterprise**.

## Elsewhere in the stack

| former name | current name |
|---|---|
| Cloud Composer | **Managed Service for Apache Airflow** |
| Cloud Dataproc | **Managed Service for Apache Spark** |
| Cloud Data Loss Prevention (DLP) | **Sensitive Data Protection** |
| Data Catalog | **Dataplex Catalog** / Dataplex Universal Catalog |

## New in the 2026 blueprint

Things that simply did not appear before, and are now examinable:

| | what it is |
|---|---|
| **Model Armor** | Safety filtering against malicious prompting and data exfiltration through an LLM. |
| **Ray on Agent Platform** | Distributed Python/ML compute, alongside Pipelines and Airflow as an orchestration option. |
| **Tabular Workflows** | Managed end-to-end tabular training pipelines. |
| **Gemini / Imagen / Veo** | Foundational models for text, image and video, tuned and served through Agent Platform. |

---

## How this repository handles it

**Current name first, former name in parentheses the first time it appears in a document**, e.g.
*Agent Platform Model Registry (formerly Vertex AI Model Registry)*.

Deliberately **not** a blanket find-and-replace. Most material you will study alongside this repo
still says Vertex AI, and stripping the old name entirely would make it harder to match this up
against a course or a practice question — which is the opposite of useful.

> **Renames are worth a mark on their own.** A question naming *Agent Platform Inference* is asking
> about model serving; if you only know it as "Vertex AI Endpoints" you can lose a question you
> fully understand.
