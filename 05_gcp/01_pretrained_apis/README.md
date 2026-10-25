# Pre-trained APIs

> **Concept reference:** Success Guide, Part 1.2 · Blueprint section 1 (~13%)
> **Runtime:** ~2 s · **Executes nothing against Google Cloud**

```bash
python 05_gcp/01_pretrained_apis/pretrained_apis.py
```

---

## The idea

The first of Google Cloud's four levels of effort: **a REST call against a model somebody else
trained, tuned and serves.** No data, no model code, no training job.

| level | your data | your model code | to first result |
|---|---|---|---|
| **Pre-trained API** | none | none | **minutes** |
| AutoML | your labels | none | hours |
| BigQuery ML | your table | SQL only | minutes |
| Custom training | your data | you write it | days |

> **The question is never "which is most accurate" — it is "how unusual is my problem".** Pick the
> least custom option that fits.

## The APIs, and what they actually do

| API | does | does **not** |
|---|---|---|
| **Vision** | OCR, labels, objects, faces, safe-search | understand document *layout* |
| **Document AI** | **layout-aware** parsing → typed entities, tables, key-value pairs | work on arbitrary photos |
| **Natural Language** | entities, sentiment, syntax, classification | domain-specific jargon without tuning |
| **Speech-to-Text** | transcription, diarisation, punctuation | separate overlapping speakers perfectly |
| **Translation** | 100+ languages, glossaries, batch | preserve formatting in complex documents |

## The distinction that gets tested

> **Vision OCR extracts text. Document AI understands layout.**
>
> *"Extract line items from invoices"* is **Document AI**, every time — it returns typed entities
> and table structure, not a flat string you then regex.
>
> And it costs roughly **an order of magnitude more per page**, which is the counter-signal when a
> question stresses volume. At high enough volume, OCR plus your own parsing can be the right
> answer.

## Cost and latency decide more architectures than accuracy

| | pre-trained API | a deployed endpoint |
|---|---|---|
| cost at rest | **nothing** | per node-hour, always |
| cost model | per request / per page | per replica-hour |
| scaling | Google's problem | `min_replica_count`, yours |

**A pre-trained API costs nothing when nobody calls it.** That single property decides a great many
architectures for low or spiky volume — see [`07_serving/01_batch_vs_online`](../../07_serving/01_batch_vs_online/).

## The exam angle

| the scenario says | the answer |
|---|---|
| common task, common data, "as fast as possible" | **pre-trained API** |
| "extract fields from forms/invoices/receipts" | **Document AI** |
| "read text from a photo" | **Vision OCR** |
| domain jargon the API gets wrong | **tune it**, or move up a level |
| "no ML expertise on the team" | *not* custom training |

## What was actually run

**Nothing against Google Cloud.** Every cloud import sits inside a function, so the script runs and
teaches with no credentials, project or billing. The code in those functions is real and
deployable; it has not been executed, and any response shown is labelled as illustrative.
