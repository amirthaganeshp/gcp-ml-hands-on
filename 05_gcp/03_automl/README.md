# AutoML

> **Concept reference:** Success Guide, Part 1.1 · Blueprint section 1 (~13%)
> **Runtime:** ~2 s · **Executes nothing against Google Cloud**

```bash
python 05_gcp/03_automl/automl.py
```

---

## The idea

AutoML sits between a pre-trained API (someone else's data *and* model) and custom training (yours
and yours). **You supply a labelled dataset; the platform searches architectures and
hyperparameters and hands back a deployable model.** You write no model code at all.

That is genuinely useful, and it is oversold — so the script spends as much time on when **not** to
use it.

## Coverage by data type

| data | tasks |
|---|---|
| **Tabular** | classification, regression, forecasting |
| **Image** | classification, object detection |
| **Text** | classification, entity extraction, sentiment |
| **Video** | classification, object tracking, action recognition |

## Data requirements — the part that decides whether it will work

Minimums are low and **the useful numbers are much higher**. Roughly:

| | absolute minimum | actually workable |
|---|---|---|
| image classification | 10 per class | **100+ per class** |
| object detection | 10 per label | 100+ per label |
| tabular | 1,000 rows | 10,000+ |

> **Class imbalance matters more than volume.** AutoML will happily train on 99/1 data and report
> excellent accuracy — the same trap as any classifier, and the reason to check per-class recall
> rather than the headline number. See [`08_monitoring/02_fairness_and_slices`](../../08_monitoring/02_fairness_and_slices/).

## Cost is the usual reason a team abandons it

Training is billed in **node-hours** and the search is not cheap. A tabular AutoML run can cost
more than the engineer-hours it saved, and unlike a notebook it is hard to stop halfway and keep
something useful.

**Set a training budget.** It is a parameter, not a suggestion.

## AutoML vs BQML vs custom

| | choose when |
|---|---|
| **BQML** | data already in BigQuery, SQL team, standard model |
| **AutoML** | **image / text / video**, or tabular with no ML staff and no SQL warehouse |
| **Custom** | custom architecture, custom loss, a framework requirement, or research |

> **For tabular data with the rows already in BigQuery, BQML usually beats AutoML on cost and
> operational simplicity.** AutoML earns its keep on unstructured data, where writing the model
> yourself is a genuine project.

## The exam angle

| the scenario says | the answer |
|---|---|
| "no ML expertise", images or text | **AutoML** |
| "no ML expertise", data in BigQuery, tabular | **BQML** |
| a novel architecture or custom loss | **custom training** |
| "we need it explainable to a regulator" | a simpler model, not AutoML's search |

## What was actually run

**Nothing against Google Cloud.** The client code is real and inside functions; no dataset was
created, no training job submitted, no billing incurred.
