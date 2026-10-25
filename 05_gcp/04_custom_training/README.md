# Custom training

> **Concept reference:** Success Guide, Part 3.2 · Blueprint section 3 (~21%)
> **Runtime:** ~2 s for `submit.py` · **the trainer itself runs locally**

```bash
python 05_gcp/04_custom_training/submit.py          # explains, submits nothing
python -m trainer.task --epochs 2                   # from 04_custom_training/, runs for real
```

---

## The idea

The most effort and the most control: **you write the model code**, package it, and the platform
runs it on hardware you choose. Reach for it when a custom architecture, a custom loss, or a
framework requirement rules the other three levels out.

## What is in this folder

| file | what it is |
|---|---|
| `trainer/task.py` | the training script — **runs locally**, which is the point |
| `trainer/__init__.py` | makes it a package, so `python -m trainer.task` works |
| `Dockerfile` | a custom container, actually built |
| `requirements.txt` | pinned, because the container must be reproducible |
| `submit.py` | the three submission paths, explained and **not executed** |

## Prebuilt vs custom container — the real decision

| | use when |
|---|---|
| **Prebuilt container** | your framework and version are on the supported list, and your deps are pip-installable |
| **Custom container** | a system library, an unusual framework version, a private package index, or anything apt-get |

**Prebuilt is not "the easy one" and custom "the hard one"** — prebuilt is a *constraint*. The
moment you need a system package, the choice is made for you.

## Hyperparameter tuning, and the metric-name trap

The tuning service optimises a metric your trainer **reports**, by name. If the name in the job
spec does not match the name your code reports, the job runs to completion and optimises
**nothing** — no error, just a search that never improves.

> Report it explicitly with `hypertune`, and make the name in the spec and the code the same
> string. This is the single most common way a tuning job wastes a day.

## Deploy vs batch — the cost question

| | |
|---|---|
| a user is waiting | deploy to an **endpoint**, pay per replica-hour |
| a table or file to score | **batch prediction**, no endpoint at all |

Quantified in [`07_serving/01_batch_vs_online`](../../07_serving/01_batch_vs_online/): an
always-warm replica costs the same as batch-scoring **48 million rows/day**.

## Local first — why, and what it catches

`python -m trainer.task --epochs 2` runs the real trainer on this laptop. That catches, in seconds
and for free, what would otherwise cost a queued cloud job each time: import errors, argument
parsing, a mis-shaped tensor, a path that only exists on your machine.

> **The cloud is a slow, expensive debugger.** Everything that can fail locally should fail
> locally first.

## The exam angle

| the scenario says | the answer |
|---|---|
| custom architecture / loss / framework | **custom training** |
| needs a system library | **custom container** |
| "the tuning job didn't improve anything" | the **metric name** does not match |
| score a table nightly | **batch prediction**, not an endpoint |
| distributed across machines | see [`06_hardware/`](../06_hardware/) |
