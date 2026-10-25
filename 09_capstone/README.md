# Capstone — one model, all the way through

> **Concept reference:** the whole repository · Blueprint: all six sections
> **Datasets:** Adult income
> **Runtime:** ~9 s on a laptop CPU

```bash
python 09_capstone/capstone.py
```

---

## The idea

**Nothing here is new.** Every step was built in an earlier part — this file just runs them in
order, which is the one thing no single example can show.

```
data  ->  split  ->  train  ->  EVALUATE  ->  serve  ->  MONITOR
            |          |           |            |          |
            |          |           |            |          +-- 08_monitoring
            |          |           |            +-- 07_serving
            |          |           +-- 01_foundations (metrics, slices)
            |          +-- 02/03/04 (model, framework)
            +-- 01_foundations (split, leakage)
```

**The two in capitals are where models actually fail.** Everyone trains. Far fewer evaluate
honestly, and fewer still monitor for the failure that never shows up in the inputs.

## 1. Evaluate honestly

| | |
|---|---|
| accuracy | 0.8507 |
| ROC-AUC | 0.9004 |
| **majority baseline** | **0.7511** |
| lift over it | **+0.0996** |

**On imbalanced data, accuracy alone is nearly meaningless.** Predicting the majority class every
time already scores 0.7511 — so the headline number carries about ten points of actual information.

And the same model, sliced:

| group | n | accuracy | **TPR** |
|---|---|---|---|
| Female | 2,905 | 0.9222 | **0.5504** |
| Male | 6,144 | 0.8169 | **0.6551** |

One aggregate number, two very different experiences. **Reporting only the first is how a model
ships with a gap nobody has looked at.**

## 2. The serving decision

Scoring 2,000,000 rows/day:

| | $/day |
|---|---|
| batch (no endpoint) | **0.25** |
| online (1 warm replica) | 6.00 |
| ratio | **24×** |

**But cost does not decide this — the requirement does.** A nightly table → batch, and never create
an endpoint. A user waiting on a screen → online, and pay for warm replicas. This model scores a
nightly customer table, so: batch.

## 3. The monitoring plan — and the seam

| what to watch | catches | here |
|---|---|---|
| input distributions | skew and data drift | **JS 0.0519** |
| **accuracy on labels** | **concept drift** | **0.8576 → 0.6559** |
| sliced metrics | a gap in one group | see §1 |
| feature attributions | right answers, new reasons | `08_monitoring` |

> **Input divergence 0.0519 — nothing to alert on. Accuracy fell 20.2 points.**
>
> That is the seam. The inputs are fine and the model is not. No input monitor would fire; only
> labels reveal it. **Every dashboard built on feature distributions would show green through
> this.**

## 4. Four ways it breaks

| | |
|---|---|
| **a) Leakage** | a feature that *is* the label → **1.0000** vs honest 0.8507. Spectacular offline, worthless in production — that column does not exist at prediction time. |
| **b) Wrong metric** | 0.8507 against a 0.7511 baseline. On rarer positives the gap closes to nothing and accuracy stops carrying information. |
| **c) Skew** | preprocessing fitted before the split, or written twice. One definition is the fix: `Pipeline`, `TRANSFORM`, or a Feature Store. |
| **d) Concept drift** | §3 — inputs unchanged, accuracy gone. Needs labels, not distributions. |

## What this demo omits — stated, not hidden

| missing | why it matters |
|---|---|
| **no cloud** | nothing deployed; the serving decision is arithmetic, not an endpoint |
| **no pipeline** | one script, not an orchestrated DAG — [`06_pipelines`](../06_pipelines/) has that |
| **no retraining** | the monitor detects; nothing acts on it |
| **no feature store** | train and serve read the same in-memory frame — exactly the luxury production lacks |
| **one model** | no registry, no versions, no rollout |
| **tabular only** | images, text and agents each change the shape |

Those omissions are the difference between an example and a system. **Naming them is more useful
than pretending otherwise.**
