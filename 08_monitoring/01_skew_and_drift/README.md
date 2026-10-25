# Skew and drift

> **Concept reference:** Success Guide, Part 6.1 · Blueprint section 6 (~13%)
> **Datasets:** Adult income, split into a training set and two serving windows
> **Runtime:** ~15 s on a laptop CPU

```bash
python 08_monitoring/01_skew_and_drift/skew_and_drift.py
```

---

## The idea

Four different failures get called "drift" casually. The exam separates them precisely, and so
does Model Monitoring.

| term | compares | meaning |
|---|---|---|
| **Training-serving skew** | serving data vs **training** data | production never matched training in the first place |
| **Prediction / data drift** | serving **now** vs serving **earlier** | the world moved after you deployed |
| **Concept drift** | the **X → y relationship** | same inputs, different correct answer |
| **Feature attribution drift** | which features the model leans on | right answers, changed reasons |

**Model Monitoring's metrics:** Jensen-Shannon divergence for **numerical** features, L-infinity
distance for **categorical** features. This script computes both.

## The headline result

| failure | worst input divergence | | accuracy |
|---|---|---|---|
| *(baseline)* | — | | 0.8008 |
| training-serving skew | 0.6378 | caught | 0.7282 |
| data drift | 0.6060 | caught | 0.8024 |
| **concept drift** | **0.0000** | **MISSED** | **0.6500** |

**Read the bottom row twice.** Every input distribution is unchanged — divergence is *exactly*
zero, because they are literally the same rows — and the model has lost **15 accuracy points**.

> **No amount of input monitoring will ever fire on this.** Detecting concept drift requires
> **ground-truth labels** and continuous evaluation of accuracy, not distribution comparison.
>
> Exam shape: *"the features look stable but accuracy is falling"* → **concept drift**, every time.

## And the opposite failure: a true alarm on a healthy model

Look at the data-drift row. Divergence **0.6060** — well over any sane threshold, so the alert
fires. Accuracy: **0.8024**, marginally *better* than baseline.

> The inputs moved substantially and the model was **fine**. Input drift is a signal to go and
> look, not proof of damage. **Alerting on divergence alone produces exactly this false positive**
> — which is the same lesson as choosing an alerting condition in the PDE repo.

## 1. Skew needs a training baseline — drift does not

That asymmetry is an exam answer discriminator:

| | baselines against | available when… |
|---|---|---|
| **skew detection** | the **training dataset** | you still have the training data |
| **drift detection** | **earlier serving traffic** | always — you always have yesterday's traffic |

> **If the original training data is unavailable, you cannot configure skew detection at all** —
> only drift. Options that offer skew detection in that scenario are eliminable on that basis
> alone.

## 2. Feature attribution drift — the subtle one

Permutation importance before and after:

| feature | before | after | change |
|---|---|---|---|
| age | 0.156 | 0.074 | **−0.082** |
| … | | | |

The model can hold roughly its accuracy while relying on **different features** — right answers
for changed reasons. That is a *leading* indicator rather than a failure you can already see.

> **"Overall accuracy hasn't moved much but its behaviour feels different"** → feature attribution
> drift.

## The exam angle

| the scenario says | the answer |
|---|---|
| features stable, **accuracy falling** | **concept drift** — needs labels, not distributions |
| serving population never matched training | training-serving **skew** |
| **training data is gone**, monitor anyway | **drift** detection only |
| accuracy steady, model "behaving differently" | **feature attribution** drift |
| numerical feature distributions | **Jensen-Shannon** divergence |
| categorical feature distributions | **L-infinity** distance |

## Things to try

- Set the concept-drift flip probability to 0.9. Accuracy collapses further and the input
  divergence is *still* exactly zero — the point gets sharper, not weaker.
- Raise `JS_ALERT` and re-run. Both the skew and drift alerts sit around 0.6 here, so no threshold
  separates the harmful case from the harmless one — which is precisely why divergence alone is a
  weak alerting signal.
