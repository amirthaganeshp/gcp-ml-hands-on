# Responsible AI — slices, proxies, and fairness measures

> **Concept reference:** Success Guide, Part 6.4 · Blueprint section 6 (~13%)
> **Datasets:** Adult income (1994 US census extract)
> **Runtime:** ~16 s on a laptop CPU

```bash
python 08_monitoring/02_fairness_and_slices/fairness_and_slices.py
```

> **On the data.** Adult income is a real census extract and its disparities are historical facts
> of that sample, not claims about people. A fairness example needs real structure — a synthetic
> dataset would only contain the bias you put into it.

---

## The idea

There is one option the exam marks wrong **every time**:

> *"Simply remove the gender column and the model is fair."*

It is wrong because other features **proxy** for the protected attribute. This script demonstrates
that rather than asserting it.

## 1. Aggregate accuracy hides the gap

**Overall accuracy: 0.8497.** One number, looks healthy, and it is what gets reported.

Sliced:

| group | n | accuracy | selection rate | **TPR** | FPR |
|---|---|---|---|---|---|
| Female | 2,905 | 0.9188 | 0.0885 | **0.5303** | 0.0285 |
| Male | 6,144 | 0.8171 | 0.2772 | **0.6520** | 0.1088 |

**Evaluate on slices, not aggregates.** The overall figure is a weighted average that can hide an
arbitrarily large gap — and here the *higher* accuracy group is the one being under-selected.

## 2. The three measures the exam names

| measure | asks | gap here |
|---|---|---|
| **Demographic parity** | are **selection rates** equal? | **+0.1887** |
| **Equality of opportunity** | are **true-positive rates** equal? | **+0.1217** |
| **Equalized odds** | are TPR **and** FPR equal? | 0.1217 |

They are different goals and **can conflict**. Demographic parity can be satisfied by promoting
unqualified candidates; equality of opportunity cannot. **Which one applies is a policy decision,
not a modelling one.**

## 3. Drop the protected attribute — watch nothing improve

Removed `sex_Male`, retrained from scratch:

| | equality-of-opportunity gap |
|---|---|
| with the column | **+0.1217** |
| without the column | **+0.1052** |
| change | −0.0165 — **barely moved** |

**The model never saw the attribute and reproduced the gap anyway.**

## 4. Why — the proxies are still there

If the remaining features carried no information about the protected attribute, you could not
predict it from them. So try:

```
Predicting 'sex' from the REMAINING features:
  accuracy          0.8358
  majority-class    0.6790
  lift over chance  +0.1568
```

Strongest proxies:

| feature | importance |
|---|---|
| `fnlwgt` | 0.1273 |
| `relationship_Wife` | 0.1251 |
| `marital_status_Married-civ-spouse` | 0.1235 |
| `age` | 0.0996 |
| `hours_per_week` | 0.0823 |

> **The strongest proxy is often not the obvious one.** `relationship` and `marital_status` carry
> it explicitly, but census **sampling weights** and **hours worked** carry it too — because they
> were themselves produced by a world in which the attribute mattered.

The model reconstructs the attribute and uses it, without ever being told. **That is why "just
drop the column" is always the wrong option.**

## 5. A mitigation that actually moves the number

Per-group decision thresholds, chosen so true-positive rates match:

| group | threshold | TPR before | TPR after |
|---|---|---|---|
| Female | 0.50 | 0.5303 | 0.5303 |
| Male | **0.64** | 0.6520 | 0.5323 |

| | before | after |
|---|---|---|
| equality-of-opportunity gap | +0.1217 | **+0.0020** |
| overall accuracy | 0.8497 | 0.8471 |

**The gap closes and accuracy pays for it** — 0.26 points. That trade is the honest part: fairness
interventions cost something, and the cost is the conversation, not a reason to skip them.

Note this is an *explicit policy applied openly*, rather than a hoped-for side effect of deleting a
column.

## The exam angle

| the option says | verdict |
|---|---|
| "drop the protected column and the model is fair" | **always wrong** — proxies |
| "evaluate on slices, not just aggregates" | **correct** (TFMA / sliced metrics) |
| equal **selection rates** | demographic parity |
| equal **true-positive rates** | equality of opportunity |
| equal TPR **and** FPR | equalized odds |

**Mitigations worth recognising:** rebalance or augment the training data, remove proxy features,
adjust per-group decision thresholds, add human review for high-impact decisions.

> **Look for bias in the training data first** — unrepresentative sampling, historical bias baked
> into the labels, proxy variables. A model trained on biased labels learns the bias no matter how
> carefully you evaluate it afterwards.

## Things to try

- Switch `PROTECTED` to `race` and re-run. The proxy structure differs, and so does which
  mitigation is defensible.
- Target demographic parity instead of equality of opportunity. The thresholds move in a different
  direction — a concrete demonstration that the two measures genuinely conflict.
