# Part 01 — Foundations

**What is a model, and how do you know whether it is any good?**

Six runnable demonstrations of the ideas every later part assumes. Nothing here needs a neural
network or a GPU — the whole part runs in about two minutes on a laptop CPU, and it is the part
worth doing first even if the deep-learning material looks more urgent.

| # | Folder | Question it answers | Runtime |
|---|---|---|---|
| 01 | [`01_train_test_split_and_cv`](01_train_test_split_and_cv/) | How do I split data honestly, and why is one split not enough? | ~33 s |
| 02 | [`02_bias_variance_tradeoff`](02_bias_variance_tradeoff/) | Why do models fail, and would more data fix it? | ~25 s |
| 03 | [`03_confusion_matrix_and_metrics`](03_confusion_matrix_and_metrics/) | Which metric, and why is accuracy so often the wrong one? | ~8 s |
| 04 | [`04_roc_vs_pr_curves`](04_roc_vs_pr_curves/) | ROC or precision–recall? | ~32 s |
| 05 | [`05_data_leakage`](05_data_leakage/) | Why was it brilliant offline and useless in production? | ~11 s |
| 06 | [`06_feature_engineering`](06_feature_engineering/) | How much is in the features rather than the model? | ~13 s |

Run any of them standalone from the repository root:

```bash
python 01_foundations/02_bias_variance_tradeoff/bias_variance.py
```

---

## The through-line

These six are not independent topics. They are one argument:

**You cannot judge a model from a single split** (01) — the same model on the same data swings 8.7
accuracy points on the luck of the draw. So you cross-validate, and report the spread.

**Once you can measure honestly, you can see the trade-off** (02). Training error falls forever;
validation error is U-shaped. Bias and variance are measurable, not metaphorical — and knowing
which one you have tells you whether more data will help.

**But "error" is not one number** (03). Four counts generate every classification metric, and
accuracy is the one that lies most readily: a model with no features scored 99% on an imbalanced
problem here.

**Which curve you use depends on how rare the positive class is** (04). Drop prevalence from 50% to
1% and ROC-AUC falls 0.16 while PR-AUC falls 0.83. Same model, same features.

**All of that assumes your validation is honest, and it usually is not** (05). Leakage produces
0.99 offline and nothing in production. The Bike Sharing file ships with a real one.

**And most of the achievable gain is in the features** (06), not the algorithm — provided every
fitted transform lives inside the pipeline, which is the lesson of 05 applied.

---

## Recurring conventions

Every folder has the same shape:

```
NN_topic/
├── README.md      the concept, the figures, what to look for, the exam angle
├── <topic>.py     one runnable, seeded, CPU-only script with narrated output
└── images/        committed figures, embedded in the README
```

Scripts print what they are doing and what each number means, so **running one teaches you
something** rather than just producing a file. Every number quoted in a README is reproducible —
the seeds are fixed.

---

## What these six deliberately do not cover

- **Regression metrics** beyond the basics — see `02_traditional_ml/01_regression`
- **Imbalance *handling*** (class weights, SMOTE, threshold moving) — 04 shows how to *measure*
  under imbalance; the fixes live with the classifiers
- **Clustering and unsupervised evaluation** — `02_traditional_ml/03_clustering`
- **Anything neural** — Part 03
