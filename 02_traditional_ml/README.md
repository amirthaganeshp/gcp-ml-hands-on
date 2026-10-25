# Traditional Machine Learning

Everything you can do before reaching for a neural network — which, on tabular data, is usually
more than people expect. The recurring theme of this section is that **the algorithm matters far
less than the framing, the features and the metric.**

---

## Contents

| # | Topic | Dataset | Runtime | The one thing to take from it |
|---|---|---|---|---|
| [01](01_regression/) | **Regression** | California Housing, Auto MPG | ~55 s | L1 zeroes coefficients, L2 only shrinks them — and gradient-boosted trees beat linear by 38% |
| [02](02_classification/) | **Classification** | Rice, plus two synthetic shapes | ~15 s | Every model family can only draw one *shape* of boundary; on clean data all seven tie |
| [03](03_clustering/) | **Clustering** | Wheat Seeds, plus three synthetic shapes | ~11 s | Internal metrics measure *separation*, not *truth* — silhouette picked k=2 against 3 real varieties |
| [04](04_dimensionality_reduction/) | **Dimensionality reduction** | Digits, Wine | ~18 s | PCA removes *redundancy*, not *irrelevance* — it spent 33% of its capacity on pure noise columns |
| [05](05_anomaly_detection/) | **Anomaly detection** | Bike Sharing, synthetic | ~11 s | A one-column z-score rule beat two of four real detectors |
| [06](06_time_series/) | **Time series** | Flights, Bike Sharing | ~17 s | A random split made the error look 2× better; trees cannot extrapolate a trend |
| [07](07_recommenders/) | **Recommenders** | MovieLens 100K | ~8 s | The RMSE winner and the ranking winner are different models — and popularity beat both |
| [08](08_self_supervised/) | **Self-supervised** | Digits (labels withheld) | ~13 s | Both pretext tasks **lost** to raw pixels at scarce labels — a negative result, kept |
| [09](09_reinforcement_learning/) | **Reinforcement learning** | hand-written GridWorld | ~6 s | ε=0 won, because zero-init + negative rewards is optimistic initialisation in disguise |

Run any of them standalone from the repository root:

```bash
python 02_traditional_ml/03_clustering/clustering.py
```

---

## The through-line

**1. Model choice is the least valuable knob.** Seven classifiers landed within 0.013 ROC-AUC of
each other on Rice, and the winner was plain logistic regression — the same model that scored
*worse than chance* on a ring-shaped problem one section earlier. What a model can do is determined
by the **shape of its assumptions**, not by a "simple → powerful" ranking.

**2. Scaling is not a formality.** It decided the answer three separate times here: KNN and SVM
need it to work at all, k-means silently clustered on whichever column had the biggest units, and
unscaled PCA turned `PC1` into a single feature wearing a disguise.

**3. Metrics encode assumptions, and will happily mislead you.** PR-AUC barely moved across three
different imbalance remedies. Silhouette correctly reported two well-separated groups where biology
says three. Accuracy is meaningless for clustering because cluster IDs are arbitrary. None of these
metrics was broken — each answered precisely the question it actually asks.

**4. Anything fitted belongs inside a `Pipeline`.** Scalers, PCA, imputers and encoders all learn
from data, so fitting them before the split leaks. The leak always flatters you, which is why it
survives to production.

**5. Honest results beat tidy ones.** Many demonstrations here contradict the usual advice, and
every one is left as it came out, with an explanation, rather than tuned until it agreed with the
textbook:

| topic | what the textbook says | what actually happened |
|---|---|---|
| Classification | reach for `class_weight` on imbalance | it scored the **worst** F1 of three techniques |
| Clustering | silhouette finds the right k | it picked k=2 against 3 real varieties — and was right about the *geometry* |
| Dim. reduction | PCA denoises and improves accuracy | it never beat doing nothing, on clean **or** noise-padded data |
| Anomaly detection | use a multivariate detector | a one-column z-score beat two of the four |
| Recommenders | personalisation beats popularity | popularity won NDCG by ~3× |
| Self-supervised | pretraining improves label efficiency | both pretext tasks **lost** at every budget up to 300 labels; rotation edged ahead (+0.007) only at 1,000 |
| RL | always explore | ε=0 won — the reward design was already exploring |

Each of those is explained rather than hidden, and in most cases the explanation is the actual
lesson of the section.

---

## Running the whole section

```bash
for f in 02_traditional_ml/*/[a-z]*.py; do python "$f"; done
```

All nine scripts, **~135 s end to end**, 35 committed figures, zero warnings.

Next: [`03_neural_networks`](../03_neural_networks/) picks up the threads left here — the two-tower
model that fixes recommender cold start, the Deep Q-Network that replaces the Q-table, and the
autoencoder that does anomaly detection and self-supervision at once.
