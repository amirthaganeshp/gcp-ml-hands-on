# Findings

Every number in this file was produced by a script in this repository, on a laptop CPU. Nothing is
quoted from a textbook.

The reason this file exists: most ML advice is passed around as folklore, and a surprising amount
of it did not survive being measured here. **The results that contradicted the expected answer are
listed first**, because they are the ones worth remembering.

---

## Results that contradicted the textbook

| # | expected | measured | where |
|---|---|---|---|
| 1 | `class_weight='balanced'` is the first move on imbalance | **worst F1 of three techniques** (0.545 vs 0.725 for threshold moving) | [02/02](../02_traditional_ml/02_classification/) |
| 2 | silhouette finds the right *k* | picked **k=2 against 3 real varieties** — and was right about the *geometry* | [02/03](../02_traditional_ml/03_clustering/) |
| 3 | PCA denoises and improves accuracy | **never beat doing nothing** — on clean data *or* with 100 junk columns | [02/04](../02_traditional_ml/04_dimensionality_reduction/) |
| 4 | use a multivariate anomaly detector | a **one-column z-score beat 2 of the 4** | [02/05](../02_traditional_ml/05_anomaly_detection/) |
| 5 | personalisation beats popularity | **popularity won NDCG@10 by ~3×** | [02/07](../02_traditional_ml/07_recommenders/) |
| 6 | self-supervised pretraining improves label efficiency | **both pretext tasks lost** to raw pixels at every budget up to 300 labels; rotation edged ahead (+0.007) only at 1,000 | [02/08](../02_traditional_ml/08_self_supervised/) |
| 7 | always explore in RL | **ε=0 won** — zero-init Q + a −1 step cost *is* optimistic initialisation | [02/09](../02_traditional_ml/09_reinforcement_learning/) |
| 8 | LSTM remembers longer than a SimpleRNN | **SimpleRNN best at every length** (1.000 / 1.000 / 0.955); LSTM erratic | [03/04](../03_neural_networks/04_rnn_lstm_gru/) |
| 9 | embeddings save parameters | at dim ≥ 32 they cost **more** than one-hot | [03/06](../03_neural_networks/06_embeddings/) |
| 10 | bigger embedding dimension is better | error rose **monotonically**; the smallest size won | [03/06](../03_neural_networks/06_embeddings/) |
| 11 | Wide & Deep beats both its halves | **lost to its own deep half**, and to XGBoost | [03/08](../03_neural_networks/08_wide_and_deep/) |
| 12 | augmentation improves accuracy | **−0.064** — though it *did* shrink the overfitting gap; 40 epochs narrowed but never closed it | [03/03](../03_neural_networks/03_cnn_images/) |

Each is explained where it occurs. In most cases the explanation is the actual lesson.

---

## The single most important discipline: give both arms a fair budget

**Four conclusions reversed once the undertrained side was given enough epochs.** This was the
dominant source of wrong answers in the whole repository.

| comparison | undertrained verdict | fair verdict |
|---|---|---|
| CNN vs dense, matched params | +0.005 (6 epochs) | **+0.024** (15 epochs) |
| LSTM on the copy task, length 40 | 0.512 (12 epochs) | **0.822** (40 epochs) |
| Transformer vs LSTM | both at chance (12) | **0.765 vs 0.513** (35) |
| Two-tower retrieval | barely off random (12) | trained (60) |

If a comparison surprises you, check convergence before you believe it.

---

## Numbers worth remembering

### Validation and metrics

- **Split lottery:** accuracy across seeds spanned **0.087**; CV collapsed it to ±0.005.
- **Random split on a time series:** MAE **20.90** (fiction) vs **72.33** (honest). Later, on bike
  data, it made the error look **2× better** than it was.
- **Prevalence sweep:** ROC-AUC fell **0.159** while PR-AUC fell **0.829** on the same model.
- **The accuracy trap:** 0.99 accuracy at **0.0 recall**.
- **Calibration:** Brier 0.1864 → **0.1363** with almost no AUC change — ranking and calibration
  are different things.

### Models

- **L1 vs L2:** Lasso zeroed **8 of 8** coefficients; Ridge zeroed **0 of 8** at any alpha.
- **Gradient boosting on tabular:** XGBoost **38%** better than linear regression.
- **Seven classifiers on a clean problem:** total ROC-AUC spread **0.013** — and plain logistic
  regression won.
- **Scaling:** KNN **+0.2048** from scaling alone; an unscaled neural net scored **exactly 0.5000**.
- **Vanishing gradient, measured:** sigmoid attenuates **5,206×** from output to input over six
  layers; ReLU is flat.
- **CNN vs dense at matched parameters** (105,720 vs 105,866): **17% lower error rate**.
- **PCA on junk columns:** **33%** of the top-20 loading mass spent on pure noise; supervised LDA
  with 9 components beat every PCA setting *and* no-reduction.
- **Isolation Forest on 784 raw pixels:** PR-AUC **0.475** — exactly random. An autoencoder got
  **0.818**.

### Frameworks

Same CNN, same data, same hyperparameters, **105,866 parameters in all three**:

| | test acc | init std | default initialiser |
|---|---|---|---|
| JAX (explicit He) | 0.8800 | 0.1177 | `he_normal` |
| TensorFlow | 0.8767 | 0.0676 | `glorot_uniform` |
| PyTorch | 0.8550 | 0.0478 | `kaiming_uniform(a=√5)` |

**The accuracy ordering matches the initialisation-scale ordering exactly.** A 2.5-point spread,
mostly produced by a default nobody chose: a controlled PyTorch run (five seeds per initialiser)
averages He 0.8783 / Glorot 0.8677 / default 0.8603. But seed and data-order noise is about a point
by itself, so a single run per framework cannot separate the smaller gaps.

---

## Decision rules that held up

**Tabular data → gradient-boosted trees first.** Logistic regression beat the MLP; XGBoost beat
Wide & Deep; trees find interactions by construction. Neural networks earn tabular data only with
high-cardinality categoricals to embed, mixed modalities to fuse, or enough rows that trees stop
scaling.

**Match the mechanism to the question.** The LSTM won forecasting by 20% and sat at chance on set
membership; the transformer did the reverse. Recurrence compresses recent history; attention
compares against everything. Neither is simply better.

**The architectural prior beats parameters.** Same capacity, same data — the CNN won because it was
told the input is a grid.

**Anything fitted goes inside the `Pipeline`.** Scalers, PCA, imputers, encoders. The leak always
flatters you, which is why it survives to production.

**Report the dumb baseline every time.** Popularity, seasonal naive, majority class, predict-the-mean.
Several of the contradictions above are just "nobody ran the baseline".

---

## Environment findings (Windows)

| finding | consequence |
|---|---|
| TensorFlow's bundled headers blow the 260-char path limit | install via `subst M: <repo>`; a broken half-install must be deleted by hand |
| Flax's `orbax` ships test fixtures ~260 chars deep | cannot be installed at all here |
| JAX has no official Windows support | crashes **4 in 70 runs** (5.7%) inside jaxlib; **0 in 50** under WSL2 Linux, which proves less than it looks — see [03_jax](../04_frameworks/03_jax/) |
| Same JAX example, Windows vs WSL2 Linux | **0.8800 on both**, and on all 50 Linux runs — identical to four decimals across operating systems |
| `shutil.rmtree` fails on read-only entries where `rm -rf` succeeds | use an `onexc` handler that clears the attribute |
| `from __future__ import annotations` breaks KFP | PEP 563 stringifies annotations; KFP reads them at runtime |
| Keras preprocessing layers cost ~38 s/epoch on CPU vs ~3 s for the model | do augmentation in NumPy via a `PyDataset` |

---

## A clean result is not a fix — the rule of three

The JAX crash investigation produced the sharpest statistical lesson in the repository, and it is
worth stating on its own because the mistake is so natural.

**Zero failures in N runs puts the 95% upper bound on the true failure rate at `3/N`.** So 50 clean
runs bounds the rate at **6%** — which is exactly the rate we were trying to disprove. Fisher's
exact test on 0/50 against 4/70 gives **p = 0.14**: not significant.

Both of the tempting readings are wrong:

- *"50 clean runs, so it's fixed"* — no; 0/50 cannot distinguish 0% from 6%.
- *"P(0 in 50 at the old rate) = 0.053, so it's nearly significant"* — no; that conditions on the
  old rate being exactly right rather than comparing two observed samples.

Earlier in the same investigation a **0/10 probe was over-read as implicating a specific code
path**, when at a 6% rate 10 clean runs happen 54% of the time by chance. The same error, one
round earlier.

**Always ask how many failures you would have expected to see anyway.** For a rare event that
number is usually smaller than the sample feels, and the answer is almost always "run more" or
"argue from mechanism instead" — here, that JAX is simply not supported on Windows.

## A note on reading any of this

Every number here is **one machine, one dataset, one configuration**. The contradictions are real
and reproducible on this setup; they are not claims that the textbook advice is universally wrong.
`class_weight` is a reasonable tool — it just lost *here*, on *this* data, and the reason it lost is
the useful part.

The transferable lesson is not any single number. It is that **these things are cheap to measure
and frequently surprising**, and that a comparison you have not run is a comparison you do not know
the answer to.
