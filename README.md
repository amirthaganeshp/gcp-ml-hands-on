# Machine Learning, End to End

A hands-on companion to the theory: every core ML concept as a small, self-contained, runnable
Python example on a standard public dataset — from a perceptron written in NumPy to a Kubeflow
pipeline on Google Cloud.

Each example is **one script you can run on a laptop CPU**, with a README that explains the
concept, tells you what to look for in the output, and points at the nuance that matters.

**Measured:** most examples finish in **under 40 seconds**; the deep-learning ones in part 03
take **1–3 minutes** (the slowest, `rnn_lstm_gru`, is 176 s). **All 41 runnable examples
together: 41/41 passing in 1,231 s (~20 minutes)** — reproduce with
`bash .github/scripts/run_list.sh full`.

---

## Why this exists

Reading about the bias–variance trade-off is not the same as watching the validation curve turn
upward on your own screen. Every folder here takes one idea and makes it observable.

The repository is ordered so it reads as a single argument:

| Part | Question it answers |
|---|---|
| `01_foundations` | What is a model, and how do you know whether it is any good? |
| `02_traditional_ml` | What can you do without neural networks? (Usually more than you think.) |
| `03_neural_networks` | How do the deep ones actually work? |
| `04_frameworks` | What do TensorFlow, PyTorch and JAX each look like for the *same* model? |
| `05_gcp` | What will Google Cloud build for you, and what must you build yourself? |
| `06_pipelines` | How does any of this become a system rather than a script? |
| `07_serving` | How does a trained model reach users — batch or online, how is it rolled out, how is it made cheaper? |
| `08_monitoring` | How do you know a deployed model is still right — skew, drift, slices and fairness? |
| `09_capstone` | What does one model look like taken all the way through: split, evaluate, serve, monitor? |
| `10_summary` | What is the whole repository on one page? |
| `99_reference` | What did all of it actually measure? |

---

## Quick start

```bash
git clone <your-repo-url>
cd machine-learning

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

python 00_setup/check_environment.py     # confirm everything works
python 00_setup/download_datasets.py     # fetch and cache the datasets
```

Then run anything:

```bash
python 02_traditional_ml/02_classification/classification.py
```

Full setup notes, including the deep-learning extras, are in **[SETUP.md](SETUP.md)**.

---

## Repository map

```
00_setup/              environment check, dataset downloader, dataset inventory
01_foundations/        train/test/CV · bias-variance · confusion matrix · ROC vs PR
                       · data leakage · feature engineering
02_traditional_ml/     regression · classification · clustering · dimensionality reduction
                       · anomaly detection · time series · recommenders
                       · self-supervised · reinforcement learning
03_neural_networks/    perceptron from scratch · MLP · CNN · RNN/LSTM/GRU · transformer
                       · embeddings · autoencoders · wide & deep · two-tower retrieval
04_frameworks/         the same CNN in TensorFlow, PyTorch and JAX
05_gcp/                pre-trained APIs · BigQuery ML · AutoML · custom training · ADK agents
06_pipelines/          Kubeflow Pipelines v2 · components · conditional deploy gate
07_serving/            batch vs online · canary/A-B/blue-green/shadow · quantization,
                       pruning, batching
08_monitoring/         training-serving skew · data drift · concept drift · attribution
                       drift · slices, proxies and fairness measures
09_capstone/           one model end to end: split, evaluate, serve, monitor
10_summary/            the whole repository on one page - a visual cheat sheet
99_reference/          every measured finding, decision tables, glossary, service renames
```

Every leaf folder follows the same shape:

```
NN_topic/
├── README.md      the concept, what to look for, the nuance that matters
├── <topic>.py     one runnable, seeded, CPU-only script
└── images/        the figures it writes, committed so the README renders on GitHub
```

---

## Conventions

- **Every script runs standalone.** No hidden state, no required execution order.
- **Everything is seeded.** Re-running reproduces the numbers quoted in the README — verified by
  running a script twice and diffing: every value identical, only the elapsed-time columns move.
  **The figures are not byte-identical between runs** (matplotlib rendering is not reproducible at
  the byte level), so a few `images/*.png` will show as modified after a full run. That is
  expected and is not a signal that anything broke.
- **CPU-only.** Parts 01–02 are seconds each; part 03 is 1–3 minutes per example. Heavier
  variants are behind a `--full` flag and default to a subsample.
- **No data in git.** `00_setup/download_datasets.py` fetches and caches everything; `data/` is
  ignored.
- **Narrated output.** Scripts print what they are doing and what the number means, so running one
  teaches you something.

---

## Datasets

All public and standard. The examples use California Housing, Wine, Penguins, Wheat Seeds, Rice,
Auto MPG, White Wine Quality, Bike Sharing, Flights, Adult Income, Digits, MNIST, Fashion-MNIST and
MovieLens 100K. The downloader also caches Iris, Banknote and Red Wine Quality, which no script
currently loads.

`00_setup/README.md` lists every one with its source, licence, shape and the scripts that use it.

---

## Status

| Phase | Contents | State |
|---|---|---|
| 0 | Scaffolding, environment check, dataset downloader | ✅ done |
| 1 | [Foundations](01_foundations/) — six concept demos, 94 s end to end | ✅ done |
| 2a | [Core traditional ML](02_traditional_ml/) — regression, classification, clustering, dimensionality reduction | ✅ done |
| 2b | Anomaly detection, time series, recommenders, self-supervised, RL | ✅ done |
| 3 | [Neural networks](03_neural_networks/) — nine topics, perceptron to two-tower | ✅ done |
| 4 | [Frameworks](04_frameworks/) — the same CNN in TensorFlow, PyTorch and JAX | ✅ done |
| 5 | [Google Cloud](05_gcp/) — pre-trained APIs, BigQuery ML, AutoML, custom training, ADK agents | ✅ done |
| 6 | [Pipelines](06_pipelines/) — KFP v2, components, conditional deploy gate | ✅ done |
| 7 | [Serving](07_serving/) — batch vs online, rollout strategies, model optimization | ✅ done |
| 8 | [Monitoring](08_monitoring/) — skew and drift, slices and fairness | ✅ done |
| 9 | [Capstone](09_capstone/) — one model end to end: split, evaluate, serve, monitor | ✅ done |
| 10 | [Summary](10_summary/) — the whole repository on one visual cheat sheet | ✅ done |

---

## Licence

Code in this repository is MIT licensed. Datasets retain their original licences — see
`00_setup/README.md` for the source and terms of each.
