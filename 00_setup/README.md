# Setup and dataset inventory

Three files live here. Run the first two once, after installing dependencies.

```bash
python 00_setup/check_environment.py     # confirms the stack works
python 00_setup/download_datasets.py     # fetches and caches the datasets
python 00_setup/loaders.py               # verifies every loader parses
```

| File | Purpose |
|---|---|
| `check_environment.py` | Versions, what is missing, and a real RandomForest fit — an import succeeding does not prove the library works |
| `download_datasets.py` | Fetches everything not built into a library, into `data/`. Idempotent; `--force` re-fetches |
| `loaders.py` | Clean `pandas` loaders with real column names, correct dtypes and each dataset's quirks documented |

`data/` is gitignored. Nothing downloaded is committed.

> **Behind a corporate proxy?** If downloads fail with `SSLError` but the URL opens in your
> browser, the proxy is inspecting TLS with a certificate Python does not trust. `truststore` is in
> `requirements.txt` and the downloader uses it automatically.

---

## Using a dataset in an example

Parsing lives in `loaders.py` once, so example scripts stay about the concept:

```python
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "00_setup"))
from loaders import load_rice

df = load_rice()          # 3,810 x 8, already tidy
```

Library built-ins are called directly — no loader needed:

```python
from sklearn.datasets import load_digits, load_wine, fetch_california_housing
from keras.datasets import mnist, fashion_mnist
```

---

## Every dataset, and where it comes from

Three sources. **The last column is the one that matters** — a dataset being "built in" does not
make it less central. Fashion-MNIST carries the CNN, the TF/PyTorch/JAX comparison and the custom
training job; it simply arrives via Keras rather than via `download_datasets.py`.

The *Used by* columns list the scripts that actually load each dataset, checked against the code.

### Downloaded into `data/`

| Dataset | Shape | Loader | Used by |
|---|---|---|---|
| **Penguins** | 344 × 7 | `load_penguins()` | Train/test split, confusion matrix, feature engineering — has real missing values |
| **Seeds** | 210 × 8 | `load_seeds()` | Clustering |
| **Rice** (Cammeo / Osmancik) | 3,810 × 8 | `load_rice()` | Confusion matrix, classification, MLP tabular — clean binary |
| **Auto MPG** | 398 × 9 | `load_auto_mpg()` | Bias–variance, regression |
| **Wine Quality** (white) | 4,898 × 12 | `load_wine_quality()` | **ROC vs PR** — 21.6% natural imbalance |
| **Bike Sharing** | 731 day / 17,379 hour × 16 | `load_bike()` | Data leakage, feature engineering, anomaly detection, time series |
| **Adult Income** | 32,561 × 15 | `load_adult()` | Wide & Deep, model optimization, skew and drift, fairness and slices, capstone |
| **Flights** | 144 × 4 | `load_flights()` | Train/test split, time series, RNN/LSTM/GRU |
| **MovieLens 100K** | 100,000 ratings | `load_movielens_ratings()` | Recommenders, embeddings, two-tower retrieval |

**Downloaded but not used by any script.** `download_datasets.py` also caches three files that
have a loader but that no example loads today — available for your own experiments.
(`python 00_setup/loaders.py` parses Iris and Banknote; it calls
`load_wine_quality()` with its default, so it checks the white file, not the red one.)

| Dataset | Shape | Loader |
|---|---|---|
| Iris (CSV) | 150 × 5 | `load_iris_df()` |
| Banknote | 1,372 × 5 | `load_banknote()` |
| Wine Quality (red) | 1,599 × 12 | `load_wine_quality("red")` |

### Built into scikit-learn

| Dataset | Shape | Call | Used by |
|---|---|---|---|
| Iris | 150 × 4 | `load_iris()` | `check_environment.py` only — the smoke-test fit |
| Wine | 178 × 13 | `load_wine()` | Dimensionality reduction |
| Digits | 1,797 × 64 | `load_digits()` | Dimensionality reduction (64→2 is visually convincing), self-supervised |
| California Housing | 20,640 × 8 | `fetch_california_housing()` | Regression |
| Synthetic | any | `make_moons`, `make_circles`, `make_blobs` | Classification, clustering, anomaly detection |

### Built into Keras — part 03 onward, needs `requirements-dl.txt`

| Dataset | Shape | Call | Used by |
|---|---|---|---|
| **Fashion-MNIST** | 70,000 × 28 × 28 | `fashion_mnist.load_data()` | **CNN, the TF/PyTorch/JAX comparison, custom training** |
| **MNIST** | 70,000 × 28 × 28 | `mnist.load_data()` | Autoencoders |

**Why Fashion-MNIST for the CNN rather than MNIST.** MNIST is easy to look at — you can see a digit
and judge a reconstruction at a glance, which suits the autoencoder. But almost anything reaches
98% on it, so it cannot tell a good architecture from a mediocre one: a dense network does about as
well as a CNN and the comparison shows nothing. Fashion-MNIST is a drop-in replacement with
identical shape and genuinely harder classes, so the same code shows a real difference.

---

## Why these datasets

Deliberately small, standard and **neutral or positive in subject matter** — flowers, grains,
wine, penguins, bicycles, cars, banknotes, films. Learning material should not require spending a
fortnight thinking about illness, and nothing here is about disease or disaster.

Everything runs on a laptop CPU in seconds, and every one is well known enough to compare your
results against published work. The point of these examples is the **mechanism**, not squeezing
out accuracy — a dataset you wait ten minutes for teaches you less, not more.

Three are chosen for a specific teaching property rather than convenience:

- **Penguins** carries genuine missing values (2 rows with no measurements, 11 with no sex),
  which is what makes it the honest feature-engineering example rather than a pre-cleaned toy.
- **Wine Quality** binarises at `quality >= 7` to **21.6% positives** — a real, unmanufactured
  class imbalance, which is exactly what the ROC-versus-precision-recall demonstration needs.
- **Bike Sharing** has `casual + registered == cnt`, a textbook **target leak** sitting in the
  raw file. The leakage example uses it rather than inventing one.

---

## Licences

| Dataset | Licence |
|---|---|
| Iris, Flights | Public domain |
| Penguins | CC0 (Palmer Station LTER) |
| Seeds, Rice, Banknote, Auto MPG, Wine Quality, Bike Sharing, Adult | CC BY 4.0 (UCI ML Repository) |
| MovieLens 100K | GroupLens research terms — free for research and education, **not redistributable**, which is why it is downloaded rather than committed |
| MNIST, Fashion-MNIST | Open, via Keras |
