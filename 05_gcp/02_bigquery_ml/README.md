# BigQuery ML

> **Concept reference:** Success Guide, Part 1.1 · Blueprint section 1 (~13%)
> **Runtime:** ~2 s · **Executes nothing** — the SQL in [`models.sql`](models.sql) is never submitted

```bash
python 05_gcp/02_bigquery_ml/bigquery_ml.py
```

---

## The idea

**If your training data is already a BigQuery table, BQML trains and serves a model without the
data ever moving** — no export, no training job, no serving endpoint, no second system to operate.

That is a narrower claim than *"it is the best model"*, and it is usually the one that decides the
architecture.

## The model-type map

| `MODEL_TYPE` | for |
|---|---|
| `LINEAR_REG` | predict a number |
| `LOGISTIC_REG` | binary or multi-class classification |
| `BOOSTED_TREE_*` | tabular workhorse, usually the accuracy answer |
| `KMEANS` | segmentation — **unsupervised, no labels, no split** |
| `MATRIX_FACTORIZATION` | recommendations from historical behaviour |
| `PCA` | dimensionality reduction |
| `ARIMA_PLUS` | forecasting — **many series in one statement** |
| `TENSORFLOW` / `ONNX` / `XGBOOST` | import a model trained elsewhere |
| `REMOTE WITH CONNECTION` | call a Vertex/Agent Platform endpoint **from SQL** |

## What it automates, and what that hides

Automatic: train/eval split, one-hot encoding of strings, standardisation, hyperparameter defaults,
and `TRANSFORM` preprocessing **stored inside the model** so serving matches training.

> **`TRANSFORM` is the training-serving-skew fix.** Preprocessing written as a separate query
> *will* drift from what the model was trained on — the same problem Feature Store solves for
> non-SQL pipelines. See [`08_monitoring`](../../08_monitoring/).

**What it hides:** `AUTO_SPLIT` changes behaviour with row count. **Under 500 rows every row trains
and no evaluation set exists** — `ML.EVALUATE` then scores the model on its own training data. The
result looks excellent, means nothing, and nothing warns you.

## The limits, stated plainly

- Narrow model catalogue — no custom architectures or losses.
- **`ML.PREDICT` is batch.** It scans, scores and returns. There is no per-row latency story; a
  user waiting for a response needs an endpoint.
- The split defaults to **random**, which **leaks the future** on time-ordered data. Use `SEQ` with
  a date column, or `CUSTOM`.

## Cost works differently here, and catches people out

Training bills **bytes processed**, like any query. A `SELECT *` into `CREATE MODEL` on a wide table
is expensive before a single gradient step. Prune columns first; the model does not need them.

## The exam angle

| the scenario says | the answer |
|---|---|
| data already in BigQuery, analyst-led, standard model | **BQML** |
| forecast many series at once | **`ARIMA_PLUS`** with `TIME_SERIES_ID_COL` |
| a user is waiting, sub-second | **not** `ML.PREDICT` — an endpoint |
| custom architecture or loss | custom training |
| "no ML expertise", data in BigQuery | **BQML** before AutoML |
