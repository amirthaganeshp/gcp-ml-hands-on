"""BigQuery ML: when the answer is SQL.

The pitch is narrow and strong. If your training data is already a BigQuery
table, BQML trains and serves a model without the data ever moving - no
export, no training job, no serving endpoint, no second system to operate.
That is a smaller claim than "it is the best model", and it is usually the
one that decides the architecture.

  1. The model-type map - which MODEL_TYPE for which question
  2. What BQML does for you automatically, and what it hides
  3. The limits, stated plainly
  4. Cost, which works differently here and catches people out

The runnable SQL is in models.sql, beside this file. Nothing has been
executed - see the note in the README.

Run:
    python 05_gcp/02_bigquery_ml/bigquery_ml.py

Concept reference: Foundations volume, Part D.2.
"""

from __future__ import annotations

from pathlib import Path

SQL = Path(__file__).parent / "models.sql"


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ============================================== 1. which MODEL_TYPE?
def model_map() -> None:
    rows = [
        ("predict a number", "LINEAR_REG", "or BOOSTED_TREE_REGRESSOR"),
        ("predict a class", "LOGISTIC_REG", "binary AND multiclass"),
        ("best tabular accuracy", "BOOSTED_TREE_*", "XGBoost under the covers"),
        ("very large tabular", "DNN_*", "rarely beats boosted trees"),
        ("both of the above", "WIDE_AND_DEEP_*", "see 03_neural_networks/08"),
        ("segment customers", "KMEANS", "set standardize_features"),
        ("forecast a series", "ARIMA_PLUS", "holidays + seasonality built in"),
        ("forecast with drivers", "ARIMA_PLUS_XREG", "needs future regressors"),
        ("recommend items", "MATRIX_FACTORIZATION", "needs a reservation"),
        ("find anomalies", "AUTOENCODER", "or KMEANS / ARIMA_PLUS"),
        ("reduce dimensions", "PCA", "ML.PRINCIPAL_COMPONENTS"),
        ("use a trained model", "TENSORFLOW / ONNX", "import a SavedModel"),
        ("call an LLM", "REMOTE (endpoint=...)", "ML.GENERATE_TEXT"),
    ]
    print(f"  {'you want to...':<26} {'MODEL_TYPE':<24} note")
    print("  " + "-" * 82)
    for a, b, c in rows:
        print(f"  {a:<26} {b:<24} {c}")

    print("\n  Three of these are the ones exam questions actually turn on:")
    print("\n    ARIMA_PLUS for forecasting. If a question says 'forecast',")
    print("    'seasonal', and 'the data is in BigQuery', this is the answer")
    print("    and a custom LSTM is the distractor. Note time_series_id_col -")
    print("    one statement fits thousands of independent series in")
    print("    parallel, which is the thing that would otherwise be hard.")
    print("\n    BOOSTED_TREE_* for tabular accuracy, for exactly the reasons")
    print("    measured in 02_traditional_ml/01. A DNN option on a mid-sized")
    print("    tabular problem is usually the wrong answer.")
    print("\n    REMOTE models. BQML can call a Vertex AI endpoint or a Gemini")
    print("    model from SQL, which is how generative AI reaches structured")
    print("    data without moving it or standing up serving infrastructure.")


# ================================== 2. what it does for you, and hides
def automatic() -> None:
    print("  BQML does a surprising amount without being asked:\n")
    for item in [
        "numeric features standardised",
        "string features one-hot encoded (no manual encoding step)",
        "timestamps expanded into useful parts",
        "NULLs imputed - numerics to the mean, categoricals to a new level",
        "train/eval split chosen (AUTO_SPLIT)",
        "hyperparameters tuned, if you set num_trials",
        "the model versioned and stored as a dataset object",
    ]:
        print(f"    - {item}")

    print("\n  That convenience is the product, and it is also the risk. Two")
    print("  of those defaults will quietly cost you:")
    print("\n    AUTO_SPLIT IS RANDOM. On time-ordered data a random split")
    print("    leaks the future into training - exactly the failure measured")
    print("    in 02_traditional_ml/06, where it made the error look 2x")
    print("    better than it was. Set data_split_method='SEQ' and")
    print("    data_split_col to a timestamp whenever time matters.")
    print("\n    NULL IMPUTATION IS SILENT. A column that is 90% NULL becomes")
    print("    a column that is 90% the mean, and nothing warns you. Check")
    print("    your nulls before training, not after the model disappoints.")
    print("\n  And one that is a genuine free win: AUTO_CLASS_WEIGHTS=TRUE on")
    print("  an imbalanced classifier. Without it a 3%-positive dataset")
    print("  happily trains a model that predicts 'no' for everything and")
    print("  reports 97% accuracy - the accuracy trap from")
    print("  01_foundations/03, reproduced at cloud scale.")


# ============================================== 3. the limits
def limits() -> None:
    print("  The honest case against BQML:\n")
    for a, b in [
        ("No custom architectures",
         "you get the listed MODEL_TYPEs and nothing else"),
        ("No custom loss functions",
         "if your problem needs one, you need custom training"),
        ("Limited to tabular-ish data",
         "images and audio need Vertex AI, not SQL"),
        ("Serving is a query",
         "ML.PREDICT is batch. Low-latency online serving means "
         "exporting to Vertex AI"),
        ("Debugging is opaque",
         "when a model underperforms there is much less to inspect"),
        ("Some types need a reservation",
         "MATRIX_FACTORIZATION will not run on on-demand pricing"),
    ]:
        print(f"    {a:<28} {b}")

    print("\n  The serving limit is the one that catches people. ML.PREDICT")
    print("  runs as a BigQuery job - seconds of latency, and priced per byte")
    print("  scanned. That is ideal for scoring a table nightly and wrong for")
    print("  answering an API call in 50 ms. For online serving you EXPORT the")
    print("  BQML model to Vertex AI and deploy it to an endpoint, which is a")
    print("  supported path and a question the exam likes.")


# ==================================================== 4. cost
def cost() -> None:
    print("  BQML pricing does not work like the rest of Vertex AI, and the")
    print("  difference is worth internalising:\n")
    print("    TRAINING   priced on BYTES PROCESSED, not on machine-hours.")
    print("               The cost is driven by how much data the SELECT")
    print("               touches, so column pruning and partition filters")
    print("               reduce the training bill directly.")
    print("    PREDICTION priced the same way - it is a query.")
    print("    AT REST    nothing. A stored model costs only its (tiny)")
    print("               storage. There is no endpoint burning money.")
    print("\n  Consequences that show up in architecture questions:")
    print("\n    SELECT * IN A CREATE MODEL IS EXPENSIVE. It scans every")
    print("    column of every row. Naming the columns you need is not style,")
    print("    it is the bill.")
    print("\n    NO IDLE COST beats a Vertex endpoint for intermittent use.")
    print("    A model called a few times a day costs essentially nothing in")
    print("    BQML and costs a full node-hour rate on a deployed endpoint.")
    print("\n    THE FREE SANDBOX is real: 1 TB of queries and 10 GB of")
    print("    storage per month with no billing account at all, which is")
    print("    enough to learn every statement in models.sql.")


def show_sql() -> None:
    if not SQL.exists():
        print("  models.sql not found.")
        return
    text = SQL.read_text(encoding="utf-8")
    statements = text.upper().count("CREATE OR REPLACE MODEL")
    lines = len(text.split("\n"))
    print(f"  models.sql: {lines} lines, {statements} CREATE MODEL statements.")
    print("  Every statement is real and runnable against your own tables -")
    print("  replace `your-project.your_dataset` and repoint the FROM clauses.")
    print("  None of it has been executed here.\n")
    # Section headings only: "-- 3. TITLE". A looser test also matched
    # prose like "-- 02_traditional_ml/01. BQML wraps XGBoost."
    import re
    for line in text.split("\n"):
        m = re.match(r"^-- (\d)\. (.+)$", line.strip())
        if m:
            print(f"    {m.group(1)}. {m.group(2)}")


def main() -> None:
    print("=" * 72)
    print("  BIGQUERY ML")
    print("=" * 72)
    print("\n  Write-only: no queries run, no project touched, no billing.")

    banner("1. Which MODEL_TYPE?")
    model_map()

    banner("2. What BQML does automatically - and what that hides")
    automatic()

    banner("3. The limits")
    limits()

    banner("4. Cost")
    cost()

    banner("5. The SQL")
    show_sql()

    banner("What to take away")
    print("  - THE PITCH IS THAT THE DATA NEVER MOVES. If the table is already")
    print("    in BigQuery, BQML removes an export, a training job and a")
    print("    serving system. That is usually what decides it, not accuracy.")
    print("  - ARIMA_PLUS is the managed forecasting answer, and")
    print("    time_series_id_col fits thousands of series in one statement.")
    print("  - BOOSTED_TREE_* for tabular accuracy; a DNN option on mid-sized")
    print("    tabular data is usually the distractor.")
    print("  - AUTO_SPLIT IS RANDOM - set SEQ and a time column for anything")
    print("    time-ordered, or you leak the future.")
    print("  - AUTO_CLASS_WEIGHTS=TRUE is a free win on imbalanced data.")
    print("  - ML.PREDICT is BATCH. Online serving means exporting the model")
    print("    to a Vertex AI endpoint.")
    print("  - Priced on BYTES SCANNED, so SELECT * in a CREATE MODEL is a")
    print("    real cost. Nothing is charged at rest, which beats an endpoint")
    print("    for intermittent use.")
    print("  - REMOTE models call Vertex AI endpoints and Gemini from SQL -")
    print("    the route to generative AI over structured data.")
    print("=" * 72)


if __name__ == "__main__":
    main()
