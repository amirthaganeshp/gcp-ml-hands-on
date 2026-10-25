"""AutoML: your data, your labels, nobody's model code.

AutoML sits between a pre-trained API (someone else's data and model) and
custom training (yours and yours). You supply a labelled dataset; Vertex AI
searches architectures and hyperparameters and hands back a deployable
model. You write no model code at all.

That is genuinely useful and it is oversold, so this file spends as much
time on when NOT to use it.

  1. What AutoML covers, by data type
  2. The workflow, in code
  3. Data requirements - the part that decides whether it will work
  4. Cost, which is the usual reason a team abandons it
  5. AutoML vs BQML vs custom - the decision, stated plainly

NOT EXECUTED. Every Vertex import is inside a function, so this runs with
no credentials, no project and no billing.

Run:
    python 05_gcp/03_automl/automl.py

Concept reference: Foundations volume, Part D.3.
"""

from __future__ import annotations

PROJECT = "your-project-id"
REGION = "us-central1"
BUCKET = "gs://your-bucket"


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ================================================== 1. what it covers
def coverage() -> None:
    rows = [
        ("Tabular", "classification, regression, forecasting",
         "1,000 rows", "competitive with a tuned GBM"),
        ("Image", "classification, object detection",
         "100/label (1,000 better)", "strong - transfer learning"),
        ("Text", "classification, entity extraction, sentiment",
         "100/label", "strong"),
        ("Video", "classification, action recognition, tracking",
         "100/label", "niche but real"),
    ]
    print(f"  {'data type':<10} {'tasks':<44} {'min data':<24} quality")
    print("  " + "-" * 96)
    for a, b, c, d in rows:
        print(f"  {a:<10} {b:<44} {c:<24} {d}")

    print("\n  AutoML Tabular deserves its reputation - it runs a real")
    print("  architecture search and an ensemble, and it is frequently")
    print("  competitive with a carefully tuned gradient-boosted model.")
    print("  Given the measurements in 02_traditional_ml/01 and")
    print("  03_neural_networks/02, that is a meaningful bar to clear.")
    print("\n  The image and text products lean on transfer learning from")
    print("  Google's pretrained backbones, which is why 100 examples per")
    print("  label is enough to get something working at all.")


# ================================================== 2. the workflow
def create_tabular_dataset():
    """Datasets are Vertex AI resources, not just files."""
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)
    return aiplatform.TabularDataset.create(
        display_name="churn",
        # either a GCS CSV or, usefully, a BigQuery table directly
        bq_source=f"bq://{PROJECT}.your_dataset.customers",
    )


def train_tabular(dataset):
    """The training job. Note what you specify and what you do not."""
    from google.cloud import aiplatform

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="churn-automl",
        optimization_prediction_type="classification",
        # THE OBJECTIVE IS A REAL DECISION, not a default to skim past.
        # 'maximize-au-prc' for imbalanced data, for the reasons measured
        # in 01_foundations/04 - ROC-AUC flatters a rare positive class.
        optimization_objective="maximize-au-prc",
        column_specs={"tenure_months": "numeric",
                      "plan_type": "categorical",
                      "signup_date": "timestamp"},
    )
    return job.run(
        dataset=dataset,
        target_column="churned",
        # budget is in MILLI-hours: 1000 = one hour of node time
        budget_milli_node_hours=1000,
        # Time-ordered data needs a chronological split, same lesson as
        # 02_traditional_ml/06. The default is random.
        predefined_split_column_name=None,
        timestamp_split_column_name="signup_date",
        model_display_name="churn-automl",
        disable_early_stopping=False,
    )


def batch_predict(model):
    from google.cloud import aiplatform  # noqa: F401
    return model.batch_predict(
        job_display_name="churn-batch",
        bigquery_source=f"bq://{PROJECT}.your_dataset.new_customers",
        bigquery_destination_prefix=f"bq://{PROJECT}.your_dataset",
        sync=True)


def workflow_notes() -> None:
    print("  Three functions above: create the dataset, train, predict.")
    print("  Four lines of real configuration. What is worth noticing is")
    print("  which four:\n")
    print("    bq_source          AutoML reads a BigQuery table directly, so")
    print("                       there is no export step. That removes most")
    print("                       of the reason people reach for BQML on a")
    print("                       tabular problem.")
    print("\n    optimization_objective  NOT a default to skim. 'maximize-au-prc'")
    print("                       for imbalanced data; ROC-AUC flatters a rare")
    print("                       positive class, as measured in")
    print("                       01_foundations/04 where it fell 0.159 while")
    print("                       PR-AUC fell 0.829 on the same model.")
    print("\n    budget_milli_node_hours  in MILLI-hours. 1000 = 1 node-hour.")
    print("                       Getting this wrong by 1000x is a real and")
    print("                       expensive mistake.")
    print("\n    timestamp_split_column_name  the chronological split. The")
    print("                       default is RANDOM, which leaks the future on")
    print("                       time-ordered data - the failure measured in")
    print("                       02_traditional_ml/06.")


# ============================================ 3. data requirements
def data_requirements() -> None:
    print("  AutoML fails on data problems, not model problems. The")
    print("  requirements that actually bite:\n")
    for a, b in [
        ("1,000 rows minimum (tabular)",
         "and that is a floor, not a target"),
        ("100 examples per label (image/text)",
         "1,000 for anything you would deploy"),
        ("Class balance matters",
         "it warns below 10 examples for a class, and struggles well above"),
        ("Labels must be consistent",
         "AutoML cannot fix disagreement between annotators"),
        ("Leakage is not detected",
         "give it a column derived from the target and it will happily use it"),
    ]:
        print(f"    {a:<38} {b}")

    print("\n  The last one is worth dwelling on. AutoML does the modelling")
    print("  and none of the thinking. Hand it the bike-sharing table from")
    print("  01_foundations/05 with `casual` and `registered` still in it and")
    print("  it will build a superb model of `cnt = casual + registered`,")
    print("  report near-perfect metrics, and be worthless. Nothing in the")
    print("  service will tell you.")
    print("\n  That is the honest division of labour: AutoML removes the")
    print("  architecture search. It does not remove the need to understand")
    print("  your data, and the failures it cannot see are exactly the ones")
    print("  that matter most.")


# ==================================================== 4. cost
def cost() -> None:
    print("  Rough and regionally variable - check before quoting:\n")
    print(f"  {'stage':<22} {'unit':<28} note")
    print("  " + "-" * 76)
    for a, b, c in [
        ("Tabular training", "~$20+ per node-hour", "budget is the cap"),
        ("Image training", "~$3-8 per node-hour", "plus a minimum"),
        ("Deployed endpoint", "per node-hour, always on", "the recurring cost"),
        ("Batch prediction", "per node-hour of the job", "nothing at rest"),
    ]:
        print(f"  {a:<22} {b:<28} {c}")

    print("\n  Two cost facts that decide whether teams keep using AutoML:")
    print("\n    TRAINING IS A ONE-OFF; THE ENDPOINT IS FOREVER. A model")
    print("    trained once for a few hundred dollars can cost that again")
    print("    every month sitting deployed. If predictions are not needed")
    print("    in a request path, use batch prediction and delete the")
    print("    endpoint - the single biggest saving available here.")
    print("\n    THE BUDGET IS A CAP, NOT A TARGET. budget_milli_node_hours")
    print("    bounds the search; early stopping often finishes sooner and")
    print("    you pay for what was used. Setting a large budget is not")
    print("    automatically expensive - but setting it 1000x too large,")
    print("    by forgetting the milli-, is.")


# ============================================ 5. the decision
def the_decision() -> None:
    print("  AutoML, BigQuery ML, or custom training?\n")
    print(f"  {'if...':<46} then")
    print("  " + "-" * 76)
    for a, b in [
        ("the data is already in BigQuery and it is tabular", "BigQuery ML"),
        ("you want it in SQL, with no new system to operate", "BigQuery ML"),
        ("images, video, or text classification", "AutoML"),
        ("tabular, no ML staff, accuracy matters more than cost", "AutoML"),
        ("you need a custom architecture or loss", "custom training"),
        ("you need to control the training loop", "custom training"),
        ("the model must be reproducible from source", "custom training"),
        ("cost per experiment matters more than time", "custom training"),
    ]:
        print(f"  {a:<46} {b}")

    print("\n  The overlap that confuses people is TABULAR, where all three")
    print("  are genuinely viable. The tiebreakers, in the order they")
    print("  usually decide it:")
    print("\n    1. Where is the data? In BigQuery -> BQML removes an export,")
    print("       a training system and a serving system.")
    print("    2. Who is doing the work? No ML engineers -> AutoML.")
    print("    3. What does it cost per iteration? AutoML is expensive to")
    print("       re-run; custom training on a small instance is not.")
    print("    4. Do you need to explain the model to a regulator? Then you")
    print("       want something you can read - see the glass-box vs")
    print("       post-hoc distinction in 02_traditional_ml/02.")
    print("\n  And the answer this repository keeps arriving at: on small,")
    print("  clean tabular data a gradient-boosted tree you train yourself is")
    print("  hard to beat, and it is the cheapest of all of these to operate.")


def main() -> None:
    print("=" * 72)
    print("  AutoML ON VERTEX AI")
    print("=" * 72)
    print("\n  Write-only: no datasets created, no jobs run, no billing.")

    banner("1. What AutoML covers")
    coverage()

    banner("2. The workflow")
    workflow_notes()

    banner("3. Data requirements - where AutoML actually fails")
    data_requirements()

    banner("4. Cost")
    cost()

    banner("5. AutoML, BigQuery ML, or custom?")
    the_decision()

    banner("What to take away")
    print("  - AutoML = YOUR data and labels, NOBODY'S model code. It sits")
    print("    between a pre-trained API and custom training.")
    print("  - AutoML Tabular is genuinely competitive with a tuned GBM.")
    print("    Image and text lean on transfer learning, which is why 100")
    print("    examples per label is enough to start.")
    print("  - SET optimization_objective DELIBERATELY. 'maximize-au-prc' for")
    print("    imbalanced data - ROC-AUC flatters a rare positive class.")
    print("  - budget_milli_node_hours is in MILLI-hours, and it is a CAP.")
    print("  - The default split is RANDOM. Set a timestamp split for")
    print("    time-ordered data or you leak the future.")
    print("  - AUTOML CANNOT SEE LEAKAGE. Hand it a column derived from the")
    print("    target and it builds a perfect, worthless model and tells you")
    print("    nothing. It removes the architecture search, not the thinking.")
    print("  - Training is one-off; A DEPLOYED ENDPOINT BILLS FOREVER. Use")
    print("    batch prediction unless a user is waiting.")
    print("=" * 72)


if __name__ == "__main__":
    main()
