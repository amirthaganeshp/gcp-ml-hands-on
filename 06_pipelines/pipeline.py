"""A Kubeflow Pipelines v2 pipeline, with a conditional deployment gate.

A pipeline is the thing that turns a notebook into a system. The model is
usually the easy part; what makes it production is that training is
reproducible, artefacts are tracked, and a model cannot reach an endpoint
without passing a check.

This file COMPILES. `kfp` turns the Python below into a YAML pipeline spec
with no cluster, no credentials and no Docker - and that compiled spec is
the real artefact, the thing you would submit to Vertex AI Pipelines. The
compile is run in this repository; the pipeline has never been submitted.

  1. Components - a step is a function plus its container and its contract
  2. Artifacts vs parameters - the distinction the DSL is built around
  3. The pipeline graph, and how KFP infers it
  4. THE DEPLOYMENT GATE - dsl.If, so a bad model cannot reach production
  5. Compile, and what is actually in the YAML

Run:
    python 06_pipelines/pipeline.py

Concept reference: Foundations volume, Part D.5.
"""

# NOTE: deliberately NO `from __future__ import annotations` here, unlike
# every other file in this repository.
#
# PEP 563 turns every annotation into a STRING at runtime, and KFP builds a
# component's interface by INSPECTING those annotations. With the future
# import it sees the string "int" where it expects the type int, and the
# compile dies with a misleading error:
#
#     TypeError: Artifacts must have both a schema_title and a
#     schema_version, separated by `@`. Got: int
#
# which points at artifacts when the real problem is string annotations.
# Any library that reads annotations at runtime - KFP, and older FastAPI
# and Pydantic - has this constraint.

from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

OUT = Path(__file__).parent / "compiled"
BASE_IMAGE = "python:3.12-slim"

# Placeholders - nothing in this repository contains a real project id.
PROJECT = "your-project-id"
REGION = "us-central1"
PIPELINE_ROOT = "gs://your-bucket/pipeline-root"

# The bar a model must clear before it is allowed near an endpoint.
ACCURACY_THRESHOLD = 0.85


# ==================================================== 1-2. components
@dsl.component(base_image=BASE_IMAGE, packages_to_install=["pandas==2.2.3"])
def ingest(row_limit: int, output_data: Output[Dataset]) -> None:
    """Every component runs in its own container.

    Three things make this a component rather than a function:

      - the decorator, which says how to containerise it
      - packages_to_install, resolved at RUNTIME inside that container. The
        function body cannot see anything imported at module level in this
        file, which is why the imports are inside.
      - typed inputs and outputs. Output[Dataset] is an ARTIFACT: KFP
        allocates a path, passes it in, and tracks whatever is written
        there in the metadata store.
    """
    import csv
    import random

    random.seed(42)
    with open(output_data.path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_a", "feature_b", "label"])
        for _ in range(row_limit):
            a = random.gauss(0, 1)
            b = random.gauss(0, 1)
            w.writerow([a, b, int(a + b > 0)])

    # Metadata travels with the artifact and shows up in the run graph.
    output_data.metadata["rows"] = row_limit
    output_data.metadata["source"] = "synthetic"


@dsl.component(base_image=BASE_IMAGE,
               packages_to_install=["pandas==2.2.3", "scikit-learn==1.6.1"])
def validate(input_data: Input[Dataset], report: Output[Metrics]) -> bool:
    """Validate BEFORE training, and return a decision.

    This component returns a bool, which is a PARAMETER rather than an
    artifact - small values passed by value. Parameters can be used in
    control flow; artifact CONTENTS cannot, because the orchestrator never
    opens them.
    """
    import pandas as pd

    df = pd.read_csv(input_data.path)
    n_rows = len(df)
    n_null = int(df.isna().sum().sum())
    balance = float(df["label"].mean())

    report.log_metric("rows", n_rows)
    report.log_metric("nulls", n_null)
    report.log_metric("class_balance", balance)

    # The checks worth making are the ones that caused real failures
    # earlier in this repository: too little data, nulls silently imputed
    # downstream, and a class balance that makes accuracy meaningless.
    ok = n_rows >= 100 and n_null == 0 and 0.2 <= balance <= 0.8
    report.log_metric("passed", int(ok))
    return ok


@dsl.component(base_image=BASE_IMAGE,
               packages_to_install=["pandas==2.2.3", "scikit-learn==1.6.1"])
def train(input_data: Input[Dataset], n_estimators: int,
          model: Output[Model], metrics: Output[Metrics]) -> float:
    """Train, evaluate, and return the metric the gate will read."""
    import pickle

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_data.path)
    X = df[["feature_a", "feature_b"]]
    y = df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    acc = float(accuracy_score(y_te, proba > 0.5))
    auc = float(roc_auc_score(y_te, proba))

    metrics.log_metric("accuracy", acc)
    metrics.log_metric("roc_auc", auc)
    metrics.log_metric("n_estimators", n_estimators)

    with open(model.path, "wb") as f:
        pickle.dump(clf, f)
    model.metadata["framework"] = "scikit-learn"
    model.metadata["accuracy"] = acc

    return acc


@dsl.component(base_image=BASE_IMAGE)
def deploy(model: Input[Model], accuracy: float,
           endpoint: Output[Model]) -> None:
    """The step the gate protects.

    Deliberately a stub. Uploading to the Vertex AI Model Registry and
    deploying to an endpoint is a real call, and this repository does not
    make real calls - see the README. What matters structurally is that
    this component only ever runs inside the dsl.If below.
    """
    print(f"Would upload model from {model.path} to the Model Registry")
    print(f"Would deploy to an endpoint - accuracy {accuracy:.4f}")
    endpoint.metadata["deployed"] = True
    endpoint.metadata["accuracy"] = accuracy


@dsl.component(base_image=BASE_IMAGE)
def alert(reason: str, accuracy: float) -> None:
    """The other branch. A pipeline that silently does nothing when the
    model is bad is worse than one that fails loudly.
    """
    print(f"BLOCKED: {reason} (accuracy {accuracy:.4f})")


# ============================================ 3-4. the pipeline graph
@dsl.pipeline(
    name="train-and-conditionally-deploy",
    description="Ingest, validate, train, and deploy only if the model "
                "clears an accuracy threshold.",
    pipeline_root=PIPELINE_ROOT,
)
def ml_pipeline(row_limit: int = 2000,
                n_estimators: int = 100,
                accuracy_threshold: float = ACCURACY_THRESHOLD) -> None:
    """The graph.

    Note there is no explicit ordering anywhere below. KFP infers the DAG
    from DATA DEPENDENCIES: train consumes ingest's output, so train runs
    after ingest. Steps with no dependency between them run in parallel.
    """
    ingest_task = ingest(row_limit=row_limit)
    ingest_task.set_display_name("Ingest data")

    validate_task = validate(input_data=ingest_task.outputs["output_data"])
    validate_task.set_display_name("Validate data")

    # Only train if validation passed. Catching bad data here rather than
    # after an hour of GPU time is most of the point of a pipeline.
    #
    # NOTE `.outputs["Output"]`, not `.output`. A task with a SINGLE output
    # exposes `.output`; validate() has two (the Metrics artifact and the
    # bool it returns), so `.output` raises and the bare return value is
    # keyed under the literal name "Output".
    with dsl.If(validate_task.outputs["Output"] == True,  # noqa: E712
                name="data-is-valid"):
        train_task = train(input_data=ingest_task.outputs["output_data"],
                           n_estimators=n_estimators)
        train_task.set_display_name("Train model")
        # Retries and resources are per-task, and this is where they go.
        train_task.set_cpu_limit("2").set_memory_limit("4G")
        train_task.set_retry(num_retries=2, backoff_duration="60s")

        # THE DEPLOYMENT GATE. The float train() returned is a PARAMETER,
        # so it can drive control flow - an artifact could not.
        accuracy = train_task.outputs["Output"]

        with dsl.If(accuracy >= accuracy_threshold,
                    name="model-is-good-enough"):
            deploy_task = deploy(model=train_task.outputs["model"],
                                 accuracy=accuracy)
            deploy_task.set_display_name("Deploy to endpoint")

        with dsl.Else(name="model-failed-the-gate"):
            alert(reason="accuracy below threshold",
                  accuracy=accuracy).set_display_name("Block deploy")


# ==================================================== 5. compile
def explain_and_compile() -> Path:
    OUT.mkdir(exist_ok=True)
    target = OUT / "pipeline.yaml"
    compiler.Compiler().compile(pipeline_func=ml_pipeline,
                                package_path=str(target))
    return target


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def main() -> None:
    print("=" * 72)
    print("  KUBEFLOW PIPELINES v2")
    print("=" * 72)
    print("\n  The compile below is real. The pipeline has never been")
    print("  submitted - no cluster, no project, no billing.")

    banner("1. Components: a function, a container, and a contract")
    print("  Five components are defined in this file. Each one becomes a")
    print("  step that runs in its OWN CONTAINER:\n")
    for c in (ingest, validate, train, deploy, alert):
        doc = (c.python_func.__doc__ or "").strip().split("\n")[0]
        print(f"    {c.python_func.__name__:<10} {doc}")

    print("\n  Three consequences of 'its own container' that catch people:")
    print("\n    THE FUNCTION BODY IS ISOLATED. It cannot see imports made at")
    print("    the top of this file, or any other function here. Every")
    print("    import must be inside the component - which looks wrong and")
    print("    is correct.")
    print("\n    DEPENDENCIES INSTALL AT RUNTIME. packages_to_install runs a")
    print("    pip install when the step starts, on every run. Pin the")
    print("    versions - an unpinned pipeline is not reproducible - and for")
    print("    anything heavy, bake a custom base image instead.")
    print("\n    STEPS SHARE NOTHING BUT ARTIFACTS. There is no shared memory")
    print("    and no shared filesystem. If a value needs to cross a step")
    print("    boundary it is either a parameter or an artifact.")

    banner("2. Artifacts vs parameters")
    print(f"  {'':<12} {'parameter':<34} artifact")
    print("  " + "-" * 74)
    for a, b, c in [
        ("what", "a small value (int, str, float, bool)",
         "a file or directory"),
        ("passed", "BY VALUE, through the orchestrator",
         "by REFERENCE - a path"),
        ("examples", "n_estimators, a threshold, a flag",
         "Dataset, Model, Metrics"),
        ("control flow", "YES - can drive dsl.If",
         "NO - contents are never read by KFP"),
        ("tracked", "as a run input", "in the metadata store, with lineage"),
    ]:
        print(f"  {a:<12} {b:<34} {c}")

    print("\n  The control-flow row is the one that shapes real pipelines.")
    print("  KFP's orchestrator never opens an artifact, so you cannot")
    print("  branch on 'the accuracy inside this Metrics file'. That is why")
    print("  train() RETURNS a float as well as logging it - the returned")
    print("  value is a parameter, and only a parameter can drive the gate.")

    banner("3. The graph")
    print("  There is not a single explicit ordering statement in the")
    print("  pipeline function. KFP infers the DAG from DATA DEPENDENCIES:")
    print("  train consumes ingest's output, so it runs after it. Steps with")
    print("  no dependency between them run in parallel automatically.")
    print("\n  (set_upstream exists for the rare case where a step depends on")
    print("  another's side effect rather than its output. Reach for it")
    print("  seldom - it is usually a sign the data flow is not modelled.)")

    banner("4. The deployment gate")
    print("  Two nested conditions, and the second is the point:\n")
    print("    dsl.If(validate_task.output == True)")
    print("        train")
    print("        dsl.If(train_task.output >= accuracy_threshold)")
    print("            deploy")
    print("        dsl.Else()")
    print("            alert")
    print(f"\n  The threshold is a PIPELINE PARAMETER (default "
          f"{ACCURACY_THRESHOLD}), not a")
    print("  constant, so it can be raised per run without touching code.")
    print("\n  This is the structural difference between a script and a")
    print("  pipeline. A script trains a model and a human decides whether")
    print("  to ship it. A pipeline makes that decision a reviewable,")
    print("  version-controlled step that cannot be skipped because someone")
    print("  was in a hurry on a Friday.")
    print("\n  Note also the validation gate BEFORE training. Catching bad")
    print("  data there rather than after an hour of GPU time is most of")
    print("  what a pipeline buys you on cost alone.")

    banner("5. Compile")
    target = explain_and_compile()
    text = target.read_text(encoding="utf-8")
    lines = text.split("\n")
    print(f"  Compiled to {target.relative_to(Path(__file__).parent)} - "
          f"{len(lines):,} lines, {target.stat().st_size / 1024:.0f} KB")

    import yaml
    spec = yaml.safe_load(text)
    comps = sorted(spec.get("components", {}))
    print(f"\n  The YAML contains {len(comps)} component definitions:")
    for c in comps:
        print(f"    {c}")
    print(f"\n  schemaVersion: {spec.get('schemaVersion')}")
    print(f"  sdkVersion:    {spec.get('sdkVersion')}")

    print("\n  THAT YAML IS THE ARTEFACT. The Python in this file is a")
    print("  convenient way to author it; what you submit, version and")
    print("  review is the compiled spec. It is self-contained - each")
    print("  component carries its base image and its pip commands - which")
    print("  is why a pipeline can be handed to a cluster that has never")
    print("  seen this repository.")

    banner("Submitting it - not done here")
    print("  Vertex AI Pipelines:")
    print("      from google.cloud import aiplatform")
    print("      aiplatform.init(project=PROJECT, location=REGION)")
    print("      aiplatform.PipelineJob(")
    print("          display_name='train-and-deploy',")
    print("          template_path='compiled/pipeline.yaml',")
    print("          pipeline_root=PIPELINE_ROOT,")
    print("          parameter_values={'n_estimators': 200},")
    print("      ).submit()")
    print("\n  The same YAML runs on any Kubeflow Pipelines v2 backend. That")
    print("  portability is real, and it is the main argument for KFP over a")
    print("  vendor-specific orchestrator.")

    banner("What to take away")
    print("  - A COMPONENT IS A FUNCTION PLUS A CONTAINER. Its body is")
    print("    isolated: every import goes inside, and it shares nothing")
    print("    with the rest of the file.")
    print("  - PIN packages_to_install. They install at runtime on every")
    print("    run, so an unpinned pipeline is not reproducible. Bake a")
    print("    custom image for anything heavy.")
    print("  - PARAMETERS are small values passed by value and CAN drive")
    print("    control flow. ARTIFACTS are paths, tracked with lineage, and")
    print("    CANNOT - the orchestrator never opens them. Return a value")
    print("    as well as logging it when you need to branch on it.")
    print("  - THE DAG IS INFERRED from data dependencies. No explicit")
    print("    ordering, and independent steps parallelise for free.")
    print("  - VALIDATE BEFORE TRAINING. Failing on bad data before an hour")
    print("    of GPU time is most of the cost argument for pipelines.")
    print("  - THE DEPLOYMENT GATE is the structural point: a model cannot")
    print("    reach an endpoint without clearing a threshold, and that")
    print("    check is in version control rather than in someone's head.")
    print("  - THE COMPILED YAML IS THE ARTEFACT, and it is portable across")
    print("    any KFP v2 backend.")
    print("=" * 72)


if __name__ == "__main__":
    main()
