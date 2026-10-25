"""Submitting the trainer package to Vertex AI - the three ways, and the
decisions each one forces.

NOT EXECUTED. Every Vertex call is inside a function and every import is
local to it, so this module runs and explains itself with no credentials,
no project and no billing. The code is real; run it against your own
project by filling in the constants and calling the functions.

  1. Prebuilt container vs custom container - the actual decision
  2. Submitting a CustomJob
  3. Hyperparameter tuning, and the metric-name trap
  4. Deploying, and the cost question that decides batch vs online
  5. Local first - why, and what it catches

Run:
    python 05_gcp/04_custom_training/submit.py

Concept reference: Foundations volume, Part D.4.
"""

from __future__ import annotations

# Replace these with your own. They are deliberately obvious placeholders -
# nothing in this repository should contain a real project id.
PROJECT = "your-project-id"
REGION = "us-central1"
BUCKET = "gs://your-bucket"
IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT}/ml-repo/trainer:latest"


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ================================ 1. prebuilt vs custom container
def container_choice() -> None:
    print("  Two ways to get your code onto a Vertex training node:\n")
    print("    PREBUILT CONTAINER + PYTHON PACKAGE")
    print("      You hand Vertex a .tar.gz of your trainer package and name")
    print("      a prebuilt image (a TF, PyTorch or sklearn version). Vertex")
    print("      pip-installs your package into it and runs your module.")
    print("      + nothing to build, nothing to push, fast to iterate")
    print("      - you get the dependencies that image has, plus whatever")
    print("        pip can add at job start")
    print("\n    CUSTOM CONTAINER")
    print("      You build a Docker image and push it to Artifact Registry.")
    print("      + exact control: system packages, CUDA versions, private")
    print("        wheels, a compiled dependency")
    print("      + the image is the reproducibility boundary - it pins")
    print("        everything, not just the Python layer")
    print("      - you now own a build, a registry and an image lifecycle")
    print("\n  The decision rule that answers most exam questions: use the")
    print("  PREBUILT container unless you need something it cannot give you.")
    print("  'We need a specific CUDA version', 'an apt package', 'a private")
    print("  wheel' - those are custom-container signals. 'We want to train a")
    print("  Keras model' is not.")


# ========================================== 2. submitting a CustomJob
def submit_python_package_job():
    """Prebuilt container route. The package is a .tar.gz in GCS."""
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)

    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name="fashion-cnn",
        python_package_gcs_uri=f"{BUCKET}/trainer/trainer-0.1.tar.gz",
        python_module_name="trainer.task",          # matches trainer/task.py
        container_uri=(
            "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest"),
        model_serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-17:latest"),
    )
    # args are exactly the flags task.py's argparse defines
    return job.run(
        model_display_name="fashion-cnn",
        args=["--epochs", "10", "--learning-rate", "0.001"],
        # task.py defaults to the torch backend; this image has only TF.
        # The custom container sets the same thing with ENV in its Dockerfile.
        environment_variables={"KERAS_BACKEND": "tensorflow"},
        replica_count=1,
        machine_type="n1-standard-4",
        # accelerator_type="NVIDIA_TESLA_T4", accelerator_count=1,
        sync=True,
    )


def submit_custom_container_job():
    """Custom container route. The image is already in Artifact Registry."""
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)
    job = aiplatform.CustomContainerTrainingJob(
        display_name="fashion-cnn-container",
        container_uri=IMAGE,
        model_serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-17:latest"),
    )
    return job.run(model_display_name="fashion-cnn",
                   args=["--epochs", "10"],
                   replica_count=1, machine_type="n1-standard-4", sync=True)


# ============================================ 3. hyperparameter tuning
def submit_tuning_job():
    """Vizier-backed search over the flags task.py exposes.

    THE TRAP: `metric_spec` names a metric, and that name must match the
    hyperparameter_metric_tag the trainer reports (see report_metric in
    trainer/task.py, which reports "accuracy"). If they disagree the job
    runs to completion, costs money, and reports nothing - with no error.
    """
    from google.cloud import aiplatform
    from google.cloud.aiplatform import hyperparameter_tuning as hpt

    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)

    worker_pool_specs = [{
        "machine_spec": {"machine_type": "n1-standard-4"},
        "replica_count": 1,
        "container_spec": {"image_uri": IMAGE},
    }]

    custom_job = aiplatform.CustomJob(
        display_name="fashion-cnn-tune",
        worker_pool_specs=worker_pool_specs,
        staging_bucket=BUCKET)

    return aiplatform.HyperparameterTuningJob(
        display_name="fashion-cnn-tune",
        custom_job=custom_job,
        metric_spec={"accuracy": "maximize"},        # must match the trainer
        parameter_spec={
            "learning-rate": hpt.DoubleParameterSpec(
                min=1e-4, max=1e-2, scale="log"),    # log scale for LR
            "batch-size": hpt.DiscreteParameterSpec(
                values=[32, 64, 128, 256], scale=None),
            "dropout": hpt.DoubleParameterSpec(min=0.0, max=0.5, scale="linear"),
        },
        max_trial_count=20,
        parallel_trial_count=4,        # fewer parallel = smarter search
    ).run(sync=True)


def tuning_notes() -> None:
    print("  Three things that decide whether a tuning job is worth its cost:")
    print("\n    THE METRIC NAME MUST MATCH. metric_spec here says 'accuracy';")
    print("    trainer/task.py reports 'accuracy'. If those disagree the job")
    print("    completes, bills you, and reports nothing - with no error.")
    print("\n    SCALE MATTERS. Learning rate must be searched on a LOG scale")
    print("    - a linear sweep from 1e-4 to 1e-2 spends about half its")
    print("    trials above 5e-3 and 91% above 1e-3 - just 9% for the whole")
    print("    1e-4 to 1e-3 decade. The measured healthy band in")
    print("    03_neural_networks/02 spanned three orders of magnitude.")
    print("\n    PARALLEL TRIALS TRADE SPEED FOR INTELLIGENCE. Vizier is")
    print("    Bayesian - it learns from completed trials. Running all 20 at")
    print("    once means 20 uninformed trials; running 4 at a time lets it")
    print("    use what it learned. Fewer parallel is usually better per")
    print("    dollar, and slower in wall-clock time.")


# ============================================= 4. deploy, or not
def deploy_model(model_resource_name: str):
    """Online serving: an endpoint with nodes that bill while they exist."""
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT, location=REGION)
    model = aiplatform.Model(model_resource_name)
    return model.deploy(
        machine_type="n1-standard-2",
        min_replica_count=1,        # 1, not 0 - see below
        max_replica_count=3,
        traffic_split={"0": 100},
    )


def batch_predict(model_resource_name: str):
    """Batch: no endpoint, no idle cost, higher latency."""
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT, location=REGION)
    model = aiplatform.Model(model_resource_name)
    return model.batch_predict(
        job_display_name="fashion-batch",
        gcs_source=f"{BUCKET}/input/instances.jsonl",
        gcs_destination_prefix=f"{BUCKET}/output/",
        machine_type="n1-standard-4",
        starting_replica_count=1, max_replica_count=5, sync=True)


def serving_choice() -> None:
    print(f"  {'':<14} {'online endpoint':<30} batch prediction")
    print("  " + "-" * 74)
    for a, b, c in [
        ("latency", "tens of ms", "minutes to hours"),
        ("cost at rest", "PAYS PER NODE-HOUR, always", "nothing"),
        ("input", "one request at a time", "a file in GCS"),
        ("scales to zero", "no (min_replica_count >= 1)", "n/a - no endpoint"),
        ("use when", "a user is waiting", "scoring a table nightly"),
    ]:
        print(f"  {a:<14} {b:<30} {c}")

    print("\n  'Cost at rest' is the row that decides most real architectures.")
    print("  A Vertex endpoint bills for its nodes whether or not anyone")
    print("  calls it, and min_replica_count cannot be 0 for a standard")
    print("  endpoint - so an internal model used twice a day still costs a")
    print("  full month of node time.")
    print("\n  A question that says 'predictions are needed once a day' or")
    print("  'scores are written back to a warehouse' is describing BATCH,")
    print("  and an endpoint is the distractor. The reverse - 'a user is")
    print("  waiting', 'sub-second' - is an endpoint.")


# =============================================== 5. local first
def local_first() -> None:
    print("  Run the trainer locally before ever submitting it:\n")
    print("    python 05_gcp/04_custom_training/trainer/task.py --epochs 3")
    print("\n  That command has been run in this repository and works - 0.8000")
    print("  test accuracy in 2.6 s. It is the only part of 05_gcp that has")
    print("  actually been executed, and it is deliberately the part where")
    print("  execution matters most.")
    print("\n  What a local run catches, cheaply, that a cloud job catches")
    print("  expensively and forty minutes later:")
    for item in [
        "a typo in an argparse flag name",
        "a module path that does not match the package layout",
        "a missing dependency",
        "an output path built wrong",
        "a metric name that does not match the tuning spec",
        "the model failing to serialise at the end of a long run",
    ]:
        print(f"    - {item}")
    print("\n  The trainer is written so the two paths are the same code:")
    print("  AIP_MODEL_DIR is read from the environment with a local")
    print("  fallback, so nothing branches on 'am I in the cloud'.")


def main() -> None:
    print("=" * 72)
    print("  CUSTOM TRAINING ON VERTEX AI")
    print("=" * 72)
    print("\n  Write-only: no jobs submitted, no project touched, no billing.")
    print("  The trainer package itself DOES run locally - see section 5.")

    banner("1. Prebuilt container or custom container?")
    container_choice()

    banner("2. Submitting the job")
    print("  Two functions are defined above:")
    print("    submit_python_package_job()     prebuilt image + a .tar.gz")
    print("    submit_custom_container_job()   your own image")
    print("\n  Both take args=[...] matching the flags trainer/task.py")
    print("  defines. That is the contract between the two files: Vertex")
    print("  passes hyperparameters as command-line arguments, so anything")
    print("  hardcoded in the trainer cannot be tuned.")

    banner("3. Hyperparameter tuning")
    tuning_notes()

    banner("4. Online endpoint or batch prediction?")
    serving_choice()

    banner("5. Run it locally first")
    local_first()

    banner("What to take away")
    print("  - A TRAINER PACKAGE is an ordinary Python package whose entry")
    print("    point takes hyperparameters as FLAGS and writes to")
    print("    AIP_MODEL_DIR. Write it so it runs locally unchanged.")
    print("  - PREBUILT CONTAINER unless you need a system package, a CUDA")
    print("    version or a private wheel. Then custom container.")
    print("  - The tuning metric name must match what the trainer reports,")
    print("    or the job silently reports nothing.")
    print("  - Search learning rate on a LOG scale.")
    print("  - Fewer PARALLEL trials lets Vizier learn - better per dollar,")
    print("    slower in wall-clock.")
    print("  - AN ENDPOINT BILLS AT REST and cannot scale to zero. Batch")
    print("    prediction has no idle cost. 'Once a day' means batch.")
    print("  - RUN IT LOCALLY FIRST. It catches the cheap mistakes before")
    print("    they become slow, billed ones.")
    print("=" * 72)


if __name__ == "__main__":
    main()
