"""A Vertex AI trainer package entry point - and it runs locally too.

This is the file Vertex AI executes when you submit a custom training job.
It is also an ordinary Python script, and running it locally first is the
single most effective way to avoid discovering a typo forty minutes into a
cloud job.

What makes it a TRAINER PACKAGE rather than just a script:

  - arguments come from the command line, because that is how Vertex passes
    hyperparameters
  - the output location comes from AIP_MODEL_DIR, an environment variable
    Vertex sets, with a local fallback
  - the model is written (model.keras) to that directory
  - metrics are reported back to the hyperparameter tuning service
  - it exits non-zero on failure, because that is how the job reports

Run locally:
    python 05_gcp/04_custom_training/trainer/task.py --epochs 3

Submit to Vertex AI:
    see ../submit.py - not executed in this repository
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402


def get_args() -> argparse.Namespace:
    """Every hyperparameter Vertex might tune must be a command-line flag.

    The hyperparameter tuning service works by launching this script many
    times with different flag values. Anything hardcoded cannot be tuned.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n-train", type=int, default=6000,
                   help="subsample size; keeps the local run quick")
    # Vertex passes the output location via AIP_MODEL_DIR. Accepting it as a
    # flag too makes local runs and cloud runs the same code path.
    p.add_argument("--model-dir", type=str,
                   default=os.environ.get("AIP_MODEL_DIR", ""))
    return p.parse_args()


def resolve_model_dir(arg: str) -> Path:
    """AIP_MODEL_DIR is a gs:// URI on Vertex and empty locally.

    Never wrap a gs:// URI in Path: pathlib collapses the double slash
    (gs://bucket/x -> gs:/bucket/x), so every later write goes to a local
    directory that does not exist - and the job trains for its full run
    before failing at model.save. Custom training jobs mount Cloud Storage
    with Cloud Storage FUSE at /gcs/<bucket>, so a gs:// URI becomes an
    ordinary file path there. Locally we fall back to a directory beside
    this package, so the script never needs to know which environment it
    is in.
    """
    if arg.startswith("gs://"):
        if not Path("/gcs").is_dir():
            # Fail before training, not after: no FUSE mount means this is
            # not a Vertex training node, and the save would fail at the end.
            raise SystemExit(f"--model-dir {arg} needs the Cloud Storage FUSE "
                             "mount at /gcs, which only exists on Vertex "
                             "training nodes. Use a local path here.")
        path = Path("/gcs") / arg[len("gs://"):]   # the FUSE mount on Vertex
    elif arg:
        path = Path(arg)
    else:
        path = Path(__file__).resolve().parents[1] / "local_output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_data(n_train: int) -> tuple:
    import keras
    import truststore
    truststore.inject_into_ssl()
    (xtr, ytr), (xte, yte) = keras.datasets.fashion_mnist.load_data()
    rng = np.random.default_rng(42)
    i = rng.choice(len(xtr), min(n_train, len(xtr)), replace=False)
    j = rng.choice(len(xte), 2000, replace=False)
    return ((xtr[i] / 255.0).astype("float32")[..., None], ytr[i],
            (xte[j] / 255.0).astype("float32")[..., None], yte[j])


def build_model(dropout: float, learning_rate: float):
    import keras
    layers = [
        keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
    ]
    if dropout:
        layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(10, activation="softmax"))
    m = keras.Sequential(layers)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def report_metric(name: str, value: float) -> None:
    """Report a metric to Vertex AI hyperparameter tuning.

    The real mechanism is cloudml-hypertune, which writes to a file Vertex
    reads. It is an optional dependency, so this degrades to a print when it
    is absent - which is exactly what happens on a local run.
    """
    try:
        import hypertune
        hypertune.HyperTune().report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=name,
            metric_value=value)
        print(f"  reported to hypertune: {name}={value:.4f}")
    except ImportError:
        print(f"  [local] would report to hypertune: {name}={value:.4f}")


def main() -> int:
    args = get_args()
    on_vertex = bool(os.environ.get("AIP_MODEL_DIR"))
    print("=" * 72)
    print("  VERTEX AI TRAINER PACKAGE")
    print("=" * 72)
    print(f"\n  Environment: {'Vertex AI' if on_vertex else 'LOCAL'}")
    print(f"  Args: epochs={args.epochs} batch={args.batch_size} "
          f"lr={args.learning_rate} dropout={args.dropout}")

    model_dir = resolve_model_dir(args.model_dir)
    print(f"  Model dir: {model_dir}")

    xtr, ytr, xte, yte = load_data(args.n_train)
    print(f"  Data: {len(xtr):,} train / {len(xte):,} test")

    import keras
    keras.utils.set_random_seed(42)
    model = build_model(args.dropout, args.learning_rate)
    print(f"  Parameters: {model.count_params():,}\n")

    t0 = time.perf_counter()
    model.fit(xtr, ytr, validation_data=(xte, yte), epochs=args.epochs,
              batch_size=args.batch_size, verbose=2)
    secs = time.perf_counter() - t0

    loss, acc = model.evaluate(xte, yte, verbose=0)
    print(f"\n  Trained in {secs:.1f}s")
    print(f"  Test accuracy: {acc:.4f}   loss: {loss:.4f}")

    # The metric name here must match the one named in the tuning job spec.
    # A mismatch is the classic reason a tuning job runs and reports nothing.
    report_metric("accuracy", float(acc))

    out = model_dir / "model.keras"
    model.save(out)
    (model_dir / "metrics.json").write_text(
        json.dumps({"accuracy": float(acc), "loss": float(loss),
                    "seconds": round(secs, 1),
                    "args": vars(args)}, indent=2), encoding="utf-8")
    print(f"  Saved {out}")
    print("\n  On Vertex AIP_MODEL_DIR is a gs:// URI, written through the")
    print("  /gcs FUSE mount, so the artefacts land in Cloud Storage where")
    print("  the Model Registry picks them up.")
    return 0


if __name__ == "__main__":
    # Exit code is how a training job reports success or failure. An
    # uncaught exception gives a non-zero exit and Vertex marks the job
    # FAILED, which is the behaviour you want - do not swallow errors here.
    sys.exit(main())
