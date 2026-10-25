"""The parts that must be IDENTICAL across all three frameworks.

The comparison in this folder is only worth anything if the data, the
architecture and the hyperparameters are the same in every implementation.
Putting them here rather than copying them into three files makes that
checkable rather than merely claimed.

What is deliberately NOT here: anything framework-specific. Each script
builds its own model, its own input pipeline and its own training loop, so
you can read them side by side and see what each framework actually asks of
you.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- the spec
SEED = 42
N_TRAIN, N_TEST = 10000, 3000      # subsampled: this stays a laptop-CPU example
EPOCHS = 8
BATCH = 128
LR = 1e-3

# The same convolutional network in all three, described once so the three
# implementations can be checked against it:
#
#   Conv2D(16, 3x3, same padding) -> ReLU -> MaxPool 2x2      28x28 -> 14x14
#   Conv2D(32, 3x3, same padding) -> ReLU -> MaxPool 2x2      14x14 ->  7x7
#   Flatten                                                    -> 1568
#   Dense(64) -> ReLU
#   Dense(10) -> softmax
#
# Expected trainable parameters: 105,866. Each script asserts this, which is
# the cheapest possible check that the three architectures really do match.
EXPECTED_PARAMS = 105_866

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

RESULTS = Path(__file__).parent / "results.json"


def load_data() -> dict[str, np.ndarray]:
    """Fashion-MNIST, subsampled identically for every framework.

    Loaded through keras.datasets purely because it is a convenient cache -
    the arrays handed back are plain NumPy, so nothing about the data
    pipeline is TensorFlow-flavoured before each script gets it.
    """
    import os
    os.environ.setdefault("KERAS_BACKEND", "torch")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import truststore
    truststore.inject_into_ssl()         # corporate TLS inspection
    import keras

    (xtr, ytr), (xte, yte) = keras.datasets.fashion_mnist.load_data()
    rng = np.random.default_rng(SEED)
    i = rng.choice(len(xtr), N_TRAIN, replace=False)
    j = rng.choice(len(xte), N_TEST, replace=False)
    return {
        "x_train": (xtr[i] / 255.0).astype("float32")[..., None],
        "y_train": ytr[i].astype("int64"),
        "x_test": (xte[j] / 255.0).astype("float32")[..., None],
        "y_test": yte[j].astype("int64"),
    }


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def header(framework: str, version: str) -> None:
    print("=" * 72)
    print(f"  {framework.upper()}")
    print("=" * 72)
    print(f"\n  {framework} {version}")
    print(f"  Fashion-MNIST {N_TRAIN:,} train / {N_TEST:,} test, "
          f"{EPOCHS} epochs, batch {BATCH}, Adam lr={LR}")


def check_params(n: int) -> None:
    """Fail loudly if an implementation drifted from the shared architecture."""
    status = "matches" if n == EXPECTED_PARAMS else "DOES NOT MATCH"
    print(f"  Trainable parameters: {n:,}  ({status} the shared spec "
          f"of {EXPECTED_PARAMS:,})")
    if n != EXPECTED_PARAMS:
        raise SystemExit(
            f"Architecture drift: {n:,} parameters, expected "
            f"{EXPECTED_PARAMS:,}. The three implementations must match or "
            f"the comparison is meaningless.")


def source_lines(path: str | Path) -> int:
    """Non-blank, non-comment, non-docstring lines - a rough 'how much did
    this framework make me write' measure. Rough on purpose: it is indicative,
    not a benchmark.
    """
    text = Path(path).read_text(encoding="utf-8").split("\n")
    count, in_doc = 0, False
    for raw in text:
        line = raw.strip()
        if line.startswith('"""') or line.startswith("'''"):
            # a one-line docstring opens and closes on the same line
            if not (len(line) > 3 and line.endswith(('"""', "'''"))):
                in_doc = not in_doc
            continue
        if in_doc or not line or line.startswith("#"):
            continue
        count += 1
    return count


def record(framework: str, accuracy: float, seconds: float, params: int,
           lines: int, notes: str = "") -> None:
    """Append this run to results.json so compare.py can tabulate all three."""
    data = {}
    if RESULTS.exists():
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
    data[framework] = {"accuracy": round(accuracy, 4),
                       "seconds": round(seconds, 1),
                       "params": params, "lines": lines, "notes": notes}
    RESULTS.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n  Recorded to {RESULTS.name}: accuracy {accuracy:.4f}, "
          f"{seconds:.1f}s, {lines} lines of code.")


def add_repo_root() -> None:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "00_setup"))
