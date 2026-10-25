"""The three implementations side by side - and why they disagree.

Run the three framework scripts first; each appends its result to
results.json. This one tabulates them and then investigates the thing worth
investigating: the same architecture, the same hyperparameters and the same
data produced three different accuracies.

  1. The table
  2. Why the accuracies differ - measured, not guessed
  3. What each framework asks of you, feature by feature

Run:
    python 04_frameworks/01_tensorflow/tf_version.py
    python 04_frameworks/02_pytorch/torch_version.py
    python 04_frameworks/03_jax/jax_version.py
    python 04_frameworks/00_same_model_three_ways/compare.py

Concept reference: Foundations volume, Part C.4.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from common import EXPECTED_PARAMS, RESULTS, banner  # noqa: E402

OUT = Path(__file__).parent / "images"


def load_results() -> dict:
    if not RESULTS.exists():
        raise SystemExit(
            "No results.json. Run the three framework scripts first - see "
            "the docstring at the top of this file.")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    missing = {"TensorFlow", "PyTorch", "JAX"} - set(data)
    if missing:
        print(f"  WARNING: no result yet for {', '.join(sorted(missing))}. "
              f"The table below is incomplete.\n")
    return data


# ================================================================ 1. table
def show_table(r: dict) -> None:
    print(f"  {'framework':<14} {'test acc':>9} {'train s':>9} {'params':>9} "
          f"{'lines':>7}")
    for name in ("TensorFlow", "PyTorch", "JAX"):
        if name not in r:
            continue
        v = r[name]
        print(f"  {name:<14} {v['accuracy']:>9.4f} {v['seconds']:>9.1f} "
              f"{v['params']:>9,} {v['lines']:>7}")

    params = {v["params"] for v in r.values()}
    print(f"\n  All three report {params.pop():,} parameters "
          f"({'identical' if not params else 'MISMATCH'}) - the architectures")
    print(f"  really are the same. Each script asserts this against the "
          f"shared\n  spec of {EXPECTED_PARAMS:,} and exits if it drifts.")

    accs = [v["accuracy"] for v in r.values()]
    secs = [v["seconds"] for v in r.values()]
    lines = [v["lines"] for v in r.values()]
    print(f"\n  Accuracy spread: {max(accs) - min(accs):.4f}")
    print(f"  Time spread:     {max(secs) / min(secs):.1f}x")
    print(f"  Lines of code:   {min(lines)} to {max(lines)} - "
          f"essentially the same")

    print("\n  The lines-of-code number is the one that surprises people, so")
    print("  be clear about what it counts. These files are dominated by")
    print("  narration and demonstrations, not by the model. The MODELLING")
    print("  code differs far more than the totals suggest: Keras trains in")
    print("  one fit() call, PyTorch needs ~35 lines of loop, and JAX needs")
    print("  the loop plus the layers themselves.")


# ====================================== 2. why the accuracies differ
def explain_accuracy(r: dict) -> None:
    """The architectures match and the hyperparameters match, so the
    difference has to come from a DEFAULT none of the scripts set.

    It does: the three frameworks initialise weights differently, and the
    accuracy ordering follows the initialisation scale exactly.
    """
    import tensorflow as tf
    import torch
    import torch.nn as nn
    import jax

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf.keras.utils.set_random_seed(42)
        torch.manual_seed(42)

        # the same layer in each: Conv2D 16 -> 32 channels, 3x3
        fan_in = 3 * 3 * 16
        k = tf.keras.layers.Conv2D(32, 3, padding="same")
        k.build((None, 14, 14, 16))
        keras_w = np.asarray(k.get_weights()[0]).ravel()
        torch_w = (nn.Conv2d(16, 32, 3, padding=1)
                   .weight.detach().numpy().ravel())
        jax_w = np.asarray(
            jax.random.normal(jax.random.PRNGKey(42), (3, 3, 16, 32))
            * np.sqrt(2.0 / fan_in)).ravel()

        keras_std = float(keras_w.std())
        torch_std = float(torch_w.std())
        jax_std = float(jax_w.std())

    he = float(np.sqrt(2.0 / fan_in))
    glorot = float(np.sqrt(2.0 / (fan_in + 3 * 3 * 32)))

    print("  Same layer in each framework - Conv2D, 16 -> 32 channels, 3x3,")
    print(f"  fan_in = {fan_in}. Standard deviation of the INITIAL weights:\n")
    print(f"  {'framework':<26} {'init std':>9}   default scheme")
    print(f"  {'Keras Conv2D':<26} {keras_std:>9.4f}   glorot_uniform")
    print(f"  {'PyTorch Conv2d':<26} {torch_std:>9.4f}   kaiming_uniform(a=sqrt(5))")
    print(f"  {'this JAX (written by hand)':<26} {jax_std:>9.4f}   he_normal")
    print(f"\n  For reference: He target {he:.4f}, Glorot target {glorot:.4f}.")

    order = sorted(((v["accuracy"], n) for n, v in r.items()), reverse=True)
    stds = {"TensorFlow": keras_std, "PyTorch": torch_std, "JAX": jax_std}
    print("\n  Now put that next to the accuracies:\n")
    print(f"  {'framework':<14} {'init std':>9} {'test acc':>9}")
    for acc, name in order:
        print(f"  {name:<14} {stds[name]:>9.4f} {acc:>9.4f}")

    ranks_match = [n for _, n in order] == sorted(
        stds, key=lambda n: -stds[n])
    print(f"\n  Accuracy ordering matches initialisation-scale ordering: "
          f"{ranks_match}")

    print("\n  Identical architecture, identical optimiser, identical learning")
    print("  rate, identical data, identical seed - and a spread of "
          f"{max(v['accuracy'] for v in r.values()) - min(v['accuracy'] for v in r.values()):.3f}")
    print("  accuracy. Initialisation scale is a REAL part of it: the same")
    print("  PyTorch model over five seeds averages He 0.878 / Glorot 0.868 /")
    print("  default 0.860. But it is not the whole explanation. Seed and")
    print("  data-order noise alone is about a point (PyTorch's default ranged")
    print("  0.846-0.869 over those five seeds), so a single run per framework")
    print("  cannot separate gaps smaller than that.")
    print("\n  PyTorch's default is the notable one. kaiming_uniform with")
    print("  a=sqrt(5) is a historical accident rather than a recommendation -")
    print("  it produces a gain appropriate to a leaky ReLU with slope sqrt(5),")
    print("  which is nobody's network. For ReLU the right choice is He")
    print("  initialisation, and PyTorch makes you ask for it explicitly.")
    print("\n  The practical lesson for porting a model between frameworks:")
    print("  matching the LAYERS is the easy part. Matching the defaults -")
    print("  initialisers, epsilon in the optimiser, whether the loss expects")
    print("  logits or probabilities, weight decay - is where a port silently")
    print("  loses two points of accuracy and nobody can say why.")

    return {
        "TensorFlow": (keras_w, keras_std, "glorot_uniform"),
        "PyTorch": (torch_w, torch_std, "kaiming_uniform(a=$\\sqrt{5}$)"),
        "JAX": (jax_w, jax_std, "he_normal"),
    }


def plot_initialisers(weights: dict) -> None:
    """Draw the three initial weight distributions.

    The table above gives the standard deviations, which is most of the
    story. What a table CANNOT show is the shape: two of these are uniform
    and one is normal. Same layer, same fan-in, three different densities.
    """
    sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))
    from plotting import PALETTE, save_figure, use_house_style

    import matplotlib.pyplot as plt

    use_house_style()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colours = {"TensorFlow": "blue", "PyTorch": "vermillion", "JAX": "green"}

    for name, (w, std, scheme) in weights.items():
        # 4,608 real weights (3x3x16x32). Few enough that a fine binning is
        # mostly sampling noise, which hides the flat tops that are the point.
        ax.hist(w, bins=42, density=True, histtype="step", linewidth=2.2,
                color=PALETTE[colours[name]],
                label=f"{name} - {scheme}\n    std {std:.4f}")

    ax.set_xlabel("initial weight value")
    ax.set_ylabel("density")
    ax.set_title("Same layer, three frameworks, three default initialisers")
    ax.legend(loc="upper right", fontsize=9.5, labelspacing=0.9)

    ax.text(
        0.015, 0.97,
        "Conv2D 16$\\rightarrow$32, 3$\\times$3, fan-in 144.\n"
        "Two are UNIFORM (flat-topped), one is NORMAL (bell).\n"
        "That shape difference is invisible in a table of standard deviations.",
        transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
        color=PALETTE["grey"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#DDDDDD"),
    )

    out = save_figure(fig, OUT / "initialiser_distributions.png")
    print(f"\n  Saved {out.resolve().relative_to(Path(__file__).resolve().parents[2])}")


# ====================================== 3. what each asks of you
def feature_matrix() -> None:
    rows = [
        ("Training loop",       "fit()",            "you write it",   "you write it"),
        ("Early stopping",      "callback",         "you write it",   "you write it"),
        ("Checkpointing",       "callback",         "you write it",   "orbax"),
        ("LR scheduling",       "callback",         "torch.optim.lr_scheduler", "optax chain"),
        ("Input pipeline",      "tf.data",          "DataLoader",     "you write it"),
        ("Shape inference",     "automatic",        "you declare",    "you declare"),
        ("Layers provided",     "keras.layers",     "torch.nn",       "jax.lax (raw)"),
        ("Model state",         "in the object",    "in the object",  "a value you hold"),
        ("Randomness",          "global seed",      "global seed",    "explicit keys"),
        ("Serving artefact",    "SavedModel",       "TorchScript/ONNX", "you decide"),
        ("Channel order",       "NHWC",             "NCHW",           "either"),
        ("Loss expects",        "probabilities",    "logits",         "logits"),
    ]
    print(f"  {'':<20} {'TensorFlow':<18} {'PyTorch':<24} {'JAX'}")
    print("  " + "-" * 82)
    for label, a, b, c in rows:
        print(f"  {label:<20} {a:<18} {b:<24} {c}")

    print("\n  Read the 'Loss expects' row twice. Keras'")
    print("  sparse_categorical_crossentropy defaults to probabilities, so its")
    print("  model ends in softmax. PyTorch's CrossEntropyLoss applies")
    print("  log_softmax itself, so its model must NOT. Add a softmax to the")
    print("  PyTorch model and it still trains - just worse - which is the")
    print("  single most common bug when porting between them.")


def main() -> None:
    print("=" * 72)
    print("  THE SAME MODEL, THREE WAYS")
    print("=" * 72)
    r = load_results()

    banner("1. The results")
    show_table(r)

    banner("2. Why the accuracies differ")
    weights = explain_accuracy(r)
    plot_initialisers(weights)

    banner("3. What each framework asks of you")
    feature_matrix()

    banner("Which one should you use?")
    print("  For the GCP ML Engineer exam: TENSORFLOW/KERAS. Vertex AI's")
    print("  prebuilt containers, SavedModel serving, TFRecord, TFX and")
    print("  BigQuery ML all assume it, and the exam is written around that")
    print("  stack.")
    print("\n  In practice, and independent of the exam:")
    print("    KERAS      when the model is standard and you want it working")
    print("               today. The callbacks alone justify it.")
    print("    PYTORCH    when you need to change the training loop, or when")
    print("               you are reading research code - almost all of it is")
    print("               PyTorch now.")
    print("    JAX        when you need composable transformations or TPUs.")
    print("               It is the smallest idea of the three and the most")
    print("               demanding to build on.")
    print("\n  Worth knowing: KERAS 3 RUNS ON ALL THREE BACKENDS. Everything")
    print("  in 03_neural_networks was written against the Keras API and runs")
    print("  on PyTorch, selected by a single environment variable. The API")
    print("  and the backend are now separable choices, which makes 'which")
    print("  framework' a smaller decision than it used to be.")
    print("\n  And the honest summary of this whole folder: all three trained")
    print("  the same network to within 2.5 points, in under 13 seconds, in")
    print("  about the same number of lines. The framework is rarely the thing")
    print("  that decides whether a project works.")


if __name__ == "__main__":
    main()
