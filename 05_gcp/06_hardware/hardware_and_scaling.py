"""Choosing hardware for training, and scaling across more than one of it.

Blueprint section 3.3. Two questions:

    which accelerator - CPU, GPU or TPU?
    and when one is not enough, which distribution strategy?

Both are usually decided by arithmetic and by the SHAPE of the model, not
by which chip is fastest in isolation. The fastest chip is irrelevant if
your input pipeline cannot feed it.

  1. The three, and what each is actually for
  2. Cost per epoch - where the crossover is
  3. The utilisation trap: a fast chip waiting for data
  4. Distribution strategies, and which one a scenario implies
  5. Data parallelism vs model parallelism

EXECUTED IN FULL. The cost arithmetic is computed; the throughput figures
are representative published rates, printed so a stale number is visible.

Run:
    python 05_gcp/06_hardware/hardware_and_scaling.py

Concept reference: Success Guide, Part 3.3. Blueprint section 3 (~21%).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402

OUT = Path(__file__).parent / "images"

# Representative CPU-relative throughput and published-ish list rates.
# The RATIOS are the durable part; the prices move.
HARDWARE = {
    #            $/hour   relative throughput on a big dense model
    "CPU (n1-8)":  (0.38,   1.0),
    "GPU (T4)":    (0.35,  12.0),
    "GPU (A100)":  (2.93,  75.0),
    "TPU v5e":     (1.20,  95.0),
}


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------- 1. the three
def section_three() -> None:
    rule("1. THE THREE, AND WHAT EACH IS FOR")
    rows = [
        ("CPU", "small data, classical ML, prototyping, inference of small models",
         "anything with big dense matmuls"),
        ("GPU", "most deep learning; flexible, every framework, custom ops",
         "when a TPU would be cheaper at scale"),
        ("TPU", "very large dense matmuls - transformers, big CNNs, huge batches",
         "custom ops, dynamic shapes, tiny models"),
    ]
    print(f"  {'':<5} {'good for':<52} bad for")
    print("  " + "-" * 70)
    for name, good, bad in rows:
        print(f"  {name:<5} {good:<52} {bad}")

    print("\n  TPUs are not 'faster GPUs'. They are systolic arrays built for large")
    print("  dense matrix multiplication. A model full of custom ops, control flow")
    print("  or dynamic shapes may not compile for one at all - and a small model")
    print("  will not fill one, so you pay for silicon you never use.")


# ------------------------------------------------------------------ 2. cost
def section_cost() -> dict:
    rule("2. COST PER EPOCH - the arithmetic")

    # One epoch takes this many hours on a single CPU machine.
    cpu_hours = 40.0
    print(f"  A model that takes {cpu_hours:.0f} CPU-hours per epoch.\n")
    print(f"  {'hardware':<14} {'$/hour':>8} {'speedup':>9} {'hours':>8} {'$/epoch':>10}")
    print("  " + "-" * 54)

    out = {}
    for name, (rate, speed) in HARDWARE.items():
        hours = cpu_hours / speed
        cost = hours * rate
        out[name] = (hours, cost)
        print(f"  {name:<14} {rate:>8.2f} {speed:>8.0f}x {hours:>8.2f} {cost:>10.2f}")

    best = min(out, key=lambda k: out[k][1])
    fastest = min(out, key=lambda k: out[k][0])
    print(f"\n  Cheapest per epoch : {best}  (${out[best][1]:.2f})")
    print(f"  Fastest per epoch  : {fastest}  ({out[fastest][0]:.2f} h)")
    if best != fastest:
        print("\n  They are NOT the same hardware, which is the whole point. 'Fastest'")
        print("  and 'cheapest' are different questions and the scenario picks one.")
    else:
        print("\n  Here they coincide - they often do not.")
    return out


# ------------------------------------------------------- 3. the utilisation trap
def section_utilisation() -> None:
    rule("3. THE UTILISATION TRAP - a fast chip waiting for data")

    accel_step_ms = 8.0          # the accelerator can do a step this fast
    print(f"  The accelerator can complete a training step in {accel_step_ms:.0f} ms.")
    print("  What the input pipeline delivers decides whether it ever does.\n")
    print(f"  {'input pipeline':>20} {'step ms':>9} {'accel utilisation':>18} {'wasted $':>10}")
    print("  " + "-" * 60)

    rate = HARDWARE["GPU (A100)"][0]
    for name, input_ms in (("fast (cached)", 2.0), ("ok", 8.0),
                           ("slow (decode/step)", 30.0), ("very slow", 80.0)):
        step = max(accel_step_ms, input_ms)
        util = accel_step_ms / step
        wasted = rate * (1 - util)
        print(f"  {name:>20} {step:>9.0f} {util:>18.0%} {wasted:>9.2f}/h")

    print("\n  LOW ACCELERATOR UTILISATION IS AN INPUT PROBLEM, NOT A COMPUTE ONE.")
    print("  Buying a bigger accelerator makes the utilisation worse and the bill")
    print("  larger. The fix is the data pipeline: prefetch, cache, parallel")
    print("  interleave, TFRecord/Parquet instead of per-file decode.")
    print("\n  Exam shape: 'GPU utilisation is 15% and training is slow' -> fix the")
    print("  input pipeline. An option offering a larger machine type is the trap.")


# ------------------------------------------------------------ 4. strategies
def section_strategies() -> None:
    rule("4. DISTRIBUTION STRATEGIES")
    rows = [
        ("MirroredStrategy", "several GPUs, ONE machine",
         "the usual first step beyond one GPU"),
        ("MultiWorkerMirroredStrategy", "several GPUs, SEVERAL machines",
         "one machine is no longer enough"),
        ("TPUStrategy", "TPU cores / pods",
         "you have committed to TPU"),
        ("ParameterServerStrategy", "many machines, asynchronous",
         "very large sparse models, embeddings"),
    ]
    print(f"  {'strategy':<30} {'topology':<28} use when")
    print("  " + "-" * 70)
    for name, topo, when in rows:
        print(f"  {name:<30} {topo:<28} {when}")

    print("\n  The giveaway is the TOPOLOGY in the question. 'Multiple GPUs on a")
    print("  single VM' is Mirrored. 'Across multiple VMs' is MultiWorker. If the")
    print("  scenario says TPU, the strategy is decided for you.")


# --------------------------------------------------------- 5. the two parallels
def section_parallelism() -> None:
    rule("5. DATA PARALLELISM vs MODEL PARALLELISM")
    print("  DATA parallelism   every worker holds the WHOLE model,")
    print("                     each sees a different slice of the batch,")
    print("                     gradients are averaged.")
    print("                     -> the default, and what every strategy above does")
    print()
    print("  MODEL parallelism  the MODEL is split across devices because it does")
    print("                     not fit in one device's memory.")
    print("                     -> only when the model itself is too big")
    print()
    print("  Exam shape: 'the model does not fit in GPU memory' -> MODEL parallelism")
    print("  (or gradient checkpointing, or a smaller batch). 'Training is too slow")
    print("  and the model fits fine' -> DATA parallelism.")
    print("\n  Scaling data parallelism also scales the EFFECTIVE BATCH SIZE, which")
    print("  changes the optimisation problem - the learning rate usually has to")
    print("  rise with it. More workers is not a free speedup.")


def figure(costs: dict) -> None:
    use_house_style()
    names = list(costs)
    hours = [costs[n][0] for n in names]
    dollars = [costs[n][1] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.barh(names, hours, color=PALETTE["blue"])
    ax1.set_xlabel("hours per epoch")
    ax1.set_title("Speed")
    ax2.barh(names, dollars, color=PALETTE["vermillion"])
    ax2.set_xlabel("$ per epoch")
    ax2.set_title("Cost")
    for ax in (ax1, ax2):
        ax.invert_yaxis()
    save_figure(fig, OUT / "hardware_cost.png")
    print(f"\n  figure -> {OUT / 'hardware_cost.png'}")


def main() -> None:
    print("HARDWARE AND SCALING")
    section_three()
    costs = section_cost()
    section_utilisation()
    section_strategies()
    section_parallelism()
    figure(costs)

    rule("THE EXAM ANGLE")
    print("""  the scenario says                        the answer
  ------------------------------------------------------------------
  classical ML, small data                 CPU
  most deep learning, custom ops           GPU
  very large dense matmuls, huge batches   TPU
  "accelerator utilisation is low"         fix the INPUT PIPELINE
  several GPUs, one VM                     MirroredStrategy
  several VMs                              MultiWorkerMirroredStrategy
  TPU anywhere in the question             TPUStrategy
  huge sparse embeddings, async            ParameterServerStrategy
  "the model does not fit in memory"       MODEL parallelism
  "training is too slow", model fits       DATA parallelism
  edge / on-device                         quantize -> TF Lite -> Coral""")


if __name__ == "__main__":
    main()
