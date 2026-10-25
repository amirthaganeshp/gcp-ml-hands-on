"""The whole repository on one page: exam weights, coverage, and the numbers.

Renders a single-page cheat sheet and prints the same content as text, so
it is useful on a screen, on paper, and in a terminal.

  1. Exam weight by blueprint section, and which part covers it
  2. The lifecycle, and where each part sits
  3. The decision rules that answer the most questions
  4. Every measured headline number in one table

EXECUTED IN FULL. The numbers are the ones measured by the other parts of
this repository, quoted here rather than recomputed - each is reproducible
by running the script named beside it.

Run:
    python 10_summary/cheatsheet.py

Concept reference: everything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402

OUT = Path(__file__).parent / "images"

# (section, weight %, repo parts, state)
BLUEPRINT = [
    ("1  Low-code AI solutions",        13, "05_gcp/01-03", "covered"),
    ("2  Data and model collaboration",  16, "01, 02, 05_gcp", "partial"),
    ("3  Scaling prototypes",            21, "02, 03, 04, 05_gcp/04+06", "covered"),
    ("4  Serving and scaling",           20, "07_serving", "covered"),
    ("5  Pipelines and automation",      18, "06_pipelines", "partial"),
    ("6  Monitoring AI solutions",       13, "08_monitoring", "covered"),
]

# (label, value, where to reproduce it)
NUMBERS = [
    ("concept drift: input divergence", "0.0000", "08_monitoring/01"),
    ("concept drift: accuracy lost", "15 pts", "08_monitoring/01"),
    ("fairness: TPR gap before / after", "0.1217 / 0.0020", "08_monitoring/02"),
    ("proxy: recover 'sex' from other cols", "0.8358", "08_monitoring/02"),
    ("batch vs online at 10k rows/day", "4800x", "07_serving/01"),
    ("always-warm replica = batch of", "48M rows/day", "07_serving/01"),
    ("A/B: detect 2% lift, per arm", "752,703", "07_serving/02"),
    ("canary 5% vs A/B 50/50 to decide", "12.2 d / 1.2 d", "07_serving/02"),
    ("INT8: size / speed on a small model", "3.7x / SLOWER", "07_serving/03"),
    ("batching 1->128: QPS / user latency", "10.4x / 4291x", "07_serving/03"),
    ("accelerator fed every 30ms", "27% utilised", "05_gcp/06"),
    ("capstone: accuracy vs majority", "0.8507 / 0.7511", "09_capstone"),
]

RULES = [
    ("a user is waiting",              "online endpoint"),
    ("a nightly table or file",        "BATCH - no endpoint"),
    ("intermittent, no idle cost",     "Cloud Run, scale to zero"),
    ("no cold starts, strict p99",     "min_replica_count >= 1"),
    ("roll out safely",                "CANARY (95/5)"),
    ("which converts better",          "A/B (50/50)"),
    ("real traffic, zero user risk",   "SHADOW"),
    ("features stable, accuracy down", "CONCEPT DRIFT - needs labels"),
    ("training data gone",             "drift detection only"),
    ("drop the protected column",      "ALWAYS WRONG - proxies"),
    ("accelerator utilisation low",    "fix the INPUT PIPELINE"),
    ("model does not fit in memory",   "MODEL parallelism"),
    ("data already in BigQuery",       "BigQuery ML"),
    ("images/text, no ML staff",       "AutoML"),
    ("custom architecture or loss",    "custom training"),
    ("extract fields from invoices",   "Document AI, not Vision OCR"),
    ("tree ensemble, need attributions", "Sampled Shapley (not IG)"),
    ("employees pasting data into LLM", "Model Armor + Sensitive Data Prot."),
]


def rule_line(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_text() -> None:
    print("PMLE CHEAT SHEET")

    rule_line("1. EXAM WEIGHT, AND WHERE IT LIVES IN THIS REPO")
    print(f"  {'blueprint section':<34} {'%':>4}  {'repo':<26} state")
    print("  " + "-" * 72)
    for name, w, parts, state in BLUEPRINT:
        print(f"  {name:<34} {w:>3}%  {parts:<26} {state}")
    print(f"\n  {'':<34} {sum(w for _, w, _, _ in BLUEPRINT):>3}%")

    rule_line("2. THE LIFECYCLE")
    print("""
    data -> split -> train -> EVALUATE -> serve -> MONITOR
              |        |         |          |         |
         01_found   02/03/04  01_found  07_serving 08_monitoring
         (leakage)  (models)  (metrics) (batch vs  (skew, drift,
                                         online)    fairness)

  The two in capitals are where models fail. Everyone trains.""")

    rule_line("3. THE DECISION RULES")
    print(f"  {'the scenario says':<36} the answer")
    print("  " + "-" * 72)
    for says, answer in RULES:
        print(f"  {says:<36} {answer}")

    rule_line("4. EVERY MEASURED NUMBER")
    print(f"  {'what':<38} {'value':<18} reproduce with")
    print("  " + "-" * 72)
    for label, value, where in NUMBERS:
        print(f"  {label:<38} {value:<18} {where}")


def render() -> None:
    use_house_style()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.30, wspace=0.18)

    fig.suptitle("Professional ML Engineer - one page",
                 fontsize=19, fontweight="bold", y=0.975)

    # ---- exam weights -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    names = [b[0] for b in BLUEPRINT][::-1]
    weights = [b[1] for b in BLUEPRINT][::-1]
    states = [b[3] for b in BLUEPRINT][::-1]
    colours = [PALETTE["blue"] if s == "covered" else PALETTE["orange"] for s in states]
    ax.barh(names, weights, color=colours)
    for i, (w, p) in enumerate(zip(weights, [b[2] for b in BLUEPRINT][::-1])):
        ax.text(w + 0.4, i, f"{w}%  {p}", va="center", fontsize=8.5)
    ax.set_xlim(0, 34)
    ax.set_xlabel("% of the exam")
    ax.set_title("Exam weight, and which part covers it", fontsize=12, loc="left")
    ax.legend(handles=[mpatches.Patch(color=PALETTE["blue"], label="runnable examples"),
                       mpatches.Patch(color=PALETTE["orange"], label="partial")],
              frameon=False, fontsize=8.5, loc="lower right")

    # ---- lifecycle --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("The lifecycle", fontsize=12, loc="left")
    steps = ["data", "split", "train", "EVALUATE", "serve", "MONITOR"]
    owners = ["", "01_found", "02/03/04", "01_found", "07_serving", "08_monitoring"]
    for i, (s, o) in enumerate(zip(steps, owners)):
        x = 0.04 + i * 0.16
        hot = s.isupper()
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.56), 0.122, 0.17, boxstyle="round,pad=0.012",
            facecolor=PALETTE["vermillion"] if hot else PALETTE["sky"],
            edgecolor="none", transform=ax.transAxes))
        ax.text(x + 0.061, 0.645, s, ha="center", va="center", fontsize=9.5,
                fontweight="bold" if hot else "normal",
                color="white" if hot else "black", transform=ax.transAxes)
        if o:
            ax.text(x + 0.061, 0.49, o, ha="center", va="center", fontsize=7.5,
                    color="#444", transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 0.156, 0.645), xytext=(x + 0.132, 0.645),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
    ax.text(0.04, 0.33,
            "The two in red are where models actually fail.\n"
            "Everyone trains. Far fewer evaluate honestly, and\n"
            "fewer still monitor for the failure that never\n"
            "shows up in the inputs.",
            fontsize=9.5, va="top", transform=ax.transAxes)

    # ---- decision rules ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_title("If the scenario says...", fontsize=12, loc="left")
    for i, (says, answer) in enumerate(RULES):
        y = 0.95 - i * 0.053
        ax.text(0.0, y, says, fontsize=8.6, va="top", transform=ax.transAxes)
        ax.text(0.52, y, answer, fontsize=8.6, va="top", fontweight="bold",
                color=PALETTE["blue"], transform=ax.transAxes)

    # ---- numbers ----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("Every measured number, and where to reproduce it",
                 fontsize=12, loc="left")
    for i, (label, value, where) in enumerate(NUMBERS):
        y = 0.95 - i * 0.079
        ax.text(0.0, y, label, fontsize=8.6, va="top", transform=ax.transAxes)
        ax.text(0.60, y, value, fontsize=8.6, va="top", fontweight="bold",
                color=PALETTE["vermillion"], transform=ax.transAxes)
        ax.text(0.82, y, where, fontsize=7.6, va="top", color="#666",
                transform=ax.transAxes)

    save_figure(fig, OUT / "cheatsheet.png")
    print(f"\n  cheat sheet -> {OUT / 'cheatsheet.png'}")


def main() -> None:
    print_text()
    render()
    rule_line("HOW TO USE THIS")
    print("""  The figure is the revision aid; the tables above are the same content
  for a terminal. Every number in section 4 is reproducible - run the
  script named beside it and you will get that figure back, because the
  data generator is seeded.

  If a number here disagrees with a fresh run, the cheat sheet is wrong.""")


if __name__ == "__main__":
    main()
