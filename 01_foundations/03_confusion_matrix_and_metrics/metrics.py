"""The confusion matrix, and every metric that comes out of it.

Four counts - TP, FP, TN, FN - generate every classification metric you will ever
report. This script computes them by hand, checks the hand calculation against
scikit-learn, and then shows what each one is for.

  1. The confusion matrix on a real binary problem, with every metric derived
     from it manually so the formulas stop being abstract
  2. The precision-recall trade-off as the decision threshold moves - the point
     being that you CHOOSE where to sit on it, and the choice is a business one
  3. The accuracy trap: a model that predicts one class for everything, scoring
     99% on an imbalanced problem
  4. Multiclass, and why macro and weighted averaging disagree

Run:
    python 01_foundations/03_confusion_matrix_and_metrics/metrics.py

Concept reference: Foundations volume, Part A.6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_penguins, load_rice  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
POSITIVE = "Cammeo"  # the minority variety, so "positive" means something
rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ------------------------------------------------------- 1. the four counts
def figure_confusion(y_true, y_pred) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["Osmancik", POSITIVE]).ravel()

    print(f"  Positive class = {POSITIVE} (the minority variety)\n")
    print("                      PREDICTED")
    print("                  Cammeo    Osmancik")
    print(f"  ACTUAL Cammeo   {tp:>6}    {fn:>8}      <- {tp + fn} real Cammeo grains")
    print(f"         Osmancik {fp:>6}    {tn:>8}      <- {fp + tn} real Osmancik grains")

    print("\n  Every metric, computed by hand and checked against scikit-learn:\n")
    manual = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }
    library = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE),
        "recall": recall_score(y_true, y_pred, pos_label=POSITIVE),
        "specificity": recall_score(y_true, y_pred, pos_label="Osmancik"),
        "f1": f1_score(y_true, y_pred, pos_label=POSITIVE),
    }
    formulas = {
        "accuracy": "(TP+TN)/all      of everything, how much was right",
        "precision": "TP/(TP+FP)       of what I flagged, how much was real",
        "recall": "TP/(TP+FN)       of what was real, how much did I catch",
        "specificity": "TN/(TN+FP)       of the negatives, how many did I pass",
        "f1": "2TP/(2TP+FP+FN)  harmonic mean of precision and recall",
    }
    print(f"  {'metric':<12} {'by hand':>9} {'sklearn':>9}   formula")
    for k in manual:
        agree = "ok" if abs(manual[k] - library[k]) < 1e-9 else "MISMATCH"
        print(f"  {k:<12} {manual[k]:>9.4f} {library[k]:>9.4f}  [{agree}] {formulas[k]}")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=[POSITIVE, "Osmancik"]),
        display_labels=[POSITIVE, "Osmancik"],
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion matrix - rice variety")
    ax.grid(False)
    print(f"\n  saved  {save_figure(fig, OUT / '1_confusion_matrix.png').name}")
    return int(tp), int(fp), int(tn), int(fn)


# ---------------------------------------- 2. precision and recall trade off
def figure_threshold(model, X_te, y_te) -> None:
    proba = model.predict_proba(X_te)[:, list(model.classes_).index(POSITIVE)]
    thresholds = np.linspace(0.02, 0.98, 97)
    prec, rec, f1s = [], [], []
    for t in thresholds:
        pred = np.where(proba >= t, POSITIVE, "Osmancik")
        prec.append(precision_score(y_te, pred, pos_label=POSITIVE, zero_division=1))
        rec.append(recall_score(y_te, pred, pos_label=POSITIVE, zero_division=0))
        f1s.append(f1_score(y_te, pred, pos_label=POSITIVE, zero_division=0))

    best_f1 = int(np.argmax(f1s))
    # the operating point a business constraint would pick, not the one F1 likes
    want_recall = 0.95
    idx_recall = int(np.argmin(np.abs(np.array(rec) - want_recall)))

    print(f"  {'threshold':>9} {'precision':>10} {'recall':>8} {'F1':>7}")
    for t in (0.10, 0.30, 0.50, 0.70, 0.90):
        i = int(np.argmin(np.abs(thresholds - t)))
        print(f"  {thresholds[i]:>9.2f} {prec[i]:>10.3f} {rec[i]:>8.3f} {f1s[i]:>7.3f}")
    print(f"\n  Best F1          at threshold {thresholds[best_f1]:.2f}  "
          f"(P {prec[best_f1]:.3f}, R {rec[best_f1]:.3f})")
    print(f"  'Catch 95% of Cammeo' at threshold {thresholds[idx_recall]:.2f}  "
          f"(P {prec[idx_recall]:.3f}, R {rec[idx_recall]:.3f})")
    print("  Same model, same data. The threshold is a BUSINESS decision, not a")
    print("  modelling one - and 0.5 is a default, not an answer.")

    fig, ax = plt.subplots()
    ax.plot(thresholds, prec, color=PALETTE["blue"], label="precision")
    ax.plot(thresholds, rec, color=PALETTE["vermillion"], label="recall")
    ax.plot(thresholds, f1s, color=PALETTE["green"], linestyle="--", label="F1")
    ax.axvline(0.5, color=PALETTE["grey"], linestyle=":", label="default 0.5")
    ax.axvline(thresholds[idx_recall], color=PALETTE["black"], linestyle="-.",
               linewidth=1.3, label=f"recall {want_recall:.0%} target")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score")
    ax.set_title("You do not get precision and recall. You choose where to sit.")
    ax.legend(loc="lower center", ncol=2)
    print(f"\n  saved  {save_figure(fig, OUT / '2_threshold_tradeoff.png').name}")


# ------------------------------------------------------ 3. the accuracy trap
def figure_accuracy_trap(X, y) -> None:
    """Thin the positive class to 1%, then predict the majority for everything."""
    pos = np.flatnonzero(y == POSITIVE)
    neg = np.flatnonzero(y != POSITIVE)
    keep_pos = rng.choice(pos, max(int(0.01 / 0.99 * len(neg)), 5), replace=False)
    idx = np.concatenate([neg, keep_pos])
    y_imb = y[idx]
    rate = (y_imb == POSITIVE).mean()

    lazy = np.full(len(y_imb), "Osmancik", dtype=object)
    acc = accuracy_score(y_imb, lazy)
    rec = recall_score(y_imb, lazy, pos_label=POSITIVE, zero_division=0)
    prec = precision_score(y_imb, lazy, pos_label=POSITIVE, zero_division=0)
    f1 = f1_score(y_imb, lazy, pos_label=POSITIVE, zero_division=0)

    print(f"  Imbalanced set: {len(y_imb):,} grains, {rate:.1%} are {POSITIVE}")
    print(f"  Model: predict 'Osmancik' for everything. It has no features and no brain.\n")
    print(f"    accuracy   {acc:.4f}   <- looks superb")
    print(f"    precision  {prec:.4f}")
    print(f"    recall     {rec:.4f}   <- it caught NOTHING")
    print(f"    F1         {f1:.4f}")
    print("\n  This is why accuracy is close to meaningless on imbalanced problems,")
    print("  and why 'our model is 99% accurate' should always prompt a second question.")

    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["accuracy", "precision", "recall", "F1"]
    vals = [acc, prec, rec, f1]
    colours = [PALETTE["vermillion"], PALETTE["blue"], PALETTE["blue"], PALETTE["blue"]]
    bars = ax.bar(names, vals, color=colours, alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title(f"A model that predicts one class for everything ({rate:.1%} positives)")
    ax.annotate("accuracy says 'excellent'\neverything else says 'useless'",
                xy=(0.30, 0.62), xycoords="axes fraction", color=PALETTE["grey"])
    print(f"\n  saved  {save_figure(fig, OUT / '3_accuracy_trap.png').name}")


# --------------------------------------------- 4. multiclass: macro vs weighted
def figure_multiclass() -> None:
    df = load_penguins().dropna(subset=["bill_length_mm", "bill_depth_mm",
                                        "flipper_length_mm", "body_mass_g", "species"])
    X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].to_numpy()
    y = df["species"].to_numpy()
    # make one species rare, so macro and weighted have something to disagree about
    keep = (y != "Chinstrap") | (rng.random(len(y)) < 0.25)
    X, y = X[keep], y[keep]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=SEED, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_tr, y_tr)
    pred = model.predict(X_te)

    counts = {c: int((y_te == c).sum()) for c in sorted(set(y_te))}
    print(f"  Test-set composition: {counts}\n")
    print(f"  {'averaging':<10} {'precision':>10} {'recall':>8} {'F1':>7}   what it does")
    notes = {
        "macro": "unweighted mean - every class counts equally",
        "weighted": "weighted by support - big classes dominate",
        "micro": "pool all TP/FP/FN - equals accuracy here",
    }
    for avg in ("macro", "weighted", "micro"):
        p = precision_score(y_te, pred, average=avg, zero_division=0)
        r = recall_score(y_te, pred, average=avg, zero_division=0)
        f = f1_score(y_te, pred, average=avg, zero_division=0)
        print(f"  {avg:<10} {p:>10.3f} {r:>8.3f} {f:>7.3f}   {notes[avg]}")

    per_class = {c: f1_score(y_te, pred, labels=[c], average="macro", zero_division=0)
                 for c in counts}
    print("\n  Per-class F1:", {k: round(v, 3) for k, v in per_class.items()})
    print("  If the rare class scores worse, MACRO exposes it and WEIGHTED hides it.")
    print("  That is why macro is the fairness-flavoured choice.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ConfusionMatrixDisplay(
        confusion_matrix(y_te, pred, labels=sorted(counts)), display_labels=sorted(counts)
    ).plot(ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title("Multiclass confusion matrix")
    axes[0].grid(False)

    ax = axes[1]
    names = list(per_class)
    ax.bar(names, [per_class[c] for c in names], color=PALETTE["blue"], alpha=0.9)
    ax.axhline(f1_score(y_te, pred, average="macro", zero_division=0),
               color=PALETTE["vermillion"], linestyle="--", label="macro F1")
    ax.axhline(f1_score(y_te, pred, average="weighted", zero_division=0),
               color=PALETTE["green"], linestyle=":", label="weighted F1")
    for i, c in enumerate(names):
        ax.text(i, per_class[c] + 0.02, f"n={counts[c]}", ha="center", fontsize=9,
                color=PALETTE["grey"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1")
    ax.set_title("Macro treats classes equally; weighted follows the crowd")
    ax.legend(loc="lower right")
    print(f"\n  saved  {save_figure(fig, OUT / '4_multiclass.png').name}")


def main() -> None:
    use_house_style()
    print("=" * 72)
    print("  THE CONFUSION MATRIX AND EVERYTHING DERIVED FROM IT")
    print("=" * 72)

    df = load_rice()
    X = df.drop(columns=["class"]).to_numpy()
    y = df["class"].to_numpy()
    print(f"\n  Rice: {len(df):,} grains, {X.shape[1]} shape measurements")
    print(f"  Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=SEED, stratify=y)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, random_state=SEED))
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    banner("1. The four counts, and every metric they generate")
    figure_confusion(y_te, y_pred)

    banner("2. Precision and recall trade against each other")
    figure_threshold(model, X_te, y_te)

    banner("3. The accuracy trap")
    figure_accuracy_trap(X, y)

    banner("4. Multiclass: macro vs weighted averaging")
    figure_multiclass()

    banner("What to take away")
    print("  - Four counts generate every metric. Learn the four, derive the rest.")
    print("  - PRECISION when a false positive is expensive (blocking a good")
    print("    customer). RECALL when a false negative is expensive (missing")
    print("    a fault). You cannot have both; the scenario tells you which.")
    print("  - The threshold is a business decision. 0.5 is a default, not an answer.")
    print("  - ACCURACY IS MEANINGLESS ON IMBALANCED DATA. A brainless model scored")
    print("    99% above. Always ask what the base rate is.")
    print("  - Macro averaging exposes a failing rare class; weighted hides it.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
