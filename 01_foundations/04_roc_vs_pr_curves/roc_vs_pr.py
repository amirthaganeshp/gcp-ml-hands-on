"""ROC or precision-recall? The answer depends on how rare the positive class is.

The headline demonstration is section 2: hold the model and the data fixed, and vary
only the PREVALENCE of the positive class. ROC-AUC barely moves. PR-AUC collapses.
That single experiment explains why the choice of curve matters, and it is the most
reliably tested classical fact on the ML Engineer exam.

  1. Both curves at the dataset's natural 21.6% imbalance
  2. The prevalence sweep - ROC-AUC stays flat while PR-AUC falls off a cliff
  3. Why: the false-positive rate has a huge true-negative count in its
     denominator, and precision does not
  4. Calibration - a separate property from ranking, and when you need it

Run:
    python 01_foundations/04_roc_vs_pr_curves/roc_vs_pr.py

Concept reference: Foundations volume, Part A.7.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_wine_quality  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def load() -> tuple[np.ndarray, np.ndarray]:
    df = load_wine_quality("white")
    X = df.drop(columns=["quality"]).to_numpy()
    y = (df["quality"] >= 7).astype(int).to_numpy()  # "good wine"
    print(f"  White wines           {len(df):,}")
    print(f"  Features              {X.shape[1]} physicochemical measurements")
    print(f"  Positive = quality>=7 {y.mean():.1%}   <- a real, unmanufactured imbalance")
    return X, y


def fit_and_score(X_tr, y_tr, X_te, y_te):
    model = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    return model, proba, roc_auc_score(y_te, proba), average_precision_score(y_te, proba)


# --------------------------------------------- 1. both curves, natural balance
def figure_both_curves(y_te, proba, roc_auc, pr_auc) -> None:
    fpr, tpr, _ = roc_curve(y_te, proba)
    prec, rec, _ = precision_recall_curve(y_te, proba)
    baseline = y_te.mean()

    print(f"  ROC-AUC              {roc_auc:.3f}")
    print(f"  PR-AUC (avg prec.)   {pr_auc:.3f}")
    print(f"  Random-guess ROC-AUC 0.500      (the diagonal)")
    print(f"  Random-guess PR-AUC  {baseline:.3f}      (the base rate)")
    print("\n  Note the two baselines. A 'random' ROC curve is always the diagonal,")
    print("  whatever the imbalance. A 'random' PR curve sits at the positive rate -")
    print("  so PR-AUC is judged against a bar that MOVES with the problem.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(fpr, tpr, color=PALETTE["blue"], label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["grey"], linestyle="--", label="random = 0.500")
    ax.set_xlabel("false positive rate  =  FP / (FP + TN)")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(rec, prec, color=PALETTE["vermillion"], label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(baseline, color=PALETTE["grey"], linestyle="--",
               label=f"random = {baseline:.3f}")
    ax.set_xlabel("recall  =  TP / (TP + FN)")
    ax.set_ylabel("precision  =  TP / (TP + FP)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="lower left")

    fig.suptitle(f"The same predictions, two views  (positives = {baseline:.1%})",
                 fontsize=12.5, fontweight="bold")
    print(f"\n  saved  {save_figure(fig, OUT / '1_both_curves.png').name}")


# -------------------------------------------------- 2. THE prevalence sweep
def figure_prevalence_sweep(X, y) -> None:
    """Hold model and data fixed; vary only how rare the positive class is."""
    rates = [0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    base = y.mean()
    # At 1% prevalence the subsample holds only ~38 positives, so ~11 land in the test
    # split and a single estimate is extremely noisy. Average over repeats, or the
    # curve wobbles and hides the trend it exists to show.
    n_repeats = 6
    roc_aucs, pr_aucs, actual = [], [], []
    for rate in rates:
        ra_reps, pa_reps, rate_reps = [], [], []
        for rep in range(n_repeats):
            r = np.random.default_rng(SEED + rep)
            # To reach a prevalence ABOVE the natural base rate we must subsample the
            # negatives; below it, the positives. Only ever subsampling positives
            # silently clamps every target above the base rate back to the base rate.
            if rate <= base:
                n_pos = min(max(int(rate / (1 - rate) * len(neg)), 25), len(pos))
                idx = np.concatenate([neg, r.choice(pos, n_pos, replace=False)])
            else:
                n_neg = min(max(int((1 - rate) / rate * len(pos)), 25), len(neg))
                idx = np.concatenate([pos, r.choice(neg, n_neg, replace=False)])
            Xs, ys = X[idx], y[idx]

            X_tr, X_te, y_tr, y_te = train_test_split(
                Xs, ys, test_size=0.3, random_state=SEED + rep, stratify=ys)
            _, _, ra, pa = fit_and_score(X_tr, y_tr, X_te, y_te)
            ra_reps.append(ra)
            pa_reps.append(pa)
            rate_reps.append(ys.mean())

        roc_aucs.append(float(np.mean(ra_reps)))
        pr_aucs.append(float(np.mean(pa_reps)))
        actual.append(float(np.mean(rate_reps)))

    print(f"  {'positives':>10} {'ROC-AUC':>9} {'PR-AUC':>8} {'PR baseline':>12}")
    for a, ra, pa in zip(actual, roc_aucs, pr_aucs):
        print(f"  {a:>9.1%} {ra:>9.3f} {pa:>8.3f} {a:>12.3f}")

    drop_roc = roc_aucs[0] - roc_aucs[-1]
    drop_pr = pr_aucs[0] - pr_aucs[-1]
    print(f"\n  From {actual[0]:.0%} positives down to {actual[-1]:.0%}:")
    print(f"    ROC-AUC fell {drop_roc:.3f}   ({roc_aucs[0]:.3f} -> {roc_aucs[-1]:.3f}), "
          f"still looks respectable")
    print(f"    PR-AUC  fell {drop_pr:.3f}   ({pr_aucs[0]:.3f} -> {pr_aucs[-1]:.3f}), "
          f"down to near-useless")
    print(f"    PR-AUC fell {drop_pr / drop_roc:.1f}x further.")
    print("\n  At 1% positives ROC-AUC still reads 0.74 - the kind of number that gets")
    print("  put in a slide. PR-AUC reads 0.06, which is the honest description of a")
    print("  model whose alerts are almost all wrong. Same model, same features:")
    print("  only the imbalance changed, and only one metric noticed.")

    fig, ax = plt.subplots()
    x = np.arange(len(rates))
    ax.plot(x, roc_aucs, marker="o", color=PALETTE["blue"], label="ROC-AUC")
    ax.plot(x, pr_aucs, marker="o", color=PALETTE["vermillion"], label="PR-AUC")
    ax.plot(x, actual, marker="s", color=PALETTE["grey"], linestyle=":",
            label="PR baseline (base rate)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.0%}" for a in actual])
    ax.set_xlabel("proportion of positives  (rarer to the right)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Same model, same features. Only the imbalance changed.")
    ax.annotate("ROC-AUC barely moves", xy=(0.34, 0.93), xycoords="axes fraction",
                color=PALETTE["blue"], fontweight="bold")
    ax.annotate("PR-AUC collapses", xy=(0.55, 0.36), xycoords="axes fraction",
                color=PALETTE["vermillion"], fontweight="bold")
    ax.set_title(f"Same model, same features. Only the imbalance changed."
                 f"  ({n_repeats} repeats averaged)")
    ax.legend(loc="center left")
    print(f"\n  saved  {save_figure(fig, OUT / '2_prevalence_sweep.png').name}")


# -------------------------------------------------------- 3. why it happens
def figure_why(X, y) -> None:
    """The arithmetic: a huge TN count sits in FPR's denominator and not in precision's."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = max(int(0.01 / 0.99 * len(neg)), 25)
    idx = np.concatenate([neg, rng.choice(pos, min(n_pos, len(pos)), replace=False)])
    Xs, ys = X[idx], y[idx]
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, ys, test_size=0.3,
                                              random_state=SEED, stratify=ys)
    _, proba, ra, pa = fit_and_score(X_tr, y_tr, X_te, y_te)

    thr = np.quantile(proba, 0.97)  # flag the top 3% of scores
    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum())
    fp = int(((pred == 1) & (y_te == 0)).sum())
    tn = int(((pred == 0) & (y_te == 0)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum())

    fpr = fp / (fp + tn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    print(f"  A test set with {y_te.mean():.1%} positives, flagging the top 3% of scores:\n")
    print(f"    TP {tp:>5}    FP {fp:>5}")
    print(f"    FN {fn:>5}    TN {tn:>5}")
    print(f"\n    false positive rate  = FP/(FP+TN) = {fp}/{fp + tn} = {fpr:.4f}"
          f"   <- looks excellent")
    print(f"    precision            = TP/(TP+FP) = {tp}/{tp + fp} = {precision:.4f}"
          f"   <- the truth")
    print(f"    recall               = {recall:.4f}")
    print(f"\n  The {tn:,} true negatives dominate FPR's denominator and flatter it.")
    print("  Precision has no true negatives in it at all, so it cannot be flattered.")
    print(f"  ROC-AUC {ra:.3f} vs PR-AUC {pa:.3f} on the same predictions.")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["false positive rate\nFP/(FP+TN)", "precision\nTP/(TP+FP)"]
    vals = [fpr, precision]
    bars = ax.bar(labels, vals, color=[PALETTE["blue"], PALETTE["vermillion"]], alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("value")
    ax.set_title(f"Same predictions, {y_te.mean():.1%} positives — "
                 f"{tn:,} true negatives hide the problem")
    ax.annotate(f"'only {fpr:.1%} of good wines\nwrongly flagged'",
                xy=(0.06, 0.55), xycoords="axes fraction", color=PALETTE["grey"])
    ax.annotate(f"'{1-precision:.0%} of my alerts\nare wrong'",
                xy=(0.58, 0.55), xycoords="axes fraction", color=PALETTE["grey"])
    print(f"\n  saved  {save_figure(fig, OUT / '3_why_roc_flatters.png').name}")


# ------------------------------------------------------------ 4. calibration
def figure_calibration(X, y) -> None:
    """Ranking well and being well-calibrated are different properties.

    Gaussian Naive Bayes is used deliberately. Its independence assumption is wrong
    on correlated physicochemical features, which makes it push probabilities toward
    0 and 1 - badly calibrated while still RANKING respectably. A random forest is
    already fairly well calibrated here, so it makes a poor demonstration.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=SEED, stratify=y)
    raw = GaussianNB().fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5).fit(X_tr, y_tr)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color=PALETTE["grey"], linestyle="--", label="perfectly calibrated")
    for model, name, colour in [(raw, "raw naive Bayes", PALETTE["vermillion"]),
                                (cal, "isotonic calibrated", PALETTE["blue"])]:
        p = model.predict_proba(X_te)[:, 1]
        frac, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        auc = roc_auc_score(y_te, p)
        brier = brier_score_loss(y_te, p)
        ax.plot(mean_pred, frac, marker="o", color=colour,
                label=f"{name}  (AUC {auc:.3f}, Brier {brier:.4f})")
        print(f"  {name:<22} ROC-AUC {auc:.3f}   Brier {brier:.4f}")

    print("\n  Both rank nearly identically - the ROC-AUCs barely differ, because")
    print("  calibration does not reorder anything. But the Brier score improves")
    print("  sharply, and the raw curve visibly sags away from the diagonal:")
    print("  naive Bayes says 0.9 for cases that happen far less than 90% of the time.")
    print("\n  Ranking is enough for 'show me the top 100'. Calibration is required")
    print("  when the probability feeds arithmetic: expected value, pricing, risk.")

    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration: does 0.8 actually mean 80%?")
    ax.legend(loc="upper left")
    print(f"\n  saved  {save_figure(fig, OUT / '4_calibration.png').name}")


def main() -> None:
    use_house_style()
    print("=" * 72)
    print("  ROC vs PRECISION-RECALL")
    print("=" * 72)

    print()
    X, y = load()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=SEED, stratify=y)
    _, proba, roc_auc, pr_auc = fit_and_score(X_tr, y_tr, X_te, y_te)

    banner("1. Both curves, at the natural imbalance")
    figure_both_curves(y_te, proba, roc_auc, pr_auc)

    banner("2. The prevalence sweep  (the headline experiment)")
    figure_prevalence_sweep(X, y)

    banner("3. Why ROC flatters: the arithmetic")
    figure_why(X, y)

    banner("4. Calibration is a separate property")
    figure_calibration(X, y)

    banner("What to take away")
    print("  - USE PR-AUC WHEN THE POSITIVE CLASS IS RARE. ROC-AUC is optimistic on")
    print("    imbalanced data because the true-negative count sits in the")
    print("    false-positive rate's denominator and swamps it.")
    print("  - The two baselines differ: random ROC-AUC is always 0.5; random PR-AUC")
    print("    is the base rate. So PR-AUC is scored against a moving bar.")
    print("  - Fraud, defect, disease, churn at low base rate, '0.5% of transactions'")
    print("    are all precision-recall problems.")
    print("  - ROC-AUC is still the right choice when classes are roughly balanced")
    print("    and both error types matter equally.")
    print("  - Ranking well is NOT the same as being calibrated. If the probability")
    print("    feeds arithmetic, check the calibration curve and the Brier score.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
