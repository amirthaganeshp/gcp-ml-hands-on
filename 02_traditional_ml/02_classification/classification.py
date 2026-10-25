"""Classification: the algorithm zoo, and what each one actually does.

  1. Decision boundaries - six classifiers on the same two features. The shapes
     they can and cannot draw explain most of their behaviour elsewhere.
  2. A fair comparison on the full feature set, cross-validated
  3. Interpretability - logistic coefficients as odds ratios, tree feature
     importance, and the difference between a glass box and a post-hoc
     explanation
  4. Class imbalance - class_weight vs threshold moving vs doing nothing

Run:
    python 02_traditional_ml/02_classification/classification.py

Concept reference: Foundations volume, Parts A.6, A.12, A.14, A.15.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_rice  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.inspection import DecisionBoundaryDisplay  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def classifiers() -> dict:
    return {
        "Logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, random_state=SEED)),
        "KNN (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
        # SVC has no native predict_proba. SVC(probability=True) was deprecated in
        # scikit-learn 1.9; CalibratedClassifierCV is the supported replacement and
        # is what produces the probabilities ROC-AUC and PR-AUC need.
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False),
        ),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                 random_state=SEED, n_jobs=-1, verbosity=0,
                                 eval_metric="logloss"),
    }


# ---------------------------------------------------- 1. decision boundaries
def figure_boundaries() -> None:
    """Boundary SHAPE, on two deliberately different problems.

    Synthetic data is used here on purpose. Rice is very close to linearly
    separable along one dominant feature, so every model draws almost the same
    vertical line and the comparison shows nothing. `make_moons` is the canonical
    dataset for this question precisely because a straight line cannot solve it,
    so the differences between model families become visible.

    Accuracy shown is HELD OUT, not in-sample - a random forest scores 1.000 on
    its own training data, which would display overfitting as success.
    """
    from sklearn.datasets import make_circles, make_moons

    problems = {
        # Noise is kept low enough that the two crescents and the ring are still
        # VISIBLE. At noise=0.28 the shapes wash out into overlapping blobs, and
        # the whole point of the panel - that the boundary must match the shape
        # of the data - stops being legible.
        "moons": make_moons(n_samples=600, noise=0.18, random_state=SEED),
        "circles": make_circles(n_samples=600, noise=0.08, factor=0.45, random_state=SEED),
    }
    show = ["Logistic", "KNN (k=15)", "SVM (RBF)", "Naive Bayes",
            "Decision Tree", "Random Forest"]

    fig, axes = plt.subplots(2, 6, figsize=(19, 6.6))
    acc = {}                       # acc[problem][model] - narrated below, never hardcoded
    for row, (pname, (X, y)) in enumerate(problems.items()):
        acc[pname] = {}
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                  random_state=SEED, stratify=y)
        print(f"  {pname}:")
        for col, name in enumerate(show):
            ax = axes[row, col]
            model = classifiers()[name]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X_tr, y_tr)
            DecisionBoundaryDisplay.from_estimator(
                model, X, ax=ax, response_method="predict",
                cmap="coolwarm", alpha=0.25, grid_resolution=200)
            ax.scatter(X_te[y_te == 1, 0], X_te[y_te == 1, 1], s=9,
                       color=PALETTE["vermillion"], alpha=0.8)
            ax.scatter(X_te[y_te == 0, 0], X_te[y_te == 0, 1], s=9,
                       color=PALETTE["blue"], alpha=0.8)
            a = model.score(X_te, y_te)
            acc[pname][name] = a
            ax.set_title(f"{name}\nheld-out {a:.3f}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            print(f"    {name:<16} held-out accuracy {a:.3f}")
        axes[row, 0].set_ylabel(pname, fontsize=11, fontweight="bold")

    fig.suptitle("The shape of boundary each model family can draw",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    lr_c, lr_m = acc["circles"]["Logistic"], acc["moons"]["Logistic"]
    nb_c, nb_m = acc["circles"]["Naive Bayes"], acc["moons"]["Naive Bayes"]
    print("\n  LOGISTIC REGRESSION draws exactly one straight line, so it cannot")
    print(f"  solve either problem - and on circles it scores {lr_c:.3f}, WORSE THAN")
    print("  CHANCE. No line separates a ring from its centre. That is pure bias,")
    print("  visible as a shape.")
    print(f"\n  NAIVE BAYES is the interesting one: {nb_c:.3f} on circles but only")
    print(f"  {nb_m:.3f} on moons - and moons is the problem that LOOKS easier.")
    print("  Gaussian Naive Bayes fits one Gaussian per class, so its boundary is")
    print("  QUADRATIC, not linear: it can wrap a ring almost perfectly. Two")
    print("  crescents are not Gaussian blobs, so it cannot describe them at all,")
    print(f"  and it beats logistic there by only {nb_m - lr_m:+.3f}. The lesson is that")
    print("  a model's limitation is the SHAPE OF ITS ASSUMPTIONS, not a single")
    print("  'simple to powerful' ranking you can memorise.")
    print("\n  The DECISION TREE draws axis-parallel rectangles, so its edges are")
    print("  always steps. RANDOM FOREST averages hundreds of those into something")
    print("  nearly smooth - variance reduction, made visual.")
    print("  SVM with an RBF kernel curves freely and sits at or near the top of")
    print("  both problems. KNN matches it, but look at HOW: its boundary hugs the")
    print("  data so closely that it is visibly ragged, and that raggedness IS its")
    print("  variance. Two models, the same score, completely different risk.")
    print(f"\n  saved  {save_figure(fig, OUT / '1_decision_boundaries.png').name}")


# ------------------------------------------------------ 2. a fair comparison
def figure_comparison(X, y) -> dict:
    print(f"  {'model':<16} {'accuracy':>9} {'F1':>7} {'ROC-AUC':>8} {'PR-AUC':>7} {'fit s':>7}")
    scores = {}
    for name, model in classifiers().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            res = cross_validate(model, X, y, cv=CV,
                                 scoring=["accuracy", "f1", "roc_auc", "average_precision"])
        scores[name] = {
            "accuracy": res["test_accuracy"].mean(),
            "f1": res["test_f1"].mean(),
            "roc_auc": res["test_roc_auc"].mean(),
            "pr_auc": res["test_average_precision"].mean(),
            "fit_time": res["fit_time"].mean(),
        }
        s = scores[name]
        print(f"  {name:<16} {s['accuracy']:>9.4f} {s['f1']:>7.4f} "
              f"{s['roc_auc']:>8.4f} {s['pr_auc']:>7.4f} {s['fit_time']:>7.3f}")

    best = max(scores, key=lambda k: scores[k]["roc_auc"])
    spread = (max(s["roc_auc"] for s in scores.values())
              - min(s["roc_auc"] for s in scores.values()))
    print(f"\n  Best ROC-AUC: {best} ({scores[best]['roc_auc']:.4f})")
    print(f"  Spread across ALL seven models: {spread:.4f}")
    print("\n  That spread is the useful number. On a clean, well-separated problem")
    print("  every reasonable algorithm lands in the same place, and the choice is")
    print("  really about training cost, interpretability and maintenance - not")
    print("  accuracy. Algorithm shopping is the least valuable thing you can do")
    print("  with a dataset like this; features and framing matter far more.")

    fig, ax = plt.subplots(figsize=(10.5, 5))
    names = list(scores)
    x = np.arange(len(names))
    width = 0.38
    ax.bar(x - width / 2, [scores[n]["roc_auc"] for n in names], width,
           label="ROC-AUC", color=PALETTE["blue"], alpha=0.9)
    ax.bar(x + width / 2, [scores[n]["f1"] for n in names], width,
           label="F1", color=PALETTE["vermillion"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("cross-validated score")
    ax.set_title("Seven algorithms, one clean problem — almost no difference")
    ax.legend(loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    print(f"\n  saved  {save_figure(fig, OUT / '2_model_comparison.png').name}")
    return scores


# ------------------------------------------------------- 3. interpretability
def figure_interpretability(X, y, names) -> None:
    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=SEED)).fit(X, y)
    coefs = logit[-1].coef_[0]
    odds = np.exp(coefs)

    print("  Logistic regression - a GLASS BOX. Every coefficient is readable:\n")
    print(f"  {'feature':<22} {'coef':>8} {'odds ratio':>11}  reading")
    order = np.argsort(-np.abs(coefs))
    for j in order:
        direction = "increases" if coefs[j] > 0 else "decreases"
        print(f"  {names[j]:<22} {coefs[j]:>8.3f} {odds[j]:>11.3f}  "
              f"+1 sd {direction} odds x{max(odds[j], 1/odds[j]):.2f}")

    forest = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X, y)
    importance = forest.feature_importances_

    # max_leaf_nodes rather than max_depth: a depth limit lets the tree spend every
    # level re-splitting its single strongest feature, producing leaves that all
    # predict the same class and look broken. Limiting LEAVES makes it spend each
    # split where it buys most. show_weights prints [n_class0, n_class1] per leaf,
    # so leaves that share a majority class still visibly differ in purity.
    tree = DecisionTreeClassifier(max_leaf_nodes=5, random_state=SEED).fit(X, y)
    print(f"\n  A 5-leaf decision tree, readable in full "
          f"(accuracy {tree.score(X, y):.3f}).")
    print("  Each leaf shows [Osmancik, Cammeo] counts:\n")
    for line in export_text(tree, feature_names=list(names),
                            show_weights=True).splitlines():
        print("    " + line)
    used = {names[i] for i in tree.tree_.feature if i >= 0}
    print(f"\n  It used {len(used)} of {len(names)} features: {', '.join(sorted(used))}.")
    print("  A tree does its own feature selection - unused columns simply never")
    print("  appear. That is part of why trees need so little preprocessing.")

    print("\n  Two DIFFERENT kinds of interpretability, and the exam separates them:")
    print("    glass box  - the model IS the explanation (linear, shallow tree)")
    print("    post-hoc   - an approximation of a model you cannot read directly")
    print("                 (feature importance, SHAP, Integrated Gradients)")
    print("  If a regulation demands explainable decision logic, a post-hoc")
    print("  approximation of a black box does not satisfy it.")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    ax.barh([names[j] for j in order][::-1], [coefs[j] for j in order][::-1],
            color=[PALETTE["vermillion"] if coefs[j] < 0 else PALETTE["blue"]
                   for j in order][::-1], alpha=0.9)
    ax.axvline(0, color=PALETTE["black"], linewidth=1)
    ax.set_xlabel("standardised coefficient")
    ax.set_title("Logistic regression: signed and readable")

    ax = axes[1]
    o2 = np.argsort(importance)
    ax.barh([names[j] for j in o2], importance[o2], color=PALETTE["green"], alpha=0.9)
    ax.set_xlabel("mean decrease in impurity")
    ax.set_title("Random forest: magnitude only, no direction")
    fig.suptitle("Glass box vs post-hoc importance", fontsize=12.5, fontweight="bold")
    print(f"\n  saved  {save_figure(fig, OUT / '3_interpretability.png').name}")


# ------------------------------------------------------ 4. handling imbalance
def figure_imbalance(X, y) -> None:
    """Thin the positive class to 5%, then try three responses."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = int(0.05 / 0.95 * len(neg))
    idx = np.concatenate([neg, rng.choice(pos, n_pos, replace=False)])
    Xi, yi = X[idx], y[idx]
    X_tr, X_te, y_tr, y_te = train_test_split(Xi, yi, test_size=0.3,
                                              random_state=SEED, stratify=yi)
    print(f"  Imbalanced to {yi.mean():.1%} positives ({len(yi):,} grains)\n")

    results = {}

    base = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=SEED)).fit(X_tr, y_tr)
    p_base = base.predict_proba(X_te)[:, 1]
    results["do nothing\n(threshold 0.5)"] = (base.predict(X_te), p_base)

    weighted = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    ).fit(X_tr, y_tr)
    results["class_weight\n='balanced'"] = (weighted.predict(X_te),
                                            weighted.predict_proba(X_te)[:, 1])

    # move the threshold on the ORIGINAL model - no retraining at all
    thr = np.quantile(p_base, 1 - y_tr.mean())
    results[f"threshold moved\nto {thr:.3f}"] = ((p_base >= thr).astype(int), p_base)

    print(f"  {'approach':<28} {'precision':>10} {'recall':>8} {'F1':>7} {'PR-AUC':>8}")
    scores = {}
    for name, (pred, proba) in results.items():
        flat = name.replace("\n", " ")
        scores[flat] = s = {
            "p": precision_score(y_te, pred, zero_division=0),
            "r": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            "ap": average_precision_score(y_te, proba)}
        print(f"  {flat:<28} {s['p']:>10.3f} {s['r']:>8.3f} {s['f1']:>7.3f} "
              f"{s['ap']:>8.3f}")
    base_s, cw_s, thr_s = scores.values()
    spread = max(s["ap"] for s in scores.values()) - min(s["ap"] for s in scores.values())

    print("\n  Read the PR-AUC column first, because it is the surprise. All three")
    print(f"  are within {spread:.3f} of each other. PR-AUC measures RANKING quality")
    print("  across every threshold, and none of these techniques improved the")
    print("  ranking - they only moved the operating point along the same curve.")
    print("\n  'Do nothing' and 'threshold moved' are literally the same model, so")
    print("  identical PR-AUC is expected. The interesting result is that")
    print("  class_weight barely changed it either.")
    print("\n  So what DID change? Where you sit on the precision/recall trade:")
    for label, s in (("class_weight='balanced'", cw_s), ("threshold moved", thr_s)):
        print(f"    {label:<24} recall {base_s['r']:.3f} -> {s['r']:.3f}, "
              f"precision {base_s['p']:.3f} -> {s['p']:.3f}")
    by_f1 = sorted(scores, key=lambda k: scores[k]["f1"])
    print(f"\n  By F1:  best   {by_f1[-1]:<28} {scores[by_f1[-1]]['f1']:.3f}")
    print(f"          worst  {by_f1[0]:<28} {scores[by_f1[0]]['f1']:.3f}")
    if by_f1[0].startswith("class_weight"):
        print("  class_weight over-corrects, buying recall far too expensively.")
        print("  That is the honest result here, and it contradicts the usual")
        print("  advice to reach for class_weight first.")
    print("\n  The real lesson: imbalance techniques mostly RELOCATE you on a curve")
    print("  you already had. If you need a better curve, you need better features")
    print("  or a better model - not reweighting. And if you only need a different")
    print("  operating point, threshold moving is free and fully controllable,")
    print("  whereas class_weight moves you somewhere you did not choose.")

    fig, ax = plt.subplots(figsize=(10, 5))
    names_ = list(results)
    x = np.arange(len(names_))
    width = 0.26
    for i, (metric, fn, colour) in enumerate([
        ("precision", precision_score, PALETTE["blue"]),
        ("recall", recall_score, PALETTE["vermillion"]),
        ("F1", f1_score, PALETTE["green"]),
    ]):
        vals = [fn(y_te, results[n][0], zero_division=0) for n in names_]
        ax.bar(x + (i - 1) * width, vals, width, label=metric, color=colour, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(names_)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Three responses to {yi.mean():.0%} positives")
    ax.legend()
    print(f"\n  saved  {save_figure(fig, OUT / '4_imbalance.png').name}")


def main() -> None:
    use_house_style()
    print("=" * 72)
    print("  CLASSIFICATION")
    print("=" * 72)

    df = load_rice()
    names = [c for c in df.columns if c != "class"]
    X = df[names].to_numpy(float)
    y = (df["class"] == "Cammeo").astype(int).to_numpy()
    print(f"\n  Rice: {len(df):,} grains, {len(names)} features, "
          f"{y.mean():.1%} Cammeo (the positive class)")

    banner("1. What boundary can each model draw?")
    figure_boundaries()

    banner("2. A fair comparison on all features")
    figure_comparison(X, y)

    banner("3. Interpretability: glass box vs post-hoc")
    figure_interpretability(X, y, names)

    banner("4. Handling class imbalance")
    figure_imbalance(X, y)

    banner("What to take away")
    print("  - Model choice matters far less than people expect on a clean problem.")
    print("    Seven algorithms landed within a few thousandths of ROC-AUC here.")
    print("    Spend the time on features, framing and the metric instead.")
    print("  - The BOUNDARY SHAPE explains the behaviour: linear models draw one")
    print("    plane, trees draw rectangles, KNN and RBF-SVM draw anything.")
    print("  - Glass box (linear, shallow tree) is NOT the same as post-hoc")
    print("    explanation (importance, SHAP). Regulations usually want the first.")
    print("  - Tree importance gives MAGNITUDE but no DIRECTION. Logistic")
    print("    coefficients give both, and exponentiate into odds ratios.")
    print("  - Imbalance techniques mostly RELOCATE you on a precision/recall curve")
    print("    you already had - PR-AUC barely moved for any of them here. Threshold")
    print("    moving is free and controllable; class_weight over-corrected and")
    print("    scored the worst F1 of the three. A genuinely better curve needs")
    print("    better features, not reweighting.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
