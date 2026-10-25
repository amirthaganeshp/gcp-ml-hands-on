"""Regression: predicting a number, and the regularisation that keeps it honest.

  1. A baseline, and why you always need one
  2. L1 vs L2 - the single most-tested regularisation fact. Lasso drives
     coefficients to EXACTLY zero (feature selection); Ridge shrinks them
     smoothly toward zero and never quite gets there. Shown as coefficient
     paths, which makes the difference impossible to misremember.
  3. Metric choice - MAE, RMSE and R2 disagree, and outliers are why
  4. Linear vs trees on tabular data, measured
  5. Residual diagnostics - what a bad fit looks like before you read a number

Run:
    python 02_traditional_ml/01_regression/regression.py

Concept reference: Foundations volume, Parts A.8 (metrics) and A.10 (features).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_auto_mpg  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.datasets import fetch_california_housing  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
CV = KFold(5, shuffle=True, random_state=SEED)
rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def load_housing():
    data = fetch_california_housing()
    X, y = data.data, data.target  # target is median house value, $100k units
    print(f"  California Housing   {X.shape[0]:,} districts x {X.shape[1]} features")
    print(f"  Target               median house value, $100k units "
          f"({y.min():.2f} to {y.max():.2f})")
    print(f"  Features             {', '.join(data.feature_names)}")
    return X, y, list(data.feature_names)


# ----------------------------------------------------------- 1. the baseline
def show_baseline(X, y) -> float:
    dummy = -cross_val_score(DummyRegressor(strategy="mean"), X, y, cv=CV,
                             scoring="neg_root_mean_squared_error").mean()
    linear = -cross_val_score(make_pipeline(StandardScaler(), LinearRegression()),
                              X, y, cv=CV,
                              scoring="neg_root_mean_squared_error").mean()
    print(f"  Predict the mean for everything   RMSE {dummy:.4f}   <- the baseline")
    print(f"  Linear regression                 RMSE {linear:.4f}")
    print(f"  The model beats the baseline by {(1 - linear / dummy):.1%}.")
    print("\n  Always compute this. A model that cannot beat the mean is not a model,")
    print("  and an R2 near zero is exactly that - R2 is defined against this baseline.")
    return dummy


# -------------------------------------------------------- 2. L1 vs L2, seen
def figure_regularisation(X, y, names) -> None:
    Xs = StandardScaler().fit_transform(X)  # coefficients only comparable when scaled

    # The two penalties are parameterised differently, so a single alpha grid cannot
    # serve both. Ridge's penalty is negligible against 20,640 samples until alpha is
    # very large; Lasso has already zeroed everything by alpha = 1. Using one shared
    # range makes the Ridge panel look deceptively flat.
    ridge_alphas = np.logspace(-2, 6, 40)
    lasso_alphas = np.logspace(-3, 1, 40)

    ridge_paths = np.array([Ridge(alpha=a).fit(Xs, y).coef_ for a in ridge_alphas])
    lasso_paths = np.array([Lasso(alpha=a, max_iter=5000).fit(Xs, y).coef_
                            for a in lasso_alphas])

    zeros_ridge = (np.abs(ridge_paths) < 1e-8).sum(axis=1)
    zeros_lasso = (np.abs(lasso_paths) < 1e-8).sum(axis=1)
    shrink_ridge = np.abs(ridge_paths).sum(axis=1)
    shrink_lasso = np.abs(lasso_paths).sum(axis=1)

    print("  Ridge, as alpha grows:")
    print(f"    {'alpha':>10}  {'sum |coef|':>11}  {'coefs == 0':>11}")
    for i in (0, 15, 25, 32, 39):
        print(f"    {ridge_alphas[i]:>10.4g}  {shrink_ridge[i]:>11.4f}  {zeros_ridge[i]:>11}")
    print("\n  Lasso, as alpha grows:")
    print(f"    {'alpha':>10}  {'sum |coef|':>11}  {'coefs == 0':>11}")
    for i in (0, 15, 22, 28, 39):
        print(f"    {lasso_alphas[i]:>10.4g}  {shrink_lasso[i]:>11.4f}  {zeros_lasso[i]:>11}")

    print(f"\n  Ridge shrank total coefficient mass from {shrink_ridge[0]:.2f} to "
          f"{shrink_ridge[-1]:.4f} -")
    print(f"  a {1 - shrink_ridge[-1] / shrink_ridge[0]:.1%} reduction - while zeroing "
          f"exactly {zeros_ridge.max()} of them.")
    print(f"  Lasso zeroed {zeros_lasso.max()} of {len(names)}.")
    print("\n  That is the whole distinction, and it is a named exam fact:")
    print("    L1 (Lasso)  ->  coefficients hit EXACTLY zero  ->  feature selection")
    print("    L2 (Ridge)  ->  coefficients shrink toward zero, never reaching it")
    print("  ElasticNet mixes the two when you want selection AND stability with")
    print("  correlated features.")

    # 8 features need 8 distinguishable colours; the house 6-colour cycle would repeat
    colours = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(names)))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    for ax, alphas_, paths, title in [
        (axes[0], ridge_alphas, ridge_paths, "Ridge (L2): shrinks, never reaches zero"),
        (axes[1], lasso_alphas, lasso_paths, "Lasso (L1): snaps to exactly zero"),
    ]:
        for j, name in enumerate(names):
            ax.plot(alphas_, paths[:, j], label=name, color=colours[j])
        ax.axhline(0, color=PALETTE["black"], linewidth=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("alpha  (regularisation strength)")
        ax.set_title(title)
    axes[0].set_ylabel("coefficient value")
    axes[0].legend(fontsize=8, ncol=2, loc="lower left")
    fig.suptitle("The same data, two penalties  (note the different alpha ranges)",
                 fontsize=12.5, fontweight="bold")
    print(f"\n  saved  {save_figure(fig, OUT / '1_l1_vs_l2.png').name}")


# ------------------------------------------------------- 3. metrics disagree
def figure_metrics() -> None:
    """Auto MPG is small enough that a handful of outliers genuinely move things."""
    df = load_auto_mpg().dropna(subset=["horsepower", "mpg"])
    feats = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
    X = df[feats].to_numpy(float)
    y = df["mpg"].to_numpy(float)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    pred = model.predict(X_te)

    clean = {"MAE": mean_absolute_error(y_te, pred), "RMSE": rmse(y_te, pred),
             "R2": r2_score(y_te, pred)}

    # corrupt five test targets - a data-entry error, an mpg recorded in the wrong unit
    y_bad = y_te.copy()
    hit = rng.choice(len(y_bad), 5, replace=False)
    y_bad[hit] = y_bad[hit] * 3
    dirty = {"MAE": mean_absolute_error(y_bad, pred), "RMSE": rmse(y_bad, pred),
             "R2": r2_score(y_bad, pred)}

    print(f"  {len(y_te)} test cars. Then 5 of them (4%) get a corrupted target.\n")
    print(f"  {'metric':<8} {'clean':>9} {'with 5 outliers':>17} {'change':>9}")
    for k in clean:
        change = (dirty[k] - clean[k]) / abs(clean[k])
        print(f"  {k:<8} {clean[k]:>9.3f} {dirty[k]:>17.3f} {change:>+8.0%}")

    print("\n  MAE moves least - it is linear in the error, so one huge miss counts")
    print("  once. RMSE squares first, so a single outlier can dominate the whole")
    print("  score. Choose MAE when a few extreme values should not set your")
    print("  direction; choose RMSE when large errors really are disproportionately")
    print("  costly. They are not interchangeable and they will rank models")
    print("  differently.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    labels = list(clean)
    x = np.arange(len(labels))
    ax.bar(x - 0.19, [clean[k] for k in labels], 0.38, label="clean",
           color=PALETTE["blue"], alpha=0.9)
    ax.bar(x + 0.19, [dirty[k] for k in labels], 0.38, label="5 outliers",
           color=PALETTE["vermillion"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("value")
    ax.set_title("Outlier sensitivity")
    ax.legend()

    ax = axes[1]
    ax.scatter(pred, y_te, s=16, color=PALETTE["blue"], alpha=0.6, label="clean targets")
    ax.scatter(pred[hit], y_bad[hit], s=48, color=PALETTE["vermillion"],
               marker="X", label="corrupted")
    lims = [min(y_te.min(), pred.min()) - 2, max(y_bad.max(), pred.max()) + 2]
    ax.plot(lims, lims, color=PALETTE["black"], linestyle="--", linewidth=1)
    ax.set_xlabel("predicted mpg"); ax.set_ylabel("actual mpg")
    ax.set_title("Five points doing all the damage")
    ax.legend()
    print(f"\n  saved  {save_figure(fig, OUT / '2_metric_choice.png').name}")


# ----------------------------------------------------- 4. linear vs trees
def figure_models(X, y) -> None:
    models = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=5000)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5,
                                                                max_iter=5000)),
        "Random Forest": RandomForestRegressor(n_estimators=120, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                random_state=SEED, n_jobs=-1, verbosity=0),
    }

    print(f"  {'model':<18} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    scores = {}
    for name, model in models.items():
        r = -cross_val_score(model, X, y, cv=CV,
                             scoring="neg_root_mean_squared_error").mean()
        m = -cross_val_score(model, X, y, cv=CV,
                             scoring="neg_mean_absolute_error").mean()
        r2 = cross_val_score(model, X, y, cv=CV, scoring="r2").mean()
        scores[name] = (r, m, r2)
        print(f"  {name:<18} {r:>8.4f} {m:>8.4f} {r2:>8.4f}")

    best = min(scores, key=lambda k: scores[k][0])
    lin = scores["Linear"][0]
    print(f"\n  Best: {best}  (RMSE {scores[best][0]:.4f}), "
          f"{(1 - scores[best][0] / lin):.0%} better than linear regression.")
    print("\n  Gradient-boosted trees usually win on tabular data. They capture")
    print("  interactions and non-linearity without you specifying any, they need")
    print("  no scaling, and they handle mixed types. That is why BigQuery ML's")
    print("  BOOSTED_TREE_* is the tabular workhorse, and why offering a deep neural")
    print("  network for a mid-sized tabular problem should make you suspicious.")
    print("\n  The regularised linear models barely differ from plain linear here -")
    print("  8 features and 20,000 rows is not a regime where overfitting bites.")
    print("  Regularisation earns its keep with many features relative to rows.")

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(scores)
    vals = [scores[n][0] for n in names]
    colours = [PALETTE["grey"] if n == "Baseline (mean)" else
               (PALETTE["green"] if n == best else PALETTE["blue"]) for n in names]
    bars = ax.bar(names, vals, color=colours, alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("cross-validated RMSE  ($100k)   lower is better")
    ax.set_title("Linear models, regularised linear models, and trees")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    print(f"\n  saved  {save_figure(fig, OUT / '3_model_comparison.png').name}")


# ------------------------------------------------------ 5. residuals talk
def figure_residuals(X, y) -> None:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    trees = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                         random_state=SEED, n_jobs=-1, verbosity=0).fit(X_tr, y_tr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model, name in [(axes[0], linear, "Linear regression"),
                            (axes[1], trees, "XGBoost")]:
        pred = model.predict(X_te)
        resid = y_te - pred
        ax.scatter(pred, resid, s=8, alpha=0.25, color=PALETTE["blue"])
        ax.axhline(0, color=PALETTE["vermillion"], linewidth=1.4)
        ax.set_xlabel("predicted value")
        ax.set_title(f"{name}  (RMSE {rmse(y_te, pred):.3f})")
        print(f"  {name:<20} RMSE {rmse(y_te, pred):.4f}   "
              f"residual mean {resid.mean():+.4f}   sd {resid.std():.4f}")
        if model is linear:
            capped = y_te >= 5.0
            c_res, c_pred = resid[capped], pred[capped]
    axes[0].set_ylabel("residual  (actual - predicted)")
    fig.suptitle("Residuals should be a formless cloud around zero",
                 fontsize=12.5, fontweight="bold")

    print("\n  Read the LEFT panel: the residuals form a diagonal edge and fan out.")
    print("  That edge is the target being capped at 5.0 ($500k) in this dataset.")
    print("  For a capped district the actual value is pinned at 5.0, so its")
    print("  residual is 5.0 - prediction: a straight line of slope -1.")
    print(f"\n    capped districts in test        {capped.sum():,}")
    print(f"    their mean residual             {c_res.mean():+.2f}")
    print(f"    positive (model predicted < 5)  {(c_res > 0).mean():.0%}")
    print(f"    predicted above the cap         {(c_pred > 5.0).sum():,}")
    print("\n  Most of the edge sits ABOVE zero: the linear model under-predicts")
    print("  the most expensive districts, whose true value is unknown beyond the")
    print("  cap. Only where it predicts above 5.0 does the line turn negative.")
    print("  A metric alone would never have told you that. Plot the residuals")
    print("  before you trust the number.")
    print(f"\n  saved  {save_figure(fig, OUT / '4_residuals.png').name}")


def main() -> None:
    use_house_style()
    print("=" * 72)
    print("  REGRESSION")
    print("=" * 72)
    print()
    X, y, names = load_housing()

    banner("1. The baseline")
    show_baseline(X, y)

    banner("2. L1 vs L2 regularisation")
    figure_regularisation(X, y, names)

    banner("3. Metric choice, and outliers")
    figure_metrics()

    banner("4. Linear models vs trees")
    figure_models(X, y)

    banner("5. Residual diagnostics")
    figure_residuals(X, y)

    banner("What to take away")
    print("  - Always beat a baseline. R2 is literally defined against 'predict the")
    print("    mean', so R2 near zero means your model is worth nothing.")
    print("  - L1 (Lasso) gives EXACT zeros and therefore feature selection.")
    print("    L2 (Ridge) shrinks smoothly and never zeroes. ElasticNet mixes them.")
    print("  - MAE is robust to outliers; RMSE is not. They rank models differently,")
    print("    so pick the one matching what actually costs money.")
    print("  - Gradient-boosted trees usually win on tabular data.")
    print("  - Plot the residuals. A capped target, heteroscedasticity or a missed")
    print("    non-linearity are all visible there and invisible in a single number.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
