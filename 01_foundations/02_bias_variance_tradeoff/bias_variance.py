"""The bias-variance trade-off, made visible.

Predicts a car's fuel efficiency from its engine power using polynomial regression,
sweeping the polynomial degree from 1 to 15. Degree is a clean, single dial for model
complexity, which makes the trade-off observable rather than theoretical.

Four things are demonstrated, in order:

  1. What under- and over-fitting look like as a fitted curve
  2. The U-curve: training error falls forever, validation error turns upward
  3. The decomposition itself - bias^2 and variance measured empirically, by
     refitting on many resamples, and shown summing to the total error
  4. Learning curves - the diagnostic that says whether more DATA would help,
     or whether you need a different MODEL

Run:
    python 01_foundations/02_bias_variance_tradeoff/bias_variance.py

Concept reference: Foundations volume, Part A.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 00_setup holds the shared dataset loaders and figure style
sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_auto_mpg  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402
from sklearn.model_selection import KFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402

SEED = 42
DEGREES = range(1, 16)
OUT = Path(__file__).parent / "images"

rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def polynomial_model(degree: int):
    """Polynomial regression. Scaling matters: degree 15 on raw horsepower
    produces values around 10^30 and the fit collapses numerically."""
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# --------------------------------------------------------------------- 0. data
def load() -> tuple[np.ndarray, np.ndarray]:
    df = load_auto_mpg()
    before = len(df)
    df = df.dropna(subset=["horsepower", "mpg"])

    print(f"  Auto MPG                    {before} cars")
    print(f"  Dropped missing horsepower  {before - len(df)} rows  (encoded as '?')")
    print(f"  Modelling                   {len(df)} cars")
    print("  Predicting                  mpg  from  horsepower")
    return df[["horsepower"]].to_numpy(float), df["mpg"].to_numpy(float)


# ------------------------------------------------- 1. what the fits look like
def figure_fits(X, y, X_tr, y_tr) -> None:
    grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    show = [1, 2, 5, 15]
    labels = {1: "degree 1 - underfits", 2: "degree 2 - about right",
              5: "degree 5 - still sensible", 15: "degree 15 - overfits"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, d in zip(axes.ravel(), show):
        model = polynomial_model(d).fit(X_tr, y_tr)
        ax.scatter(X, y, s=12, color=PALETTE["grey"], alpha=0.45, label="all cars")
        ax.plot(grid, model.predict(grid), color=PALETTE["vermillion"], label=f"degree {d}")
        ax.set_title(labels[d])
        ax.set_ylim(5, 50)
    for ax in axes[1]:
        ax.set_xlabel("horsepower")
    for ax in axes[:, 0]:
        ax.set_ylabel("mpg")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Same data, four model complexities", fontsize=13, fontweight="bold")
    print(f"  saved  {save_figure(fig, OUT / '1_fits.png').name}")


# ----------------------------------------------------------- 2. the U-curve
def figure_u_curve(X, y) -> tuple[int, int, int]:
    """10-fold cross-validation, not a single split.

    One 70/30 split of 392 rows is noisy enough that the U-curve comes out lumpy and
    the 'best' degree moves if you change the seed. Averaging over 10 folds gives a
    curve you can actually read - and is the honest way to choose a hyperparameter.
    """
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    train_rmse, valid_rmse, valid_sd = [], [], []

    for d in DEGREES:
        scores = cross_validate(
            polynomial_model(d), X, y, cv=cv,
            scoring="neg_root_mean_squared_error", return_train_score=True,
        )
        train_rmse.append(float(-scores["train_score"].mean()))
        valid_rmse.append(float(-scores["test_score"].mean()))
        valid_sd.append(float(scores["test_score"].std()))

    best = int(np.argmin(valid_rmse)) + 1

    # The ONE-STANDARD-ERROR RULE: among models statistically indistinguishable from
    # the best, prefer the simplest. Picking the raw argmin overfits the validation
    # set itself - a difference of 0.01 RMSE against a standard error of 0.18 is not
    # a real difference.
    v = np.array(valid_rmse)
    se = np.array(valid_sd) / np.sqrt(10)
    threshold = v[best - 1] + se[best - 1]
    parsimonious = int(np.argmax(v <= threshold)) + 1
    widest = int(np.flatnonzero(v <= threshold)[-1]) + 1

    print(f"  {'degree':>6}  {'train RMSE':>11}  {'valid RMSE':>11}  {'+/- sem':>8}  {'gap':>7}")
    for d, tr, va, s in zip(DEGREES, train_rmse, valid_rmse, se):
        mark = ""
        if d == best:
            mark = "  <- lowest"
        if d == parsimonious:
            mark += "  <- simplest within 1 se"
        print(f"  {d:>6}  {tr:>11.3f}  {va:>11.3f}  {s:>8.3f}  {va - tr:>7.3f}{mark}")

    print(f"\n  Lowest validation error : degree {best}  ({v[best-1]:.3f})")
    print(f"  Within one standard error: degree {parsimonious} and above "
          f"(threshold {threshold:.3f})")
    print(f"  -> Prefer degree {parsimonious}. The gap between it and degree {best} is "
          f"{v[parsimonious-1] - v[best-1]:.3f} RMSE,")
    print(f"     against a standard error of {se[best-1]:.3f}. That is not a real difference.")

    # Standard ERROR of the mean, not the standard deviation across folds. The sd
    # (~0.5) reflects how much folds differ from each other and dwarfs the gaps
    # between degrees; the sem (sd/sqrt(k)) is the right band for "how precisely do
    # we know this mean?", which is the question the curve is actually answering.
    valid, sem = v, se

    fig, ax = plt.subplots()
    ax.fill_between(list(DEGREES), valid - sem, valid + sem,
                    color=PALETTE["vermillion"], alpha=0.18,
                    label="validation $\\pm$ 1 standard error")
    ax.plot(list(DEGREES), train_rmse, marker="o", color=PALETTE["blue"], label="training error")
    ax.plot(list(DEGREES), valid, marker="o", color=PALETTE["vermillion"], label="validation error")
    ax.axvline(best, color=PALETTE["green"], linestyle="--", linewidth=1.5)
    ax.axvline(parsimonious, color=PALETTE["sky"], linestyle=":", linewidth=1.8)

    # axes-fraction coordinates, so annotations cannot escape the plot area
    ax.annotate(f"lowest: degree {best}", xy=(0.5, 0.06), xycoords="axes fraction",
                ha="center", color=PALETTE["green"], fontweight="bold")
    ax.annotate(f"simplest within 1 se: degree {parsimonious}", xy=(0.5, 0.13),
                xycoords="axes fraction", ha="center", color=PALETTE["sky"],
                fontweight="bold")
    ax.annotate("underfitting\n(high bias)", xy=(0.03, 0.62), xycoords="axes fraction",
                color=PALETTE["grey"], fontsize=9.5)
    ax.annotate("overfitting\n(high variance)", xy=(0.74, 0.62), xycoords="axes fraction",
                color=PALETTE["grey"], fontsize=9.5)

    ax.set_xlabel("polynomial degree  (model complexity)")
    ax.set_ylabel("RMSE  (mpg)   -   10-fold cross-validated")
    ax.set_title("Training error falls forever. Validation error turns upward.")
    ax.legend(loc="upper center")
    print(f"\n  saved  {save_figure(fig, OUT / '2_u_curve.png').name}")
    return best, parsimonious, widest


# ------------------------------- 3. bias and variance, measured not asserted
def figure_decomposition(X_tr, X_te, y_tr, y_te, n_resamples: int = 300) -> None:
    """Refit on many bootstrap resamples and measure the two error sources directly.

    For each validation point, across resamples:
        variance      = spread of the predictions
        bias^2+noise  = (mean prediction - observed y)^2

    An honest caveat, and a teaching point in itself: with a single observed y at each
    x we CANNOT separate bias from irreducible noise. The true decomposition is
        error = bias^2 + variance + noise
    but the measurable quantity lumps the first and last together. So the blue line
    below is bias^2 PLUS noise, and it is why it flattens rather than reaching zero -
    a car's mpg is not a deterministic function of its horsepower.
    """
    n = len(X_tr)
    bias2_plus_noise, variance, total = [], [], []

    for d in DEGREES:
        preds = np.empty((n_resamples, len(X_te)))
        for i in range(n_resamples):
            idx = rng.integers(0, n, n)  # bootstrap: sample with replacement
            preds[i] = polynomial_model(d).fit(X_tr[idx], y_tr[idx]).predict(X_te)

        mean_pred = preds.mean(axis=0)
        bias2_plus_noise.append(float(np.mean((mean_pred - y_te) ** 2)))
        variance.append(float(np.mean(preds.var(axis=0))))
        total.append(float(np.mean((preds - y_te) ** 2)))

    print(f"  {'degree':>6}  {'bias^2+noise':>13}  {'variance':>10}  {'total':>10}")
    for d, b, v, t in zip(DEGREES, bias2_plus_noise, variance, total):
        print(f"  {d:>6}  {b:>13.2f}  {v:>10.2f}  {t:>10.2f}")
    print("\n  Note: total = (bias^2 + noise) + variance exactly, because with one")
    print("  observation per x the noise cannot be split out of the bias term.")

    fig, ax = plt.subplots()
    ax.plot(list(DEGREES), bias2_plus_noise, marker="o", color=PALETTE["blue"],
            label="bias$^2$ + irreducible noise  (falls, then flattens)")
    ax.plot(list(DEGREES), variance, marker="o", color=PALETTE["vermillion"],
            label="variance  (rises, then explodes)")
    ax.plot(list(DEGREES), total, marker="s", color=PALETTE["black"], linestyle="--",
            linewidth=1.6, label="total error  (their sum)")
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("squared error  (log scale)")
    ax.set_title(f"Where the error comes from - measured over {n_resamples} bootstrap refits")
    ax.legend(loc="upper left")
    print(f"\n  saved  {save_figure(fig, OUT / '3_decomposition.png').name}")


# --------------------------- 4. would more data help? the learning-curve test
def figure_learning_curves(X_tr, X_te, y_tr, y_te) -> None:
    """Error against training-set size, for a high-bias and a high-variance model.

    Y limits are fixed deliberately. A degree-15 polynomial fitted to ~40 points is
    close to singular and can score an RMSE in the thousands; left on a shared free
    axis it flattens everything else to a line at zero. Those points are allowed to
    run off the top of the chart and are reported in the caption instead - the fact
    that they exist at all is the variance lesson.
    """
    Y_TOP = 9.0
    sizes = np.linspace(0.18, 1.0, 12)
    cases = [(1, "High bias: degree 1"), (15, "High variance: degree 15")]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    for ax, (degree, title) in zip(axes, cases):
        tr_curve, te_curve, counts = [], [], []
        for frac in sizes:
            # a degree-d polynomial needs comfortably more than d+1 points to be stable
            k = max(int(frac * len(X_tr)), 2 * (degree + 1))
            reps = []
            for _ in range(12):
                idx = rng.choice(len(X_tr), k, replace=False)
                m = polynomial_model(degree).fit(X_tr[idx], y_tr[idx])
                reps.append((rmse(y_tr[idx], m.predict(X_tr[idx])), rmse(y_te, m.predict(X_te))))
            tr_curve.append(float(np.mean([r[0] for r in reps])))
            te_curve.append(float(np.mean([r[1] for r in reps])))
            counts.append(k)

        ax.plot(counts, tr_curve, marker="o", color=PALETTE["blue"], label="training")
        ax.plot(counts, te_curve, marker="o", color=PALETTE["vermillion"], label="validation")
        ax.set_ylim(3.5, Y_TOP)
        ax.set_title(title)
        ax.set_xlabel("training examples")

        gap = te_curve[-1] - tr_curve[-1]
        verdict = "more data will NOT help" if degree == 1 else "more data WILL help"
        offscale = sum(v > Y_TOP for v in te_curve)
        note = f"final gap {gap:.2f}\n{verdict}"
        if offscale:
            note += f"\n({offscale} early point(s) off-scale,\n up to RMSE {max(te_curve):,.0f})"
        ax.annotate(note, xy=(0.40, 0.66), xycoords="axes fraction",
                    color=PALETTE["grey"], fontsize=9.5)
        print(f"  degree {degree:>2}: final train {tr_curve[-1]:.2f}  "
              f"valid {te_curve[-1]:.2f}  gap {gap:.2f}  ->  {verdict}")
        if offscale:
            print(f"            {offscale} small-sample point(s) off-scale, "
                  f"worst RMSE {max(te_curve):,.0f}  <- variance, vividly")

    axes[0].set_ylabel("RMSE (mpg)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Learning curves: the diagnostic for 'should we collect more data?'",
                 fontsize=12.5, fontweight="bold")
    print(f"\n  saved  {save_figure(fig, OUT / '4_learning_curves.png').name}")


def main() -> None:
    use_house_style()
    print("=" * 72)
    print("  THE BIAS-VARIANCE TRADE-OFF")
    print("=" * 72)

    banner("0. Data")
    X, y = load()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)
    print(f"  Split                       {len(X_tr)} train / {len(X_te)} validation")

    banner("1. What under- and over-fitting look like")
    figure_fits(X, y, X_tr, y_tr)

    banner("2. The U-curve  (10-fold cross-validated)")
    best, lo, hi = figure_u_curve(X, y)

    banner("3. Bias and variance, measured directly")
    figure_decomposition(X_tr, X_te, y_tr, y_te)

    banner("4. Would more data help?")
    figure_learning_curves(X_tr, X_te, y_tr, y_te)

    banner("What to take away")
    print("  - Training error falls monotonically with complexity. On its own it is")
    print("    evidence of nothing.")
    print(f"  - Cross-validated error is U-shaped, with its minimum at degree {best}.")
    print(f"  - But look at the error band: degrees {lo} to {hi} are within one")
    print(f"    standard error of the best. Reporting 'degree {best} is best' is")
    print("    over-precise - that is fitting the validation set. The one-standard-")
    print("    error rule says take the SIMPLEST model within 1 se of the minimum.")
    print("  - Bias falls quickly and then flattens - it cannot reach zero, because")
    print("    mpg is not a deterministic function of horsepower. Variance starts")
    print("    negligible and ends up dominating by three orders of magnitude.")
    print("  - Variance is not just large at high degree, it is ERRATIC: one unlucky")
    print("    resample throws a wild curve. That instability IS the phenomenon.")
    print("  - MORE DATA REDUCES VARIANCE, NOT BIAS. The degree-1 learning curves")
    print("    converge high and flat: no amount of extra data fixes a model that")
    print("    cannot represent the pattern. The degree-15 curves are still")
    print("    separating, so more data would genuinely help there.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
