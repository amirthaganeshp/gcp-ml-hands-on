"""How to split data, and why a single split lies to you.

Four demonstrations:

  1. The split lottery  - the same model, the same data, 300 different random
     splits. Test accuracy swings by more than 10 points. A single number from a
     single split is a sample of size one.
  2. Cross-validation   - averaging over folds shrinks that uncertainty, and the
     spread across folds tells you how much to trust the mean.
  3. Stratification     - why the class balance must be preserved in every fold,
     and what breaks when it is not.
  4. Temporal splitting - random splitting on a time series lets the model
     interpolate between points it has already seen. The error it reports is
     fiction.

Run:
    python 01_foundations/01_train_test_split_and_cv/splitting.py

Concept reference: Foundations volume, Part A.4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from loaders import load_flights, load_penguins  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.metrics import accuracy_score, mean_absolute_error  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)

SEED = 42
OUT = Path(__file__).parent / "images"
rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def penguin_xy() -> tuple[np.ndarray, np.ndarray]:
    df = load_penguins().dropna(subset=["bill_length_mm", "bill_depth_mm",
                                        "flipper_length_mm", "body_mass_g", "species"])
    X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].to_numpy()
    y = df["species"].to_numpy()
    return X, y


# ------------------------------------------------------- 1. the split lottery
def figure_split_lottery(X, y, n_trials: int = 300) -> tuple[float, float]:
    """Same model, same data, 300 different random splits."""
    scores = []
    for seed in range(n_trials):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
        model = RandomForestClassifier(n_estimators=60, random_state=SEED).fit(X_tr, y_tr)
        scores.append(accuracy_score(y_te, model.predict(X_te)))

    scores = np.array(scores)
    lo, hi = scores.min(), scores.max()
    print(f"  {n_trials} different random 70/30 splits of the SAME data")
    print(f"  worst split   {lo:.3f}")
    print(f"  best  split   {hi:.3f}")
    print(f"  spread        {hi - lo:.3f}   <- pure luck of the draw")
    print(f"  mean +/- sd   {scores.mean():.3f} +/- {scores.std():.3f}")

    fig, ax = plt.subplots()
    ax.hist(scores, bins=25, color=PALETTE["blue"], alpha=0.85, edgecolor="white")
    ax.axvline(scores.mean(), color=PALETTE["black"], linestyle="--",
               label=f"mean {scores.mean():.3f}")
    ax.axvline(lo, color=PALETTE["vermillion"], linestyle=":", label=f"worst {lo:.3f}")
    ax.axvline(hi, color=PALETTE["green"], linestyle=":", label=f"best {hi:.3f}")
    ax.set_xlabel("test accuracy")
    ax.set_ylabel(f"count  (out of {n_trials} splits)")
    ax.set_title("The split lottery: one number from one split is a sample of size one")
    ax.legend()
    print(f"\n  saved  {save_figure(fig, OUT / '1_split_lottery.png').name}")
    return float(lo), float(hi)


# ------------------------------------------------ 2. cross-validation instead
def figure_cv(X, y, lo: float, hi: float) -> None:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    model = RandomForestClassifier(n_estimators=60, random_state=SEED)
    folds = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    sem = folds.std() / np.sqrt(len(folds))

    print(f"  10-fold CV per-fold accuracy: {np.round(folds, 3)}")
    print(f"  mean          {folds.mean():.3f}")
    print(f"  std dev       {folds.std():.3f}   spread BETWEEN folds")
    print(f"  std error     {sem:.3f}   uncertainty in the MEAN")
    print(f"\n  Single-split range was {lo:.3f} to {hi:.3f}.")
    print(f"  CV gives one number with a stated uncertainty instead of a lottery ticket.")

    fig, ax = plt.subplots()
    ax.bar(range(1, len(folds) + 1), folds, color=PALETTE["blue"], alpha=0.85)
    ax.axhline(folds.mean(), color=PALETTE["black"], linestyle="--",
               label=f"CV mean {folds.mean():.3f}")
    ax.axhspan(lo, hi, color=PALETTE["vermillion"], alpha=0.12,
               label=f"single-split range ({lo:.2f}-{hi:.2f})")
    ax.set_ylim(min(folds.min(), lo) - 0.03, 1.005)
    ax.set_xlabel("fold")
    ax.set_ylabel("accuracy")
    ax.set_title("Ten folds, one answer, and an honest error bar")
    ax.legend(loc="lower right")
    print(f"\n  saved  {save_figure(fig, OUT / '2_cross_validation.png').name}")


# ------------------------------------------------------- 3. why stratify
def figure_stratification(X, y) -> None:
    """Make the problem imbalanced, then split with and without stratification."""
    keep = (y == "Adelie") | (rng.random(len(y)) < 0.12)  # thin the other species
    Xi, yi = X[keep], y[keep]
    binary = np.where(yi == "Adelie", "Adelie", "Other")
    overall = (binary == "Other").mean()
    print(f"  Imbalanced subset: {len(binary)} penguins, {overall:.1%} minority class")

    rows = []
    for name, splitter in [("plain KFold", KFold(5, shuffle=True, random_state=SEED)),
                           ("StratifiedKFold", StratifiedKFold(5, shuffle=True, random_state=SEED))]:
        for k, (_, test_idx) in enumerate(splitter.split(Xi, binary), start=1):
            rows.append({"splitter": name, "fold": k,
                         "minority_pct": (binary[test_idx] == "Other").mean() * 100})
    df = pd.DataFrame(rows)

    for name, grp in df.groupby("splitter"):
        pcts = grp["minority_pct"].to_numpy()
        print(f"  {name:<16} minority % per fold: "
              f"{np.round(pcts, 1)}   spread {pcts.max() - pcts.min():.1f} pts")

    fig, ax = plt.subplots()
    width = 0.38
    for i, (name, colour) in enumerate([("plain KFold", PALETTE["vermillion"]),
                                        ("StratifiedKFold", PALETTE["blue"])]):
        grp = df[df.splitter == name]
        ax.bar(grp["fold"] + (i - 0.5) * width, grp["minority_pct"],
               width=width, label=name, color=colour, alpha=0.9)
    ax.axhline(overall * 100, color=PALETTE["black"], linestyle="--",
               label=f"true rate {overall*100:.1f}%")
    ax.set_xlabel("fold")
    ax.set_ylabel("minority class in the fold  (%)")
    ax.set_title("Stratification keeps every fold representative")
    ax.legend()
    print(f"\n  saved  {save_figure(fig, OUT / '3_stratification.png').name}")


# ------------------------------------------- 4. time series must not shuffle
def figure_timeseries() -> None:
    """Random splitting on a time series is interpolation, not forecasting."""
    df = load_flights()
    df["t"] = np.arange(len(df))
    X = df[["t"]].to_numpy()
    y = df["passengers"].to_numpy(float)
    model = RandomForestRegressor(n_estimators=120, random_state=SEED)

    random_mae = -cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=SEED),
                                  scoring="neg_mean_absolute_error").mean()
    ordered_mae = -cross_val_score(model, X, y, cv=TimeSeriesSplit(5),
                                   scoring="neg_mean_absolute_error").mean()

    print(f"  Random KFold  MAE  {random_mae:6.2f} passengers   <- FICTION")
    print(f"  TimeSeriesSplit MAE {ordered_mae:6.2f} passengers   <- honest")
    print(f"  Random splitting looks {ordered_mae / random_mae:.1f}x better than reality,")
    print("  because every test point sits BETWEEN two training points it can interpolate.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # left: what the two schemes actually do
    ax = axes[0]
    for k, (tr, te) in enumerate(TimeSeriesSplit(5).split(X)):
        ax.plot(tr, [k] * len(tr), "|", color=PALETTE["blue"], markersize=7)
        ax.plot(te, [k] * len(te), "|", color=PALETTE["vermillion"], markersize=7)
    rand_tr, rand_te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=SEED)
    ax.plot(rand_tr, [5.6] * len(rand_tr), "|", color=PALETTE["blue"], markersize=7)
    ax.plot(rand_te, [5.6] * len(rand_te), "|", color=PALETTE["vermillion"], markersize=7)
    ax.set_yticks(list(range(5)) + [5.6])
    ax.set_yticklabels([f"TS fold {i+1}" for i in range(5)] + ["random split"])
    ax.set_xlabel("time index (months)")
    ax.set_title("Blue = train, orange = validate")
    ax.grid(False)

    # right: the cost of getting it wrong
    ax = axes[1]
    bars = ax.bar(["random\nKFold", "TimeSeries\nSplit"], [random_mae, ordered_mae],
                  color=[PALETTE["vermillion"], PALETTE["blue"]], alpha=0.9)
    for b, v in zip(bars, [random_mae, ordered_mae]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontweight="bold")
    ax.set_ylabel("MAE (passengers)")
    ax.set_title("Same data, same model, different split")
    fig.suptitle("Random splitting on a time series reports fiction",
                 fontsize=12.5, fontweight="bold")
    print(f"\n  saved  {save_figure(fig, OUT / '4_timeseries_split.png').name}")


def main() -> None:
    use_house_style()
    print("=" * 72)
    print("  SPLITTING DATA: TRAIN, VALIDATION, TEST AND CROSS-VALIDATION")
    print("=" * 72)

    X, y = penguin_xy()
    print(f"\n  Penguins: {len(X)} rows, {X.shape[1]} features, {len(set(y))} species")

    banner("1. The split lottery")
    lo, hi = figure_split_lottery(X, y)

    banner("2. Cross-validation instead")
    figure_cv(X, y, lo, hi)

    banner("3. Why stratify")
    figure_stratification(X, y)

    banner("4. Time series must not be shuffled")
    figure_timeseries()

    banner("What to take away")
    print("  - A single train/test split is ONE DRAW from a distribution. Reporting")
    print(f"    it as 'the' accuracy hides a {hi - lo:.2f} swing that is pure luck.")
    print("  - Cross-validation averages that away and gives you an error bar.")
    print("    Report the mean AND the spread; the spread is the honesty.")
    print("  - Stratify for classification, always. It costs nothing.")
    print("  - Group your splits when rows repeat an entity, or the same customer")
    print("    appears in train and test and the score is inflated.")
    print("  - NEVER shuffle a time series. Train on the past, validate on the")
    print("    future - anything else is interpolation dressed up as forecasting.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
