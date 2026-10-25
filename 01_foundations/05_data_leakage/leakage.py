"""Data leakage: the bug that gives you 0.99 offline and nothing in production.

Leakage is information reaching the model that would not be available at prediction
time. It produces spectacular validation scores and total production failure, and it
is the single highest-severity bug in applied machine learning.

Three kinds, each demonstrated and then fixed:

  1. TARGET leakage - a feature that is a consequence of the label. The Bike
     Sharing file ships with a real one: casual + registered == cnt exactly.
  2. PREPROCESSING leakage - fitting a scaler or an imputer before splitting, so
     statistics from the validation rows bleed into training. Fixed with Pipeline.
  3. TEMPORAL leakage - training on data from after the moment you predict.

Run:
    python 01_foundations/05_data_leakage/leakage.py

Concept reference: Foundations volume, Part A.11.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from loaders import load_bike  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
rng = np.random.default_rng(SEED)

HONEST = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
          "weathersit", "temp", "atemp", "hum", "windspeed"]


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ------------------------------------------------------- 1. target leakage
def figure_target_leakage(df: pd.DataFrame) -> None:
    """The leak is in the raw file: casual + registered == cnt, exactly."""
    check = (df["casual"] + df["registered"] - df["cnt"]).abs().max()
    corr_cas = df["casual"].corr(df["cnt"])
    corr_reg = df["registered"].corr(df["cnt"])

    print(f"  max |casual + registered - cnt|  =  {check}")
    print("  The target is the SUM of two columns sitting in the same file.\n")
    print(f"  correlation(casual, cnt)      {corr_cas:.3f}")
    print(f"  correlation(registered, cnt)  {corr_reg:.3f}   <- suspiciously high")

    y = df["cnt"].to_numpy(float)
    cv = KFold(5, shuffle=True, random_state=SEED)
    results = {}
    for name, cols in [
        ("with the leak", HONEST + ["casual", "registered"]),
        ("honest features", HONEST),
    ]:
        X = df[cols].to_numpy(float)
        model = RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=-1)
        r2 = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
        mae = -cross_val_score(model, X, y, cv=cv,
                               scoring="neg_mean_absolute_error").mean()
        results[name] = (r2, mae)
        print(f"\n  {name:<18} R2 = {r2:.4f}   MAE = {mae:6.1f} rentals/day")

    r2_leak, mae_leak = results["with the leak"]
    r2_ok, mae_ok = results["honest features"]
    print(f"\n  The leak buys {r2_leak - r2_ok:.3f} R2 and makes the error "
          f"{mae_ok / mae_leak:.0f}x smaller.")
    print("  None of it is real. At prediction time - forecasting tomorrow's demand -")
    print("  you do not know tomorrow's casual and registered counts. They ARE the")
    print("  answer, split in two.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = list(results)
    for ax, metric, idx, ylab in [(axes[0], "R2", 0, "R² (higher looks better)"),
                                  (axes[1], "MAE", 1, "MAE, rentals/day (lower looks better)")]:
        vals = [results[n][idx] for n in names]
        bars = ax.bar(names, vals, color=[PALETTE["vermillion"], PALETTE["blue"]], alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02,
                    f"{v:.3f}" if idx == 0 else f"{v:.0f}", ha="center", fontweight="bold")
        ax.set_ylabel(ylab)
        ax.set_title(metric)
    fig.suptitle("Target leakage: casual + registered == cnt, in the raw file",
                 fontsize=12.5, fontweight="bold")
    print(f"\n  saved  {save_figure(fig, OUT / '1_target_leakage.png').name}")


# ------------------------------------------------ 2. preprocessing leakage
def figure_preprocessing_leakage() -> None:
    """Selecting features on the whole dataset, before splitting.

    Uses PURE NOISE with no relationship to the target. An honest procedure must
    score ~0.5 (chance). Anything above that is leakage, by construction.
    """
    n_samples, n_features = 120, 6000
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, n_samples)  # labels independent of X - no signal exists

    # WRONG: choose the best features using ALL the data, then cross-validate
    selector = SelectKBest(f_classif, k=20).fit(X, y)
    X_pre = selector.transform(X)
    leaked = cross_val_score(LogisticRegression(max_iter=1000), X_pre, y,
                             cv=KFold(5, shuffle=True, random_state=SEED)).mean()

    # RIGHT: selection happens inside the pipeline, refitted on each training fold
    honest_pipe = Pipeline([
        ("select", SelectKBest(f_classif, k=20)),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    honest = cross_val_score(honest_pipe, X, y,
                             cv=KFold(5, shuffle=True, random_state=SEED)).mean()

    print(f"  {n_samples} rows, {n_features:,} PURE NOISE features, random labels.")
    print("  There is no signal. Any honest method must score about 0.50.\n")
    print(f"  Select features on ALL data, then CV   {leaked:.3f}   <- leakage")
    print(f"  Select inside the Pipeline (per fold)  {honest:.3f}   <- honest")
    print(f"\n  The leaky version invented {leaked - 0.5:+.3f} of accuracy out of")
    print("  nothing. The selector saw the validation labels while choosing which")
    print("  columns to keep, so the 'best' 20 of 6,000 noise columns were the ones")
    print("  that happened to correlate with the labels it was about to be tested on.")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(["select before split\n(leaky)", "select inside Pipeline\n(honest)"],
                  [leaked, honest], color=[PALETTE["vermillion"], PALETTE["blue"]], alpha=0.9)
    ax.axhline(0.5, color=PALETTE["black"], linestyle="--", label="true value: chance = 0.50")
    for b, v in zip(bars, [leaked, honest]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("cross-validated accuracy")
    ax.set_title("Feature selection on pure noise — the leak is the whole score")
    ax.legend()
    print(f"\n  saved  {save_figure(fig, OUT / '2_preprocessing_leakage.png').name}")


# ------------------------------------------------------ 3. temporal leakage
def figure_temporal_leakage(df: pd.DataFrame) -> None:
    """Predicting the past from the future looks easy and is worth nothing."""
    d = df.sort_values("dteday").reset_index(drop=True)
    X = d[HONEST].to_numpy(float)
    y = d["cnt"].to_numpy(float)
    cut = int(len(d) * 0.75)

    model = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)

    # WRONG: random split - training rows come from after the validation rows
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    rand_mae = mean_absolute_error(yr_te, model.fit(Xr_tr, yr_tr).predict(Xr_te))

    # RIGHT: train on the first 75% of days, forecast the rest
    time_mae = mean_absolute_error(y[cut:], model.fit(X[:cut], y[:cut]).predict(X[cut:]))

    print(f"  Random split      MAE {rand_mae:6.1f} rentals/day   <- FICTION")
    print(f"  Chronological     MAE {time_mae:6.1f} rentals/day   <- honest")
    print(f"  Random splitting looks {time_mae / rand_mae:.1f}x better.\n")
    print("  This dataset spans two years with strong growth. A random split lets the")
    print("  model learn 'year 2 is busier' from year-2 rows and then apply it to")
    print("  other year-2 rows. A real forecaster standing at the cut has never seen")
    print("  year 2 at all.")

    fig, ax = plt.subplots()
    ax.plot(d["dteday"], y, color=PALETTE["grey"], linewidth=1.0, label="actual rentals")
    ax.axvline(d["dteday"].iloc[cut], color=PALETTE["black"], linestyle="--",
               label="chronological cut")
    ax.plot(d["dteday"].iloc[cut:], model.predict(X[cut:]),
            color=PALETTE["blue"], linewidth=1.6, label=f"honest forecast (MAE {time_mae:.0f})")
    ax.fill_between(d["dteday"].iloc[:cut], y[:cut].min(), y[:cut].max(),
                    color=PALETTE["blue"], alpha=0.06)
    ax.annotate("train", xy=(0.28, 0.93), xycoords="axes fraction", color=PALETTE["grey"])
    ax.annotate("forecast", xy=(0.85, 0.93), xycoords="axes fraction", color=PALETTE["grey"])
    ax.set_ylabel("daily rentals")
    ax.set_title(f"Chronological split MAE {time_mae:.0f}  vs  random split MAE "
                 f"{rand_mae:.0f} (fiction)")
    ax.legend(loc="upper left")
    print(f"\n  saved  {save_figure(fig, OUT / '3_temporal_leakage.png').name}")


def main() -> None:
    use_house_style()
    print("=" * 72)
    print("  DATA LEAKAGE")
    print("=" * 72)

    df = load_bike("day")
    print(f"\n  Bike Sharing: {len(df)} days, "
          f"{df['dteday'].min():%Y-%m-%d} to {df['dteday'].max():%Y-%m-%d}")

    banner("1. Target leakage - a feature that IS the answer")
    figure_target_leakage(df)

    banner("2. Preprocessing leakage - fitting before splitting")
    figure_preprocessing_leakage()

    banner("3. Temporal leakage - predicting the past from the future")
    figure_temporal_leakage(df)

    banner("What to take away")
    print("  - THE TELL: '0.99 offline, useless in production' is almost always")
    print("    leakage. Not overfitting, not drift.")
    print("  - THE TEST: for every feature ask - would I actually know this value")
    print("    at the moment I need the prediction? If not, or only after the")
    print("    outcome, it leaks.")
    print("  - Anything fitted - scaler, imputer, encoder, feature selector - must")
    print("    be fitted INSIDE the cross-validation loop. Pipeline does this for")
    print("    you and is the reason Pipeline exists.")
    print("  - Never shuffle a time series.")
    print("  - Watch for group leakage too: several rows per customer, split")
    print("    randomly, puts the same customer on both sides.")
    print("  - Production equivalent: joining a feature table at TODAY's values onto")
    print("    HISTORICAL labels. That is what point-in-time feature retrieval exists")
    print("    to prevent.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
