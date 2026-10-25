"""Feature engineering: usually a bigger lever than the choice of algorithm.

Five things, each measured rather than asserted:

  1. Missing values - four strategies compared, including the one people forget
     (impute AND add an indicator, because missingness is itself information)
  2. Categorical encoding - one-hot vs ordinal, and why ordinal quietly invents
     an ordering that does not exist
  3. Scaling - it transforms distance- and gradient-based models and does almost
     nothing for trees. Knowing which is which saves pointless preprocessing.
  4. Cyclical encoding - hour 23 and hour 0 are adjacent, and an integer column
     says they are 23 apart
  5. All of it assembled in a ColumnTransformer + Pipeline, which is the only
     leak-free way to do any of the above

Run:
    python 01_foundations/06_feature_engineering/feature_engineering.py

Concept reference: Foundations volume, Part A.10.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from loaders import load_bike, load_penguins  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

SEED = 42
OUT = Path(__file__).parent / "images"
NUM = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CAT = ["island", "sex"]
rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def penguins() -> pd.DataFrame:
    return load_penguins().dropna(subset=["species"])


def cv_score(pipe, X, y, stratified: bool = True) -> float:
    cv = (StratifiedKFold(5, shuffle=True, random_state=SEED) if stratified
          else KFold(5, shuffle=True, random_state=SEED))
    return float(cross_val_score(pipe, X, y, cv=cv).mean())


# ----------------------------------------------------- 1. missing values
def figure_missing(df: pd.DataFrame) -> None:
    missing = df[NUM + CAT].isna().sum()
    print("  Missing values in Penguins (genuine, not injected):")
    for col, n in missing.items():
        if n:
            print(f"    {col:<20} {n:>3}  ({n/len(df):.1%})")

    y = df["species"].to_numpy()
    results = {}

    # (a) drop rows with any missing value
    complete = df.dropna(subset=NUM + CAT)
    results["drop rows"] = (
        cv_score(Pipeline([("sc", StandardScaler()),
                           ("m", LogisticRegression(max_iter=2000))]),
                 complete[NUM].to_numpy(), complete["species"].to_numpy()),
        len(complete),
    )

    for name, imputer_kwargs in [
        ("impute mean", {"strategy": "mean"}),
        ("impute median", {"strategy": "median"}),
    ]:
        pipe = Pipeline([("im", SimpleImputer(**imputer_kwargs)),
                         ("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000))])
        results[name] = (cv_score(pipe, df[NUM].to_numpy(), y), len(df))

    # (d) impute AND flag - missingness may itself carry signal
    pipe = Pipeline([("im", SimpleImputer(strategy="median", add_indicator=True)),
                     ("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=2000))])
    results["impute + indicator"] = (cv_score(pipe, df[NUM].to_numpy(), y), len(df))

    print(f"\n  {'strategy':<22} {'accuracy':>9} {'rows used':>10}")
    for name, (score, n) in results.items():
        print(f"  {name:<22} {score:>9.4f} {n:>10}")
    print("\n  Dropping rows is not free: it costs data, and it silently assumes the")
    print("  missingness is random. If it is not - if a measurement is missing")
    print("  BECAUSE of something - dropping introduces bias and the indicator")
    print("  column is the feature you actually wanted.")

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(results)
    scores = [results[n][0] for n in names]
    bars = ax.bar(names, scores, color=PALETTE["blue"], alpha=0.9)
    for b, v, n in zip(bars, scores, names):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.02, f"n={results[n][1]}",
                ha="center", fontsize=9, color="white")
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("cross-validated accuracy")
    ax.set_title("Missing-value strategies")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    print(f"\n  saved  {save_figure(fig, OUT / '1_missing_values.png').name}")


# ------------------------------------------------- 2. categorical encoding
def figure_encoding(df: pd.DataFrame) -> None:
    d = df.dropna(subset=NUM + CAT)
    y = d["species"].to_numpy()
    X = d[CAT]

    print(f"  Encoding {CAT} for a LINEAR model:\n")
    scores = {}
    for name, enc in [
        ("ordinal (0,1,2)", OrdinalEncoder()),
        ("one-hot", OneHotEncoder(handle_unknown="ignore")),
    ]:
        pipe = Pipeline([("enc", enc), ("m", LogisticRegression(max_iter=2000))])
        scores[name] = cv_score(pipe, X, y)
        print(f"    {name:<18} {scores[name]:.4f}")

    print("\n  Ordinal encoding turns Biscoe/Dream/Torgersen into 0/1/2, which tells a")
    print("  linear model that Dream sits exactly halfway between the other two and")
    print("  that Torgersen is 'twice' Dream. None of that is true - the ordering is")
    print("  alphabetical. One-hot makes no such claim.")
    print("\n  Ordinal IS right when the order is real: S < M < L, or low/medium/high.")

    # trees are far less bothered - they can split a single ordinal column repeatedly
    tree_scores = {}
    for name, enc in [("ordinal (0,1,2)", OrdinalEncoder()),
                      ("one-hot", OneHotEncoder(handle_unknown="ignore"))]:
        pipe = Pipeline([("enc", enc),
                         ("m", RandomForestClassifier(n_estimators=150, random_state=SEED))])
        tree_scores[name] = cv_score(pipe, X, y)
    print(f"\n  The same two encodings with a RANDOM FOREST:")
    for k, v in tree_scores.items():
        print(f"    {k:<18} {v:.4f}")
    print("  Much closer - a tree can split the same ordinal column several times and")
    print("  carve out any grouping it likes, so the false ordering costs it little.")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(2)
    width = 0.36
    ax.bar(x - width / 2, [scores["ordinal (0,1,2)"], tree_scores["ordinal (0,1,2)"]],
           width, label="ordinal", color=PALETTE["vermillion"], alpha=0.9)
    ax.bar(x + width / 2, [scores["one-hot"], tree_scores["one-hot"]],
           width, label="one-hot", color=PALETTE["blue"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["logistic regression\n(linear)", "random forest\n(trees)"])
    ax.set_ylabel("cross-validated accuracy")
    ax.set_title("A false ordering hurts a linear model and barely troubles a tree")
    ax.legend()
    print(f"\n  saved  {save_figure(fig, OUT / '2_encoding.png').name}")


# ------------------------------------------------------------- 3. scaling
def figure_scaling(df: pd.DataFrame) -> None:
    d = df.dropna(subset=NUM)
    X, y = d[NUM].to_numpy(), d["species"].to_numpy()
    print("  Feature ranges - body_mass_g is ~1000x bill_depth_mm:")
    for col in NUM:
        print(f"    {col:<20} {d[col].min():8.1f} to {d[col].max():8.1f}")

    models = {
        "KNN (distance)": KNeighborsClassifier(5),
        "Logistic (gradient)": LogisticRegression(max_iter=2000),
        "Random forest (trees)": RandomForestClassifier(n_estimators=150, random_state=SEED),
    }
    print(f"\n  {'model':<24} {'unscaled':>9} {'scaled':>9} {'change':>9}")
    raw_s, scaled_s = [], []
    convergence_failures = 0
    for name, model in models.items():
        # The unscaled logistic regression genuinely fails to converge - lbfgs cannot
        # handle features spanning 13 to 6300. That warning IS the lesson, so it is
        # captured and reported once rather than printed forty times to stderr.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            raw = cv_score(Pipeline([("m", model)]), X, y)
        failures = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        convergence_failures += failures

        sca = cv_score(Pipeline([("sc", StandardScaler()), ("m", model)]), X, y)
        raw_s.append(raw)
        scaled_s.append(sca)
        flag = f"   ({failures} folds failed to converge)" if failures else ""
        print(f"  {name:<24} {raw:>9.4f} {sca:>9.4f} {sca - raw:>+9.4f}{flag}")

    if convergence_failures:
        print(f"\n  Note: {convergence_failures} unscaled fits hit the iteration limit without")
        print("  converging. scikit-learn's own warning text suggests scaling the data -")
        print("  the optimiser is telling you the same thing this section is.")

    print("\n  KNN moves the most: it measures Euclidean distance, so body_mass_g")
    print("  (range ~3600) drowns bill_depth_mm (range ~8) entirely until scaled.")
    print("  Trees barely move - a split is a threshold, and rescaling does not")
    print("  change the ORDER of values, so the same splits remain available.")

    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(models))
    width = 0.36
    ax.bar(x - width / 2, raw_s, width, label="unscaled",
           color=PALETTE["vermillion"], alpha=0.9)
    ax.bar(x + width / 2, scaled_s, width, label="standardised",
           color=PALETTE["blue"], alpha=0.9)
    for i, (r, s) in enumerate(zip(raw_s, scaled_s)):
        ax.text(i, max(r, s) + 0.02, f"{s - r:+.3f}", ha="center",
                fontweight="bold", color=PALETTE["grey"])
    ax.set_xticks(x)
    ax.set_xticklabels(list(models))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("cross-validated accuracy")
    ax.set_title("Scaling transforms distance-based models and barely touches trees")
    ax.legend(loc="lower right")
    print(f"\n  saved  {save_figure(fig, OUT / '3_scaling.png').name}")


# --------------------------------------------------- 4. cyclical encoding
def figure_cyclical() -> None:
    d = load_bike("hour")
    y = d["cnt"].to_numpy(float)

    raw = d[["hr", "temp", "hum", "windspeed", "workingday"]].to_numpy(float)
    cyc = np.column_stack([
        np.sin(2 * np.pi * d["hr"] / 24), np.cos(2 * np.pi * d["hr"] / 24),
        d["temp"], d["hum"], d["windspeed"], d["workingday"],
    ])

    cv = KFold(5, shuffle=True, random_state=SEED)
    r_raw = cross_val_score(Ridge(), raw, y, cv=cv, scoring="r2").mean()
    r_cyc = cross_val_score(Ridge(), cyc, y, cv=cv, scoring="r2").mean()

    print(f"  Predicting hourly rentals with a LINEAR model:\n")
    print(f"    hour as an integer 0-23     R2 = {r_raw:.4f}")
    print(f"    hour as sin/cos pair        R2 = {r_cyc:.4f}")
    print(f"    improvement                 {r_cyc - r_raw:+.4f}")
    print("\n  As an integer, hour 23 and hour 0 are 23 apart - the model is told")
    print("  midnight is maximally distant from 11pm. The sin/cos pair places them")
    print("  adjacent on a circle, which is what a clock actually is.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    hours = np.arange(24)
    ax = axes[0]
    ax.plot(hours, hours, marker="o", color=PALETTE["vermillion"])
    ax.annotate("23 and 0 are\nfar apart here", xy=(0.05, 0.72),
                xycoords="axes fraction", color=PALETTE["grey"])
    ax.set_xlabel("hour")
    ax.set_ylabel("encoded value")
    ax.set_title("Integer encoding: a straight line")

    ax = axes[1]
    ax.plot(np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
            marker="o", color=PALETTE["blue"])
    for h in (0, 6, 12, 18, 23):
        ax.annotate(str(h), (np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("sin(2π·hour/24)")
    ax.set_ylabel("cos(2π·hour/24)")
    ax.set_title("Cyclical encoding: 23 sits next to 0")
    ax.set_aspect("equal")
    fig.suptitle(f"Hour of day, two ways   (linear-model R²: "
                 f"{r_raw:.3f} → {r_cyc:.3f})", fontsize=12.5, fontweight="bold")
    print(f"\n  saved  {save_figure(fig, OUT / '4_cyclical.png').name}")


# --------------------------------------- 5. all of it, assembled leak-free
def demo_pipeline(df: pd.DataFrame) -> None:
    y = df["species"].to_numpy()
    X = df[NUM + CAT]

    pre = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), NUM),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT),
    ])
    pipe = Pipeline([("prep", pre), ("model", LogisticRegression(max_iter=2000))])
    score = cv_score(pipe, X, y)

    print("  ColumnTransformer + Pipeline, numeric and categorical handled separately:\n")
    print("    numeric      -> median impute (+ missing indicator) -> standardise")
    print("    categorical  -> most-frequent impute -> one-hot")
    print(f"\n    cross-validated accuracy  {score:.4f}   on all {len(df)} rows")
    print("\n  Every fitted step - the median, the mean and variance, the vocabulary -")
    print("  is refitted on each training fold and applied to the held-out fold.")
    print("  That is the ONLY leak-free way to do this, and it is a single object")
    print("  you can save and deploy, so training and serving cannot drift apart.")


def main() -> None:
    use_house_style()
    print("=" * 72)
    print("  FEATURE ENGINEERING")
    print("=" * 72)

    df = penguins()
    print(f"\n  Penguins: {len(df)} rows, {len(NUM)} numeric + {len(CAT)} categorical")

    banner("1. Missing values")
    figure_missing(df)

    banner("2. Categorical encoding")
    figure_encoding(df)

    banner("3. Scaling: who needs it")
    figure_scaling(df)

    banner("4. Cyclical features")
    figure_cyclical()

    banner("5. Assembling it without leaking")
    demo_pipeline(df)

    banner("What to take away")
    print("  - Feature engineering usually beats algorithm choice. Spend time here.")
    print("  - Dropping rows costs data and assumes missingness is random. Impute,")
    print("    and consider adding an indicator - the fact that a value is missing")
    print("    is often signal in itself.")
    print("  - One-hot for unordered categories; ordinal ONLY when the order is")
    print("    real. Trees tolerate a false ordering; linear models do not.")
    print("  - Scale for distance- and gradient-based models. Trees do not need it,")
    print("    and an option insisting on it for XGBoost is noise.")
    print("  - Encode cyclical features as sin/cos, or midnight is 23 hours from 11pm.")
    print("  - Put all of it in a Pipeline. Anything fitted must be fitted inside")
    print("    the cross-validation loop - see 05_data_leakage.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
