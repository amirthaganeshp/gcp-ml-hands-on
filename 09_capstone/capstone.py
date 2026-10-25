"""A small model, taken all the way: train, evaluate, serve, monitor.

Nothing here is new. Every step was built in an earlier part - this file
just runs them in order, which is the thing no single example can show.

The exam's harder questions, and production, are about the SEAMS: the
split you chose, the metric you reported, the serving mode you picked,
and the monitor that would have caught it. Not about any one model.

  1. The lifecycle, and where each part of this repo sits in it
  2. Data and the split - the first decision that can invalidate everything
  3. Train, and evaluate HONESTLY (aggregate hides; slices do not)
  4. The serving decision, computed
  5. The monitoring plan, with the failure each monitor catches
  6. Four ways it breaks, run rather than described
  7. What this demo omits - stated, not hidden

EXECUTED IN FULL. Every number below was measured on this machine.

Run:
    python 09_capstone/capstone.py

Concept reference: the whole repository. Blueprint: all six sections.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_adult  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
rng = np.random.default_rng(SEED)


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ------------------------------------------------------------ 1. lifecycle
def section_lifecycle() -> None:
    rule("1. THE LIFECYCLE, AND WHERE EACH PART SITS")
    print("""
    data  ->  split  ->  train  ->  EVALUATE  ->  serve  ->  MONITOR
                |           |          |            |           |
                |           |          |            |           +-- 08_monitoring
                |           |          |            +-- 07_serving
                |           |          +-- 01_foundations (metrics, slices)
                |           +-- 02/03/04 (model, framework)
                +-- 01_foundations (split, leakage)

  The two in capitals are where models actually fail. Everyone trains.
  Far fewer evaluate honestly, and fewer still monitor for the failure
  that does not show up in the inputs.""")


# ---------------------------------------------------------------- 2. data
def section_data():
    rule("2. DATA AND THE SPLIT")
    df = load_adult().dropna()
    y = df["income"].astype(str).str.contains(">50K").astype(int).to_numpy()
    groups = df["sex"].astype(str).to_numpy()
    X = pd.get_dummies(df.drop(columns=["income"]))

    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        X, y, groups, test_size=0.3, random_state=SEED, stratify=y)

    print(f"  Rows           {len(df):,}")
    print(f"  Features       {X.shape[1]} after one-hot encoding")
    print(f"  Positive rate  {y.mean():.3f}")
    print(f"  Train / test   {len(Xtr):,} / {len(Xte):,}  (stratified)")
    print("\n  STRATIFIED because the classes are imbalanced. On time-ordered data")
    print("  this would have to be CHRONOLOGICAL instead - a random split there")
    print("  leaks the future and inflates every number that follows.")
    return Xtr, Xte, ytr, yte, gte


# ------------------------------------------------------- 3. train, evaluate
def section_train(Xtr, Xte, ytr, yte, gte):
    rule("3. TRAIN, THEN EVALUATE HONESTLY")

    model = RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]

    acc = float((pred == yte).mean())
    auc = float(roc_auc_score(yte, proba))
    majority = float(max(yte.mean(), 1 - yte.mean()))

    print(f"  Accuracy          {acc:.4f}")
    print(f"  ROC-AUC           {auc:.4f}")
    print(f"  Majority baseline {majority:.4f}   <- the number accuracy must beat")
    print(f"  Lift over it      {acc - majority:+.4f}")
    print("\n  On imbalanced data ACCURACY ALONE IS NEARLY MEANINGLESS. Predicting")
    print(f"  the majority class every time already scores {majority:.4f}.\n")

    print("  And the same model, sliced:")
    print(f"    {'group':<10} {'n':>7} {'accuracy':>10} {'TPR':>8}")
    for lvl in sorted(set(gte)):
        m = gte == lvl
        pos = yte[m] == 1
        tpr = float(pred[m][pos].mean())
        print(f"    {lvl:<10} {m.sum():>7,} {float((pred[m] == yte[m]).mean()):>10.4f} {tpr:>8.4f}")
    print("\n  One aggregate number, two very different experiences. Reporting only")
    print("  the first is how a model ships with a gap nobody has looked at.")
    return model, pred, proba, acc


# --------------------------------------------------------- 4. serving mode
def section_serving() -> None:
    rule("4. THE SERVING DECISION, COMPUTED")

    rows_per_day = 2_000_000
    replica_hour, node_hour = 0.25, 0.25
    batch_rows_hour = 2_000_000

    batch = (rows_per_day / batch_rows_hour) * node_hour
    online = 24 * replica_hour     # one always-warm replica, minimum

    print(f"  Scoring {rows_per_day:,} rows/day.\n")
    print(f"  Batch  (no endpoint)      ${batch:>7,.2f}/day")
    print(f"  Online (1 warm replica)   ${online:>7,.2f}/day")
    print(f"  Ratio                      {online / batch:>7,.1f}x\n")
    print("  But cost does not decide this - the REQUIREMENT does:")
    print("    a nightly table or file      -> batch, and never create an endpoint")
    print("    a user waiting for a screen  -> online, and pay for warm replicas")
    print("\n  This model scores a nightly customer table, so: BATCH.")
    print("  Full arithmetic in 07_serving/01_batch_vs_online.")


# ------------------------------------------------------------ 5. monitoring
def section_monitoring(model, Xte, yte) -> dict:
    rule("5. THE MONITORING PLAN")

    # Two serving windows: one clean, one with a changed X->y relationship.
    half = len(Xte) // 2
    Xa, Xb = Xte.iloc[:half], Xte.iloc[half:]
    ya, yb = yte[:half], yte[half:].copy()

    acc_a = float(model.score(Xa, ya))

    # Concept drift: identical inputs, changed correct answer.
    edu = Xb["education_num"].to_numpy()
    flip = (edu >= np.median(edu)) & (rng.random(len(yb)) < 0.55)
    yb_drift = yb.copy()
    yb_drift[flip] = 1 - yb_drift[flip]
    acc_b = float(model.score(Xb, yb_drift))

    num_cols = [c for c in Xa.columns if Xa[c].dtype != bool][:6]
    worst = 0.0
    for c in num_cols:
        a, b = Xa[c].to_numpy(float), Xb[c].to_numpy(float)
        edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 31)
        pa = np.histogram(a, bins=edges)[0]
        pb = np.histogram(b, bins=edges)[0]
        worst = max(worst, float(jensenshannon(pa / pa.sum(), pb / pb.sum(), base=2)))

    print(f"  {'what to watch':<26} {'catches':<28} here")
    print("  " + "-" * 70)
    print(f"  {'input distributions':<26} {'skew and data drift':<28} JS {worst:.4f}")
    print(f"  {'ACCURACY on labels':<26} {'CONCEPT drift':<28} {acc_a:.4f} -> {acc_b:.4f}")
    print(f"  {'sliced metrics':<26} {'a gap in one group':<28} see section 3")
    print(f"  {'feature attributions':<26} {'right answers, new reasons':<28} 08_monitoring")

    print(f"\n  Input divergence {worst:.4f} - nothing to alert on.")
    print(f"  Accuracy fell {(acc_a - acc_b) * 100:.1f} points.")
    print("\n  THAT IS THE SEAM. The inputs are fine and the model is not. No input")
    print("  monitor would fire; only labels reveal it. Every dashboard built on")
    print("  feature distributions would show green through this.")
    return dict(acc_a=acc_a, acc_b=acc_b, js=worst)


# --------------------------------------------------------- 6. how it breaks
def section_failures(model, Xtr, ytr, Xte, yte, acc) -> None:
    rule("6. FOUR WAYS IT BREAKS")

    # a) leakage
    Xl = Xtr.copy()
    Xl["leak"] = ytr
    leaky = RandomForestClassifier(n_estimators=60, random_state=SEED, n_jobs=1)
    leaky.fit(Xl, ytr)
    Xte_l = Xte.copy()
    Xte_l["leak"] = yte
    print(f"  a) LEAKAGE      a feature that IS the label -> "
         f"{leaky.score(Xte_l, yte):.4f} vs honest {acc:.4f}")
    print("                  Spectacular offline, worthless in production, because")
    print("                  that column does not exist at prediction time.")

    # b) imbalance / wrong metric
    majority = float(max(yte.mean(), 1 - yte.mean()))
    print(f"\n  b) WRONG METRIC accuracy {acc:.4f} against a majority baseline of")
    print(f"                  {majority:.4f}. On rarer positives the gap closes to")
    print("                  nothing and accuracy stops carrying information.")

    # c) train/serve skew from preprocessing
    print("\n  c) SKEW         preprocessing fitted before the split, or written")
    print("                  twice - once for training, once for serving. The")
    print("                  fix is one definition: Pipeline, TRANSFORM, or a")
    print("                  Feature Store.")

    # d) concept drift
    print("\n  d) CONCEPT      measured in section 5: inputs unchanged, accuracy")
    print("                  gone. Needs labels, not distributions.")


def figure(stats: dict, acc: float) -> None:
    use_house_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["input\ndivergence", "accuracy\nwindow A", "accuracy\nwindow B"]
    vals = [stats["js"], stats["acc_a"], stats["acc_b"]]
    colours = [PALETTE["blue"], PALETTE["green"], PALETTE["vermillion"]]
    ax.bar(labels, vals, color=colours)
    ax.axhline(0.10, color=PALETTE["orange"], ls="--", lw=1.2,
               label="typical drift alert threshold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("value")
    ax.set_title("The inputs look fine. The model is not.")
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, OUT / "capstone.png")
    print(f"\n  figure -> {OUT / 'capstone.png'}")


def main() -> None:
    print("CAPSTONE - one model, all the way through")
    section_lifecycle()
    Xtr, Xte, ytr, yte, gte = section_data()
    model, pred, proba, acc = section_train(Xtr, Xte, ytr, yte, gte)
    section_serving()
    stats = section_monitoring(model, Xte, yte)
    section_failures(model, Xtr, ytr, Xte, yte, acc)
    figure(stats, acc)

    rule("7. WHAT THIS DEMO OMITS - stated, not hidden")
    print("""  no cloud            nothing was deployed; the serving decision is
                      arithmetic, not an endpoint
  no pipeline         this is one script, not an orchestrated DAG
                      (06_pipelines has the Kubeflow version)
  no retraining       the monitor detects; nothing acts on it
  no feature store    train and serve read the same in-memory frame,
                      which is exactly the luxury production lacks
  one model           no registry, no versions, no rollout
  tabular only        images, text and agents each change the shape

  Those omissions are the difference between an example and a system,
  and naming them is more useful than pretending otherwise.""")


if __name__ == "__main__":
    main()
