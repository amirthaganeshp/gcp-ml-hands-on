"""Responsible AI: why aggregate accuracy hides bias, and why dropping the
protected column does not fix it.

There is one option the exam marks wrong every single time:

    "Simply remove the gender column and the model is fair."

It is wrong because other features PROXY for the protected attribute. This
file demonstrates that rather than asserting it: drop the column, watch the
disparity survive, then train a model to recover the dropped attribute from
what is left - which is exactly what the main model is quietly doing.

  1. Aggregate accuracy, which looks fine
  2. The same model, sliced - which does not
  3. The three fairness measures the exam names
  4. Drop the protected attribute. Watch nothing improve.
  5. Prove the proxies exist
  6. A mitigation that actually moves the number

EXECUTED IN FULL. Every number below was measured on this machine.

NOTE ON THE DATA: Adult income is a real 1994 census extract and its
disparities are historical facts of that sample, not claims about people.
It is used here because a fairness example needs real structure - a
synthetic dataset would only contain the bias I put in it.

Run:
    python 08_monitoring/02_fairness_and_slices/fairness_and_slices.py

Concept reference: Success Guide, Part 6.4. Blueprint section 6 (~13%).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_adult  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
PROTECTED = "sex"


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot the categoricals, leave the numerics."""
    return pd.get_dummies(df, drop_first=True)


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pos = y_true == 1
    neg = ~pos
    return {
        "n": len(y_true),
        "accuracy": float((y_true == y_pred).mean()),
        "selection_rate": float(y_pred.mean()),               # demographic parity
        "tpr": float(y_pred[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(y_pred[neg].mean()) if neg.any() else float("nan"),
    }


def main() -> None:
    print("RESPONSIBLE AI - slices, proxies, and the option that is always wrong")

    df = load_adult().dropna()
    y = df["income"].astype(str).str.contains(">50K").astype(int).to_numpy()
    groups = df[PROTECTED].astype(str).to_numpy()
    X_full = encode(df.drop(columns=["income"]))

    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        X_full, y, groups, test_size=0.3, random_state=SEED, stratify=y)

    # ------------------------------------------------------- 1. aggregate
    rule("1. AGGREGATE ACCURACY - the number on the dashboard")
    model = RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"  Overall accuracy  {(pred == yte).mean():.4f}")
    print(f"  Test rows         {len(yte):,}")
    print("\n  One number, and it looks healthy. This is what gets reported.")

    # ---------------------------------------------------------- 2. sliced
    rule("2. THE SAME MODEL, SLICED")
    levels = sorted(set(gte))
    print(f"  {'group':<10} {'n':>7} {'accuracy':>9} {'selection':>10} {'TPR':>8} {'FPR':>8}")
    print("  " + "-" * 56)
    stats: dict[str, dict[str, float]] = {}
    for lvl in levels:
        m = gte == lvl
        stats[lvl] = group_metrics(yte[m], pred[m])
        s = stats[lvl]
        print(f"  {lvl:<10} {s['n']:>7,} {s['accuracy']:>9.4f} "
              f"{s['selection_rate']:>10.4f} {s['tpr']:>8.4f} {s['fpr']:>8.4f}")

    print("\n  Evaluate on SLICES, not aggregates. The overall number above is a")
    print("  weighted average that can hide an arbitrarily large gap.")

    # --------------------------------------------------- 3. the measures
    rule("3. THE THREE FAIRNESS MEASURES THE EXAM NAMES")
    a, b = levels[0], levels[1]
    dp = stats[b]["selection_rate"] - stats[a]["selection_rate"]
    eo = stats[b]["tpr"] - stats[a]["tpr"]
    odds = max(abs(eo), abs(stats[b]["fpr"] - stats[a]["fpr"]))

    print(f"  Demographic parity      selection rates equal?")
    print(f"    {a} {stats[a]['selection_rate']:.4f} vs {b} {stats[b]['selection_rate']:.4f}"
          f"   gap {dp:+.4f}")
    print(f"\n  Equality of opportunity  TRUE-POSITIVE rates equal?")
    print(f"    {a} {stats[a]['tpr']:.4f} vs {b} {stats[b]['tpr']:.4f}"
          f"   gap {eo:+.4f}")
    print(f"\n  Equalized odds           TPR *and* FPR equal?")
    print(f"    worst of the two gaps: {odds:.4f}")
    print("\n  They are different goals and can conflict. Demographic parity can be")
    print("  satisfied by promoting unqualified candidates; equality of opportunity")
    print("  cannot. Which one applies is a POLICY decision, not a modelling one.")
    baseline_eo = eo

    # ------------------------------------------- 4. drop the attribute
    rule("4. DROP THE PROTECTED ATTRIBUTE. WATCH NOTHING IMPROVE.")
    drop_cols = [c for c in X_full.columns if c.startswith(f"{PROTECTED}_")]
    Xtr_d = Xtr.drop(columns=drop_cols)
    Xte_d = Xte.drop(columns=drop_cols)
    print(f"  Removed {len(drop_cols)} column(s): {drop_cols}\n")

    blind = RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=1)
    blind.fit(Xtr_d, ytr)
    pred_b = blind.predict(Xte_d)

    print(f"  {'group':<10} {'accuracy':>9} {'selection':>10} {'TPR':>8}")
    print("  " + "-" * 40)
    blind_stats = {}
    for lvl in levels:
        m = gte == lvl
        blind_stats[lvl] = group_metrics(yte[m], pred_b[m])
        s = blind_stats[lvl]
        print(f"  {lvl:<10} {s['accuracy']:>9.4f} {s['selection_rate']:>10.4f} {s['tpr']:>8.4f}")

    eo_blind = blind_stats[b]["tpr"] - blind_stats[a]["tpr"]
    print(f"\n  Equality-of-opportunity gap")
    print(f"    with the column    {baseline_eo:+.4f}")
    print(f"    without the column {eo_blind:+.4f}")
    change = abs(eo_blind) - abs(baseline_eo)
    print(f"    change             {change:+.4f}"
          f"  ({'WORSE' if change > 0 else 'barely moved' if abs(change) < 0.02 else 'better'})")
    print("\n  The model never saw the attribute and reproduced the gap anyway.")

    # ------------------------------------------------- 5. prove the proxies
    rule("5. WHY - THE PROXIES ARE STILL THERE")
    print("  If the remaining features really carried no information about the")
    print("  protected attribute, you could not predict it from them. Try:\n")

    g_bin = (gte == b).astype(int)
    g_tr = (gtr == b).astype(int)
    prox = RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=1)
    prox.fit(Xtr_d, g_tr)
    prox_acc = prox.score(Xte_d, g_bin)
    majority = max(g_bin.mean(), 1 - g_bin.mean())

    print(f"  Predicting '{PROTECTED}' from the REMAINING features:")
    print(f"    accuracy          {prox_acc:.4f}")
    print(f"    majority-class    {majority:.4f}")
    print(f"    lift over chance  {prox_acc - majority:+.4f}")

    imp = pd.Series(prox.feature_importances_, index=Xtr_d.columns).nlargest(5)
    print("\n  The strongest proxies:")
    for name, val in imp.items():
        print(f"    {name:<34} {val:.4f}")

    top = ", ".join(str(n) for n in imp.index[:3])
    print(f"\n  That is the whole answer: {top} - each")
    print("  encodes part of the attribute you deleted. The model reconstructs it")
    print("  and uses it, without ever being told.")
    print("\n  Note the strongest proxy is often not the obvious one. 'relationship'")
    print("  and 'marital status' carry it explicitly (Wife, Husband), but census")
    print("  sampling weights and hours worked carry it too, because they were")
    print("  themselves produced by a world in which the attribute mattered.")
    print("\n  SO: 'just drop the column' is always the wrong option.")

    # --------------------------------------------------- 6. a real mitigation
    rule("6. A MITIGATION THAT ACTUALLY MOVES THE NUMBER")
    print("  Per-group decision thresholds: pick each group's threshold so their")
    print("  true-positive rates match. This is an explicit policy choice, applied")
    print("  openly, rather than a hoped-for side effect of deleting a column.\n")

    proba = model.predict_proba(Xte)[:, 1]
    target_tpr = min(stats[a]["tpr"], stats[b]["tpr"])
    thresholds, tuned = {}, np.zeros_like(pred)
    for lvl in levels:
        m = gte == lvl
        pos = yte[m] == 1
        cand = np.linspace(0.05, 0.95, 91)
        tprs = [(proba[m][pos] >= t).mean() for t in cand]
        thresholds[lvl] = float(cand[int(np.argmin(np.abs(np.array(tprs) - target_tpr)))])
        tuned[m] = (proba[m] >= thresholds[lvl]).astype(int)

    print(f"  {'group':<10} {'threshold':>10} {'TPR before':>12} {'TPR after':>11}")
    print("  " + "-" * 46)
    after = {}
    for lvl in levels:
        m = gte == lvl
        after[lvl] = group_metrics(yte[m], tuned[m])
        print(f"  {lvl:<10} {thresholds[lvl]:>10.2f} {stats[lvl]['tpr']:>12.4f} "
              f"{after[lvl]['tpr']:>11.4f}")

    eo_after = after[b]["tpr"] - after[a]["tpr"]
    acc_after = float((tuned == yte).mean())
    print(f"\n  Equality-of-opportunity gap  {baseline_eo:+.4f} -> {eo_after:+.4f}")
    print(f"  Overall accuracy             {(pred == yte).mean():.4f} -> {acc_after:.4f}")
    print("\n  The gap closes and accuracy pays for it. That trade is the honest")
    print("  part: fairness interventions COST something, and the cost is the")
    print("  conversation, not a reason to skip them.")

    # ---------------------------------------------------------------- figure
    use_house_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    idx = np.arange(len(levels))
    w = 0.35
    ax1.bar(idx - w / 2, [stats[l]["tpr"] for l in levels], w,
            label="with attribute", color=PALETTE["blue"])
    ax1.bar(idx + w / 2, [blind_stats[l]["tpr"] for l in levels], w,
            label="attribute dropped", color=PALETTE["vermillion"])
    ax1.set_xticks(idx); ax1.set_xticklabels(levels)
    ax1.set_ylabel("true-positive rate")
    ax1.set_title("Dropping the column changes nothing")
    ax1.legend(frameon=False, fontsize=9)

    ax2.bar(idx - w / 2, [stats[l]["tpr"] for l in levels], w,
            label="before", color=PALETTE["blue"])
    ax2.bar(idx + w / 2, [after[l]["tpr"] for l in levels], w,
            label="per-group thresholds", color=PALETTE["green"])
    ax2.set_xticks(idx); ax2.set_xticklabels(levels)
    ax2.set_ylabel("true-positive rate")
    ax2.set_title("A mitigation that does")
    ax2.legend(frameon=False, fontsize=9)
    save_figure(fig, OUT / "fairness_slices.png")
    print(f"\n  figure -> {OUT / 'fairness_slices.png'}")

    rule("THE EXAM ANGLE")
    print("""  "drop the protected column and the model is fair"   ALWAYS WRONG - proxies
  "evaluate on slices, not just aggregates"           correct (TFMA / sliced metrics)
  "equal selection rates across groups"               demographic parity
  "equal TRUE-POSITIVE rates across groups"           equality of opportunity
  "equal TPR and FPR"                                 equalized odds
  mitigations                                         rebalance/augment data,
                                                      remove proxy features,
                                                      per-group thresholds,
                                                      human review for high impact

  And look for bias in the TRAINING DATA first: unrepresentative sampling,
  historical bias baked into the labels, proxy variables. A model trained on
  biased labels learns the bias no matter how you evaluate it.""")


if __name__ == "__main__":
    main()
