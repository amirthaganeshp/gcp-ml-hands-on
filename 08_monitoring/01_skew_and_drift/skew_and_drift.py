"""Skew and drift: four different failures that get called the same thing.

The exam separates these precisely, and so does Model Monitoring:

    TRAINING-SERVING SKEW   serving data vs TRAINING data
                            production never matched training in the first place
    PREDICTION / DATA DRIFT serving data NOW vs serving data EARLIER
                            the world moved after you deployed
    CONCEPT DRIFT           the relationship between X and y changed
                            same inputs, different correct answer
    FEATURE ATTRIBUTION     the model is leaning on different features than before
    DRIFT

The one that matters most is concept drift, because it is INVISIBLE to
input monitoring. This file makes that visible: a serving window whose
feature distributions are statistically identical to training, and whose
accuracy has collapsed.

  1. A baseline model and a clean serving window
  2. Training-serving skew, measured with the metrics Model Monitoring uses
  3. Data drift - the world moved
  4. CONCEPT DRIFT - inputs unchanged, accuracy gone
  5. Feature attribution drift
  6. Which monitor would have caught which

EXECUTED IN FULL. Every number below was measured on this machine.

Run:
    python 08_monitoring/01_skew_and_drift/skew_and_drift.py

Concept reference: Success Guide, Part 6.1. Blueprint section 6 (~13%).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_adult  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
rng = np.random.default_rng(SEED)

# Model Monitoring's published thresholds are configurable; these are the
# defaults people usually start from.
JS_ALERT = 0.10          # Jensen-Shannon divergence, numerical features
LINF_ALERT = 0.10        # L-infinity distance, categorical features


def col(names: list[str], want: str) -> int:
    """Index of a named column, loudly.

    An earlier version used `names.index(x) if x in names else <number>`. The
    Adult columns use UNDERSCORES (hours_per_week), the lookups used hyphens,
    and every fallback silently fired - so the drift section scaled `fnlwgt`, a
    sampling weight, while the narration claimed it had changed hours worked.
    The demo still produced a plausible number for the wrong reason, which is
    the worst way to be wrong. Fail instead.
    """
    if want not in names:
        raise KeyError(f"expected column {want!r}; have {names}")
    return names.index(want)


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ------------------------------------------------------------- the metrics
def js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    """Jensen-Shannon divergence - what Model Monitoring uses for NUMERICAL."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(jensenshannon(pa, pb, base=2))


def l_infinity(a: np.ndarray, b: np.ndarray) -> float:
    """L-infinity distance - what Model Monitoring uses for CATEGORICAL."""
    cats = np.union1d(np.unique(a), np.unique(b))
    pa = np.array([(a == c).mean() for c in cats])
    pb = np.array([(b == c).mean() for c in cats])
    return float(np.max(np.abs(pa - pb)))


def report(name: str, train: np.ndarray, serve: np.ndarray,
           feature_names: list[str]) -> float:
    """Print per-feature divergence and return the worst."""
    worst = 0.0
    print(f"  {name}")
    print(f"    {'feature':<18} {'JS divergence':>14}  {'':<6}")
    for i, fname in enumerate(feature_names):
        d = js_divergence(train[:, i], serve[:, i])
        worst = max(worst, d)
        flag = "ALERT" if d > JS_ALERT else ""
        print(f"    {fname:<18} {d:>14.4f}  {flag:<6}")
    return worst


def main() -> None:
    print("SKEW AND DRIFT - four failures, one careless name")

    # ---------------------------------------------------------- 1. baseline
    rule("1. BASELINE - train a model, then serve it clean traffic")

    df = load_adult().dropna()
    y_all = df["income"].astype(str).str.contains(">50K").astype(int).to_numpy()
    num = df.select_dtypes(include=[np.number])
    feature_names = list(num.columns)
    X_all = num.to_numpy(dtype=np.float64)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X_all, y_all, test_size=0.6, random_state=SEED, stratify=y_all)
    # Two disjoint serving windows drawn from the SAME distribution.
    X_s0, X_s1, y_s0, y_s1 = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=SEED, stratify=y_rest)

    model = RandomForestClassifier(n_estimators=120, random_state=SEED, n_jobs=1)
    model.fit(X_train, y_train)

    acc0 = model.score(X_s0, y_s0)
    print(f"  Features            {feature_names}")
    print(f"  Train rows          {len(X_train):,}")
    print(f"  Serving window 0    {len(X_s0):,} rows, accuracy {acc0:.4f}")
    print("\n  This is the reference. Everything below is compared against it.")

    # ------------------------------------------------------------- 2. skew
    rule("2. TRAINING-SERVING SKEW - production never matched training")

    # Skew is a MISMATCH THAT WAS ALWAYS THERE: serving sees a different
    # population than the one the model was fitted on. Simulate by sampling
    # serving traffic biased towards older applicants.
    age_i = col(feature_names, "age")
    older = X_s0[:, age_i] > np.percentile(X_train[:, age_i], 60)
    X_skew = X_s0[older]
    y_skew = y_s0[older]

    print("  Serving traffic is biased towards older applicants - a population")
    print("  the training set under-represented.\n")
    worst_skew = report("train -> serving", X_train, X_skew, feature_names)
    acc_skew = model.score(X_skew, y_skew)
    print(f"\n    accuracy {acc_skew:.4f}  (baseline {acc0:.4f})")
    print(f"    worst JS divergence {worst_skew:.4f}"
          f"{'  -> ALERT' if worst_skew > JS_ALERT else ''}")
    print("\n  SKEW NEEDS A TRAINING BASELINE. If the original training data is")
    print("  gone, you cannot configure skew detection at all - only drift.")
    print("  That constraint is an answer discriminator on the exam.")

    # ------------------------------------------------------------ 3. drift
    rule("3. DATA DRIFT - the world moved after deployment")

    # Drift compares serving-now against serving-earlier. Simulate an economy
    # where hours worked fell and capital gains rose.
    X_drift = X_s1.copy()
    X_drift[:, col(feature_names, "hours_per_week")] *= 0.75
    X_drift[:, col(feature_names, "capital_gain")] *= 3.0

    worst_drift = report("serving-earlier -> serving-now", X_s0, X_drift, feature_names)
    acc_drift = model.score(X_drift, y_s1)
    print(f"\n    accuracy {acc_drift:.4f}  (baseline {acc0:.4f})")
    print(f"    worst JS divergence {worst_drift:.4f}"
          f"{'  -> ALERT' if worst_drift > JS_ALERT else ''}")
    print("\n  DRIFT NEEDS NO TRAINING DATA - it baselines on earlier serving")
    print("  traffic, which you always have. That is why drift detection is")
    print("  available when skew detection is not.")

    # Say what actually happened to accuracy, rather than assuming drift hurt.
    if acc_drift >= acc0:
        print(f"\n  NOTE what accuracy did here: {acc_drift:.4f} against a baseline")
        print(f"  of {acc0:.4f} - the inputs moved substantially and the model was")
        print("  FINE. Input drift is a signal to go and look, not proof of damage.")
        print("  Alerting on divergence alone produces exactly this false positive.")
    else:
        print(f"\n  Accuracy fell to {acc_drift:.4f} from {acc0:.4f}, so here the")
        print("  drift alarm and the real damage agree. They do not always.")

    # -------------------------------------------------------- 4. the big one
    rule("4. CONCEPT DRIFT - the one input monitoring CANNOT see")

    # The inputs are untouched. Only the RELATIONSHIP changes: a competitor
    # cuts prices, and the rule connecting features to the outcome flips for
    # part of the population. Same X, different correct y.
    X_concept = X_s1.copy()                      # <- identical inputs
    y_concept = y_s1.copy()
    edu_i = col(feature_names, "education_num")
    affected = X_concept[:, edu_i] >= np.median(X_concept[:, edu_i])
    flip = affected & (rng.random(len(y_concept)) < 0.55)
    y_concept[flip] = 1 - y_concept[flip]        # the answer changed, not the input

    worst_concept = report("serving-earlier -> serving-now", X_s1, X_concept, feature_names)
    acc_concept = model.score(X_concept, y_concept)

    print(f"\n    accuracy {acc_concept:.4f}  (baseline {acc0:.4f})   "
          f"DOWN {(acc0 - acc_concept) * 100:.1f} points")
    print(f"    worst JS divergence {worst_concept:.4f}"
          f"{'  -> ALERT' if worst_concept > JS_ALERT else '  -> no alert'}")

    print("\n  READ THOSE TWO LINES TOGETHER. Every input distribution is")
    print("  unchanged - divergence is exactly zero, because the inputs are the")
    print("  SAME ROWS - and the model has lost a large chunk of its accuracy.")
    print("\n  No amount of input monitoring will ever fire. Detecting concept")
    print("  drift requires GROUND-TRUTH LABELS and continuous evaluation of")
    print("  accuracy, not distribution comparison.")
    print("\n  Exam shape: 'features look stable but accuracy is falling'")
    print("  -> concept drift, every time.")

    # ------------------------------------------- 5. attribution drift
    rule("5. FEATURE ATTRIBUTION DRIFT - right answers, changed reasons")

    base_imp = permutation_importance(model, X_s0, y_s0, n_repeats=5,
                                      random_state=SEED, n_jobs=1).importances_mean
    new_imp = permutation_importance(model, X_concept, y_concept, n_repeats=5,
                                     random_state=SEED, n_jobs=1).importances_mean

    def norm(v):
        v = np.clip(v, 0, None)
        return v / v.sum() if v.sum() else v

    b, n = norm(base_imp), norm(new_imp)
    attrib_js = float(jensenshannon(b, n, base=2))

    print(f"    {'feature':<18} {'before':>9} {'after':>9} {'change':>9}")
    for i, fname in enumerate(feature_names):
        print(f"    {fname:<18} {b[i]:>9.3f} {n[i]:>9.3f} {n[i] - b[i]:>+9.3f}")
    print(f"\n    attribution JS divergence {attrib_js:.4f}")
    print("\n  This is the subtle one: the model can keep roughly its accuracy")
    print("  while relying on different features - right answers for changed")
    print("  reasons, which is a leading indicator of trouble rather than a")
    print("  failure you can already see.")

    # ------------------------------------------------------------ 6. summary
    rule("6. WHICH MONITOR WOULD HAVE CAUGHT WHICH")

    rows = [
        ("training-serving skew", worst_skew, acc_skew, "training data baseline"),
        ("data drift",            worst_drift, acc_drift, "earlier serving traffic"),
        ("concept drift",         worst_concept, acc_concept, "GROUND-TRUTH LABELS"),
    ]
    print(f"  {'failure':<24} {'input divergence':>17} {'accuracy':>10}  needs")
    print("  " + "-" * 72)
    print(f"  {'(baseline)':<24} {'-':>17} {acc0:>10.4f}  -")
    for name, div, acc, needs in rows:
        caught = "caught" if div > JS_ALERT else "MISSED"
        print(f"  {name:<24} {div:>13.4f} {caught:>4} {acc:>10.4f}  {needs}")

    print("\n  The bottom row is the point of this file. Input monitoring MISSES")
    print("  concept drift completely, and concept drift is the failure that")
    print("  quietly destroys a model that looks healthy on every dashboard.")

    # ---------------------------------------------------------------- figure
    use_house_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    names = ["baseline", "skew", "data drift", "concept drift"]
    divs = [0.0, worst_skew, worst_drift, worst_concept]
    accs = [acc0, acc_skew, acc_drift, acc_concept]
    colours = [PALETTE["blue"], PALETTE["orange"], PALETTE["orange"], PALETTE["vermillion"]]

    ax1.bar(names, divs, color=colours)
    ax1.axhline(JS_ALERT, color=PALETTE["vermillion"], ls="--", lw=1.2,
                label=f"alert threshold {JS_ALERT}")
    ax1.set_ylabel("worst input JS divergence")
    ax1.set_title("What input monitoring sees")
    ax1.legend(frameon=False, fontsize=9)
    ax1.tick_params(axis="x", rotation=20)

    ax2.bar(names, accs, color=colours)
    ax2.axhline(acc0, color=PALETTE["blue"], ls="--", lw=1.2, label="baseline accuracy")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_title("What is actually happening")
    ax2.legend(frameon=False, fontsize=9)
    ax2.tick_params(axis="x", rotation=20)

    save_figure(fig, OUT / "skew_and_drift.png")
    print(f"\n  figure -> {OUT / 'skew_and_drift.png'}")

    rule("THE FOUR, IN ONE TABLE")
    print("""  term                     compares                        detect with
  ----------------------------------------------------------------------------
  training-serving skew    serving vs TRAINING data        JS / L-infinity,
                                                           needs training baseline
  prediction / data drift  serving NOW vs serving EARLIER  JS / L-infinity,
                                                           no training data needed
  concept drift            the X -> y relationship         LABELS + continuous
                                                           evaluation. Input
                                                           monitoring cannot see it.
  feature attribution      which features the model uses   attribution monitoring
    drift

  Model Monitoring's metrics: Jensen-Shannon for numerical features,
  L-infinity for categorical.""")


if __name__ == "__main__":
    main()
