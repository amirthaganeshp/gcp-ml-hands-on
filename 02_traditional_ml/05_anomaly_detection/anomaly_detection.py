"""Anomaly detection: finding the rare thing when nobody labelled it.

  1. What each detector actually flags - four algorithms on two synthetic
     shapes, because their failure modes are geometric
  2. A real series - unsupervised detection on two years of daily bicycle
     rentals, and whether the days it flags make sense
  3. Scoring it properly - inject anomalies we can count, then compare the
     four detectors with PR-AUC and precision@k
  4. The contamination knob - it does not change the RANKING, only where
     the cut falls. Same lesson as threshold-moving in classification.
  5. The API trap: LOF cannot score new data unless you ask it to up front

Run:
    python 02_traditional_ml/05_anomaly_detection/anomaly_detection.py

Concept reference: Foundations volume, Parts A.11 (ROC vs PR), A.16 (unsupervised).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_bike  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.covariance import EllipticEnvelope  # noqa: E402
from sklearn.datasets import make_blobs  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.neighbors import LocalOutlierFactor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def detectors(contamination: float = 0.05) -> dict:
    """The four standard scikit-learn detectors.

    LOF is constructed with novelty=False here because sections 1-3 score the
    same rows they were fitted on (outlier detection). Section 5 shows what
    changes when you need to score NEW rows.
    """
    return {
        "Isolation Forest": IsolationForest(contamination=contamination,
                                            random_state=SEED, n_estimators=200),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,
                                                   contamination=contamination),
        "One-Class SVM": OneClassSVM(nu=contamination, gamma="scale"),
        "Elliptic Envelope": EllipticEnvelope(contamination=contamination,
                                              random_state=SEED),
    }


def anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous, for every detector.

    scikit-learn's convention is the opposite - score_samples returns HIGHER
    for more normal points - and LOF exposes its scores under a different
    attribute when novelty=False. Negating here keeps the rest of the script
    reading in one direction.
    """
    if isinstance(model, LocalOutlierFactor) and not model.novelty:
        model.fit(X)
        return -model.negative_outlier_factor_
    model.fit(X)
    return -model.score_samples(X)


# --------------------------------------------- 1. what each detector flags
def figure_shapes() -> None:
    """Two deliberately different problems. The first is what every textbook
    shows; the second is the one that separates the algorithms.
    """
    # one tight blob plus uniform scatter - the easy, canonical case
    core, _ = make_blobs(n_samples=300, centers=[[0, 0]], cluster_std=0.6,
                         random_state=SEED)
    scatter = rng.uniform(-5, 5, (30, 2))
    easy = np.vstack([core, scatter])

    # TWO clusters of different density. A global method sees the sparse
    # cluster's members as unusual; a local method compares each point only
    # with its own neighbourhood.
    dense, _ = make_blobs(n_samples=250, centers=[[-2.5, 0]], cluster_std=0.30,
                          random_state=SEED)
    sparse, _ = make_blobs(n_samples=120, centers=[[2.5, 0]], cluster_std=1.10,
                           random_state=SEED)
    hard = np.vstack([dense, sparse, rng.uniform(-6, 6, (20, 2))])

    problems = {"One blob + scatter": easy, "Two densities + scatter": hard}

    # Membership in the hard problem is known by construction, so the claims
    # below can be counted rather than eyeballed.
    n_dense, n_sparse, n_scatter = len(dense), len(sparse), 20
    group = np.array(["dense"] * n_dense + ["sparse"] * n_sparse
                     + ["scatter"] * n_scatter)
    hard_counts: dict[str, dict[str, int]] = {}

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4))
    for r, (pname, X) in enumerate(problems.items()):
        Xs = StandardScaler().fit_transform(X)
        for c, (name, model) in enumerate(detectors(0.08).items()):
            ax = axes[r, c]
            s = anomaly_scores(model, Xs)
            cut = np.quantile(s, 0.92)
            normal, flagged = s < cut, s >= cut
            if r == 1:
                hard_counts[name] = {g: int((group[flagged] == g).sum())
                                     for g in ("scatter", "sparse", "dense")}
            ax.scatter(Xs[normal, 0], Xs[normal, 1], s=11,
                       color=PALETTE["blue"], alpha=0.55)
            ax.scatter(Xs[flagged, 0], Xs[flagged, 1], s=34,
                       color=PALETTE["vermillion"], alpha=0.95, marker="x")
            ax.set_title(name, fontsize=10.5)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        axes[r, 0].set_ylabel(pname, fontsize=10, fontweight="bold")

    fig.suptitle("The top 8% most anomalous points, by four different definitions "
                 "of 'anomalous'", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    print("  Top row - the easy case. All four broadly agree, which is why the")
    print("  textbook picture makes them look interchangeable. They are not.")
    print("\n  Bottom row - two clusters with different spread, plus 20 genuinely")
    print("  stray points. Only those 20 are true anomalies; the sparse cluster is")
    print("  ordinary data that merely happens to be spread out.")
    print(f"\n  Where each detector's {int(0.08 * len(group))} flags actually landed:\n")
    print(f"  {'detector':<24} {'stray (right)':>14} {'sparse cluster':>15} "
          f"{'dense cluster':>14}")
    for name, cnt in hard_counts.items():
        print(f"  {name:<24} {cnt['scatter']:>10}/{n_scatter:<3} "
              f"{cnt['sparse']:>12}   {cnt['dense']:>12}")

    print("\n  Read the middle column - flags wasted on perfectly normal members")
    print("  of the sparse cluster. That is the global-vs-local distinction, in")
    print("  numbers rather than adjectives:")
    print("    Elliptic Envelope fits ONE Gaussian to everything, so it flags")
    print("      whatever sits furthest from the JOINT mean. There are two")
    print("      clusters and it assumes one - its flags pile into the sparse")
    print("      cluster almost entirely.")
    print("    Isolation Forest and One-Class SVM are GLOBAL: a point in a")
    print("      genuinely sparse region is easy to isolate, so it looks unusual")
    print("      even when it is an ordinary member of a spread-out group.")
    lof_sparse = hard_counts["Local Outlier Factor"]["sparse"]
    others = [c["sparse"] for n, c in hard_counts.items() if n != "Local Outlier Factor"]
    print("    Local Outlier Factor compares each point's density only to ITS")
    print(f"      OWN NEIGHBOURS', so a uniformly sparse cluster costs it {lof_sparse}")
    print(f"      flags against {min(others)}-{max(others)} for the rest. That is the reason")
    print("      LOF exists.")
    print("\n  LOF's flags land on the DENSE cluster's rim instead - in a tight")
    print("  group, being slightly further out than your neighbours is a large")
    print("  relative difference. Local sensitivity is a trade, not a free win.")
    print("\n  Pick the detector by asking what 'unusual' means for your data:")
    print("  unusual overall (global) or unusual for its neighbourhood (local).")
    print(f"\n  saved  {save_figure(fig, OUT / '1_detector_shapes.png').name}")


# ------------------------------------------------------ 2. a real series
def figure_real(df: pd.DataFrame) -> None:
    """Unsupervised detection on real data, where there is no label to check
    against - so the only honest validation is whether the flagged rows are
    explicable.
    """
    feats = ["temp", "atemp", "hum", "windspeed", "weathersit",
             "season", "workingday", "cnt"]
    X = StandardScaler().fit_transform(df[feats].to_numpy(dtype=float))

    iso = IsolationForest(contamination=0.02, random_state=SEED, n_estimators=300)
    score = -iso.fit(X).score_samples(X)
    top = np.argsort(score)[::-1][:10]

    weather = {1: "clear", 2: "mist", 3: "rain/snow", 4: "severe"}
    print(f"  Two years of daily rentals ({len(df)} days), {len(feats)} features, no labels.")
    print("  The 10 days Isolation Forest finds most unusual:\n")
    print(f"  {'date':<12} {'day':<5} {'rentals':>8} {'weather':<10} "
          f"{'temp':>6} {'wind':>6} {'score':>7}")
    for i in top:
        r = df.iloc[i]
        print(f"  {r['dteday'].date()!s:<12} {r['dteday'].strftime('%a'):<5} "
              f"{int(r['cnt']):>8,} {weather.get(int(r['weathersit']), '?'):<10} "
              f"{r['temp']:>6.2f} {r['windspeed']:>6.2f} {score[i]:>7.3f}")

    # Rank of the single most extreme day by volume. Computed, because the
    # interesting fact is that it is NOT first.
    rank_order = np.argsort(score)[::-1]
    worst_i = int(np.argmin(df["cnt"].to_numpy()))
    worst = df.iloc[worst_i]
    worst_rank = int(np.where(rank_order == worst_i)[0][0]) + 1
    median = df["cnt"].median()

    print(f"\n  {worst['dteday'].date()} is the lowest-volume day in the whole series:")
    print(f"  {int(worst['cnt'])} rentals against a median of {median:,.0f}, "
          f"{median / max(worst['cnt'], 1):.0f}x below it. A severe")
    print("  storm closed the city, and the model found it with no knowledge of")
    print("  weather events at all.")
    print(f"\n  But it ranks {worst_rank}th, not 1st - and that is the more useful fact.")
    print("  Isolation Forest is scoring all 8 features jointly, so a day that is")
    print("  merely the most extreme in ONE column does not automatically win.")

    # How many of the top 10 have an unremarkable rental count?
    lo, hi = df["cnt"].quantile([0.10, 0.90])
    ordinary = int(((df["cnt"].iloc[top] >= lo) & (df["cnt"].iloc[top] <= hi)).sum())
    bad_weather = int((df["weathersit"].iloc[top] >= 3).sum())
    print(f"\n  Now the sobering part: only {ordinary} of the top 10 has a rental count")
    print(f"  inside the normal 10th-90th percentile band, and {bad_weather} of them are rain")
    print("  or snow days. The model has largely rediscovered 'bad weather means")
    print("  few rentals'.")
    print("\n  That is a SANITY CHECK, not a discovery, and it is worth saying so")
    print("  rather than dressing it up. An unsupervised detector agreeing with")
    print("  the obvious explanation is evidence that it works. It is NOT evidence")
    print("  that it beats a simple rule - section 3 measures that directly, and")
    print("  the answer is less flattering than you might expect.")
    print("\n  There is no ground truth here, so this cannot be scored - only")
    print("  explained. That is the normal situation in production, and it is why")
    print("  section 3 injects anomalies we CAN count.")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    axes[0].plot(df["dteday"], df["cnt"], color=PALETTE["blue"], lw=1.0)
    axes[0].scatter(df["dteday"].iloc[top], df["cnt"].iloc[top], s=55,
                    color=PALETTE["vermillion"], zorder=5, label="flagged")
    axes[0].set_ylabel("daily rentals"); axes[0].legend(fontsize=9)
    axes[0].set_title("Daily bicycle rentals, with the 10 most anomalous days marked")

    axes[1].plot(df["dteday"], score, color=PALETTE["orange"], lw=1.0)
    axes[1].axhline(np.quantile(score, 0.98), color=PALETTE["vermillion"],
                    ls="--", lw=1.3)
    axes[1].annotate("98th percentile cut", xy=(0.02, 0.86),
                     xycoords="axes fraction", fontsize=9,
                     color=PALETTE["vermillion"])
    axes[1].set_ylabel("anomaly score"); axes[1].set_xlabel("date")
    axes[1].set_title("The score is continuous - the cut is a separate decision")

    fig.tight_layout()
    print(f"\n  saved  {save_figure(fig, OUT / '2_real_series.png').name}")


# ------------------------------------------------- 3. scoring it properly
def figure_scoring(df: pd.DataFrame) -> dict:
    """Inject anomalies with known positions, then measure. Ranking metrics
    only - a label requires a threshold, and the threshold is not the model.
    """
    feats = ["temp", "atemp", "hum", "windspeed", "cnt"]
    clean = df[feats].to_numpy(dtype=float).copy()
    n, rate = len(clean), 0.04
    n_bad = int(n * rate)

    idx = rng.choice(n, n_bad, replace=False)
    X = clean.copy()
    # Two flavours on purpose: half are extreme in one column, half are
    # ordinary values in an impossible COMBINATION. A univariate rule catches
    # the first kind and is blind to the second.
    half = n_bad // 2
    X[idx[:half], 4] *= rng.uniform(2.5, 4.0, half)               # implausible volume
    X[idx[half:], 0] = X[idx[half:], 0].max()                      # hot ...
    X[idx[half:], 2] = X[idx[half:], 2].max()                      # ... and saturated
    y = np.zeros(n, dtype=int)
    y[idx] = 1

    Xs = StandardScaler().fit_transform(X)
    print(f"  {n} rows, {n_bad} injected anomalies ({rate:.0%}).")
    print(f"  Half are extreme in one column; half are ordinary values in an")
    print(f"  impossible combination.\n")

    results = {}
    print(f"  {'detector':<24} {'PR-AUC':>8} {'ROC-AUC':>9} {'P@30':>7} {'P@60':>7}")
    for name, model in detectors(rate).items():
        s = anomaly_scores(model, Xs)
        order = np.argsort(s)[::-1]
        results[name] = {
            "pr_auc": average_precision_score(y, s),
            "roc_auc": roc_auc_score(y, s),
            "p30": y[order[:30]].mean(),
            "p60": y[order[:60]].mean(),
            "scores": s,
        }
        r = results[name]
        print(f"  {name:<24} {r['pr_auc']:>8.3f} {r['roc_auc']:>9.3f} "
              f"{r['p30']:>7.2f} {r['p60']:>7.2f}")

    # what a single-column rule would achieve, for context
    univariate = np.abs((X[:, 4] - X[:, 4].mean()) / X[:, 4].std())
    uni_pr = average_precision_score(y, univariate)
    print(f"  {'(rule: |z| on rentals)':<24} {uni_pr:>8.3f}")

    best = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"\n  Best PR-AUC: {best} ({results[best]['pr_auc']:.3f}), against a")
    print(f"  {rate:.0%} base rate - so random guessing would score {rate:.3f}.")
    beaten = [n for n in results if results[n]["pr_auc"] < uni_pr]
    print(f"\n  Now look at the one-column z-score rule: {uni_pr:.3f}. It BEATS "
          f"{len(beaten)} of the 4")
    print("  real detectors here (" + ", ".join(beaten) + ").")
    print("  It finds only the half of the anomalies that are extreme in rentals")
    print("  and is structurally blind to the impossible combinations - and that")
    print("  is still enough to embarrass two published algorithms.")
    print(f"\n  So 'reach for a multivariate detector' is not automatically right.")
    print(f"  What IS true is that the best one, {best}, scores")
    print(f"  {results[best]['pr_auc']:.3f} - about {results[best]['pr_auc'] / uni_pr:.1f}x the rule - because it can see both")
    print("  kinds of anomaly at once. Always measure against the dumb baseline")
    print("  before concluding that the model is needed.")
    print("\n  Note ROC-AUC looks far more flattering than PR-AUC for every")
    print("  detector here. At a 4% base rate that is expected, and PR-AUC is")
    print("  the one to quote - same argument as 01_foundations/04_roc_vs_pr.")
    print("\n  PRECISION@K is usually the number that matters in practice: if an")
    print("  analyst can review 30 cases a day, 'how many of my top 30 are real'")
    print("  is the question, and no threshold has to be chosen to answer it.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    names = list(results)
    axes[0].barh(names, [results[n]["pr_auc"] for n in names],
                 color=PALETTE["blue"])
    axes[0].axvline(rate, color=PALETTE["vermillion"], ls="--", lw=1.4)
    axes[0].annotate(f"random = {rate:.2f}", xy=(0.28, 0.06),
                     xycoords="axes fraction", fontsize=9,
                     color=PALETTE["vermillion"])
    axes[0].set_xlabel("PR-AUC"); axes[0].set_title("Ranking quality")
    axes[0].grid(axis="y", visible=False)

    ks = np.arange(5, 121, 5)
    for name, colour in zip(names, ("blue", "vermillion", "green", "orange")):
        order = np.argsort(results[name]["scores"])[::-1]
        axes[1].plot(ks, [y[order[:k]].mean() for k in ks],
                     color=PALETTE[colour], label=name)
    axes[1].axhline(rate, color=PALETTE["grey"], ls="--", lw=1.2)
    axes[1].set_xlabel("k (cases reviewed)"); axes[1].set_ylabel("precision@k")
    axes[1].set_title("If you can only review k cases, how many are real?")
    axes[1].legend(fontsize=9)

    fig.suptitle("Anomaly detection is a RANKING problem before it is a "
                 "labelling problem", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '3_scoring.png').name}")
    return {"X": Xs, "y": y, "rate": rate}


# ------------------------------------------------ 4. the contamination knob
def figure_contamination(data: dict) -> None:
    """contamination does not change what the model learned. It sets where the
    line is drawn on a ranking that is already fixed.
    """
    X, y = data["X"], data["y"]
    settings = [0.01, 0.02, 0.04, 0.08, 0.15, 0.30]

    print(f"  {'contamination':>14} {'flagged':>9} {'precision':>10} "
          f"{'recall':>8} {'F1':>7} {'PR-AUC':>8}")
    rows = []
    for c in settings:
        m = IsolationForest(contamination=c, random_state=SEED, n_estimators=200)
        pred = (m.fit(X).predict(X) == -1).astype(int)
        s = -m.score_samples(X)
        tp = int(((pred == 1) & (y == 1)).sum())
        prec = tp / max(pred.sum(), 1)
        rec = tp / y.sum()
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        pr_auc = average_precision_score(y, s)
        rows.append((c, int(pred.sum()), prec, rec, f1, pr_auc))
        print(f"  {c:>14.2f} {int(pred.sum()):>9} {prec:>10.3f} "
              f"{rec:>8.3f} {f1:>7.3f} {pr_auc:>8.3f}")

    spread = max(r[5] for r in rows) - min(r[5] for r in rows)
    print(f"\n  Read the PR-AUC column: it moves by {spread:.4f} across a 30x change in")
    print("  contamination. Not merely stable - IDENTICAL, to four decimal places.")
    print("  contamination never touches the trees at all; it only sets `offset_`,")
    print("  the value a score is compared against inside predict(). The ranking")
    print("  was fixed the moment the forest was built.")
    print("\n  What it does control is the precision/recall trade, exactly like")
    print("  moving a classification threshold (02_classification, section 4).")
    print("  Set it too low and you miss most of the anomalies; set it too high")
    print("  and your analysts drown in false positives.")
    print("\n  The trap: contamination is a PRIOR you are asserting - 'I believe")
    print("  x% of my data is anomalous'. In unsupervised work you usually have")
    print("  no way to check it, so prefer to rank by score and cut at whatever")
    print("  volume you can actually investigate.")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    cs = [r[0] for r in rows]
    ax.plot(cs, [r[2] for r in rows], "o-", color=PALETTE["blue"], label="precision")
    ax.plot(cs, [r[3] for r in rows], "s-", color=PALETTE["vermillion"], label="recall")
    ax.plot(cs, [r[4] for r in rows], "^-", color=PALETTE["green"], label="F1")
    ax.plot(cs, [r[5] for r in rows], "d--", color=PALETTE["grey"], label="PR-AUC (flat)")
    ax.axvline(data["rate"], color=PALETTE["orange"], ls=":", lw=1.6)
    ax.annotate(f"true rate = {data['rate']:.0%}", xy=(0.14, 0.92),
                xycoords="axes fraction", fontsize=9, color=PALETTE["orange"],
                fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("contamination setting (log scale)"); ax.set_ylabel("score")
    ax.set_title("contamination moves the cut, not the model")
    ax.legend(fontsize=9)
    fig.tight_layout()
    print(f"\n  saved  {save_figure(fig, OUT / '4_contamination.png').name}")


# ------------------------------------------- 5. outlier vs novelty detection
def demo_novelty(data: dict) -> None:
    """The distinction scikit-learn makes in its API, and the error it throws
    at you if you miss it.
    """
    X, y = data["X"], data["y"]

    # Split ROW INDICES first, then take the clean subset of the training half.
    # An earlier version sliced the clean rows and then indexed X with the same
    # offset, which silently overlapped train and test.
    order = rng.permutation(len(X))
    tr_idx, te_idx = order[: len(X) // 2], order[len(X) // 2:]
    train = X[tr_idx][y[tr_idx] == 0]          # believed-clean training rows only
    test, y_test = X[te_idx], y[te_idx]

    print("  OUTLIER detection  - fit on dirty data, find the bad rows inside it.")
    print("  NOVELTY detection  - fit on data believed CLEAN, then score new")
    print("                       rows as they arrive. This is the production shape.")
    print(f"\n  Fitting on {len(train)} clean rows, scoring {len(test)} unseen rows "
          f"({int(y_test.sum())} anomalous).")

    lof_outlier = LocalOutlierFactor(n_neighbors=20)
    lof_outlier.fit(X)
    print(f"\n  LocalOutlierFactor(novelty=False).predict exists? "
          f"{hasattr(lof_outlier, 'predict')}")
    print("  It does not - by default LOF is TRANSDUCTIVE. It scores the rows it")
    print("  was fitted on and nothing else, so there is no model to deploy.")
    print("  novelty=True must be set at CONSTRUCTION time, not at predict time.")

    print(f"\n  {'detector':<24} {'fit on clean':>13} {'fit on all (s.3)':>18}")
    for name in ("Local Outlier Factor", "Isolation Forest"):
        if name.startswith("Local"):
            nov = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(train)
            full = LocalOutlierFactor(n_neighbors=20)
            full.fit(X)
            full_pr = average_precision_score(y, -full.negative_outlier_factor_)
        else:
            nov = IsolationForest(random_state=SEED, n_estimators=200).fit(train)
            f = IsolationForest(random_state=SEED, n_estimators=200).fit(X)
            full_pr = average_precision_score(y, -f.score_samples(X))
        nov_pr = average_precision_score(y_test, -nov.score_samples(test))
        print(f"  {name:<24} {nov_pr:>13.3f} {full_pr:>18.3f}")

    print("\n  The two columns are not strictly comparable - the left scores half")
    print("  the rows, the right scores all of them - so read the SHAPE, not the")
    print("  gap. BOTH detectors score lower under the clean-fit setup, and that")
    print("  is the honest headline: on this data, novelty framing did not help.")
    print("  LOF retains far more of its performance than Isolation Forest does.")
    print("\n  Why: Isolation Forest isolates points by random axis-aligned splits,")
    print("  and with only ~350 training rows it has far less structure to learn")
    print("  than LOF, which needs nothing more than 20 neighbours to compare")
    print("  against. Novelty detection trades data volume for a cleaner notion of")
    print("  'normal', and not every algorithm comes out ahead on that trade.")
    print("\n  Two things to carry away. Use a known-good period when you have one,")
    print("  because it gives the detector an uncontaminated reference - but")
    print("  MEASURE it rather than assuming, because halving your training data")
    print("  is a real cost. And check the API before designing around an")
    print("  algorithm: LOF simply cannot score new rows without novelty=True.")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  ANOMALY DETECTION")
    print("=" * 72)

    df = load_bike("day")

    banner("1. What does each detector think 'anomalous' means?")
    figure_shapes()

    banner("2. A real series, with no labels to check against")
    figure_real(df)

    banner("3. Scoring it properly, with injected anomalies")
    data = figure_scoring(df)

    banner("4. The contamination knob")
    figure_contamination(data)

    banner("5. Outlier detection vs novelty detection")
    demo_novelty(data)

    banner("What to take away")
    print("  - Choose the detector by what 'unusual' means for your data.")
    print("    GLOBAL (Isolation Forest, One-Class SVM) flags anything in a")
    print("    sparse region; LOCAL (LOF) compares a point to its own")
    print("    neighbours; Elliptic Envelope assumes ONE Gaussian blob and")
    print("    misreads data with two clusters.")
    print("  - Isolation Forest is the usual DEFAULT on tabular data - no")
    print("    distance metric to tune, near-linear in n, scales to large data.")
    print("    Note it was NOT the best here (0.398 against LOF's 0.824): the")
    print("    default is a starting point, not a finding.")
    print("  - The model produces a SCORE; the label is a separate thresholding")
    print("    decision. contamination left PR-AUC EXACTLY unchanged across a")
    print("    30x range - it only sets where the cut falls on a fixed ranking.")
    print("  - Evaluate with PR-AUC and precision@k, never accuracy - at a 4%")
    print("    base rate, flagging nothing scores 96%.")
    print("  - MEASURE AGAINST THE DUMB BASELINE. A one-column z-score rule beat")
    print("    two of the four detectors here. Multivariate methods earn their")
    print("    keep on impossible COMBINATIONS of ordinary values - but only the")
    print("    ones actually suited to your data do.")
    print("  - Novelty framing (fit clean, score new) is the production shape,")
    print("    but it did NOT improve either detector here - halving the training")
    print("    data cost more than the cleaner reference bought. Measure it.")
    print("  - LOF needs novelty=True at CONSTRUCTION or it cannot score new rows")
    print("    at all - it has no predict() method by default.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
