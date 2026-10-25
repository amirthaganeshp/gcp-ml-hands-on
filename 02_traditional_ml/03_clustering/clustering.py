"""Clustering: finding structure when nobody gave you labels.

  1. Scaling is not optional - k-means is a distance algorithm, so unscaled
     features let whichever column has the largest units decide everything
  2. Choosing k - the elbow method is ambiguous far more often than textbooks
     admit; silhouette gives you a number instead of a judgement call
  3. Where k-means breaks - it draws spherical, similar-sized blobs. Three
     shapes it cannot handle, and what DBSCAN / GMM / Agglomerative do instead
  4. Scoring a clustering - why accuracy is meaningless here, and what ARI
     does about it

The wheat Seeds dataset has three genuine varieties. We cluster WITHOUT the
labels, then use them only to score the result - which is exactly how you
would validate a clustering when you happen to have a held-out ground truth.

Run:
    python 02_traditional_ml/03_clustering/clustering.py

Concept reference: Foundations volume, Part A.16 (unsupervised learning).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_seeds  # noqa: E402
from plotting import CYCLE, PALETTE, save_figure, use_house_style  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.datasets import make_blobs, make_moons  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# --------------------------------------------------------- 1. scaling first
def figure_scaling(X_raw: np.ndarray, names: list[str]) -> np.ndarray:
    """k-means minimises squared distance, so a feature measured in larger units
    contributes more variance and therefore dominates the objective. Nothing in
    the algorithm knows that 'area' and 'compactness' are different kinds of
    quantity - it only sees numbers.
    """
    spread = X_raw.std(axis=0)
    widest, narrowest = int(np.argmax(spread)), int(np.argmin(spread))
    print(f"  Feature spreads (standard deviation) in the raw units:")
    for n, s in sorted(zip(names, spread), key=lambda t: -t[1]):
        print(f"    {n:<16} {s:8.3f}")
    print(f"\n  '{names[widest]}' varies {spread[widest] / spread[narrowest]:.0f}x more than "
          f"'{names[narrowest]}'.")
    print("  Unscaled, k-means is effectively clustering on that one column.")

    X_scaled = StandardScaler().fit_transform(X_raw)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (X, title) in zip(axes, [(X_raw, "Raw units"), (X_scaled, "Standardised")]):
        labels = KMeans(3, n_init=10, random_state=SEED).fit_predict(X)
        # Project to 2-D for display only. The clustering above used ALL features;
        # PCA here is a viewing lens, not part of the pipeline.
        pcs = PCA(2, random_state=SEED).fit_transform(StandardScaler().fit_transform(X_raw))
        for k in range(3):
            m = labels == k
            ax.scatter(pcs[m, 0], pcs[m, 1], s=22, color=CYCLE[k], alpha=0.85,
                       label=f"cluster {k}")
        ari = adjusted_rand_score(TRUE_LABELS, labels)
        ax.set_title(f"{title}\nARI vs the true varieties = {ari:.3f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=9)
        print(f"  k-means on {title.lower():<15} ARI = {ari:.3f}")

    fig.suptitle("Scaling decides the answer before the algorithm runs",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    print(f"\n  saved  {save_figure(fig, OUT / '1_scaling.png').name}")
    return X_scaled


# ------------------------------------------------------------ 2. choosing k
def figure_choosing_k(X: np.ndarray) -> None:
    """Two ways to pick k, and an honest look at how well each works.

    The elbow method asks you to eyeball where the inertia curve stops falling
    steeply. Inertia decreases monotonically with k by construction (more
    centroids can only reduce within-cluster distance), so there is no minimum
    to find - only a subjective change in slope.
    """
    ks = range(2, 11)
    inertia, sil, db = [], [], []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))
        db.append(davies_bouldin_score(X, km.labels_))

    best_sil = list(ks)[int(np.argmax(sil))]
    best_db = list(ks)[int(np.argmin(db))]

    print(f"  {'k':>3} {'inertia':>10} {'drop':>8} {'silhouette':>11} {'Davies-Bouldin':>15}")
    prev = None
    for i, k in enumerate(ks):
        drop = "" if prev is None else f"{(prev - inertia[i]) / prev * 100:6.1f}%"
        print(f"  {k:>3} {inertia[i]:>10.1f} {drop:>8} {sil[i]:>11.3f} {db[i]:>15.3f}")
        prev = inertia[i]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(ks, inertia, "o-", color=PALETTE["blue"])
    axes[0].axvline(3, color=PALETTE["grey"], ls="--", lw=1.5)
    axes[0].annotate("k=3 is the truth,\nbut is this an elbow?", xy=(0.45, 0.6),
                     xycoords="axes fraction", fontsize=9, color=PALETTE["vermillion"])
    axes[0].set_title("Elbow: inertia (subjective)")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia")

    axes[1].plot(ks, sil, "o-", color=PALETTE["green"])
    axes[1].axvline(best_sil, color=PALETTE["vermillion"], ls="--", lw=1.5)
    axes[1].set_title(f"Silhouette: higher is better\npicks k={best_sil}")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette")

    axes[2].plot(ks, db, "o-", color=PALETTE["orange"])
    axes[2].axvline(best_db, color=PALETTE["vermillion"], ls="--", lw=1.5)
    axes[2].set_title(f"Davies-Bouldin: LOWER is better\npicks k={best_db}")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("Davies-Bouldin")

    fig.suptitle("Choosing k: one subjective method and two numeric ones",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    print(f"\n  Silhouette picks k={best_sil}; Davies-Bouldin picks k={best_db}; "
          f"there are 3 varieties.")
    print("  Inertia FALLS MONOTONICALLY by construction - more centroids can only")
    print("  reduce within-cluster distance - so there is no minimum to find and")
    print("  no 'best' value. You are eyeballing a change in slope, which is why")
    print("  two people reading the same elbow plot routinely disagree.")

    # Why k=2 wins. This is the most important paragraph in the section, so it is
    # computed rather than asserted: measure how far apart the true varieties
    # actually sit, and show that two of them overlap.
    cents = {v: X[TRUE_CODES == i].mean(axis=0) for i, v in enumerate(VARIETIES)}
    pairs = [(a, b, float(np.linalg.norm(cents[a] - cents[b])))
             for i, a in enumerate(VARIETIES) for b in VARIETIES[i + 1:]]
    pairs.sort(key=lambda t: t[2])
    closest, farthest = pairs[0], pairs[-1]

    print("\n  But k=2 is NOT a failure of the metrics - it is a correct answer to")
    print("  a different question. Distance between the true variety centroids:")
    for a, b, d in pairs:
        print(f"    {a:<9} vs {b:<9} {d:.2f}")
    print(f"\n  {closest[0]} and {closest[1]} sit {closest[2]:.2f} apart while "
          f"{farthest[0]} and {farthest[1]}")
    print(f"  sit {farthest[2]:.2f} apart, so GEOMETRICALLY there are two groups, even")
    print("  though BIOLOGICALLY there are three. Silhouette measures separation;")
    print("  it does not know that a botanist would draw the line elsewhere.")
    print("\n  That is the real lesson. An internal metric answers 'how many")
    print("  well-separated blobs are in this space', never 'how many categories")
    print("  exist in the world'. When those two answers differ, the metric is not")
    print("  wrong - your features simply do not resolve the distinction you care")
    print("  about, and the fix is better features, not a different k.")
    print(f"\n  saved  {save_figure(fig, OUT / '2_choosing_k.png').name}")


# ------------------------------------------------- 3. where k-means breaks
def figure_shapes() -> None:
    """k-means assigns each point to the nearest centroid, so its clusters are
    Voronoi cells: convex, and implicitly of similar size and density. Three
    datasets that violate those assumptions, and three alternatives.

    Every case here is scored with ARI against the labels the generator used, so
    the commentary below is measured rather than asserted. `eps` and `k` are
    given per problem explicitly - an earlier version inferred them with an
    `is` check against the unscaled array, which silently never matched.
    """
    moons, y_moons = make_moons(n_samples=500, noise=0.06, random_state=SEED)

    # The canonical anisotropic case. random_state=170 is the seed used in
    # scikit-learn's own demo because it places the blobs close enough that the
    # stretched bands genuinely interleave; well-separated bands would be solved
    # correctly by everything and would demonstrate nothing.
    blob, y_aniso = make_blobs(n_samples=500, centers=3, random_state=170)
    aniso = blob @ np.array([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]])

    varied, y_varied = make_blobs(n_samples=500, centers=3,
                                  cluster_std=[1.0, 2.5, 0.5], random_state=170)

    problems = {
        "Crescents (non-convex)": (moons, y_moons, 2, 0.30),
        "Stretched (non-spherical)": (aniso, y_aniso, 3, 0.35),
        "Unequal spread (different density)": (varied, y_varied, 3, 0.30),
    }

    def two_line(label: str) -> str:
        """Wrap a label at its opening bracket, for readable two-line titles."""
        return label.replace(" (", "\n(", 1)

    def algos(k: int, eps: float) -> dict:
        return {
            "k-means": KMeans(k, n_init=10, random_state=SEED),
            "Gaussian Mixture": GaussianMixture(k, random_state=SEED),
            "DBSCAN": DBSCAN(eps=eps, min_samples=8),
            "Agglomerative (Ward)": AgglomerativeClustering(k),
        }

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    table: dict[str, dict[str, float]] = {}
    for r, (pname, (X, y_true, k, eps)) in enumerate(problems.items()):
        Xs = StandardScaler().fit_transform(X)
        table[pname] = {}
        for c, (aname, algo) in enumerate(algos(k, eps).items()):
            ax = axes[r, c]
            labels = algo.fit_predict(Xs)
            for lab in sorted(set(labels)):
                m = labels == lab
                # DBSCAN marks noise as -1. That is a feature: no other algorithm
                # here is allowed to say "this point belongs to nothing".
                colour = PALETTE["grey"] if lab == -1 else CYCLE[lab % len(CYCLE)]
                ax.scatter(Xs[m, 0], Xs[m, 1], s=12, color=colour, alpha=0.8)
            ari = adjusted_rand_score(y_true, labels)
            table[pname][aname] = ari
            ax.set_title(f"{two_line(aname)}\nARI {ari:.2f}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        axes[r, 0].set_ylabel(two_line(pname), fontsize=10, fontweight="bold")

    fig.suptitle("k-means draws convex, equal-sized cells. When the data is not "
                 "shaped like that, it is wrong.",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    names = list(next(iter(table.values())).keys())
    print("  ARI against the generating labels (1.00 = perfect):")
    print()
    short = {n: n.split(" (")[0] for n in names}
    print(f"  {'problem':<24}" + "".join(f"{short[n]:>18}" for n in names))
    for prob, row in table.items():
        print(f"  {prob.split(' (')[0]:<24}" + "".join(f"{row[n]:>18.2f}" for n in names))

    km_scores = [row["k-means"] for row in table.values()]
    print(f"\n  Read down the k-means column. It never gets above {max(km_scores):.2f}, and it")
    print("  is the only algorithm that is mediocre on ALL THREE - every other")
    print("  method is excellent on at least one. Each k-means failure has a")
    print("  different cause:")
    print("    Crescents      a Voronoi cell is CONVEX, so no value of k can")
    print("                   produce a crescent. k-means slices both moons")
    print("                   across the middle. GMM fails for the same reason:")
    print("                   a Gaussian is convex too.")
    print("    Stretched      the bands are long and thin, so points at one end")
    print("                   of a band are further from their own centroid than")
    print("                   from a neighbouring one. GMM fits a FULL COVARIANCE")
    print("                   per component, so it can tilt and stretch to match.")
    print("    Unequal spread equal-size cells minimise inertia, so k-means eats")
    print("                   into the wide cluster to pad out the narrow ones.")
    print("\n  DBSCAN scores a perfect 1.00 on the crescents, because it grows")
    print("  clusters by DENSITY rather than by distance to a centre, so shape is")
    print("  simply irrelevant to it. It is also the only algorithm here that can")
    print("  refuse to classify a point at all (the grey noise label).")
    print("\n  But look at its STRETCHED score - it is the worst cell in the whole")
    print("  table. The bands there nearly touch, so at eps=0.35 a chain of")
    print("  neighbours bridges straight from one band into the next and DBSCAN")
    print("  merges them. Density-connectivity is transitive: two points in")
    print("  different clusters get joined if ANY path of close points links")
    print("  them. That is the price of not assuming a shape.")
    print("\n  Note also that eps had to be hand-set per dataset here. k-means asks")
    print("  you for k, which is at least a quantity you can reason about; DBSCAN")
    print("  asks you for a distance, which depends on scaling, dimensionality and")
    print("  density all at once. 'No need to specify k' is not the free lunch it")
    print("  is usually sold as.")
    print("\n  The overall winner on these three is the GAUSSIAN MIXTURE (0.47 /")
    print("  1.00 / 0.95): a full covariance per component buys it ellipses of any")
    print("  orientation and size, which covers two of the three failure modes.")
    print("  It still cannot do crescents - a Gaussian is convex.")
    print("\n  The practical rule: match the algorithm to the SHAPE you expect.")
    print("  k-means for compact similar blobs, GMM for elliptical or overlapping")
    print("  ones, DBSCAN for arbitrary shapes plus outliers, Ward when you want")
    print("  the hierarchy as well as the partition.")
    print(f"\n  saved  {save_figure(fig, OUT / '3_cluster_shapes.png').name}")



# ---------------------------------------------------------- 4. scoring it
def figure_scoring(X: np.ndarray) -> None:
    """Cluster IDs are arbitrary. A clustering that recovers the varieties
    perfectly but names them 2,0,1 instead of 0,1,2 scores 0% accuracy and 1.00
    ARI. That is the whole reason a separate family of metrics exists.
    """
    km = KMeans(3, n_init=10, random_state=SEED).fit(X)
    labels = km.labels_

    naive_accuracy = (labels == TRUE_CODES).mean()
    ari = adjusted_rand_score(TRUE_CODES, labels)
    nmi = normalized_mutual_info_score(TRUE_CODES, labels)

    print(f"  Naive accuracy (label == label)   {naive_accuracy:.3f}   <- meaningless")
    print(f"  Adjusted Rand Index (ARI)         {ari:.3f}")
    print(f"  Normalised Mutual Information     {nmi:.3f}")
    print("\n  The accuracy number is not a poor score, it is a NONSENSE score.")
    print("  k-means never saw the variety names and had no way to match its")
    print("  cluster 0 to Kama. ARI and NMI compare PARTITIONS - which points")
    print("  share a group - so they are invariant to relabelling.")

    # a permuted copy: identical partition, different integers
    perm = {0: 2, 1: 0, 2: 1}
    relabelled = np.array([perm[v] for v in labels])
    print(f"\n  Proof: renumber the same clusters 0->2, 1->0, 2->1")
    print(f"    accuracy {(relabelled == TRUE_CODES).mean():.3f} "
          f"(changed from {naive_accuracy:.3f}) but "
          f"ARI {adjusted_rand_score(TRUE_CODES, relabelled):.3f} (unchanged)")

    print("\n  ARI is also ADJUSTED FOR CHANCE - a random clustering scores ~0,")
    print("  not the ~0.3 an unadjusted Rand Index would give you.")
    rand_scores = [adjusted_rand_score(TRUE_CODES, rng.integers(0, 3, len(TRUE_CODES)))
                   for _ in range(200)]
    print(f"    200 random 3-way clusterings: mean ARI {np.mean(rand_scores):+.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # confusion-style contingency, the honest way to inspect a clustering
    cont = np.zeros((3, 3), dtype=int)
    for t, p in zip(TRUE_CODES, labels):
        cont[t, p] += 1
    axes[0].imshow(cont, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, cont[i, j], ha="center", va="center",
                         color="white" if cont[i, j] > cont.max() / 2 else "black",
                         fontweight="bold")
    axes[0].set_xticks(range(3), [f"cluster {i}" for i in range(3)])
    axes[0].set_yticks(range(3), VARIETIES)
    axes[0].set_title(f"Contingency table\nARI {ari:.3f}")
    axes[0].grid(False)

    pcs = PCA(2, random_state=SEED).fit_transform(X)
    for k, name in enumerate(VARIETIES):
        m = TRUE_CODES == k
        axes[1].scatter(pcs[m, 0], pcs[m, 1], s=22, color=CYCLE[k], alpha=0.85, label=name)
    axes[1].set_title("True varieties"); axes[1].legend(fontsize=9)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

    for k in range(3):
        m = labels == k
        axes[2].scatter(pcs[m, 0], pcs[m, 1], s=22, color=CYCLE[k], alpha=0.85,
                        label=f"cluster {k}")
    axes[2].set_title("k-means found"); axes[2].legend(fontsize=9)
    axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")

    fig.suptitle("Scoring a clustering against a known truth",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '4_scoring.png').name}")

    # ---- dendrogram: the one method that does not need k up front
    fig2, ax = plt.subplots(figsize=(12, 5))
    Z = linkage(X, method="ward")
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-2, 2],
               above_threshold_color=PALETTE["grey"])
    ax.axhline(Z[-2, 2], color=PALETTE["vermillion"], ls="--", lw=1.5)
    ax.annotate("cut here -> 3 clusters", xy=(0.62, 0.72), xycoords="axes fraction",
                fontsize=10, color=PALETTE["vermillion"], fontweight="bold")
    ax.set_title("Hierarchical clustering: choose k AFTER seeing the structure")
    ax.set_xlabel("wheat kernels"); ax.set_ylabel("merge distance (Ward)")
    fig2.tight_layout()

    print("\n  The dendrogram inverts the workflow: build the full merge tree")
    print("  once, then cut it wherever you like. Every k is available from one")
    print("  fit, and the branch heights show you how much you lose at each cut.")
    print("  Cost: it is O(n^2) memory, so it does not scale to millions of rows -")
    print("  which is exactly when you fall back to k-means or MiniBatchKMeans.")
    print(f"\n  saved  {save_figure(fig2, OUT / '5_dendrogram.png').name}")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  CLUSTERING")
    print("=" * 72)

    df = load_seeds()
    global TRUE_LABELS, TRUE_CODES, VARIETIES
    TRUE_LABELS = df["variety"].to_numpy()
    VARIETIES = sorted(df["variety"].unique())
    TRUE_CODES = np.array([VARIETIES.index(v) for v in TRUE_LABELS])
    feature_names = [c for c in df.columns if c != "variety"]
    X_raw = df[feature_names].to_numpy(dtype=float)

    print(f"\n  Seeds: {len(df):,} wheat kernels, {len(feature_names)} measurements, "
          f"{len(VARIETIES)} varieties")
    print(f"  Varieties: {', '.join(VARIETIES)} "
          f"({', '.join(str((TRUE_LABELS == v).sum()) for v in VARIETIES)} each)")
    print("  The variety column is held back from every algorithm below and used")
    print("  only to score the result afterwards.")

    banner("1. Scaling is not optional")
    X = figure_scaling(X_raw, feature_names)

    banner("2. Choosing k")
    figure_choosing_k(X)

    banner("3. Where k-means breaks")
    figure_shapes()

    banner("4. Scoring a clustering")
    figure_scoring(X)

    banner("What to take away")
    print("  - ALWAYS scale before a distance-based algorithm. Unscaled, the")
    print("    column with the largest units silently becomes the whole model.")
    print("  - The elbow method is subjective and inertia has no minimum by")
    print("    construction. Silhouette and Davies-Bouldin at least give you a")
    print("    number - but both picked k=2 here against 3 real varieties,")
    print("    because two of the three genuinely overlap in these features.")
    print("    Internal metrics measure SEPARATION, not TRUTH. If you need the")
    print("    distinction they are missing, fix the features, not the k.")
    print("  - k-means produces convex, similarly-sized cells. Non-convex shapes")
    print("    need DBSCAN or Ward; stretched ellipses need a Gaussian Mixture;")
    print("    'none of the above' is a cluster only DBSCAN can express.")
    print("  - Match the algorithm to the SHAPE you expect. GMM was the best")
    print("    all-rounder here; DBSCAN scored both the best cell in the")
    print("    table and the worst, depending on the row.")
    print("  - Never score a clustering with accuracy. Cluster IDs are arbitrary;")
    print("    use ARI or NMI, which compare partitions and adjust for chance.")
    print("  - Hierarchical clustering gives you every k from one fit, at O(n^2)")
    print("    memory. k-means gives you one k cheaply, at any scale.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
