"""Dimensionality reduction: fewer columns, and what it costs you.

  1. Scaling changes what PCA even means - on mixed units, PC1 becomes
     whichever column happens to have the largest numbers
  2. How many components? The scree plot, and seeing the information
     physically disappear from reconstructed digits
  3. PCA vs t-SNE vs Isomap - linear projection against manifold learning,
     and why only one of them belongs in a model pipeline
  4. The compression trade - and a measured test of the claim that PCA
     improves accuracy by removing noise, which does not survive it
  5. PCA is a fitted transform, so fitting it before the split leaks

Run:
    python 02_traditional_ml/04_dimensionality_reduction/dimensionality_reduction.py

Concept reference: Foundations volume, Parts A.10 (features), A.16 (unsupervised).
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from plotting import CYCLE, PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.datasets import load_digits, load_wine  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.manifold import TSNE, Isomap  # noqa: E402
from sklearn.model_selection import cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ------------------------------------------------- 1. PCA is scale-dependent
def figure_scaling() -> None:
    """PCA finds the directions of greatest VARIANCE. Variance is measured in the
    squared units of each feature, so a column recorded in larger numbers
    contributes more variance for no substantive reason at all.

    Wine is the clean demonstration: proline runs to ~1,500 while most other
    features sit below 5, so unscaled PCA is almost entirely a proline detector.
    """
    wine = load_wine()
    X, y, names = wine.data, wine.target, list(wine.feature_names)

    raw = PCA(2, random_state=SEED).fit(X)
    scl = PCA(2, random_state=SEED).fit(StandardScaler().fit_transform(X))

    top_raw = names[int(np.argmax(np.abs(raw.components_[0])))]
    load_raw = np.abs(raw.components_[0]).max()

    print(f"  Wine: {X.shape[0]} bottles, {X.shape[1]} chemical measurements.")
    print(f"  Largest feature range: {names[int(np.argmax(X.std(0)))]} "
          f"(std {X.std(0).max():.1f})")
    print(f"  Smallest:              {names[int(np.argmin(X.std(0)))]} "
          f"(std {X.std(0).min():.3f})")

    print(f"\n  UNSCALED  PC1 loads {load_raw:.3f} on '{top_raw}' alone, and explains "
          f"{raw.explained_variance_ratio_[0]:.1%}")
    print(f"            of the variance. PC1 IS that one column, renamed.")
    top3 = sorted(zip(names, np.abs(scl.components_[0])), key=lambda t: -t[1])[:3]
    print(f"    SCALED  PC1 explains {scl.explained_variance_ratio_[0]:.1%}, spread across "
          f"many features:")
    for n, w in top3:
        print(f"            {n:<28} loading {w:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (model, Xin, title) in zip(
        axes,
        [(raw, X, "Unscaled"),
         (scl, StandardScaler().fit_transform(X), "Standardised")],
    ):
        pcs = model.transform(Xin)
        for c in range(3):
            m = y == c
            ax.scatter(pcs[m, 0], pcs[m, 1], s=26, color=CYCLE[c], alpha=0.85,
                       label=wine.target_names[c])
        ax.set_title(f"{title}\nPC1 explains {model.explained_variance_ratio_[0]:.1%}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=9)

    fig.suptitle("PCA maximises variance, and variance depends on your units",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print("\n  Unscaled, the three grape varieties barely separate - the projection")
    print("  spent both of its dimensions describing proline. Scaled, they fall")
    print("  into three clear groups. Same data, same algorithm, different answer.")
    print("  ALWAYS StandardScaler before PCA unless every feature is already in")
    print("  the same unit (image pixels, for instance).")
    print(f"\n  saved  {save_figure(fig, OUT / '1_pca_scaling.png').name}")


# ------------------------------------------- 2. how many components to keep
def figure_scree(X: np.ndarray, images: np.ndarray) -> PCA:
    """Two views of the same question. The scree plot is the abstract answer;
    reconstructing the digits is the same answer you can actually see.
    """
    pca = PCA(random_state=SEED).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    need = {t: int(np.searchsorted(cum, t) + 1) for t in (0.80, 0.90, 0.95, 0.99)}

    print(f"  {X.shape[1]} original features (8x8 pixels). Components needed to retain:")
    for t, n in need.items():
        print(f"    {t:.0%} of variance -> {n:>2} components  "
              f"({n / X.shape[1]:.0%} of the original width)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(range(1, 31), pca.explained_variance_ratio_[:30],
                color=PALETTE["blue"])
    axes[0].set_title("Scree: variance explained per component")
    axes[0].set_xlabel("component"); axes[0].set_ylabel("share of variance")

    axes[1].plot(range(1, len(cum) + 1), cum, color=PALETTE["blue"])
    for t, colour in zip((0.80, 0.90, 0.95), ("green", "orange", "vermillion")):
        axes[1].axhline(t, color=PALETTE[colour], ls="--", lw=1.2)
        axes[1].annotate(f"{t:.0%} at {need[t]} comps", xy=(need[t] + 1, t - 0.06),
                         fontsize=9, color=PALETTE[colour])
    axes[1].set_title("Cumulative variance explained")
    axes[1].set_xlabel("components kept"); axes[1].set_ylabel("cumulative share")

    fig.suptitle("How many components? Read it off the curve, not from a rule of thumb",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    print(f"\n  saved  {save_figure(fig, OUT / '2_scree.png').name}")

    # ---- the same question, visually
    keeps = [2, 5, 10, 20, 40, 64]
    idx = [np.where(DIGIT_TARGET == d)[0][0] for d in (0, 3, 8)]
    fig2, axes2 = plt.subplots(len(idx), len(keeps) + 1, figsize=(14, 6.4))
    for r, i in enumerate(idx):
        axes2[r, 0].imshow(images[i], cmap="gray_r", aspect="equal")
        axes2[r, 0].set_ylabel(f"digit {DIGIT_TARGET[i]}", fontsize=10,
                               fontweight="bold")
        if r == 0:
            axes2[r, 0].set_title("original\n64 values", fontsize=10)
        axes2[r, 0].set_xticks([]); axes2[r, 0].set_yticks([]); axes2[r, 0].grid(False)
        for c, k in enumerate(keeps, start=1):
            p = PCA(k, random_state=SEED).fit(X)
            rec = p.inverse_transform(p.transform(X[i:i + 1]))[0].reshape(8, 8)
            axes2[r, c].imshow(rec, cmap="gray_r", aspect="equal")
            if r == 0:
                axes2[r, c].set_title(f"{k} comps\n{cum[k - 1]:.0%} var", fontsize=10)
            axes2[r, c].set_xticks([]); axes2[r, c].set_yticks([]); axes2[r, c].grid(False)

    fig2.suptitle("The same compression, reconstructed - where does the digit stop "
                  "being readable?", fontsize=13, fontweight="bold")
    fig2.subplots_adjust(left=0.045, right=0.99, top=0.82, bottom=0.03,
                         wspace=0.10, hspace=0.10)

    print("\n  In the reconstruction figure the digits become recognisable somewhere")
    print(f"  around 10-20 components - roughly the {need[0.90]}-component, 90% mark - and")
    print("  stop improving visibly well before all 64. That gap is the compression")
    print("  you are being offered. Note that PCA is LOSSY but INVERTIBLE-ish:")
    print("  inverse_transform gets you back to 64 pixels, just not the same ones.")
    print(f"\n  saved  {save_figure(fig2, OUT / '3_reconstruction.png').name}")
    return pca


# ----------------------------------- 3. linear projection vs manifold learning
def figure_manifold(X: np.ndarray, y: np.ndarray) -> None:
    """PCA can only rotate and project - it is a linear map. t-SNE and Isomap
    can unroll a curved manifold, which makes far prettier pictures and comes
    with a restriction people routinely ignore.
    """
    Xs = StandardScaler().fit_transform(X)
    methods = {
        "PCA\n(linear)": PCA(2, random_state=SEED),
        "t-SNE\n(manifold)": TSNE(2, random_state=SEED, init="pca", perplexity=30),
        "Isomap\n(manifold)": Isomap(n_components=2, n_neighbors=12),
        "LDA\n(supervised)": LinearDiscriminantAnalysis(n_components=2),
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, (name, model) in zip(axes, methods.items()):
        t0 = time.perf_counter()
        # LDA is the odd one out: it is the only method here that is given y.
        Z = (model.fit_transform(Xs, y) if isinstance(model, LinearDiscriminantAnalysis)
             else model.fit_transform(Xs))
        secs = time.perf_counter() - t0
        for d in range(10):
            m = y == d
            ax.scatter(Z[m, 0], Z[m, 1], s=7, alpha=0.65,
                       color=plt.cm.tab10(d / 10))
        ax.set_title(f"{name}\n{secs:.1f}s", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        print(f"  {name.replace(chr(10), ' '):<22} fitted in {secs:5.1f}s")

    fig.suptitle("Four ways to get to two dimensions (digits 0-9, coloured by true label)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    print("\n  t-SNE and Isomap separate the ten digits far better than PCA, because")
    print("  PCA can only ROTATE AND PROJECT - it is a linear map, and the digit")
    print("  classes are not linearly arranged. LDA does well too, but note it is")
    print("  the only method here that was shown the labels: it maximises class")
    print("  separation rather than variance, so it is supervised reduction and")
    print("  is capped at (n_classes - 1) components.")

    print("\n  THE RESTRICTION THAT MATTERS, and the most common mistake with t-SNE:")
    print("    - Distances BETWEEN t-SNE clusters are not meaningful. Two blobs")
    print("      drawn far apart are not more different than two drawn close.")
    print("    - Cluster SIZES are not meaningful either; t-SNE equalises density.")
    print("    - It has no transform() for new data. Each run refits everything,")
    print("      and a different seed or perplexity gives a different picture.")
    print("  So t-SNE and UMAP are for LOOKING AT data, not for generating features.")
    print("  PCA is the one that belongs inside a model pipeline: it is a fitted")
    print("  linear transform with a real transform() you can apply to test rows.")
    print(f"\n  saved  {save_figure(fig, OUT / '4_manifold.png').name}")


# ------------------------------------------------ 4. the compression trade
def figure_tradeoff(X: np.ndarray, y: np.ndarray) -> None:
    """What do you actually buy by dropping components? Accuracy against
    components, measured rather than assumed - including the textbook claim
    that PCA improves accuracy by removing noise, which does NOT reproduce
    here and is worth understanding rather than repeating.
    """
    grid = [2, 5, 10, 15, 20, 30, 40, 64]

    def sweep(Xin: np.ndarray, ks: list[int]) -> list[tuple[int, float, float]]:
        out = []
        for k in ks:
            pipe = make_pipeline(StandardScaler(), PCA(k, random_state=SEED),
                                 SVC(random_state=SEED))
            t0 = time.perf_counter()
            out.append((k, cross_val_score(pipe, Xin, y, cv=5).mean(),
                        time.perf_counter() - t0))
        return out

    def no_pca(Xin: np.ndarray) -> float:
        return cross_val_score(make_pipeline(StandardScaler(), SVC(random_state=SEED)),
                               Xin, y, cv=5).mean()

    rows = sweep(X, grid)
    base_acc = no_pca(X)

    print(f"  {'components':>11} {'accuracy':>9} {'vs all 64':>10} {'cv seconds':>11}")
    for k, acc, secs in rows:
        print(f"  {k:>11} {acc:>9.4f} {acc - base_acc:>+10.4f} {secs:>11.2f}")
    print(f"  {'no PCA':>11} {base_acc:>9.4f}")

    best = max(rows, key=lambda r: r[1])
    thrifty = next(k for k, acc, _ in rows if acc >= base_acc - 0.01)

    print(f"\n  Best accuracy is {best[1]:.4f} at {best[0]} components against "
          f"{base_acc:.4f} with no")
    print("  PCA at all, so compression never actually WINS on clean digits. What")
    print(f"  it does is stay within one point all the way down to {thrifty} components,")
    print(f"  {thrifty / X.shape[1]:.0%} of the original width, for a model that is much narrower to")
    print("  store, serve and keep clean in production.")
    print("\n  Note the timings barely move either. At 64 features an SVC is already")
    print("  fast and PCA itself costs something; the speed argument only starts")
    print("  paying at far higher dimensionality.")

    # ---- the claim that does not survive contact with a measurement
    print("\n  You will often read that PCA raises accuracy by discarding noise.")
    print("  Testing it: add 100 columns of pure Gaussian junk to the 64 real")
    print("  pixels and see whether PCA can find its way back.")

    rng = np.random.default_rng(SEED)
    n_junk = 100
    X_junk = np.hstack([X, rng.normal(0, 5.0, (X.shape[0], n_junk))])
    junk_grid = [10, 20, 30, 40, 64]
    rows_j = sweep(X_junk, junk_grid)
    base_junk = no_pca(X_junk)
    best_j = max(rows_j, key=lambda r: r[1])

    # LDA is supervised, so unlike PCA it can see which directions separate the
    # classes. n_components is capped at n_classes - 1 = 9.
    lda_acc = cross_val_score(
        make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=9),
                      SVC(random_state=SEED)),
        X_junk, y, cv=5).mean()

    print(f"\n  {'approach':<34} {'accuracy':>9}")
    print(f"  {'no reduction (164 columns)':<34} {base_junk:>9.4f}")
    for k, acc, _ in rows_j:
        print(f"  {'PCA, ' + str(k) + ' components':<34} {acc:>9.4f}")
    print(f"  {'LDA, 9 components (SUPERVISED)':<34} {lda_acc:>9.4f}")

    # Why PCA cannot help: measure how much of its loading mass it spends on junk.
    Xs = StandardScaler().fit_transform(X_junk)
    comps = PCA(20, random_state=SEED).fit(Xs).components_
    mass = comps ** 2
    junk_share = mass[:, X.shape[1]:].sum() / mass.sum()

    print(f"\n  PCA never beats doing nothing ({best_j[1]:.4f} at {best_j[0]} components "
          f"vs {base_junk:.4f}),")
    print(f"  and supervised LDA with just 9 components beats every PCA setting at "
          f"{lda_acc:.4f}.")
    print(f"\n  Here is the mechanism. Of the squared loading mass in the top 20")
    print(f"  components, {junk_share:.1%} sits on the junk columns. PCA is UNSUPERVISED:")
    print("  it maximises VARIANCE, and it has no idea which variance is useful.")
    print("  After standardising, a pure-noise column has variance 1 - exactly")
    print("  like a real pixel - so PCA cheerfully spends components describing it.")
    print("\n  The rule that follows: PCA removes REDUNDANCY (correlated columns),")
    print("  not IRRELEVANCE. If your problem is irrelevant features, you need")
    print("  something that looks at y - LDA, or an explicit feature-selection")
    print("  step - because no amount of variance-maximising will find the signal.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    axes[0].plot([r[0] for r in rows], [r[1] for r in rows], "o-",
                 color=PALETTE["blue"])
    axes[0].axhline(base_acc, color=PALETTE["grey"], ls="--", lw=1.3)
    axes[0].annotate(f"no PCA: {base_acc:.3f}", xy=(0.42, 0.30),
                     xycoords="axes fraction", fontsize=9, color=PALETTE["grey"])
    axes[0].axvline(thrifty, color=PALETTE["vermillion"], ls=":", lw=1.6)
    axes[0].annotate(f"{thrifty} components is within\n1 point of the full model",
                     xy=(0.28, 0.55), xycoords="axes fraction", fontsize=9,
                     color=PALETTE["vermillion"], fontweight="bold")
    axes[0].set_title("Clean digits:\ncompression costs a little, never wins")
    axes[0].set_xlabel("PCA components kept"); axes[0].set_ylabel("5-fold accuracy")

    axes[1].plot([r[0] for r in rows_j], [r[1] for r in rows_j], "o-",
                 color=PALETTE["vermillion"], label="PCA")
    axes[1].axhline(base_junk, color=PALETTE["grey"], ls="--", lw=1.3,
                    label="no reduction")
    axes[1].axhline(lda_acc, color=PALETTE["green"], ls="-.", lw=1.6,
                    label="LDA, 9 comps (supervised)")
    axes[1].annotate(f"{junk_share:.0%} of PCA's top-20 loading\nmass is spent on junk",
                     xy=(0.30, 0.30), xycoords="axes fraction", fontsize=9,
                     color=PALETTE["vermillion"], fontweight="bold")
    axes[1].set_title("Plus 100 junk columns:\nPCA cannot tell signal from noise")
    axes[1].set_xlabel("PCA components kept"); axes[1].set_ylabel("5-fold accuracy")
    axes[1].legend(fontsize=9, loc="lower right")

    fig.suptitle("PCA removes redundancy, not irrelevance",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    print(f"\n  saved  {save_figure(fig, OUT / '5_tradeoff.png').name}")
# ------------------------------------------------------ 5. PCA leaks, too
def demo_leakage(X: np.ndarray, y: np.ndarray) -> None:
    """PCA is FITTED - it learns component directions from the data it sees. Fit
    it on everything and the test rows have already influenced the axes the
    model is trained on. Same trap as scaling, same fix.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=SEED, stratify=y)

    # WRONG: components learned from train + test together
    leaky = PCA(20, random_state=SEED).fit(StandardScaler().fit_transform(X))
    Xa = leaky.transform(StandardScaler().fit(X).transform(X_tr))
    Xb = leaky.transform(StandardScaler().fit(X).transform(X_te))
    m1 = LogisticRegression(max_iter=3000, random_state=SEED).fit(Xa, y_tr)
    leaky_acc = m1.score(Xb, y_te)

    # RIGHT: everything fitted inside the pipeline, on training rows only
    pipe = make_pipeline(StandardScaler(), PCA(20, random_state=SEED),
                         LogisticRegression(max_iter=3000, random_state=SEED))
    pipe.fit(X_tr, y_tr)
    clean_acc = pipe.score(X_te, y_te)

    print(f"  PCA fitted on train + test (WRONG)   test accuracy {leaky_acc:.4f}")
    print(f"  PCA fitted inside a Pipeline (RIGHT) test accuracy {clean_acc:.4f}")
    print(f"  Difference: {leaky_acc - clean_acc:+.4f}")
    print("\n  The gap here is small, and that is exactly what makes this dangerous.")
    print("  Digits is large, clean and balanced, so the components barely move")
    print("  when you add the test rows. On a small or high-dimensional dataset")
    print("  the same mistake can be worth several points - and it always flatters")
    print("  you, so nothing looks wrong until production.")
    print("\n  The rule is mechanical: ANY step that LEARNS from data - scaler,")
    print("  PCA, imputer, target encoder, feature selector - goes inside the")
    print("  Pipeline, so cross_val_score refits it on each training fold.")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  DIMENSIONALITY REDUCTION")
    print("=" * 72)

    digits = load_digits()
    global DIGIT_TARGET
    DIGIT_TARGET = digits.target
    X, y, imgs = digits.data, digits.target, digits.images

    print(f"\n  Digits: {X.shape[0]:,} handwritten numerals as 8x8 grids "
          f"= {X.shape[1]} features, {len(set(y))} classes")

    banner("1. PCA is scale-dependent")
    figure_scaling()

    banner("2. How many components should you keep?")
    figure_scree(X, imgs)

    banner("3. Linear projection vs manifold learning")
    figure_manifold(X, y)

    banner("4. The compression trade")
    figure_tradeoff(X, y)

    banner("5. PCA leaks if you fit it before the split")
    demo_leakage(X, y)

    banner("What to take away")
    print("  - Scale before PCA. Variance is unit-dependent, so without scaling")
    print("    PC1 is just whichever column has the biggest numbers.")
    print("  - Choose k from the cumulative variance curve, not a rule of thumb.")
    print("    Digits needs ~21 components for 90% of 64 - a 3x reduction.")
    print("  - PCA is linear: rotate and project, nothing else. Manifold methods")
    print("    can unfold a curved surface - t-SNE split the digits into ten")
    print("    clean islands where PCA produced one cloud.")
    print("  - But t-SNE/UMAP are for LOOKING, not for features: between-cluster")
    print("    distances and cluster sizes are not meaningful, and there is no")
    print("    transform() for new rows. PCA and LDA are the ones you can deploy.")
    print("  - LDA is SUPERVISED reduction - it uses y, maximises class")
    print("    separation rather than variance, and caps at n_classes - 1.")
    print("  - Compression did NOT raise accuracy here, on clean data or on")
    print("    data padded with 100 junk columns. The payoff is a narrower")
    print("    model to store, serve and keep clean - not accuracy, and at this")
    print("    width not speed either.")
    print("  - PCA removes REDUNDANCY, not IRRELEVANCE. It spent 33% of its top-20")
    print("    loading mass on pure noise columns, because after standardising")
    print("    they have exactly as much variance as a real pixel. Irrelevant")
    print("    features need something that looks at y - LDA or feature selection.")
    print("  - PCA is a fitted transform, so it belongs inside the Pipeline.")
    print("    Fitting it before the split leaks, and it always flatters you.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
