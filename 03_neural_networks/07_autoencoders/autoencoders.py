"""Autoencoders: compression as a learning objective.

An autoencoder is trained to reproduce its own input through a narrow
bottleneck. Because the bottleneck cannot carry everything, the network must
decide what matters - and the code it settles on is a learned, non-linear
alternative to PCA.

  1. Autoencoder vs PCA at the same bottleneck width, which is the only
     fair comparison and the one usually skipped
  2. Denoising: corrupt the input, ask for the clean output
  3. Reconstruction error as an anomaly score, against the Isolation
     Forest from 02_traditional_ml/05

Run:
    python 03_neural_networks/07_autoencoders/autoencoders.py

Concept reference: Foundations volume, Parts B.21-B.22.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
N_TRAIN, N_TEST, EPOCHS = 12000, 2500, 25


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def seed_everything() -> None:
    keras.utils.set_random_seed(SEED)


def build_autoencoder(n_in: int, bottleneck: int,
                      deep: bool = True) -> tuple[keras.Model, keras.Model]:
    """Returns (autoencoder, encoder).

    With `deep=False` there are no hidden layers and no non-linearity, which
    makes the autoencoder mathematically equivalent to PCA - a useful control.
    """
    inp = keras.layers.Input((n_in,))
    if deep:
        h = keras.layers.Dense(128, activation="relu")(inp)
        h = keras.layers.Dense(64, activation="relu")(h)
        code = keras.layers.Dense(bottleneck, activation="relu", name="code")(h)
        h = keras.layers.Dense(64, activation="relu")(code)
        h = keras.layers.Dense(128, activation="relu")(h)
        out = keras.layers.Dense(n_in, activation="sigmoid")(h)
    else:
        code = keras.layers.Dense(bottleneck, activation=None, name="code")(inp)
        out = keras.layers.Dense(n_in, activation=None)(code)
    ae = keras.Model(inp, out)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, keras.Model(inp, code)


# ======================================== 1. autoencoder vs PCA, fairly
def figure_vs_pca(xtr: np.ndarray, xte: np.ndarray) -> None:
    widths = [2, 8, 32]
    print(f"  Reconstruction MSE on held-out digits, same bottleneck width.\n")
    print(f"  {'bottleneck':>11} {'PCA':>10} {'linear AE':>11} "
          f"{'deep AE':>10} {'deep vs PCA':>13}")

    rows, recons = [], {}
    for w in widths:
        p = PCA(w, random_state=SEED).fit(xtr)
        pca_mse = float(np.mean((p.inverse_transform(p.transform(xte))
                                 - xte) ** 2))

        seed_everything()
        lin, _ = build_autoencoder(xtr.shape[1], w, deep=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lin.fit(xtr, xtr, epochs=EPOCHS, batch_size=256, verbose=0)
        lin_mse = float(np.mean((lin.predict(xte, verbose=0) - xte) ** 2))

        seed_everything()
        deep, _ = build_autoencoder(xtr.shape[1], w, deep=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deep.fit(xtr, xtr, epochs=EPOCHS, batch_size=256, verbose=0)
        deep_mse = float(np.mean((deep.predict(xte, verbose=0) - xte) ** 2))

        rows.append((w, pca_mse, lin_mse, deep_mse))
        recons[w] = (p.inverse_transform(p.transform(xte[:6])),
                     deep.predict(xte[:6], verbose=0))
        print(f"  {w:>11} {pca_mse:>10.5f} {lin_mse:>11.5f} {deep_mse:>10.5f} "
              f"{(deep_mse / pca_mse - 1) * 100:>12.0f}%")

    print("\n  The LINEAR autoencoder column is the control. With no hidden")
    print("  layers and no activation it can only learn a linear projection,")
    print("  so it should match PCA - and it roughly does. Any difference is")
    print("  optimisation noise, not a difference in what the model CAN")
    print("  represent. An autoencoder without non-linearity IS PCA, spanning")
    print("  the same subspace (though not necessarily with orthogonal axes).")

    wins = [r for r in rows if r[3] < r[1]]
    print(f"\n  The DEEP autoencoder beat PCA at {len(wins)} of {len(rows)} widths.")
    print("  Where it wins, it wins because it can bend: a curved manifold")
    print("  needs fewer non-linear coordinates than linear ones. Where it")
    print("  does not, the reason is usually that PCA is an exact closed-form")
    print("  optimum for its objective while the autoencoder is doing gradient")
    print("  descent on a harder one and gets a fixed epoch budget.")
    print("\n  That last point is worth keeping: PCA cannot be undertrained.")
    print("  An autoencoder can, and frequently is.")

    fig, axes = plt.subplots(len(widths) * 2 + 1, 6, figsize=(11, 11))
    for j in range(6):
        axes[0, j].imshow(xte[j].reshape(28, 28), cmap="gray", aspect="equal")
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([]); axes[0, j].grid(False)
    axes[0, 0].set_ylabel("original", fontsize=9, fontweight="bold")
    for i, w in enumerate(widths):
        for k, (label, arr) in enumerate([(f"PCA {w}", recons[w][0]),
                                          (f"AE {w}", recons[w][1])]):
            row = 1 + i * 2 + k
            for j in range(6):
                axes[row, j].imshow(np.clip(arr[j], 0, 1).reshape(28, 28),
                                    cmap="gray", aspect="equal")
                axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
                axes[row, j].grid(False)
            axes[row, 0].set_ylabel(label, fontsize=9, fontweight="bold")
    fig.suptitle("Same bottleneck width, two ways to compress",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.01,
                        wspace=0.05, hspace=0.05)
    print(f"\n  saved  {save_figure(fig, OUT / '1_vs_pca.png').name}")


# ================================================= 2. denoising
def figure_denoising(xtr: np.ndarray, xte: np.ndarray) -> None:
    rng = np.random.default_rng(SEED)
    noise = 0.4
    xtr_n = np.clip(xtr + rng.normal(0, noise, xtr.shape), 0, 1).astype("float32")
    xte_n = np.clip(xte + rng.normal(0, noise, xte.shape), 0, 1).astype("float32")

    seed_everything()
    ae, _ = build_autoencoder(xtr.shape[1], 32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ae.fit(xtr_n, xtr, epochs=EPOCHS, batch_size=256, verbose=0)
    clean = ae.predict(xte_n, verbose=0)

    print(f"  Input: digits + N(0, {noise}) noise. Target: the CLEAN digit.")
    print("  The only change from section 1 is that input and target differ.\n")
    print(f"  MSE of the noisy input against the clean digit:  "
          f"{np.mean((xte_n - xte) ** 2):.5f}")
    print(f"  MSE of the reconstruction against the clean digit: "
          f"{np.mean((clean - xte) ** 2):.5f}")

    print("\n  A denoising autoencoder cannot succeed by copying its input,")
    print("  because its input is wrong. It has to learn what a digit actually")
    print("  looks like in order to produce one - which is a far stronger")
    print("  constraint than reconstruction alone, and the reason denoising is")
    print("  the standard trick for stopping an autoencoder learning identity.")
    print("\n  This is also SELF-SUPERVISED LEARNING (02_traditional_ml/08): the")
    print("  noise was added by us, so the clean target is free. Masked")
    print("  autoencoders for images and masked language modelling for text are")
    print("  the same idea with a different corruption.")

    fig, axes = plt.subplots(3, 8, figsize=(13, 5.2))
    for j in range(8):
        for row, (arr, label) in enumerate([(xte, "clean"), (xte_n, "noisy input"),
                                            (clean, "reconstructed")]):
            axes[row, j].imshow(np.clip(arr[j], 0, 1).reshape(28, 28),
                                cmap="gray", aspect="equal")
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
            axes[row, j].grid(False)
            if j == 0:
                axes[row, 0].set_ylabel(label, fontsize=9.5, fontweight="bold")
    fig.suptitle("Denoising: the input is corrupted, the target is not",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.02,
                        wspace=0.06, hspace=0.06)
    print(f"\n  saved  {save_figure(fig, OUT / '2_denoising.png').name}")


# ========================================= 3. anomaly detection by error
def figure_anomaly(xtr: np.ndarray, ytr: np.ndarray,
                   xte: np.ndarray, yte: np.ndarray) -> None:
    """Train on ordinary digits only; score everything by how badly it
    reconstructs. This is novelty detection from 02_traditional_ml/05, with a
    neural network supplying the score.
    """
    normal_digits = list(range(5))            # 0-4 are "normal"
    tr = xtr[np.isin(ytr, normal_digits)]
    is_anom = (~np.isin(yte, normal_digits)).astype(int)

    seed_everything()
    ae, _ = build_autoencoder(xtr.shape[1], 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ae.fit(tr, tr, epochs=EPOCHS, batch_size=256, verbose=0)

    err = np.mean((ae.predict(xte, verbose=0) - xte) ** 2, axis=1)
    ae_ap = average_precision_score(is_anom, err)

    iso = IsolationForest(random_state=SEED, n_estimators=200).fit(tr)
    iso_ap = average_precision_score(is_anom, -iso.score_samples(xte))

    base = is_anom.mean()
    print(f"  Trained on digits {normal_digits} only ({len(tr):,} images).")
    print(f"  Test set is {base:.0%} anomalous (digits 5-9, never seen).\n")
    print(f"  {'method':<28} {'PR-AUC':>8}")
    print(f"  {'random baseline':<28} {base:>8.3f}")
    print(f"  {'Isolation Forest on pixels':<28} {iso_ap:>8.3f}")
    print(f"  {'autoencoder reconstruction':<28} {ae_ap:>8.3f}")

    better = "autoencoder" if ae_ap > iso_ap else "Isolation Forest"
    print(f"\n  {better} wins here.")
    print("\n  The idea is simple: train only on normal data, so the network")
    print("  learns to reconstruct normal things well and has never had reason")
    print("  to learn anything else. Reconstruction error then IS the anomaly")
    print("  score - no labels required at any point.")
    print("\n  The catch is the one every autoencoder detector has: a")
    print("  sufficiently large bottleneck learns to reconstruct ANYTHING,")
    print("  including the anomalies, and the score collapses. The bottleneck")
    print("  is doing the work, so it is the hyperparameter that matters most -")
    print("  and you usually cannot tune it, because you have no anomaly")
    print("  labels to tune against. Same bind as `contamination` in")
    print("  02_traditional_ml/05.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(err[is_anom == 0], bins=50, alpha=0.7, density=True,
                 color=PALETTE["blue"], label="normal (digits 0-4)")
    axes[0].hist(err[is_anom == 1], bins=50, alpha=0.7, density=True,
                 color=PALETTE["vermillion"], label="anomalous (digits 5-9)")
    axes[0].set_xlabel("reconstruction MSE"); axes[0].set_ylabel("density")
    axes[0].set_title("Reconstruction error separates them - partially")
    axes[0].legend(fontsize=9)

    axes[1].barh(["random", "Isolation\nForest", "autoencoder"],
                 [base, iso_ap, ae_ap],
                 color=[PALETTE["grey"], PALETTE["orange"], PALETTE["blue"]])
    axes[1].set_xlabel("PR-AUC")
    axes[1].set_title("Anomaly ranking quality")
    axes[1].grid(axis="y", visible=False)

    fig.suptitle("Reconstruction error as an anomaly score",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '3_anomaly.png').name}")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  AUTOENCODERS")
    print("=" * 72)
    print(f"\n  Keras {keras.__version__}, backend: {keras.backend.backend()}")

    import truststore
    truststore.inject_into_ssl()
    (xtr, ytr), (xte, yte) = keras.datasets.mnist.load_data()
    rng = np.random.default_rng(SEED)
    i, j = (rng.choice(len(xtr), N_TRAIN, replace=False),
            rng.choice(len(xte), N_TEST, replace=False))
    xtr_f = (xtr[i] / 255.0).reshape(N_TRAIN, -1).astype("float32")
    xte_f = (xte[j] / 255.0).reshape(N_TEST, -1).astype("float32")
    print(f"  MNIST, {N_TRAIN:,} train / {N_TEST:,} test, flattened to 784.")

    banner("1. Autoencoder vs PCA at the same bottleneck")
    figure_vs_pca(xtr_f, xte_f)

    banner("2. Denoising")
    figure_denoising(xtr_f, xte_f)

    banner("3. Reconstruction error as an anomaly score")
    figure_anomaly(xtr_f, ytr[i], xte_f, yte[j])

    banner("What to take away")
    print("  - An autoencoder is trained to reproduce its input through a")
    print("    bottleneck. The bottleneck is the whole design: it forces the")
    print("    network to choose what to keep.")
    print("  - A LINEAR AUTOENCODER IS PCA. Without activations it spans the")
    print("    same subspace, which is why the linear column above tracks the")
    print("    PCA column. The non-linearity is the only thing that can buy")
    print("    you anything over PCA.")
    print("  - PCA cannot be undertrained and has a closed-form optimum. An")
    print("    autoencoder is gradient descent on a harder problem, so compare")
    print("    them at a matched bottleneck and give the network enough epochs")
    print("    before concluding anything.")
    print("  - DENOISING is the standard fix for an autoencoder that would")
    print("    otherwise learn the identity function. Corrupt the input, keep")
    print("    the target clean - and note this is self-supervision, the same")
    print("    idea as masked language modelling.")
    print("  - RECONSTRUCTION ERROR IS AN ANOMALY SCORE when you train on")
    print("    normal data only. No labels needed - but a bottleneck that is")
    print("    too wide reconstructs the anomalies too, and you generally")
    print("    cannot tune it because you have no anomaly labels.")
    print("  - Other uses of the same skeleton: a VARIATIONAL autoencoder adds")
    print("    a distribution over the code so you can sample new data; a")
    print("    SEQUENCE autoencoder is the encoder-decoder that machine")
    print("    translation was built on before attention replaced it.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
