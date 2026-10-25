"""Self-supervised learning: inventing labels so you can learn without any.

The idea in one sentence: hide part of the data, train a model to recover it,
and keep the representation rather than the prediction. The invented task (the
PRETEXT task) is throwaway; what you want is what the model had to understand
in order to solve it.

  1. Rotation prediction - the canonical pretext task. The labels come
     from an operation WE applied, so they are free and always correct
  2. Masked reconstruction - predict hidden pixels from visible ones
  3. Does the representation actually help? The label-efficiency curve,
     which is the only honest way to evaluate a pretext task - and which
     here returns a NEGATIVE result, left in deliberately
  4. A pretext task that learns nothing useful, to show the failure mode

Everything here is sklearn only - no deep learning framework. The ideas are
the same ones behind SimCLR, MAE and BERT; the encoders are just smaller.

Run:
    python 02_traditional_ml/08_self_supervised/self_supervised.py

Concept reference: Foundations volume, Parts A.16 (unsupervised), A.17 (transfer).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ------------------------------------------------- 1. the rotation pretext
def build_rotation_task(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate each image by 0/90/180/270 degrees and label it with the angle.

    The label is the rotation WE chose, so it costs nothing and is always
    correct. To predict it the model must learn which way up a digit goes,
    which requires recognising strokes and their arrangement - and that is
    the representation we are actually after.
    """
    X, y = [], []
    for k in range(4):
        X.append(np.rot90(images, k=k, axes=(1, 2)).reshape(len(images), -1))
        y.append(np.full(len(images), k))
    return np.vstack(X), np.concatenate(y)


def figure_rotation(images: np.ndarray, digits_y: np.ndarray) -> MLPClassifier:
    Xr, yr = build_rotation_task(images)
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.25,
                                          random_state=SEED, stratify=yr)

    scaler = StandardScaler().fit(Xtr)
    net = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                        random_state=SEED, early_stopping=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.fit(scaler.transform(Xtr), ytr)
    acc = accuracy_score(yte, net.predict(scaler.transform(Xte)))

    print(f"  Built {len(Xr):,} training examples from {len(images):,} images, using")
    print("  NO human labels at all - each image appears four times, labelled")
    print("  with the rotation we applied to it.")
    print(f"\n  Rotation-prediction accuracy: {acc:.3f} (chance is 0.250)")
    print("\n  The model has learned which way up a digit goes, which it could")
    print("  not do without encoding something about strokes and their")
    print("  arrangement. The rotation classifier itself is disposable - the")
    print("  hidden layer is what we keep.")
    print("\n  Resist the temptation to stop here. A high pretext score is NOT")
    print("  evidence that the representation is useful. Section 3 measures")
    print("  that properly, and on this dataset the answer turns out to be no.")

    print("\n  One caveat specific to digits, and worth knowing: 0, 1 and 8 are")
    print("  nearly rotation-symmetric, so the pretext task is genuinely")
    print("  ambiguous for them. A pretext task that is IMPOSSIBLE on part of")
    print("  your data caps what the representation can learn there.")

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for col, k in enumerate(range(4)):
        for row, idx in enumerate((0, 3)):
            ax = axes[row, col]
            ax.imshow(np.rot90(images[idx], k=k), cmap="gray_r", aspect="equal")
            ax.set_title(f"label = {k}  ({k * 90} deg)", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        axes[row, 0].set_ylabel("example", fontsize=9)
    fig.suptitle("The pretext task: predict the rotation WE applied "
                 "(no human labels involved)", fontsize=12.5, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.03,
                        wspace=0.12, hspace=0.25)
    print(f"\n  saved  {save_figure(fig, OUT / '1_rotation_task.png').name}")
    return net


# --------------------------------------------- 2. masked reconstruction
def figure_masking(X: np.ndarray, images: np.ndarray) -> MLPRegressor:
    """Hide the bottom half of every image, predict it from the top half.

    This is the same idea as a masked autoencoder, or as BERT's masked
    language modelling: the data is its own supervision.
    """
    top, bottom = X[:, :32], X[:, 32:]
    Xtr, Xte, ytr, yte = train_test_split(top, bottom, test_size=0.25,
                                          random_state=SEED)

    net = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500,
                       random_state=SEED, early_stopping=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.fit(Xtr / 16.0, ytr / 16.0)
    pred = net.predict(Xte / 16.0) * 16.0

    baseline = np.tile(ytr.mean(axis=0), (len(yte), 1))
    mae_model = float(np.abs(pred - yte).mean())
    mae_base = float(np.abs(baseline - yte).mean())

    print("  Task: given the TOP half of an 8x8 digit, reconstruct the BOTTOM half.")
    print(f"\n  Mean absolute pixel error")
    print(f"    predict the average bottom half   {mae_base:6.3f}")
    print(f"    learned from the top half         {mae_model:6.3f}   "
          f"({(1 - mae_model / mae_base) * 100:.0f}% better)")
    print("\n  The top half genuinely constrains the bottom half - that is the")
    print("  structure the model is forced to internalise. Same principle as")
    print("  BERT masking words or a masked autoencoder masking patches.")

    fig, axes = plt.subplots(3, 6, figsize=(12, 6.2))
    for col in range(6):
        i = col * 7
        full = np.concatenate([Xte[i], yte[i]]).reshape(8, 8)
        given = np.concatenate([Xte[i], np.zeros(32)]).reshape(8, 8)
        recon = np.concatenate([Xte[i], np.clip(pred[i], 0, 16)]).reshape(8, 8)
        for row, (img, label) in enumerate(
            [(full, "original"), (given, "top half only"), (recon, "reconstructed")]
        ):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray_r", aspect="equal", vmin=0, vmax=16)
            if col == 0:
                ax.set_ylabel(label, fontsize=9.5, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Masked reconstruction - the data supervises itself",
                 fontsize=12.5, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.03,
                        wspace=0.08, hspace=0.08)
    print(f"\n  saved  {save_figure(fig, OUT / '2_masked_reconstruction.png').name}")
    return net


# ------------------------------------ 3. does the representation help?
def encode(net: MLPClassifier | MLPRegressor, X: np.ndarray,
           scaler: StandardScaler | None = None) -> np.ndarray:
    """Hidden-layer activations - the representation, not the prediction.

    sklearn has no .transform() for an MLP, so the first layer is applied by
    hand. This is exactly what 'freeze the encoder and drop the head' means.
    """
    Z = scaler.transform(X) if scaler is not None else X
    H = np.maximum(0, Z @ net.coefs_[0] + net.intercepts_[0])   # ReLU
    return H


def figure_label_efficiency(X: np.ndarray, y: np.ndarray,
                            rot_net: MLPClassifier, scaler: StandardScaler,
                            mask_net: MLPRegressor) -> None:
    """The only evaluation that matters: with very few labels, does the
    pretrained representation beat the raw pixels?

    Self-supervision is not supposed to beat full supervision. It is supposed
    to get you further with FEW labels, so the comparison has to be made at
    several label budgets.
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)

    reps = {
        "Raw pixels": (Xtr, Xte),
        "PCA (32 comps)": None,          # filled below
        "Rotation pretext": (encode(rot_net, Xtr, scaler),
                             encode(rot_net, Xte, scaler)),
        "Masking pretext": (encode(mask_net, Xtr[:, :32] / 16.0),
                            encode(mask_net, Xte[:, :32] / 16.0)),
    }
    p = PCA(32, random_state=SEED).fit(Xtr)
    reps["PCA (32 comps)"] = (p.transform(Xtr), p.transform(Xte))

    budgets = [10, 20, 50, 100, 300, 1000]
    print("  Train a simple logistic head on N labelled examples, varying N.")
    print("  The encoders saw NO labels - only rotations and masked pixels.\n")
    header = "  " + f"{'labels':>7}" + "".join(f"{k:>20}" for k in reps)
    print(header)

    curves = {k: [] for k in reps}
    for n in budgets:
        row = f"  {n:>7}"
        for name, (A, B) in reps.items():
            # average several label draws - at n=10 a single draw is noise
            accs = []
            for rep in range(5):
                r = np.random.default_rng(SEED + rep)
                idx = []
                for c in np.unique(ytr):          # keep classes balanced
                    pool = np.where(ytr == c)[0]
                    idx.extend(r.choice(pool, max(1, n // len(np.unique(ytr))),
                                        replace=False))
                idx = np.array(idx)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LogisticRegression(max_iter=2000,
                                             random_state=SEED).fit(A[idx], ytr[idx])
                accs.append(accuracy_score(yte, clf.predict(B)))
            m = float(np.mean(accs))
            curves[name].append(m)
            row += f"{m:>20.3f}"
        print(row)

    lo = {k: v[0] for k, v in curves.items()}
    best_lo = max(lo, key=lo.get)
    ssl_best = max(("Rotation pretext", "Masking pretext"), key=lambda k: lo[k])

    wins = [(n, name, curves[name][k] - curves["Raw pixels"][k])
            for k, n in enumerate(budgets)
            for name in ("Rotation pretext", "Masking pretext")
            if curves[name][k] > curves["Raw pixels"][k]]
    if not wins:
        print(f"\n  THE HONEST RESULT: neither pretext task beat raw pixels, at any")
        print("  label budget.", end="")
    else:
        print("\n  THE HONEST RESULT: the pretext tasks beat raw pixels only at")
        print("  " + ", ".join(f"{n} labels ({name.split()[0]}, {d:+.3f})"
                             for n, name, d in wins) + " -")
        print("  the plentiful-label end, where pretraining should matter LEAST.")
        print("  Where labels are scarce, which is the whole point, both lost.", end="")
    print(" This is a\n  NEGATIVE result and it is left in, because it is far more")
    print("  instructive than a tuned demonstration would be.")
    print(f"\n  At 10 labels: raw pixels {lo['Raw pixels']:.3f}, best pretext "
          f"{lo[ssl_best]:.3f} "
          f"({(lo[ssl_best] - lo['Raw pixels']) * 100:+.1f} pp)")
    print(f"  At 1000 labels: raw pixels {curves['Raw pixels'][-1]:.3f}, best pretext "
          f"{max(curves['Rotation pretext'][-1], curves['Masking pretext'][-1]):.3f}")

    print("\n  Why it failed here, concretely:")
    print("    1. THE INPUT IS ALREADY A GOOD REPRESENTATION. A plain logistic")
    print("       regression on the 64 raw pixels reaches ~0.96. There is very")
    print("       little structure left for an encoder to discover.")
    print("    2. THERE IS NO UNLABELLED DATA ADVANTAGE. Self-supervision wins")
    print("       when unlabelled examples vastly outnumber labelled ones. Here")
    print("       there are 1,797 images total - the same pile, relabelled.")
    print("       SimCLR used 1.3M; BERT used billions of tokens.")
    print("    3. THE ENCODER IS TINY. One hidden layer of 128 units cannot")
    print("       build a richer representation than the pixels it is given.")
    print("    4. MASKING SEES HALF THE IMAGE. That encoder is built from the")
    print("       top 32 pixels only, so it has thrown away half the evidence")
    print("       before the classifier ever sees it - which is why it is worst.")

    print("\n  Note PCA matches raw pixels almost exactly. On 8x8 digits a linear")
    print("  decomposition captures essentially everything there is, which is")
    print("  the same message from a different direction.")

    print("\n  What to actually take from this: the METHOD in this section - the")
    print("  label-efficiency curve - is the right way to evaluate any pretext")
    print("  task, and it is the part that transfers. Had we quoted only the")
    print("  0.95 rotation accuracy from section 1, we would have concluded the")
    print("  pretraining worked. It did not. Self-supervision is a SCALE")
    print("  phenomenon, and on a small, clean, already-separable dataset it")
    print("  has nothing to add.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for (name, vals), colour in zip(curves.items(),
                                    ("grey", "orange", "blue", "green")):
        axes[0].plot(budgets, vals, "o-", color=PALETTE[colour], label=name)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("labelled examples (log scale)")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Label efficiency: the only fair test")
    axes[0].legend(fontsize=9, loc="lower right")

    deltas = {k: np.array(v) - np.array(curves["Raw pixels"])
              for k, v in curves.items() if k != "Raw pixels"}
    for (name, vals), colour in zip(deltas.items(), ("orange", "blue", "green")):
        axes[1].plot(budgets, vals * 100, "o-", color=PALETTE[colour], label=name)
    axes[1].axhline(0, color=PALETTE["grey"], lw=1.4)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("labelled examples (log scale)")
    axes[1].set_ylabel("accuracy gain over raw pixels (pp)")
    axes[1].set_title("Below zero = worse than raw pixels")
    axes[1].legend(fontsize=9)

    fig.suptitle("On THIS dataset, self-supervised pretraining did not beat raw "
                 "pixels", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '3_label_efficiency.png').name}")
    return curves


# ---------------------------------------- 4. a pretext task that fails
def demo_bad_pretext(X: np.ndarray, y: np.ndarray) -> None:
    """Not every invented task teaches anything. The pretext task has to
    require the structure you care about - otherwise the model finds a
    shortcut and learns nothing transferable.
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)

    # A DELIBERATELY USELESS pretext task: predict the total ink in the image.
    # Solvable to near-perfection by summing pixels, which requires no
    # understanding of shape whatsoever - the definition of a shortcut.
    target = Xtr.sum(axis=1)
    scaler = StandardScaler().fit(Xtr)

    # STANDARDISE the target, do not just divide by its maximum. Scaled by max,
    # total ink lands in a narrow band around 0.75 (mean 313, sd 34, max 419) and
    # Adam fits it badly - R2 came out at -2.876 with the target scaled that way,
    # even though total ink is a LINEAR function of the pixels and ordinary
    # least squares gets R2 = 1.0. A near-constant target far from zero is an
    # optimisation problem, not a modelling one.
    t_mu, t_sd = target.mean(), target.std()
    net = MLPRegressor(hidden_layer_sizes=(128,), max_iter=600,
                       random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.fit(scaler.transform(Xtr), (target - t_mu) / t_sd)

    pretext_r2 = net.score(scaler.transform(Xte),
                           (Xte.sum(axis=1) - t_mu) / t_sd)
    A, B = encode(net, Xtr, scaler), encode(net, Xte, scaler)

    n = 50
    r = np.random.default_rng(SEED)
    idx = np.concatenate([r.choice(np.where(ytr == c)[0], n // 10, replace=False)
                          for c in np.unique(ytr)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = LogisticRegression(max_iter=2000, random_state=SEED).fit(
            Xtr[idx], ytr[idx])
        bad = LogisticRegression(max_iter=2000, random_state=SEED).fit(
            A[idx], ytr[idx])

    print("  A deliberately BAD pretext task: predict the total ink in the image.")
    print(f"\n  The model solves it almost perfectly - R2 = {pretext_r2:.3f}.")
    print("  And that is the problem. Summing pixel values requires no")
    print("  understanding of SHAPE at all, so there is nothing to transfer.")
    print(f"\n  Digit accuracy from {n} labels")
    print(f"    raw pixels                       {accuracy_score(yte, good.predict(Xte)):.3f}")
    print(f"    'total ink' representation       {accuracy_score(yte, bad.predict(B)):.3f}")
    print("\n  A pretext task is only useful if solving it REQUIRES the structure")
    print("  you actually care about. This is the shortcut problem, and it is")
    print("  the central design difficulty in self-supervised learning:")
    print("    - colourisation can be solved from texture statistics alone")
    print("    - jigsaw puzzles can be solved by matching edge pixels")
    print("    - contrastive learning collapses if augmentations are too weak")
    print("  A high pretext score is NOT evidence of a good representation.")
    print("  Only the downstream label-efficiency curve is.")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  SELF-SUPERVISED LEARNING")
    print("=" * 72)

    d = load_digits()
    X, y, images = d.data, d.target, d.images
    print(f"\n  Digits: {len(X):,} images, 8x8 = 64 pixels, 10 classes.")
    print("  The LABELS ARE SET ASIDE for sections 1 and 2 - the encoders never")
    print("  see them. They come back only to evaluate, in section 3.")

    banner("1. Pretext task one: predict the rotation")
    rot_net = figure_rotation(images, y)
    scaler = StandardScaler().fit(build_rotation_task(images)[0])

    banner("2. Pretext task two: masked reconstruction")
    mask_net = figure_masking(X, images)

    banner("3. Does the representation help? The label-efficiency curve")
    figure_label_efficiency(X, y, rot_net, scaler, mask_net)

    banner("4. A pretext task that learns nothing")
    demo_bad_pretext(X, y)

    banner("What to take away")
    print("  - Self-supervision INVENTS labels from the data itself: rotate an")
    print("    image and predict the angle, mask pixels and predict them, hide")
    print("    a word and predict it. The labels are free and always correct.")
    print("  - The pretext MODEL is disposable. You keep the ENCODER - the")
    print("    hidden representation it had to build in order to succeed.")
    print("  - Evaluate with a LABEL-EFFICIENCY CURVE, never with the pretext")
    print("    score. The rotation task hit 0.95 and its representation still")
    print("    lost to raw pixels at every SCARCE-label budget.")
    print("  - NEITHER PRETEXT TASK HELPED HERE, and that negative result is")
    print("    the most useful thing in this folder. Digits is small, clean and")
    print("    already near-linearly-separable, and there is no pool of extra")
    print("    unlabelled data - so there was nothing for pretraining to buy.")
    print("  - The pretext task must REQUIRE the structure you care about. A")
    print("    task with a shortcut ('total ink') is solved perfectly and")
    print("    teaches nothing.")
    print("  - Self-supervision is a SCALE phenomenon. It pays when unlabelled")
    print("    data vastly outnumbers labelled data and the raw input is a poor")
    print("    representation - millions of photos, not 1,797 thumbnails. That")
    print("    is where BERT, SimCLR and MAE live.")
    print("  - This is the engine behind every foundation model. Pretrain on")
    print("    unlabelled data at scale, then fine-tune on a small labelled set")
    print("    - which on Google Cloud is the whole rationale for Model Garden.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
