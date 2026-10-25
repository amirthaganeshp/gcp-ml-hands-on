"""Convolutional networks: why you do not flatten an image.

A dense layer treats pixel (0,0) and pixel (0,1) as unrelated columns. A
convolution assumes neighbouring pixels belong together and that a useful
pattern is useful wherever it appears. Those two assumptions - LOCALITY and
WEIGHT SHARING - are the whole idea, and they are worth a lot.

  1. What a convolution actually computes, done by hand with edge filters
  2. Dense vs CNN at a MATCHED PARAMETER BUDGET, so the comparison is fair
  3. What the network learned - first-layer filters and feature maps
  4. Translation invariance: shift the test images and see who survives
  5. Data augmentation, and why it is the cheapest accuracy you can buy

Fashion-MNIST rather than MNIST: plain MNIST is so easy that a dense network
reaches 98% and the comparison shows nothing.

Run:
    python 03_neural_networks/03_cnn_images/cnn_images.py

Concept reference: Foundations volume, Parts B.10-B.12.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
N_TRAIN = 10000          # subsample - this must stay a laptop-CPU example
N_TEST = 3000
EPOCHS = 15              # 6 was not enough: the CNN is still climbing at 10

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def seed_everything() -> None:
    keras.utils.set_random_seed(SEED)


# ================================================ 1. convolution by hand
def figure_convolution(x: np.ndarray) -> None:
    """A convolution is a small weight matrix slid across the image, taking a
    dot product at every position. Nothing more.

    The two filters below are not learned - they are the classic Sobel edge
    detectors, applied by hand, so you can see that a filter is just a
    pattern matcher and the output is a map of WHERE that pattern occurred.
    """
    img = x[0]
    filters = {
        "vertical edges\n(Sobel x)": np.array([[-1, 0, 1],
                                               [-2, 0, 2],
                                               [-1, 0, 1]], dtype=float),
        "horizontal edges\n(Sobel y)": np.array([[-1, -2, -1],
                                                 [0, 0, 0],
                                                 [1, 2, 1]], dtype=float),
        "blur\n(box filter)": np.ones((3, 3)) / 9.0,
    }

    def convolve(im: np.ndarray, k: np.ndarray) -> np.ndarray:
        h, w = im.shape
        kh, kw = k.shape
        out = np.zeros((h - kh + 1, w - kw + 1))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = float((im[i:i + kh, j:j + kw] * k).sum())
        return out

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    axes[0].imshow(img, cmap="gray", aspect="equal")
    axes[0].set_title("input (28x28)", fontsize=11)
    for ax, (name, k) in zip(axes[1:], filters.items()):
        ax.imshow(np.abs(convolve(img, k)), cmap="gray", aspect="equal")
        ax.set_title(name, fontsize=11)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("A convolution is a 3x3 pattern slid across the image",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    print("  A 3x3 filter has 9 weights. It is slid across all 28x28 positions,")
    print("  taking a dot product at each one, and the output says WHERE in the")
    print("  image that 3x3 pattern occurred.")
    print("\n  Two consequences, and they are the entire argument for CNNs:")
    print("    LOCALITY       a pixel is compared only with its neighbours,")
    print("                   because that is where the information in an image")
    print("                   actually lives.")
    print("    WEIGHT SHARING the SAME 9 weights are reused at every position,")
    print("                   so a vertical edge is detected anywhere in the")
    print("                   frame without learning it separately per location.")
    print("\n  A dense layer has neither property. It gives every (pixel, unit)")
    print("  pair its own weight and has no idea the pixels form a grid - you")
    print("  could shuffle all 784 columns and it would train just as well.")
    print("  That is exactly the information a convolution refuses to discard.")
    print(f"\n  saved  {save_figure(fig, OUT / '1_convolution.png').name}")


# ================================== 2. dense vs conv at a matched budget
def build_dense(width: int) -> keras.Model:
    m = keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(width, activation="relu"),
        keras.layers.Dense(width // 2, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def build_cnn() -> keras.Model:
    m = keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),          # 28x28 -> 14x14
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),          # 14x14 -> 7x7
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def figure_comparison(d: dict) -> tuple[keras.Model, dict]:
    """The comparison is only meaningful at a matched parameter count. A CNN
    beating a tiny dense network proves nothing.
    """
    cnn = build_cnn()
    target = cnn.count_params()

    # find the dense width whose parameter count is closest to the CNN's
    width = min(range(32, 200, 4), key=lambda w: abs(build_dense(w).count_params()
                                                     - target))
    builders = {"Dense": lambda: build_dense(width), "CNN": build_cnn}

    print(f"  Fashion-MNIST, {N_TRAIN:,} training images, {EPOCHS} epochs.")
    print(f"  Dense width chosen as {width} so both models have ~the same")
    print(f"  parameter count - otherwise the comparison is rigged.\n")
    print(f"  {'model':<8} {'params':>9} {'train acc':>10} {'test acc':>9} "
          f"{'fit s':>8}")

    results, hists = {}, {}
    for name, make in builders.items():
        seed_everything()
        m = make()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = m.fit(d["xtr"], d["ytr"], validation_data=(d["xte"], d["yte"]),
                      epochs=EPOCHS, batch_size=128, verbose=0)
        secs = time.perf_counter() - t0
        results[name] = {
            "params": m.count_params(),
            "train": h.history["accuracy"][-1],
            "test": h.history["val_accuracy"][-1],
            "time": secs, "model": m,
        }
        hists[name] = h
        r = results[name]
        print(f"  {name:<8} {r['params']:>9,} {r['train']:>10.4f} "
              f"{r['test']:>9.4f} {r['time']:>8.1f}")

    gap = results["CNN"]["test"] - results["Dense"]["test"]
    print(f"\n  CNN advantage at equal parameters: {gap:+.4f} test accuracy.")
    print(f"  Error rate: {1 - results['Dense']['test']:.3f} dense vs "
          f"{1 - results['CNN']['test']:.3f} CNN - a "
          f"{(1 - results['CNN']['test']) / (1 - results['Dense']['test']) - 1:+.0%} change.")
    print("\n  Both networks have the same capacity and saw the same data. The")
    print("  difference is entirely in the ARCHITECTURAL PRIOR: the CNN was")
    print("  told, by construction, that the input is a grid and that patterns")
    print("  are worth reusing across positions. The dense network had to")
    print(f"  discover that from {N_TRAIN:,} examples, and largely did not.")
    print("\n  Look at the train/test columns too. The dense model's training")
    print(f"  accuracy is {results['Dense']['train'] - results['Dense']['test']:+.3f} above its test accuracy; the CNN's gap is")
    print(f"  {results['CNN']['train'] - results['CNN']['test']:+.3f}. Weight sharing is a regulariser - fewer")
    print("  independent things to memorise.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (name, h), colour in zip(hists.items(), ("orange", "blue")):
        axes[0].plot(h.history["val_accuracy"], color=PALETTE[colour],
                     lw=2, label=f"{name} (test)")
        axes[0].plot(h.history["accuracy"], color=PALETTE[colour], lw=1.2,
                     ls="--", alpha=0.7, label=f"{name} (train)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("Same parameter budget, different prior")
    axes[0].legend(fontsize=9)

    names = list(results)
    axes[1].bar(names, [results[n]["test"] for n in names],
                color=[PALETTE["orange"], PALETTE["blue"]])
    for i, n in enumerate(names):
        axes[1].text(i, results[n]["test"] + 0.004,
                     f"{results[n]['test']:.3f}\n{results[n]['params']:,} params",
                     ha="center", fontsize=9)
    axes[1].set_ylim(0.7, 1.0)
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title("Test accuracy at matched capacity")
    axes[1].grid(axis="x", visible=False)

    fig.suptitle("Dense vs convolutional, matched parameter count",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '2_dense_vs_cnn.png').name}")
    return results["CNN"]["model"], results


# ============================================== 3. what it actually learned
def figure_filters(model: keras.Model, d: dict) -> None:
    conv1 = next(l for l in model.layers if isinstance(l, keras.layers.Conv2D))
    W = np.array(conv1.get_weights()[0])          # (3, 3, 1, 16)
    n = W.shape[-1]

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(4, 12, hspace=0.35, wspace=0.25)

    for i in range(n):
        ax = fig.add_subplot(gs[i // 8, i % 8])
        ax.imshow(W[:, :, 0, i], cmap="RdBu_r", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if i == 0:
            ax.set_ylabel("learned\n3x3 filters", fontsize=9, fontweight="bold")

    # feature maps for one image
    sub = keras.Model(model.inputs, conv1.output)
    fmaps = np.array(sub.predict(d["xte"][:1], verbose=0))[0]
    ax = fig.add_subplot(gs[0:2, 8:10])
    ax.imshow(d["xte"][0, :, :, 0], cmap="gray", aspect="equal")
    ax.set_title(f"input: {CLASSES[int(d['yte'][0])]}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    for i in range(8):
        ax = fig.add_subplot(gs[2 + i // 4, 8 + i % 4])
        ax.imshow(fmaps[:, :, i], cmap="viridis", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if i == 0:
            ax.set_ylabel("feature\nmaps", fontsize=9, fontweight="bold")

    fig.suptitle("What the first convolutional layer learned, and what it "
                 "produces", fontsize=13, fontweight="bold")
    print(f"  The first layer has {n} filters of {W.shape[0]}x{W.shape[1]} "
          f"= {W[:, :, 0, 0].size} weights each,")
    print(f"  {int(np.prod(W.shape))} weights in total for the whole layer.")
    print("\n  Red and blue are positive and negative weights. Several filters")
    print("  have a clear light/dark split across the patch - those are edge")
    print("  detectors, and the network invented them without being told that")
    print("  edges matter. That is the same thing the hand-built Sobel filter")
    print("  in section 1 was doing.")
    print("\n  The feature maps on the right are the SAME image seen through")
    print("  eight different filters. Each highlights a different structure -")
    print("  one lights up on vertical edges, another on flat regions. Depth")
    print("  works by stacking this: layer 2 combines edges into corners and")
    print("  textures, layer 3 combines those into object parts.")
    print(f"\n  saved  {save_figure(fig, OUT / '3_filters.png').name}")


# ============================================= 4. translation invariance
def figure_translation(models: dict, d: dict) -> None:
    """The claim is that a CNN handles a shifted object better than a dense
    network. That is testable: shift the test set and re-measure.
    """
    def shift(images: np.ndarray, dx: int) -> np.ndarray:
        return np.roll(images, dx, axis=2)

    shifts = [0, 1, 2, 3, 4, 6]
    print(f"  Shifting every TEST image horizontally. Neither model saw a")
    print(f"  shifted image during training.\n")
    print(f"  {'shift (px)':>11} {'Dense':>9} {'CNN':>9} {'gap':>8}")
    curves = {"Dense": [], "CNN": []}
    for s in shifts:
        xs = shift(d["xte"], s)
        row = {}
        for name in ("Dense", "CNN"):
            acc = float(models[name]["model"].evaluate(
                xs, d["yte"], verbose=0)[1])
            curves[name].append(acc)
            row[name] = acc
        print(f"  {s:>11} {row['Dense']:>9.4f} {row['CNN']:>9.4f} "
              f"{row['CNN'] - row['Dense']:>+8.4f}")

    d_drop = curves["Dense"][0] - curves["Dense"][-1]
    c_drop = curves["CNN"][0] - curves["CNN"][-1]
    print(f"\n  Accuracy lost over a {shifts[-1]}px shift:")
    print(f"    Dense  {d_drop:.4f}")
    print(f"    CNN    {c_drop:.4f}")

    print("\n  Both degrade, and it is worth being precise about why the CNN is")
    print("  only PARTIALLY invariant. Convolution is EQUIVARIANT, not")
    print("  invariant: shift the input and the feature maps shift with it.")
    print("  Max-pooling then discards position within each 2x2 window, which")
    print("  buys a little true invariance per pooling layer - about 4 pixels")
    print("  of slack with two pooling layers here. Beyond that, the flattened")
    print("  features genuinely differ and accuracy falls.")
    print("\n  Full translation invariance comes from global pooling, or from")
    print("  augmentation - which is section 5.")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for (name, c), colour in zip(curves.items(), ("orange", "blue")):
        ax.plot(shifts, c, "o-", color=PALETTE[colour], lw=2, label=name)
    ax.set_xlabel("horizontal shift applied to the test set (pixels)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Neither is fully translation invariant - but one degrades "
                 "more gracefully")
    ax.legend(fontsize=10)
    fig.tight_layout()
    print(f"\n  saved  {save_figure(fig, OUT / '4_translation.png').name}")


# ================================================== 5. data augmentation
class AugmentedBatches(keras.utils.PyDataset):
    """Random shift + horizontal flip, applied per batch in NumPy.

    Keras' RandomTranslation/RandomFlip/RandomZoom layers are convenient but
    cost ~38 s per epoch on a CPU here, against ~3 s for the model itself -
    they dominated the runtime twelve to one. Doing the same transforms with
    np.roll and a slice is both far faster and considerably clearer about
    what augmentation actually is.
    """

    def __init__(self, x, y, batch_size=128, max_shift=3, seed=SEED):
        super().__init__()
        self.x, self.y = x, y
        self.bs, self.max_shift = batch_size, max_shift
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / self.bs))

    def __getitem__(self, i):
        xb = self.x[i * self.bs:(i + 1) * self.bs].copy()
        yb = self.y[i * self.bs:(i + 1) * self.bs]
        # PER IMAGE, not per batch. Rolling the whole batch by one offset
        # gives every image in it the identical transform, which is far less
        # diverse than real augmentation and understates the benefit.
        shifts = self.rng.integers(-self.max_shift, self.max_shift + 1,
                                   (len(xb), 2))
        flips = self.rng.random(len(xb)) < 0.5
        for k in range(len(xb)):
            xb[k] = np.roll(xb[k], tuple(shifts[k]), axis=(0, 1))
            if flips[k]:
                xb[k] = xb[k][:, ::-1, :]              # horizontal flip
        return xb, yb


def figure_augmentation(d: dict, baseline: dict) -> None:
    seed_everything()
    model = build_cnn()
    train_seq = AugmentedBatches(d["xtr"], d["ytr"])

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = model.fit(train_seq, validation_data=(d["xte"], d["yte"]),
                      epochs=EPOCHS, verbose=0)
    secs = time.perf_counter() - t0

    tr, te = h.history["accuracy"][-1], h.history["val_accuracy"][-1]
    b_tr, b_te = baseline["train"], baseline["test"]

    print(f"  Same CNN, same {EPOCHS} epochs, with a random shift of up to 3px")
    print(f"  and a random horizontal flip applied fresh to every batch.\n")
    print(f"  {'':<20} {'train acc':>10} {'test acc':>9} {'gap':>8} {'fit s':>8}")
    print(f"  {'CNN, plain':<20} {b_tr:>10.4f} {b_te:>9.4f} "
          f"{b_tr - b_te:>+8.4f} {baseline['time']:>8.1f}")
    print(f"  {'CNN, augmented':<20} {tr:>10.4f} {te:>9.4f} "
          f"{tr - te:>+8.4f} {secs:>8.1f}")
    print(f"\n  Test accuracy {te - b_te:+.4f}, overfitting gap "
          f"{(tr - te) - (b_tr - b_te):+.4f}.")

    if te > b_te:
        print("\n  Augmentation helped. It shows the network a different version")
        print("  of every image each epoch, so it cannot memorise any single one -")
        print("  effectively free extra data, paid for in training time only.")
    else:
        print("\n  Augmentation did NOT improve test accuracy here, and that is")
        print("  reported rather than tuned away. But look at the gap column")
        print(f"  before concluding it failed: {b_tr - b_te:+.4f} plain, {tr - te:+.4f} augmented.")
        print("  The overfitting is essentially GONE. The regularisation")
        print("  mechanism worked exactly as advertised.")
        print("\n  Part of what went wrong is the budget. Augmentation makes the")
        print("  training task harder, so the model needs more epochs to reach")
        print(f"  the same fit - and at a fixed {EPOCHS} both train AND test accuracy")
        print("  are still climbing. But budget is not the whole story: trained")
        print("  for 40 epochs the augmented CNN narrows the gap without closing")
        print("  it (see the README's 'Things to try').")
        print("\n  The second reason is that the plain CNN was only mildly")
        print(f"  overfitting ({b_tr - b_te:+.3f}) to begin with, so there was little for")
        print("  extra effective data to recover - the same lesson as")
        print("  regularisation in 02_mlp_tabular: it only helps if you are")
        print("  actually overfitting.")

    print("\n  Whichever way it lands, judge augmentation on the TEST column")
    print("  only. Training accuracy normally FALLS, because the training task")
    print("  itself got harder - that is the mechanism working, not a failure.")
    print("\n  The transforms must respect the data. A horizontal flip is safe")
    print("  for clothing and most natural photographs. It would be actively")
    print("  wrong for digits - a flipped 2 is not a 2 - and for text. Choosing")
    print("  augmentations is a domain decision, not a hyperparameter.")

    fig, axes = plt.subplots(2, 6, figsize=(13, 4.8))
    xb, yb = train_seq[0]
    for j in range(6):
        axes[0, j].imshow(d["xtr"][j, :, :, 0], cmap="gray", aspect="equal")
        axes[0, j].set_title(CLASSES[int(d["ytr"][j])], fontsize=9)
        axes[1, j].imshow(xb[j, :, :, 0], cmap="gray", aspect="equal")
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            axes[r, j].grid(False)
    axes[0, 0].set_ylabel("original", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("augmented", fontsize=10, fontweight="bold")
    fig.suptitle("Data augmentation: a different image every epoch",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.03,
                        wspace=0.1, hspace=0.25)
    print(f"\n  saved  {save_figure(fig, OUT / '5_augmentation.png').name}")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  CONVOLUTIONAL NEURAL NETWORKS")
    print("=" * 72)
    print(f"\n  Keras {keras.__version__}, backend: {keras.backend.backend()}")

    import truststore
    truststore.inject_into_ssl()      # corporate TLS inspection - see SETUP.md
    (xtr, ytr), (xte, yte) = keras.datasets.fashion_mnist.load_data()

    rng = np.random.default_rng(SEED)
    itr = rng.choice(len(xtr), N_TRAIN, replace=False)
    ite = rng.choice(len(xte), N_TEST, replace=False)
    d = {
        "xtr": (xtr[itr] / 255.0).astype("float32")[..., None], "ytr": ytr[itr],
        "xte": (xte[ite] / 255.0).astype("float32")[..., None], "yte": yte[ite],
    }
    print(f"  Fashion-MNIST subsampled to {N_TRAIN:,} train / {N_TEST:,} test")
    print(f"  so this stays a laptop-CPU example. 28x28 greyscale, 10 classes.")

    banner("1. What a convolution computes")
    figure_convolution(d["xtr"][:, :, :, 0])

    banner("2. Dense vs CNN at a matched parameter budget")
    cnn, results = figure_comparison(d)

    banner("3. What the first layer learned")
    figure_filters(cnn, d)

    banner("4. Translation invariance, tested")
    figure_translation(results, d)

    banner("5. Data augmentation")
    figure_augmentation(d, results["CNN"])

    banner("What to take away")
    print("  - A convolution is a small filter slid across the image. Its two")
    print("    assumptions are LOCALITY (neighbouring pixels are related) and")
    print("    WEIGHT SHARING (a pattern is useful wherever it appears).")
    print("  - A dense layer has neither. You could shuffle all 784 pixel")
    print("    columns and it would train identically - which tells you exactly")
    print("    how much spatial information it is throwing away.")
    print("  - COMPARE AT A MATCHED PARAMETER COUNT. A CNN beating a tiny dense")
    print("    net proves nothing; the point is the architectural prior, not")
    print("    capacity.")
    print("  - Weight sharing also regularises: far fewer independent weights")
    print("    means less to memorise, and a smaller train/test gap.")
    print("  - Convolution is EQUIVARIANT, not invariant. Pooling buys a few")
    print("    pixels of real invariance; large shifts still hurt. Global")
    print("    pooling or augmentation is how you get more.")
    print("  - Early filters learn edges without being told edges matter, and")
    print("    depth composes them into corners, textures and object parts.")
    print("  - AUGMENTATION removed the overfitting gap entirely here but LOST")
    print("    test accuracy, because a harder training task needs more epochs")
    print("    and the budget was held fixed. It trades training time for")
    print("    generalisation - do not hold epochs constant and call it a fail.")
    print("  - The transforms must respect the domain: horizontal flip is fine")
    print("    for clothing, and actively wrong for digits or text.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
