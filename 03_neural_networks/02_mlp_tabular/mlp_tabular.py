"""A multilayer perceptron on tabular data - and whether it was worth it.

This is the first Keras model in the repository, and the most important
question it answers is not "how do I build an MLP" but "should I". On
tabular data the honest answer is usually no, and the comparison against
gradient boosting is run properly rather than skipped.

  1. The same network, now in Keras instead of NumPy - what the framework
     is and is not doing for you
  2. MLP vs XGBoost vs logistic regression on real tabular data
  3. Regularisation: dropout, batch normalisation, early stopping, and
     which of them actually earned its place here
  4. Learning rate - the one hyperparameter that matters most
  5. Preprocessing is not optional for a neural network, and the failure
     mode when you skip it

Run:
    python 03_neural_networks/02_mlp_tabular/mlp_tabular.py

Concept reference: Foundations volume, Parts B.6-B.9.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

# Keras 3 is backend-agnostic. TensorFlow will not install on a deep Windows
# path (see requirements-dl.txt), so the torch backend is the default here.
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_rice  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def seed_everything() -> None:
    keras.utils.set_random_seed(SEED)


def build_mlp(n_features: int, hidden=(64, 32), dropout: float = 0.0,
              batchnorm: bool = False, lr: float = 1e-3) -> keras.Model:
    """The Keras equivalent of the NumPy network in 01_perceptron_from_scratch.

    Everything the framework adds is visible here: it derives the gradients,
    runs Adam instead of plain SGD, batches the data, and handles the
    train/eval distinction for dropout and batch norm. It does NOT choose the
    architecture, the preprocessing or the learning rate.
    """
    layers = [keras.layers.Input((n_features,))]
    for h in hidden:
        layers.append(keras.layers.Dense(h, activation="relu"))
        if batchnorm:
            layers.append(keras.layers.BatchNormalization())
        if dropout:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(1, activation="sigmoid"))

    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def fit_quiet(model: keras.Model, Xtr, ytr, Xva, yva, epochs: int = 60,
              patience: int | None = 10) -> keras.callbacks.History:
    cbs = []
    if patience:
        cbs.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
                         batch_size=64, verbose=0, callbacks=cbs)


# ============================================ 1-2. does the MLP earn its place
def figure_comparison(data: dict) -> dict:
    Xtr, Xva, Xte = data["Xtr"], data["Xva"], data["Xte"]
    ytr, yva, yte = data["ytr"], data["yva"], data["yte"]

    print(f"  Rice: {len(Xtr):,} train / {len(Xva):,} val / {len(Xte):,} test, "
          f"{Xtr.shape[1]} features.\n")

    results = {}

    seed_everything()
    mlp = build_mlp(Xtr.shape[1])
    t0 = time.perf_counter()
    hist = fit_quiet(mlp, Xtr, ytr, Xva, yva)
    mlp_time = time.perf_counter() - t0
    p = mlp.predict(Xte, verbose=0).ravel()
    results["MLP (64-32)"] = {
        "acc": accuracy_score(yte, p > 0.5), "auc": roc_auc_score(yte, p),
        "time": mlp_time, "params": mlp.count_params(),
        "epochs": len(hist.history["loss"]),
    }

    t0 = time.perf_counter()
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, verbosity=0,
                        eval_metric="logloss").fit(Xtr, ytr)
    px = xgb.predict_proba(Xte)[:, 1]
    results["XGBoost"] = {
        "acc": accuracy_score(yte, px > 0.5), "auc": roc_auc_score(yte, px),
        "time": time.perf_counter() - t0,
        "params": int(sum(t.count("\n") for t in xgb.get_booster().get_dump())),
        "epochs": 300,
    }

    t0 = time.perf_counter()
    lr_model = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)
    pl = lr_model.predict_proba(Xte)[:, 1]
    results["Logistic regression"] = {
        "acc": accuracy_score(yte, pl > 0.5), "auc": roc_auc_score(yte, pl),
        "time": time.perf_counter() - t0, "params": Xtr.shape[1] + 1, "epochs": 1,
    }

    print(f"  {'model':<22} {'accuracy':>9} {'ROC-AUC':>9} {'params':>9} "
          f"{'fit s':>8} {'epochs':>7}")
    for name, r in results.items():
        print(f"  {name:<22} {r['acc']:>9.4f} {r['auc']:>9.4f} "
              f"{r['params']:>9,} {r['time']:>8.2f} {r['epochs']:>7}")

    best = max(results, key=lambda k: results[k]["auc"])
    mlp_r, lr_r = results["MLP (64-32)"], results["Logistic regression"]
    print(f"\n  Best ROC-AUC: {best}")
    print(f"  Spread across all three: "
          f"{max(r['auc'] for r in results.values()) - min(r['auc'] for r in results.values()):.4f}")

    print(f"\n  The MLP has {mlp_r['params']:,} parameters and took "
          f"{mlp_r['time'] / lr_r['time']:.0f}x longer to fit than")
    print(f"  logistic regression, which has {lr_r['params']}. For that it delivered")
    print(f"  {mlp_r['auc'] - lr_r['auc']:+.4f} ROC-AUC.")
    print("\n  This is the honest state of neural networks on small tabular data:")
    print("  they are competitive, not dominant, and they cost far more in")
    print("  tuning, training time and operational complexity. Gradient-boosted")
    print("  trees remain the default for a reason - they need no scaling, no")
    print("  architecture choice, and no learning-rate search.")
    print("\n  Neural networks earn their place on tabular data when you have")
    print("  very high-cardinality categoricals worth EMBEDDING (see 06), when")
    print("  you need to fuse tabular data with text or images in one model, or")
    print("  when the dataset is big enough that trees stop scaling.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hist.history["loss"], color=PALETTE["blue"], label="train")
    axes[0].plot(hist.history["val_loss"], color=PALETTE["vermillion"],
                 label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("binary cross-entropy")
    axes[0].set_title(f"MLP training ({mlp_r['epochs']} epochs, early stopped)")
    axes[0].legend(fontsize=9)

    names = list(results)
    axes[1].barh(names, [results[n]["auc"] for n in names], color=PALETTE["blue"])
    axes[1].set_xlim(0.95, 1.0)
    axes[1].set_xlabel("ROC-AUC")
    axes[1].set_title("Three models, one clean tabular problem")
    axes[1].grid(axis="y", visible=False)

    fig.suptitle("Does the neural network earn its place here?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '1_mlp_vs_trees.png').name}")
    return results


# ================================================== 3. regularisation, tested
def figure_regularisation(data: dict) -> None:
    """Four variants, measured rather than assumed. Regularisation is only
    useful if the model is actually overfitting, and on a clean 3,800-row
    dataset it may well not be.
    """
    Xtr, Xva, Xte = data["Xtr"], data["Xva"], data["Xte"]
    ytr, yva, yte = data["ytr"], data["yva"], data["yte"]

    variants = {
        "Plain (64-32)": dict(dropout=0.0, batchnorm=False),
        "+ Dropout 0.3": dict(dropout=0.3, batchnorm=False),
        "+ BatchNorm": dict(dropout=0.0, batchnorm=True),
        "+ Both": dict(dropout=0.3, batchnorm=True),
        "Oversized (512-512-512)": dict(dropout=0.0, batchnorm=False,
                                        hidden=(512, 512, 512)),
    }

    print(f"  {'variant':<26} {'val loss':>9} {'test AUC':>9} {'params':>9} "
          f"{'epochs':>7} {'gap':>7}")
    histories = {}
    for name, kw in variants.items():
        seed_everything()
        m = build_mlp(Xtr.shape[1], **kw)
        h = fit_quiet(m, Xtr, ytr, Xva, yva, epochs=60)
        p = m.predict(Xte, verbose=0).ravel()
        gap = h.history["val_loss"][-1] - h.history["loss"][-1]
        histories[name] = h
        print(f"  {name:<26} {min(h.history['val_loss']):>9.4f} "
              f"{roc_auc_score(yte, p):>9.4f} {m.count_params():>9,} "
              f"{len(h.history['loss']):>7} {gap:>+7.3f}")

    print("\n  The 'gap' column is validation loss minus training loss - the")
    print("  direct measure of overfitting. A small or negative gap means the")
    print("  model is NOT overfitting, and regularising it further can only")
    print("  cost you.")
    print("\n  On this dataset the plain network is already fine. Rice has 3,800")
    print("  rows, 7 well-behaved features and two well-separated classes;")
    print("  there is very little to overfit TO. Dropout and batch norm are")
    print("  answers to a problem this data does not have.")
    print("\n  The oversized network is the interesting row. It has orders of")
    print("  magnitude more parameters than training rows and still does not")
    print("  collapse - early stopping catches it first. Capacity alone does")
    print("  not cause overfitting; capacity plus enough epochs does.")
    print("\n  What each one actually does:")
    print("    DROPOUT randomly zeroes activations during training only, so no")
    print("      unit can rely on any other. At predict time it is off and the")
    print("      activations are rescaled - which is why train and test loss")
    print("      are not comparable when dropout is on.")
    print("    BATCH NORM standardises each layer's pre-activations using the")
    print("      statistics of the current BATCH, then learns a scale and shift.")
    print("      It mainly stabilises and accelerates training; the")
    print("      regularisation is a side effect of the batch noise.")
    print("    EARLY STOPPING is the cheapest regulariser there is, and it is")
    print("      on in every run above - watch the epoch counts differ.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (name, h), colour in zip(histories.items(),
                                 ("blue", "vermillion", "green", "orange", "pink")):
        axes[0].plot(h.history["val_loss"], color=PALETTE[colour], label=name)
        axes[1].plot(np.array(h.history["val_loss"]) - np.array(h.history["loss"]),
                     color=PALETTE[colour], label=name)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation loss")
    axes[0].set_title("Validation loss")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val loss - train loss")
    axes[1].set_title("The overfitting gap (above 0 = overfitting)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Regularisation only helps if you are actually overfitting",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '2_regularisation.png').name}")


# ================================================== 4. the learning rate
def figure_learning_rate(data: dict) -> None:
    Xtr, Xva = data["Xtr"], data["Xva"]
    ytr, yva = data["ytr"], data["yva"]

    rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    epochs = 30
    curves, finals, bests = {}, {}, {}
    for lr in rates:
        seed_everything()
        m = build_mlp(Xtr.shape[1], lr=lr)
        h = fit_quiet(m, Xtr, ytr, Xva, yva, epochs=epochs, patience=None)
        curves[lr] = h.history["loss"]
        finals[lr] = h.history["loss"][-1]
        bests[lr] = min(h.history["val_loss"])

    # Classify RELATIVE to the best run, after seeing them all. Judging each
    # rate against a fixed threshold mislabels them - an absolute cut called
    # lr=1e-5 "healthy" at loss 0.47 and lr=1.0 "too slow" at loss 6.90.
    chance = float(-np.log(0.5))          # loss of a model that always says 0.5
    best_final = min(v for v in finals.values() if np.isfinite(v))

    print(f"  Training {epochs} epochs at each rate, no early stopping, so the")
    print(f"  curves are comparable. Chance-level loss is {chance:.3f}.\n")
    print(f"  {'learning rate':>14} {'final train loss':>18} "
          f"{'best val loss':>15}  behaviour")
    for lr in rates:
        f, b = finals[lr], bests[lr]
        if not np.isfinite(f):
            note = "DIVERGED to NaN"
        elif f >= chance:
            note = "BROKEN - no better than guessing"
        elif f > 1.5 * best_final:
            note = "too slow - still descending"
        else:
            note = "healthy"
        print(f"  {lr:>14.0e} {f:>18.4f} {b:>15.4f}  {note}")

    healthy = [lr for lr in rates
               if np.isfinite(finals[lr]) and finals[lr] < 1.5 * best_final]
    print(f"\n  The healthy band here runs {min(healthy):.0e} to {max(healthy):.0e} - "
          f"{np.log10(max(healthy) / min(healthy)):.0f} orders of")
    print("  magnitude wide. That is wider than people expect, and it is why")
    print("  Adam's 1e-3 default so often just works.")
    print(f"\n  At 1e-5 the steps are so small that {epochs} epochs is not enough to")
    print("  leave the starting region. It looks like a bad model and is not -")
    print("  it is an under-trained one, and more epochs would fix it.")
    print(f"\n  At 1.0 the loss sits at {finals[1.0]:.1f}, far ABOVE chance. Each step")
    print("  overshoots the minimum so badly that the weights are thrown into a")
    print("  region worse than random initialisation, and Adam cannot recover.")
    print("  This is the failure to recognise: a loss that rises, plateaus high,")
    print("  or goes NaN is almost always the learning rate - not the")
    print("  architecture, not the data. Check it before changing anything else.")

    fig, ax = plt.subplots(figsize=(10, 5.4))
    for (lr, c), colour in zip(curves.items(),
                               ("grey", "blue", "green", "orange",
                                "vermillion", "pink")):
        ax.plot(c, color=PALETTE[colour], lw=1.8, label=f"lr = {lr:g}")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_yscale("log")
    ax.set_title("Six learning rates, same network, same data")
    ax.legend(fontsize=9)
    fig.tight_layout()
    print(f"\n  saved  {save_figure(fig, OUT / '3_learning_rate.png').name}")


# ============================================ 5. preprocessing is mandatory
def demo_scaling(raw: dict) -> None:
    """Trees are invariant to feature scale. Neural networks are emphatically
    not, and the failure is quiet rather than loud.
    """
    print("  Rice's features range from 'extent' (~0.5) to 'area' (~15,000).\n")
    print(f"  {'input':<22} {'test ROC-AUC':>13} {'epochs run':>12}")
    for name, (Xtr, Xva, Xte) in raw.items():
        seed_everything()
        m = build_mlp(Xtr.shape[1])
        h = fit_quiet(m, Xtr, raw_y["ytr"], Xva, raw_y["yva"], epochs=60)
        p = m.predict(Xte, verbose=0).ravel()
        print(f"  {name:<22} {roc_auc_score(raw_y['yte'], p):>13.4f} "
              f"{len(h.history['loss']):>12}")

    print("\n  Unscaled, the first layer receives inputs spanning five orders of")
    print("  magnitude. The gradient with respect to a weight is proportional")
    print("  to its input, so 'area' produces gradients ~30,000x larger than")
    print("  'extent'. One learning rate cannot suit both: it is either far too")
    print("  large for one weight or far too small for the other.")
    print("\n  Note this is the SAME lesson as k-means in 02_traditional_ml/03")
    print("  and PCA in /04, arriving for a third time by a different route.")
    print("  Any method that combines features by weighted sums or distances")
    print("  needs them on comparable scales. Trees, which only ever compare a")
    print("  feature to a threshold, are the exception.")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  A MULTILAYER PERCEPTRON ON TABULAR DATA")
    print("=" * 72)
    print(f"\n  Keras {keras.__version__}, backend: {keras.backend.backend()}")

    df = load_rice()
    feats = [c for c in df.columns if c != "class"]
    X = df[feats].to_numpy(dtype="float32")
    y = (df["class"] == "Cammeo").to_numpy(dtype="float32")

    X_tmp, Xte_raw, y_tmp, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    Xtr_raw, Xva_raw, ytr, yva = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=SEED, stratify=y_tmp)

    # The scaler is fitted on TRAINING rows only - see 01_foundations/05.
    scaler = StandardScaler().fit(Xtr_raw)
    data = {
        "Xtr": scaler.transform(Xtr_raw).astype("float32"),
        "Xva": scaler.transform(Xva_raw).astype("float32"),
        "Xte": scaler.transform(Xte_raw).astype("float32"),
        "ytr": ytr, "yva": yva, "yte": yte,
    }

    banner("1-2. MLP vs XGBoost vs logistic regression")
    figure_comparison(data)

    banner("3. Regularisation, measured rather than assumed")
    figure_regularisation(data)

    banner("4. The learning rate")
    figure_learning_rate(data)

    banner("5. Preprocessing is not optional")
    global raw_y
    raw_y = {"ytr": ytr, "yva": yva, "yte": yte}
    demo_scaling({
        "Standardised": (data["Xtr"], data["Xva"], data["Xte"]),
        "Raw (unscaled)": (Xtr_raw, Xva_raw, Xte_raw),
    })

    banner("What to take away")
    print("  - On small tabular data an MLP is COMPETITIVE, NOT DOMINANT. It")
    print("    cost thousands of parameters and far more fitting time to match")
    print("    logistic regression. Gradient-boosted trees remain the default.")
    print("  - Neural nets win on tabular data when you have high-cardinality")
    print("    categoricals to embed, mixed modalities to fuse, or enough rows")
    print("    that trees stop scaling.")
    print("  - REGULARISE ONLY IF YOU ARE OVERFITTING. Measure the gap between")
    print("    validation and training loss first; dropout and batch norm are")
    print("    answers to a problem clean data may not have.")
    print("  - Dropout is train-time only. Batch norm standardises using BATCH")
    print("    statistics and mostly stabilises training. Early stopping is the")
    print("    cheapest regulariser and should essentially always be on.")
    print("  - LEARNING RATE IS THE FIRST THING TO TUNE and the usual cause of")
    print("    a network that will not train. Diverging loss? Lower it.")
    print("  - SCALE YOUR INPUTS. A network weights features by magnitude, so")
    print("    unscaled columns make one learning rate serve incompatible")
    print("    gradient scales. Trees are the only common exception.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
