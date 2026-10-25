"""Wide & Deep: memorisation and generalisation in one model.

Google's 2016 recommender paper made an argument that is easy to state and
easy to test. A WIDE linear model over crossed categorical features
MEMORISES specific combinations that were seen in training. A DEEP model
over embeddings GENERALISES to combinations that were not. Each fails where
the other succeeds, so train them jointly.

This example builds all three on the Census Income dataset - the same data
the paper's tutorial used - and checks whether the joint model actually
earns its extra complexity.

  1. The wide part: feature crosses, and what memorisation means concretely
  2. The deep part: embeddings, and where generalisation comes from
  3. All three head to head, plus the baselines that keep it honest

Run:
    python 03_neural_networks/08_wide_and_deep/wide_and_deep.py

Concept reference: Foundations volume, Parts B.23-B.24.
"""

from __future__ import annotations

import os
import sys
import warnings
import zlib
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_adult  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
EPOCHS = 12

CATEGORICAL = ["workclass", "education", "marital_status", "occupation",
               "relationship", "race", "sex", "native_country"]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss",
           "hours_per_week"]
CROSSES = [("education", "occupation"), ("marital_status", "relationship"),
           ("occupation", "sex")]


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def seed_everything() -> None:
    keras.utils.set_random_seed(SEED)


def prepare(df: pd.DataFrame, n_bins: int = 1000) -> dict:
    """Integer-encode categoricals, hash the crosses, standardise numerics.

    The crosses are HASHED into a fixed number of buckets. A raw cross of two
    16-level columns is 256 columns; three crosses of realistic cardinality
    would be tens of thousands. Hashing caps the width at the cost of
    occasional collisions, and it is what production systems actually do.

    The hash must be STABLE. Python's built-in hash() of a string is salted
    per process (PYTHONHASHSEED), so the same cross lands in a different
    bucket in every run - train in one process and serve in another and
    every crossed weight is applied to the wrong combination. zlib.crc32 is
    the same everywhere.

    Missing values ('?' in the raw file) get an explicit out-of-vocabulary
    index. pd.factorize codes NaN as -1, and an Embedding lookup at -1
    silently wraps to the LAST row - another category's vector.
    """
    out = {}
    for c in CATEGORICAL:
        codes, uniq = pd.factorize(df[c])
        oov = len(uniq)                          # one index past the vocabulary
        out[c] = np.where(codes < 0, oov, codes).astype("int32")
        out[f"_n_{c}"] = len(uniq) + 1           # vocabulary + the OOV row

    for a, b in CROSSES:
        key = f"{a}_x_{b}"
        # fill first: under pandas 3, astype(str) keeps NaN as NaN, and
        # NaN + "|" + str is NaN again - a float, not a string to hash
        joined = df[a].fillna("?").astype(str) + "|" + df[b].fillna("?").astype(str)
        out[key] = np.array([zlib.crc32(v.encode()) % n_bins for v in joined],
                            dtype="int32")
        out[f"_n_{key}"] = n_bins

    out["numeric"] = StandardScaler().fit_transform(
        df[NUMERIC].to_numpy(dtype="float32")).astype("float32")
    return out


def split_dict(d: dict, idx: np.ndarray) -> dict:
    return {k: (v[idx] if isinstance(v, np.ndarray) else v)
            for k, v in d.items()}


def build_model(meta: dict, mode: str) -> keras.Model:
    """mode is 'wide', 'deep' or 'both'."""
    inputs, wide_parts, deep_parts = {}, [], []

    num_in = keras.layers.Input((len(NUMERIC),), name="numeric")
    inputs["numeric"] = num_in
    if mode in ("deep", "both"):
        deep_parts.append(num_in)

    for c in CATEGORICAL:
        ci = keras.layers.Input((1,), dtype="int32", name=c)
        inputs[c] = ci
        if mode in ("deep", "both"):
            # embedding dimension by the usual heuristic, capped
            dim = int(min(16, round(meta[f"_n_{c}"] ** 0.25 * 2) + 1))
            deep_parts.append(keras.layers.Flatten()(
                keras.layers.Embedding(meta[f"_n_{c}"], dim)(ci)))

    for a, b in CROSSES:
        key = f"{a}_x_{b}"
        xi = keras.layers.Input((1,), dtype="int32", name=key)
        inputs[key] = xi
        if mode in ("wide", "both"):
            # ONE weight per crossed bucket - this is the memorisation term
            wide_parts.append(keras.layers.Flatten()(
                keras.layers.Embedding(meta[f"_n_{key}"], 1)(xi)))

    pieces = []
    if wide_parts:
        pieces.append(keras.layers.Add()(wide_parts) if len(wide_parts) > 1
                      else wide_parts[0])
    if deep_parts:
        x = keras.layers.Concatenate()(deep_parts)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        pieces.append(keras.layers.Dense(1)(x))

    logit = (keras.layers.Add()(pieces) if len(pieces) > 1 else pieces[0])
    out = keras.layers.Activation("sigmoid")(logit)
    m = keras.Model(inputs, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


def feed(d: dict) -> dict:
    keys = ["numeric"] + CATEGORICAL + [f"{a}_x_{b}" for a, b in CROSSES]
    return {k: d[k] for k in keys}


# ================================================== 1. what a cross is
def demo_cross(df: pd.DataFrame) -> None:
    print("  A linear model sees 'education' and 'occupation' as two separate")
    print("  effects it adds together. It cannot express 'a Masters degree is")
    print("  worth a lot in Exec-managerial and little in Handlers-cleaners' -")
    print("  that is an INTERACTION, and a sum cannot represent one.\n")

    sub = df[df.education.isin(["Masters", "HS-grad"])
             & df.occupation.isin(["Exec-managerial", "Handlers-cleaners"])]
    tab = (sub.assign(rich=(sub.income == ">50K").astype(float))
           .pivot_table(index="education", columns="occupation", values="rich"))
    print("  Share earning >50K:\n")
    print("                     " + "".join(f"{c:>22}" for c in tab.columns))
    for e in tab.index:
        print(f"  {e:<17}" + "".join(f"{tab.loc[e, c]:>22.3f}"
                                      for c in tab.columns))

    m_e = tab.loc["Masters"].mean() - tab.loc["HS-grad"].mean()
    diff = ((tab.loc["Masters", "Exec-managerial"]
             - tab.loc["HS-grad", "Exec-managerial"])
            - (tab.loc["Masters", "Handlers-cleaners"]
               - tab.loc["HS-grad", "Handlers-cleaners"]))
    print(f"\n  Average education effect: {m_e:+.3f}")
    print(f"  But the effect differs by occupation by {diff:+.3f} - that gap is")
    print("  exactly what an additive model throws away.")
    print("\n  A FEATURE CROSS creates one new categorical whose levels are the")
    print("  observed combinations: 'Masters|Exec-managerial' becomes its own")
    print("  feature with its own weight. The linear model can now give that")
    print("  specific pairing whatever value the data says it has.")
    print("\n  That is MEMORISATION, in the precise sense the paper means: the")
    print("  model stores a value for a combination it has seen. Its weakness")
    print("  follows immediately - a combination absent from training gets no")
    print("  weight at all, and the wide model has nothing to say about it.")
    print("\n  Crosses are hashed into fixed buckets here. Three crosses of")
    print("  realistic cardinality would otherwise run to tens of thousands of")
    print("  columns; hashing caps the width and accepts some collisions.")
    print("  The hash is zlib.crc32, not Python's hash(): the built-in is salted")
    print("  per process, so it would put a cross in a different bucket at")
    print("  serving time than at training time - a silent train/serve skew.")


# ============================================== 2-3. the three models
def figure_comparison(data: dict, y: np.ndarray, meta: dict) -> None:
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, random_state=SEED,
                              stratify=y)
    dtr, dte = split_dict(data, tr), split_dict(data, te)
    ytr, yte = y[tr], y[te]

    print(f"  Census Income: {len(tr):,} train / {len(te):,} test, "
          f"{y.mean():.1%} earn >50K.\n")
    print(f"  {'model':<26} {'params':>9} {'test ROC-AUC':>13} {'accuracy':>10}")

    results, hists = {}, {}
    for mode, label in (("wide", "Wide only (crosses)"),
                        ("deep", "Deep only (embeddings)"),
                        ("both", "Wide & Deep")):
        seed_everything()
        m = build_model(meta, mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = m.fit(feed(dtr), ytr, validation_data=(feed(dte), yte),
                      epochs=EPOCHS, batch_size=256, verbose=0)
        p = m.predict(feed(dte), verbose=0).ravel()
        results[label] = {"auc": roc_auc_score(yte, p),
                          "acc": float((p > 0.5).astype(int).__eq__(yte).mean()),
                          "params": m.count_params()}
        hists[label] = h
        r = results[label]
        print(f"  {label:<26} {r['params']:>9,} {r['auc']:>13.4f} "
              f"{r['acc']:>10.4f}")

    # the baselines that keep this honest
    Xtr = np.hstack([dtr["numeric"]] + [dtr[c].reshape(-1, 1) for c in CATEGORICAL])
    Xte = np.hstack([dte["numeric"]] + [dte[c].reshape(-1, 1) for c in CATEGORICAL])
    lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, verbosity=0,
                        eval_metric="logloss").fit(Xtr, ytr)
    # Integer codes tell a linear model that category 7 is seven times
    # category 1. Trees don't care; a linear model does. One-hot is the fair
    # linear baseline.
    onehot = OneHotEncoder(handle_unknown="ignore").fit(
        np.column_stack([dtr[c] for c in CATEGORICAL]))
    Xtr_oh = sparse.hstack([dtr["numeric"], onehot.transform(
        np.column_stack([dtr[c] for c in CATEGORICAL]))]).tocsr()
    Xte_oh = sparse.hstack([dte["numeric"], onehot.transform(
        np.column_stack([dte[c] for c in CATEGORICAL]))]).tocsr()
    lr_oh = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr_oh, ytr)
    for name, model, X in (("Log. reg. (integer codes)", lr, Xte),
                           ("Log. reg. (one-hot)", lr_oh, Xte_oh),
                           ("XGBoost", xgb, Xte)):
        p = model.predict_proba(X)[:, 1]
        results[name] = {"auc": roc_auc_score(yte, p),
                         "acc": float((p > 0.5).astype(int).__eq__(yte).mean()),
                         "params": np.nan}
        print(f"  {name:<26} {'-':>9} {results[name]['auc']:>13.4f} "
              f"{results[name]['acc']:>10.4f}")

    w = results["Wide only (crosses)"]["auc"]
    d = results["Deep only (embeddings)"]["auc"]
    b = results["Wide & Deep"]["auc"]
    best = max(results, key=lambda k: results[k]["auc"])

    print(f"\n  Wide {w:.4f} | Deep {d:.4f} | Both {b:.4f}")
    print(f"  Joint model vs the better of its two halves: {b - max(w, d):+.4f}")
    print(f"\n  Best overall: {best} ({results[best]['auc']:.4f})")

    lr_int = results["Log. reg. (integer codes)"]["auc"]
    lr_oh = results["Log. reg. (one-hot)"]["auc"]
    print(f"\n  Logistic regression on integer codes {lr_int:.4f}, on one-hot "
          f"{lr_oh:.4f}.")
    print("  Integer codes impose a fake order on categories, which a linear")
    print("  model takes literally. Against the integer-code baseline the wide")
    print(f"  part looks good ({w - lr_int:+.4f}); against the fair one-hot")
    print(f"  baseline it is {w - lr_oh:+.4f}.")

    if b > max(w, d):
        print("\n  The joint model beat both halves, which is the paper's claim")
        print("  and is what should happen: the wide part handles combinations")
        print("  it has memorised, the deep part covers everything else, and")
        print("  the two error patterns are different enough to combine.")
    else:
        print("\n  The joint model did NOT beat both halves here. Worth saying")
        print("  plainly rather than assuming the architecture works because")
        print("  it is famous.")

    print(f"\n  Now look at XGBoost: {results['XGBoost']['auc']:.4f}.")
    if results["XGBoost"]["auc"] >= b:
        print("  It matches or beats the whole Wide & Deep construction, with")
        print("  no embeddings, no crosses, no architecture and no epochs to")
        print("  tune. A tree finds interactions BY CONSTRUCTION - every split")
        print("  below another split is a conditional effect - which is exactly")
        print("  the job the wide part was added to do.")
        print("\n  That is the honest framing for this dataset. Wide & Deep was")
        print("  designed for Google Play's recommender: hundreds of millions")
        print("  of rows, categorical vocabularies in the millions, and a")
        print("  serving budget that rules out big models. On 32,000 rows with")
        print("  eight modest categoricals, gradient boosting is simply the")
        print("  right tool, and the architecture is solving a problem this")
        print("  data does not have.")
    else:
        print("  The neural architecture stayed ahead of gradient boosting.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (label, h), colour in zip(hists.items(),
                                  ("orange", "green", "blue")):
        axes[0].plot(h.history["val_loss"], color=PALETTE[colour], lw=2,
                     label=label)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation loss")
    axes[0].set_title("Training")
    axes[0].legend(fontsize=9)

    names = list(results)
    axes[1].barh(names, [results[n]["auc"] for n in names],
                 color=[PALETTE["orange"], PALETTE["green"], PALETTE["blue"],
                        PALETTE["grey"], PALETTE["sky"], PALETTE["vermillion"]])
    axes[1].set_xlim(0.84, 0.94)
    axes[1].set_xlabel("test ROC-AUC")
    axes[1].set_title("Against the baselines that keep it honest")
    axes[1].grid(axis="y", visible=False)

    fig.suptitle("Wide, Deep, both - and whether any of it beat XGBoost",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '1_wide_and_deep.png').name}")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  WIDE & DEEP")
    print("=" * 72)
    print(f"\n  Keras {keras.__version__}, backend: {keras.backend.backend()}")

    df = load_adult()
    y = (df["income"].str.strip() == ">50K").to_numpy(dtype="float32")
    data = prepare(df)
    meta = {k: v for k, v in data.items() if k.startswith("_n_")}

    banner("1. What a feature cross is, and what memorisation means")
    demo_cross(df)

    banner("2-3. Wide, Deep and joint - against honest baselines")
    figure_comparison(data, y, meta)

    banner("What to take away")
    print("  - WIDE = a linear model over CROSSED categorical features. It")
    print("    memorises specific combinations it has seen, and has nothing to")
    print("    say about combinations it has not.")
    print("  - DEEP = embeddings plus an MLP. It generalises to unseen")
    print("    combinations because similar categories sit near each other in")
    print("    embedding space - but it can over-generalise and recommend")
    print("    things that are merely plausible.")
    print("  - The two are trained JOINTLY, not stacked or ensembled: one loss,")
    print("    one backward pass, the wide logit and the deep logit added")
    print("    before the sigmoid.")
    print("  - Crosses are HASHED into fixed buckets in production, because a")
    print("    raw cross of two high-cardinality columns is unbounded. Use a")
    print("    STABLE hash (crc32, FarmHash), never a per-process salted one.")
    print("  - ALWAYS RUN THE BASELINES. On 32,000 rows with modest")
    print("    cardinality, gradient boosting finds interactions by")
    print("    construction and is hard to beat. Wide & Deep was built for")
    print("    hundreds of millions of rows and million-level vocabularies,")
    print("    and that is where its trade-offs make sense.")
    print("  - The idea outlived the specific architecture: DeepFM, DCN and")
    print("    two-tower retrieval (see 09) all keep the memorise-plus-")
    print("    generalise split in some form.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
