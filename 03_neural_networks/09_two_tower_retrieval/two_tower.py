"""Two-tower retrieval: the architecture that made recommendation servable.

A matrix-factorisation recommender (02_traditional_ml/07, 03/06) learns one
vector per user id and one per item id. That works until a new user arrives,
at which point there is no row to look up - the cold-start failure measured
in 02_traditional_ml/07.

A two-tower model fixes that by embedding FEATURES instead of ids. One tower
turns user features into a vector, another turns item features into a
vector, and relevance is their dot product. Because the two towers never
touch until that final dot product, item vectors can be computed once,
indexed, and searched with approximate nearest neighbours - which is what
makes it servable at all.

  1. The architecture, and why the towers must stay separate
  2. In-batch negatives - how you train a retrieval model without
     ever showing it a negative example
  3. Retrieval quality against the baselines from 02_traditional_ml/07
  4. Cold start: the thing id-based factorisation cannot do

Run:
    python 03_neural_networks/09_two_tower_retrieval/two_tower.py

Concept reference: Foundations volume, Parts B.25-B.26.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_movielens_movies, load_movielens_ratings  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
# 60 epochs, not 12: at 12 the in-batch loss had moved only 6.22 -> 5.79
# against a random-guess value of ln(batch) = 6.24, i.e. barely trained.
DIM, EPOCHS = 32, 60

GENRES = ["unknown", "action", "adventure", "animation", "childrens", "comedy",
          "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
          "musical", "mystery", "romance", "sci_fi", "thriller", "war",
          "western"]


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def seed_everything() -> None:
    keras.utils.set_random_seed(SEED)


# ================================================== the two towers
def build_towers(n_users: int, n_genres: int, n_item_feats: int):
    """Two independent encoders, joined only by a dot product.

    The separation is the entire point. Because the item tower never sees
    the user, every item vector can be precomputed once and stored in an ANN
    index. At serving time you encode ONE user and do a nearest-neighbour
    lookup, instead of scoring the user against every item.
    """
    # --- user tower: id + mean-genre-taste profile
    u_id = keras.layers.Input((1,), dtype="int32", name="user_id")
    u_taste = keras.layers.Input((n_genres,), name="user_taste")
    u = keras.layers.Concatenate()([
        keras.layers.Flatten()(keras.layers.Embedding(n_users, 16)(u_id)),
        u_taste,
    ])
    u = keras.layers.Dense(64, activation="relu")(u)
    u_vec = keras.layers.Dense(DIM)(u)
    u_vec = keras.layers.UnitNormalization()(u_vec)
    user_tower = keras.Model([u_id, u_taste], u_vec, name="user_tower")

    # --- item tower: CONTENT FEATURES ONLY, no item id.
    # This is what makes a brand-new film representable: it has genres and a
    # release year the moment it is added, and never needs an id embedding.
    i_feat = keras.layers.Input((n_item_feats,), name="item_features")
    i = keras.layers.Dense(64, activation="relu")(i_feat)
    i_vec = keras.layers.Dense(DIM)(i)
    i_vec = keras.layers.UnitNormalization()(i_vec)
    item_tower = keras.Model(i_feat, i_vec, name="item_tower")

    return user_tower, item_tower


class TwoTower(keras.Model):
    """Trained with IN-BATCH NEGATIVES.

    There are no explicit negative examples. For a batch of B observed
    (user, item) pairs, the B x B matrix of dot products has the true pairs
    on its diagonal and B-1 other items per row that this user did NOT
    interact with. Softmax across each row and the correct label is simply
    the row index - the other examples in the batch supply the negatives for
    free. This is the standard sampled-softmax trick, and it is why
    retrieval models can be trained on positives alone.
    """

    def __init__(self, user_tower, item_tower, temperature=0.05, **kw):
        super().__init__(**kw)
        self.user_tower, self.item_tower = user_tower, item_tower
        self.temperature = temperature
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

    def call(self, inputs, training=False):
        u_id, u_taste, i_feat = inputs
        uv = self.user_tower([u_id, u_taste], training=training)
        iv = self.item_tower(i_feat, training=training)
        # B x B similarity matrix: row i is user i scored against every item
        # in the batch. The true pair sits on the diagonal.
        return keras.ops.matmul(uv, keras.ops.transpose(iv)) / self.temperature

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None,
                     **kwargs):
        # The label for row i is simply i - the diagonal. Overriding
        # compute_loss rather than writing a custom train_step keeps this
        # backend-agnostic; Keras still handles the backward pass.
        labels = keras.ops.arange(keras.ops.shape(y_pred)[0])
        return self.loss_fn(labels, y_pred)


def demo_architecture() -> None:
    print("  user features ---> [ USER TOWER ] ---> u  (32-d, unit length)")
    print("                                          |")
    print("                                      dot product  --> relevance")
    print("                                          |")
    print("  item features ---> [ ITEM TOWER ] ---> v  (32-d, unit length)")
    print("\n  The towers never see each other's input. That constraint is the")
    print("  whole design, and it buys two things:")
    print("\n    SERVABILITY   every item vector can be computed ONCE, offline,")
    print("                  and put in an approximate-nearest-neighbour index.")
    print("                  At request time you encode one user and do a")
    print("                  lookup. Scoring a million items against a user")
    print("                  with a joint model would mean a million forward")
    print("                  passes; here it is one, plus an ANN query.")
    print("\n    COLD START    the item tower takes FEATURES, not an id. A film")
    print("                  added this morning has genres and a release year,")
    print("                  so it gets a vector immediately - no retraining,")
    print("                  no id embedding to learn. Section 4 tests this.")
    print("\n  The cost is that the towers cannot model interactions between")
    print("  user and item features - all their communication happens through")
    print("  one dot product. That is why production systems use two-tower")
    print("  RETRIEVAL to narrow millions of candidates to hundreds, then a")
    print("  richer cross-feature RANKER (a Wide & Deep descendant, see 08)")
    print("  on the survivors. Retrieve cheap, rank expensively.")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  TWO-TOWER RETRIEVAL")
    print("=" * 72)
    print(f"\n  Keras {keras.__version__}, backend: {keras.backend.backend()}")

    r = load_movielens_ratings()
    movies = load_movielens_movies()

    # item features: genres + a normalised release year. NO item id.
    m = movies.set_index("item_id")
    year = (pd.to_datetime(m["release_date"], format="%d-%b-%Y",
                           errors="coerce").dt.year.fillna(1990))
    item_feats = np.hstack([
        m[GENRES].to_numpy(dtype="float32"),
        ((year - year.mean()) / year.std()).to_numpy(dtype="float32")[:, None],
    ]).astype("float32")
    item_ids = m.index.to_numpy()
    pos = {int(i): k for k, i in enumerate(item_ids)}

    # keep only positives - a retrieval model trains on interactions
    liked = r[r.rating >= 4].copy()
    liked = liked[liked.item_id.isin(pos)]
    liked = liked.sort_values("timestamp")
    test = liked.groupby("user_id").tail(1)
    train = liked.drop(test.index)

    n_users = int(r.user_id.max()) + 1
    # user taste profile = mean genre vector of the films they liked in TRAIN
    taste = np.zeros((n_users, len(GENRES)), dtype="float32")
    g = m[GENRES].to_numpy(dtype="float32")
    for uid, grp in train.groupby("user_id"):
        taste[uid] = g[[pos[int(i)] for i in grp.item_id]].mean(axis=0)

    print(f"  MovieLens: {len(liked):,} positive interactions (rating >= 4),")
    print(f"  {len(train):,} train / {len(test):,} test (each user's last like).")
    print(f"  Item features: {item_feats.shape[1]} (19 genres + release year), NO item id.")

    banner("1. The architecture")
    demo_architecture()

    banner("2-3. Training with in-batch negatives, and retrieval quality")
    seed_everything()
    ut, it_ = build_towers(n_users, len(GENRES), item_feats.shape[1])
    model = TwoTower(ut, it_)
    model.compile(optimizer=keras.optimizers.Adam(1e-3))

    u_id = train.user_id.to_numpy().astype("int32").reshape(-1, 1)
    u_t = taste[train.user_id.to_numpy()]
    i_f = item_feats[[pos[int(i)] for i in train.item_id]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = model.fit((u_id, u_t, i_f), np.zeros(len(u_id)),
                         epochs=EPOCHS, batch_size=512, verbose=0)
    print(f"  In-batch-negative loss: {hist.history['loss'][0]:.4f} -> "
          f"{hist.history['loss'][-1]:.4f} over {EPOCHS} epochs.\n")

    # ---- retrieval evaluation
    all_item_vecs = np.array(it_.predict(item_feats, verbose=0))
    te_u = test.user_id.to_numpy().astype("int32").reshape(-1, 1)
    te_vecs = np.array(ut.predict([te_u, taste[test.user_id.to_numpy()]],
                                  verbose=0))
    scores = te_vecs @ all_item_vecs.T

    # never recommend something the user already liked in training
    seen = {u: set(grp.item_id) for u, grp in train.groupby("user_id")}
    for row, u in enumerate(test.user_id.to_numpy()):
        for i in seen.get(u, ()):
            scores[row, pos[int(i)]] = -np.inf

    truth = np.array([pos[int(i)] for i in test.item_id])
    order = np.argsort(-scores, axis=1)

    popularity = train.groupby("item_id").size()
    pop_vec = np.zeros(len(item_ids))
    for i, c in popularity.items():
        pop_vec[pos[int(i)]] = c

    def recall_at(k: int, ranking: np.ndarray) -> float:
        return float(np.mean([truth[i] in ranking[i, :k]
                              for i in range(len(truth))]))

    pop_order = np.tile(np.argsort(-pop_vec), (len(truth), 1))
    rng = np.random.default_rng(SEED)
    rnd_order = np.array([rng.permutation(len(item_ids))
                          for _ in range(len(truth))])

    print(f"  {'model':<22} {'Recall@10':>10} {'Recall@50':>10} {'Recall@100':>11}")
    for name, o in (("Random", rnd_order), ("Popularity", pop_order),
                    ("Two-tower", order)):
        print(f"  {name:<22} {recall_at(10, o):>10.4f} {recall_at(50, o):>10.4f} "
              f"{recall_at(100, o):>11.4f}")

    tt10, pop10 = recall_at(10, order), recall_at(10, pop_order)
    print(f"\n  Two-tower vs popularity at k=10: {tt10 - pop10:+.4f}")
    if tt10 > pop10:
        print("  The two-tower model beats the popularity baseline.")
    else:
        print("  Popularity WINS at k=10 - the same uncomfortable result as")
        print("  02_traditional_ml/07, and for the same reason: with one")
        print("  held-out item per user and a heavy popularity bias in what")
        print("  people watch, an offline recall metric flatters the popular.")
    print("\n  Note what this model is working with: 19 genre flags and a year.")
    print("  It has no cast, no synopsis, no images - so the ceiling on how")
    print("  well it can distinguish two comedies from 1995 is low. Production")
    print("  item towers consume text embeddings, images and dozens of")
    print("  attributes, which is where the approach earns its reputation.")

    banner("4. Cold start - the thing id-based factorisation cannot do")
    new_film = np.zeros((1, item_feats.shape[1]), dtype="float32")
    new_film[0, GENRES.index("animation")] = 1
    new_film[0, GENRES.index("childrens")] = 1
    # Use the LARGEST year actually present (z = +0.60), not an extrapolated
    # value. An earlier version used z = +2.0, which is 3.3 sd beyond anything
    # in training: the year feature then dominated the dot product and every
    # "nearest" film came back as an obscure late-90s title at an identical
    # 0.873, with the genres ignored entirely. Feeding a standardised feature
    # a value outside its training range is a real serving bug, and this is
    # what it looks like.
    year_max = float(item_feats[:, -1].max())
    new_film[0, -1] = year_max
    v = np.array(it_.predict(new_film, verbose=0))

    print("  Invent a film that appears nowhere in the training data:")
    print("  genres = animation + children's, release year set to the newest")
    print(f"  value actually present in the data (z = {year_max:+.2f}).\n")
    print(f"  The item tower produced a {v.shape[1]}-d vector for it immediately.")
    print("  No retraining, no id, no interaction history required.\n")

    sims = (all_item_vecs @ v.T).ravel()
    titles = dict(zip(movies.item_id.astype(int), movies.title))
    print("  Nearest existing films to this brand-new one:")
    for j in np.argsort(-sims)[:5]:
        print(f"    {sims[j]:.3f}  {titles.get(int(item_ids[j]), '?')}")

    print("\n  Compare with matrix factorisation in 02_traditional_ml/07: a new")
    print("  item there has no row in the factorisation, so it has no vector,")
    print("  so it can never be recommended, so it never acquires the")
    print("  interactions it would need to get a row. The two-tower model")
    print("  breaks that loop by never depending on the id in the first place.")
    print("\n  The same argument applies to a new USER, provided the user tower")
    print("  is built from features rather than an id. This one still takes a")
    print("  user id, so it is only half cold-start-proof - a deliberate")
    print("  compromise, and worth noticing rather than glossing over.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hist.history["loss"], color=PALETTE["blue"], lw=2)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("in-batch softmax loss")
    axes[0].set_title("Training with in-batch negatives")

    ks = [1, 5, 10, 25, 50, 100, 200]
    for (name, o), colour in zip((("Random", rnd_order),
                                  ("Popularity", pop_order),
                                  ("Two-tower", order)),
                                 ("grey", "orange", "blue")):
        axes[1].plot(ks, [recall_at(k, o) for k in ks], "o-",
                     color=PALETTE[colour], lw=2, label=name)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("k (candidates retrieved)"); axes[1].set_ylabel("recall@k")
    axes[1].set_title("Retrieval quality")
    axes[1].legend(fontsize=9)

    fig.suptitle("Two-tower retrieval on MovieLens",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '1_two_tower.png').name}")

    banner("What to take away")
    print("  - TWO SEPARATE ENCODERS joined only by a dot product. The")
    print("    separation is the design, not an accident.")
    print("  - It buys SERVABILITY: item vectors are precomputed once and put")
    print("    in an ANN index, so serving is one user encode plus a")
    print("    nearest-neighbour lookup instead of a forward pass per item.")
    print("  - It buys COLD START, provided the towers consume FEATURES rather")
    print("    than ids. A film created this morning gets a vector at once.")
    print("  - IN-BATCH NEGATIVES: the other examples in the batch are the")
    print("    negatives. A B x B score matrix, softmax per row, label = the")
    print("    diagonal. No negative sampling pipeline needed.")
    print("  - The cost is no user-item feature interaction - everything")
    print("    passes through one dot product. Hence RETRIEVE THEN RANK:")
    print("    two-tower narrows millions to hundreds, a richer cross-feature")
    print("    model (08) reorders the survivors.")
    print("  - Evaluate with RECALL@K, and keep the popularity baseline in the")
    print("    table. Offline retrieval metrics flatter popular items.")
    print("  - On Google Cloud this maps directly onto Vertex AI Vector Search")
    print("    (Agent Retrieval in the 2026 naming): you generate embeddings,")
    print("    index them, and query by nearest neighbour.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
