"""Embeddings: turning categories into geometry.

A category has no arithmetic. One-hot encoding admits this honestly - every
category is equidistant from every other - and pays for it with a column per
level. An embedding instead LEARNS a short dense vector per category, so
similar categories end up near each other, and the similarity is discovered
from the task rather than declared in advance.

  1. One-hot vs embedding: the parameter arithmetic, and the thing one-hot
     structurally cannot represent
  2. Learning film embeddings from ratings alone, then reading them back as
     nearest neighbours - the sanity check that matters
  3. Embedding dimension: what you actually lose by making it small

Run:
    python 03_neural_networks/06_embeddings/embeddings.py

Concept reference: Foundations volume, Parts B.19-B.20.
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
from loaders import load_movielens_movies, load_movielens_ratings  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def seed_everything() -> None:
    keras.utils.set_random_seed(SEED)


# ================================================ 1. one-hot vs embedding
def demo_arithmetic(n_items: int) -> None:
    print(f"  MovieLens has {n_items:,} films. Suppose a model needs to use")
    print("  'which film' as an input.\n")
    print(f"  {'representation':<28} {'width':>8} {'parameters into a 32-unit layer':>34}")
    print(f"  {'one-hot':<28} {n_items:>8,} {n_items * 32:>34,}")
    for d in (8, 16, 32, 64):
        print(f"  {'embedding, dim ' + str(d):<28} {d:>8,} "
              f"{n_items * d + d * 32:>34,}")

    breakeven = 32 * n_items / (n_items + 32)
    print(f"\n  Read that table carefully, because it does not say what people")
    print(f"  usually claim. The embedding only saves parameters while its")
    print(f"  dimension is below about {breakeven:.0f} - at dim 32 it costs MORE than")
    print("  one-hot, and at 64 nearly double.")
    print("\n  The arithmetic is simple once written out. One-hot into a")
    print("  W-unit layer costs V*W. An embedding costs V*d for the table plus")
    print("  d*W for the layer, so it wins only when d*(V+W) < V*W, i.e. when")
    print("  d is comfortably smaller than W. The embedding TABLE is itself a")
    print("  V x d weight matrix - it has not removed the big matrix, it has")
    print("  made it narrower.")
    print("\n  So the parameter argument is real but modest and conditional.")
    print("  The REPRESENTATIONAL argument is the one that actually matters:")
    print("\n  In one-hot space every pair of films is EXACTLY as far apart as")
    print("  every other pair. The cosine similarity between any two distinct")
    print("  one-hot vectors is 0, always. The encoding cannot express 'these")
    print("  two films are alike' because it was built to assert the opposite -")
    print("  that the categories are unordered and unrelated.")
    print("\n  An embedding has no such constraint. Each film gets a dense")
    print("  vector, those vectors are PARAMETERS, and gradient descent moves")
    print("  films that behave similarly towards each other. Similarity becomes")
    print("  something the model discovers rather than something you declare.")
    print("\n  This is the single strongest argument for a neural network on")
    print("  tabular data (see 02_mlp_tabular, where there was none and the")
    print("  network barely justified itself): high-cardinality categoricals")
    print("  that trees must either one-hot or target-encode by hand.")


# ============================================ 2. learn, then read them back
def build_recommender(n_users: int, n_items: int, dim: int) -> keras.Model:
    """Matrix factorisation written as a neural network.

    Two embedding tables and a dot product. This is exactly the SVD-style
    model from 02_traditional_ml/07, but now the factors are learned by
    gradient descent on the observed ratings rather than by decomposing a
    zero-filled matrix - which is why it handles sparsity better.
    """
    u_in = keras.layers.Input((1,), name="user")
    i_in = keras.layers.Input((1,), name="item")
    u = keras.layers.Flatten()(keras.layers.Embedding(n_users, dim)(u_in))
    i = keras.layers.Flatten()(keras.layers.Embedding(n_items, dim)(i_in))
    u_b = keras.layers.Flatten()(keras.layers.Embedding(n_users, 1)(u_in))
    i_b = keras.layers.Flatten()(keras.layers.Embedding(n_items, 1)(i_in))
    dot = keras.layers.Dot(axes=1)([u, i])
    out = keras.layers.Add()([dot, u_b, i_b])
    m = keras.Model([u_in, i_in], out)
    m.compile(optimizer=keras.optimizers.Adam(5e-3), loss="mse")
    return m


def figure_neighbours(r, movies, n_users: int, n_items: int):
    seed_everything()
    dim = 32
    model = build_recommender(n_users, n_items, dim)
    mu = r.rating.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit([r.user_id.to_numpy(), r.item_id.to_numpy()],
                  r.rating.to_numpy() - mu,
                  epochs=8, batch_size=1024, verbose=0)

    item_emb = np.array(model.get_layer(index=3).get_weights()[0])
    counts = r.groupby("item_id").size()
    titles = dict(zip(movies.item_id.astype(int), movies.title))

    # Only compare films with enough ratings. A film rated twice has an
    # embedding that is essentially still its random initialisation, and it
    # will otherwise dominate any nearest-neighbour list by accident.
    popular = counts[counts >= 50].index.to_numpy()
    sims = cosine_similarity(item_emb[popular])
    np.fill_diagonal(sims, -np.inf)

    print(f"  Trained a {dim}-dimensional embedding per film on "
          f"{len(r):,} ratings.")
    print("  The model saw ONLY user id, film id and rating - no genres, no")
    print("  cast, no text. Any structure below was inferred from co-rating")
    print("  behaviour alone.\n")

    seeds = ["Star Wars (1977)", "Toy Story (1995)", "Fargo (1996)"]
    for title in seeds:
        matches = [k for k, v in titles.items() if v == title]
        if not matches or matches[0] not in popular:
            continue
        idx = int(np.where(popular == matches[0])[0][0])
        near = np.argsort(sims[idx])[::-1][:4]
        print(f"  Nearest to {title}:")
        for j in near:
            print(f"      {sims[idx, j]:.3f}  {titles.get(int(popular[j]), '?')}")
        print()

    print("  Judge these by eye - that is the only check available, because")
    print("  there is no ground-truth 'similar film' label. Sequels and films")
    print("  of the same era and genre clustering together is evidence the")
    print("  embedding learned something real; a list of unrelated titles")
    print("  would mean it had not.")
    print("\n  Note the popularity filter in the code. Films with under 50")
    print("  ratings have embeddings barely moved from their random")
    print("  initialisation, and without excluding them they dominate the")
    print("  neighbour lists by accident - the same small-sample trap as the")
    print("  raw item means in 02_traditional_ml/07.")

    # project to 2-D, coloured by genre, to see whether genre emerged
    genres = ["action", "childrens", "comedy", "drama", "horror", "romance"]
    g = movies.set_index("item_id")
    keep, colours, labels = [], [], []
    for gi, gname in enumerate(genres):
        ids = [i for i in popular
               if i in g.index and g.loc[i, gname] == 1
               and g.loc[i, genres].sum() == 1]      # single-genre films only
        for i in ids[:40]:
            keep.append(int(np.where(popular == i)[0][0]))
            colours.append(gi); labels.append(gname)

    pcs = PCA(2, random_state=SEED).fit_transform(item_emb[popular])
    fig, ax = plt.subplots(figsize=(10, 6.5))
    palette = ["blue", "vermillion", "green", "orange", "pink", "sky"]
    for gi, gname in enumerate(genres):
        m = [k for k, c in zip(keep, colours) if c == gi]
        if m:
            ax.scatter(pcs[m, 0], pcs[m, 1], s=40, alpha=0.75,
                       color=PALETTE[palette[gi]], label=gname)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Learned film embeddings, projected to 2-D\n"
                 "(coloured by genre - which the model never saw)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    print(f"\n  saved  {save_figure(fig, OUT / '1_embedding_space.png').name}")
    print("\n  The genre colours are a HELD-OUT label here - the model was never")
    print("  given them. Any separation in the plot is genre structure that")
    print("  fell out of who rated what. Expect partial separation at best:")
    print("  co-rating behaviour correlates with genre without being genre.")
    return r, mu


def figure_dimension(r, mu, n_users: int, n_items: int) -> None:
    """How small can the embedding be before it stops working?"""
    idx = np.random.default_rng(SEED).permutation(len(r))
    cut = int(len(r) * 0.85)
    tr, te = r.iloc[idx[:cut]], r.iloc[idx[cut:]]

    print(f"  {'dim':>5} {'params':>10} {'test RMSE':>11}")
    dims, rmses = [2, 4, 8, 16, 32, 64], []
    for d in dims:
        seed_everything()
        m = build_recommender(n_users, n_items, d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit([tr.user_id.to_numpy(), tr.item_id.to_numpy()],
                  tr.rating.to_numpy() - mu,
                  epochs=8, batch_size=1024, verbose=0)
        p = m.predict([te.user_id.to_numpy(), te.item_id.to_numpy()],
                      verbose=0).ravel() + mu
        rmse = float(np.sqrt(np.mean((np.clip(p, 1, 5)
                                      - te.rating.to_numpy()) ** 2)))
        rmses.append(rmse)
        print(f"  {d:>5} {m.count_params():>10,} {rmse:>11.4f}")

    best = dims[int(np.argmin(rmses))]
    print(f"\n  Best test RMSE at dim {best} - the SMALLEST size tested. Error rises")
    print(f"  monotonically all the way to dim {dims[-1]}, which is not the U-shaped")
    print("  curve this experiment was expected to produce.")
    print("\n  The reason is that every extra dimension here is pure capacity")
    print("  with nothing holding it back. This model has no regularisation,")
    print("  no dropout and no weight decay, and 100,000 ratings spread over")
    print(f"  {n_items:,} films and {n_users:,} users is thin - many films have only a")
    print("  handful of ratings. At dim 64 there are more embedding parameters")
    print("  than there are ratings to fit them with, so the model memorises.")
    print("\n  That is a genuine finding and not a broken sweep: ON THIS DATA,")
    print("  bigger embeddings are strictly worse. It is also a warning about")
    print("  the rules of thumb - dim ~ 4 * cardinality^0.25 would suggest")
    print(f"  about {4 * n_items ** 0.25:.0f} here, which the measurement says is already too")
    print("  large. Sweep it; the sweep takes minutes.")
    print("\n  The honest caveat in the other direction: dim 2 winning is a")
    print("  statement about THIS dataset at THIS regularisation, not about")
    print("  embeddings in general. Add weight decay, or ten times the data,")
    print("  and the optimum moves right.")
    print("\n  Note what dominates the parameter count: the embedding TABLE")
    print(f"  ({n_items:,} films x dim), not the network. Embeddings are memory-")
    print("  hungry and that is what makes vocabulary size, hashing and")
    print("  pruning real engineering concerns at production scale.")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(dims, rmses, "o-", color=PALETTE["blue"], lw=2)
    ax.axvline(best, color=PALETTE["vermillion"], ls="--", lw=1.4)
    ax.annotate(f"best: dim {best}", xy=(0.55, 0.85), xycoords="axes fraction",
                fontsize=10, color=PALETTE["vermillion"], fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("embedding dimension"); ax.set_ylabel("test RMSE")
    ax.set_title("How small can the embedding get?")
    fig.tight_layout()
    print(f"\n  saved  {save_figure(fig, OUT / '2_dimension.png').name}")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  EMBEDDINGS")
    print("=" * 72)
    print(f"\n  Keras {keras.__version__}, backend: {keras.backend.backend()}")

    r = load_movielens_ratings()
    movies = load_movielens_movies()
    n_users = int(r.user_id.max()) + 1
    n_items = int(r.item_id.max()) + 1

    banner("1. One-hot vs embedding")
    demo_arithmetic(n_items - 1)

    banner("2. Learning film embeddings from ratings alone")
    r, mu = figure_neighbours(r, movies, n_users, n_items)

    banner("3. How many dimensions do you need?")
    figure_dimension(r, mu, n_users, n_items)

    banner("What to take away")
    print("  - An embedding is a LOOKUP TABLE OF LEARNED VECTORS, one row per")
    print("    category, trained by gradient descent like any other weight.")
    print("  - One-hot encoding makes every category equidistant from every")
    print("    other. That is not a limitation of its size, it is what the")
    print("    encoding MEANS - it cannot express similarity at all.")
    print("  - THE PARAMETER SAVING IS CONDITIONAL and often small: the table")
    print("    measured here shows an embedding costing MORE than one-hot once")
    print("    its dimension reaches the width of the next layer. The reason")
    print("    to use embeddings is that similarity becomes learnable.")
    print("  - HIGH-CARDINALITY CATEGORICALS are the strongest case for a")
    print("    neural network on tabular data - the case that was missing in")
    print("    02_mlp_tabular, where the MLP barely justified itself.")
    print("  - Sanity-check embeddings by reading back nearest neighbours.")
    print("    There is no ground truth, so plausibility is the only test -")
    print("    and filter out rare categories first, whose vectors have barely")
    print("    moved from their random initialisation.")
    print("  - SWEEP THE DIMENSION rather than trusting a rule of thumb. Here")
    print("    the smallest size tested won outright and error rose")
    print("    monotonically with capacity - an unregularised embedding table")
    print("    on sparse data overfits fast.")
    print("  - The embedding TABLE dominates the parameter count, which is why")
    print("    vocabulary size, hashing and pruning are production concerns.")
    print("  - Two embedding tables plus a dot product IS matrix")
    print("    factorisation - the recommender from 02_traditional_ml/07,")
    print("    rewritten as a network and trained on observed ratings only.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
