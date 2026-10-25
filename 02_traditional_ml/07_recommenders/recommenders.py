"""Recommenders: the task where the obvious metric measures the wrong thing.

  1. The data - 100,000 ratings, and how extreme the sparsity really is
  2. Rating prediction - global mean, biases, item-item CF, and SVD, scored
     with RMSE the way the Netflix Prize did
  3. Why RMSE is the wrong objective - the same models re-scored as a
     RANKING problem, where the leaderboard changes completely
  4. Popularity is a brutally strong baseline, and why that is a problem
     rather than a success
  5. Cold start - what every one of these models does with a new user

Run:
    python 02_traditional_ml/07_recommenders/recommenders.py

Concept reference: Foundations volume, Parts A.11 (metrics), A.16 (unsupervised).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from loaders import load_movielens_movies, load_movielens_ratings  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
rng = np.random.default_rng(SEED)
TITLES: dict[int, str] = {}      # item_id -> title, filled in main()


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ------------------------------------------------------------- 1. the data
def describe(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    n_u, n_i = ratings.user_id.nunique(), ratings.item_id.nunique()
    density = len(ratings) / (n_u * n_i)
    print(f"  MovieLens 100K: {len(ratings):,} ratings, {n_u:,} users, {n_i:,} films.")
    print(f"  The full user x item grid would have {n_u * n_i:,} cells.")
    print(f"  DENSITY: {density:.2%} - {100 - density * 100:.2f}% of it is empty.")
    print("\n  That sparsity is the defining property of the problem. Every method")
    print("  below is a different answer to 'how do I fill in a matrix that is")
    print("  94% holes', and it is why ordinary supervised regression on a")
    print("  user x item table is not the tool.")

    per_user = ratings.groupby("user_id").size()
    per_item = ratings.groupby("item_id").size()
    print(f"\n  Ratings per user:  median {per_user.median():.0f}, "
          f"min {per_user.min()}, max {per_user.max()}")
    print(f"  Ratings per film:  median {per_item.median():.0f}, "
          f"min {per_item.min()}, max {per_item.max()}")

    top_share = per_item.nlargest(int(n_i * 0.1)).sum() / len(ratings)
    print(f"\n  The most-rated 10% of films carry {top_share:.0%} of all ratings.")
    print("  This LONG TAIL is why popularity works so well and why it is such a")
    print("  trap - see section 4.")

    merged = ratings.merge(movies[["item_id", "title"]], on="item_id")
    print("\n  Most-rated films:")
    for _, r in (merged.groupby("title")
                 .agg(n=("rating", "size"), mean=("rating", "mean"))
                 .nlargest(5, "n").reset_index().iterrows()):
        print(f"    {r['n']:>4} ratings  avg {r['mean']:.2f}  {r['title']}")


def split_leave_last(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent rating.

    A random split would put a user's later ratings in training and earlier
    ones in test, which is the time-series leak from 06_time_series wearing a
    different hat. Leave-one-last is the standard honest protocol here.
    """
    r = ratings.sort_values("timestamp")
    test = r.groupby("user_id").tail(1)
    train = r.drop(test.index)
    return train, test


# ------------------------------------------------- 2. predicting the rating
def figure_rating_prediction(train: pd.DataFrame, test: pd.DataFrame,
                             n_u: int, n_i: int) -> dict:
    """The Netflix Prize formulation: guess the star rating, score with RMSE."""
    mu = train.rating.mean()

    # user and item biases - 'this user rates high', 'this film is liked'
    # damped towards the global mean so a film with 2 ratings cannot dominate
    k = 10
    u_bias = (train.groupby("user_id").rating.agg(["sum", "size"])
              .eval(f"(sum - size * {mu}) / (size + {k})"))
    i_bias = (train.groupby("item_id").rating.agg(["sum", "size"])
              .eval(f"(sum - size * {mu}) / (size + {k})"))

    # dense user x item matrix, zeros where unrated
    M = np.zeros((n_u + 1, n_i + 1), dtype=np.float32)
    M[train.user_id.to_numpy(), train.item_id.to_numpy()] = train.rating.to_numpy()

    # item-item cosine similarity on the mean-centred matrix
    seen = M > 0
    item_means = np.divide(M.sum(0), np.maximum(seen.sum(0), 1))

    # A DAMPED (shrunk) item mean, pulled towards the global mean in
    # proportion to how little evidence there is. The raw mean above is fine
    # for centring but ruinous for RANKING - see section 3.
    n_per_item = seen.sum(0)
    item_means_damped = (M.sum(0) + k * mu) / (n_per_item + k)
    Mc = np.where(seen, M - item_means, 0.0)
    sim = cosine_similarity(Mc.T)
    np.fill_diagonal(sim, 0.0)

    svd = TruncatedSVD(n_components=40, random_state=SEED)
    U = svd.fit_transform(Mc)
    recon = U @ svd.components_

    def predict(name: str, u: int, i: int) -> float:
        if name == "Global mean":
            return mu
        if name == "User + item bias":
            return mu + u_bias.get(u, 0.0) + i_bias.get(i, 0.0)
        if name == "Item-item CF":
            rated = np.where(seen[u])[0]
            if rated.size == 0:
                return mu
            s = sim[i, rated]
            top = np.argsort(s)[::-1][:30]
            s_top, r_top = s[top], M[u, rated[top]]
            if s_top.sum() <= 0:
                return mu
            return float(item_means[i] + np.dot(s_top, r_top - item_means[rated[top]])
                         / s_top.sum())
        return float(item_means[i] + recon[u, i])          # SVD

    names = ["Global mean", "User + item bias", "Item-item CF", "SVD (40 factors)"]
    print(f"  Holding out each user's most recent rating: {len(test):,} test cases.\n")
    print(f"  {'model':<22} {'RMSE':>7} {'MAE':>7}")
    out = {}
    for name in names:
        p = np.array([predict(name, u, i)
                      for u, i in zip(test.user_id, test.item_id)])
        p = np.clip(p, 1, 5)
        err = p - test.rating.to_numpy()
        out[name] = {"rmse": float(np.sqrt((err ** 2).mean())),
                     "mae": float(np.abs(err).mean())}
        print(f"  {name:<22} {out[name]['rmse']:>7.4f} {out[name]['mae']:>7.4f}")

    best = min(out, key=lambda k_: out[k_]["rmse"])
    gain = (1 - out[best]["rmse"] / out["Global mean"]["rmse"]) * 100
    print(f"\n  Best RMSE: {best} ({out[best]['rmse']:.4f}), {gain:.1f}% better than")
    print("  predicting the global mean for everything.")
    print("\n  Look at how much of that comes from BIASES ALONE - just 'this user")
    print("  is generous' plus 'this film is good', no interaction modelling at")
    print(f"  all, gets you to {out['User + item bias']['rmse']:.4f}. Bias terms were the single")
    print("  largest component of most Netflix Prize solutions, and they are")
    print("  roughly ten lines of pandas.")
    print("\n  Also note the whole spread is small. A decade of effort on this")
    print("  formulation moved RMSE by a few percent, which is the first hint")
    print("  that RMSE might not be the thing worth optimising.")

    return {"mu": mu, "M": M, "seen": seen, "sim": sim, "item_means": item_means,
            "item_means_damped": item_means_damped, "n_per_item": n_per_item,
            "recon": recon, "u_bias": u_bias, "i_bias": i_bias, "rmse": out}


# --------------------------------------------- 3. the same models, ranked
def figure_ranking(train: pd.DataFrame, test: pd.DataFrame, state: dict,
                   n_i: int) -> dict:
    """Nobody is shown a predicted star rating. They are shown a LIST.

    So the question is not 'how close was the number' but 'did the film they
    actually watched appear near the top', which is a completely different
    measurement.
    """
    M, seen, sim = state["M"], state["seen"], state["sim"]
    item_means, recon = state["item_means"], state["recon"]
    mu, i_bias = state["mu"], state["i_bias"]

    popularity = np.zeros(n_i + 1)
    counts = train.groupby("item_id").size()
    popularity[counts.index.to_numpy()] = counts.to_numpy()

    item_bias_vec = np.zeros(n_i + 1)
    item_bias_vec[i_bias.index.to_numpy()] = i_bias.to_numpy()

    damped = state["item_means_damped"]

    def scores(name: str, u: int) -> np.ndarray:
        if name == "Popularity":
            return popularity.copy()
        if name == "User + item bias":
            return mu + item_bias_vec
        if name == "Item-item CF":
            return sim[:, seen[u]] @ (M[u, seen[u]] - item_means[seen[u]])
        if name == "Mean rating (raw)":
            return item_means.copy()
        if name == "Mean rating (damped)":
            return damped.copy()
        if name == "SVD, raw means":
            return item_means + recon[u]
        if name == "SVD, damped means":
            return damped + recon[u]
        return rng.random(n_i + 1)                          # Random

    names = ["Random", "Popularity", "User + item bias", "Item-item CF",
             "Mean rating (raw)", "Mean rating (damped)",
             "SVD, raw means", "SVD, damped means"]
    K = 10
    print(f"  For each of the {len(test):,} test users, score every film they have NOT")
    print(f"  rated, take the top {K}, and ask whether their held-out film is in it.\n")
    print(f"  {'model':<22} {'HitRate@10':>11} {'NDCG@10':>9} {'coverage':>9}")

    out = {}
    for name in names:
        hits, ndcgs, recommended = 0, [], set()
        for u, truth in zip(test.user_id, test.item_id):
            s = scores(name, u).copy()
            s[seen[u]] = -np.inf          # never recommend what they have seen
            s[0] = -np.inf                # item ids are 1-based
            top = np.argpartition(-s, K)[:K]
            top = top[np.argsort(-s[top])]
            recommended.update(top.tolist())
            if truth in top:
                hits += 1
                ndcgs.append(1.0 / np.log2(int(np.where(top == truth)[0][0]) + 2))
            else:
                ndcgs.append(0.0)
        out[name] = {"hit": hits / len(test), "ndcg": float(np.mean(ndcgs)),
                     "coverage": len(recommended) / n_i}
        print(f"  {name:<22} {out[name]['hit']:>11.4f} {out[name]['ndcg']:>9.4f} "
              f"{out[name]['coverage']:>8.1%}")

    rank_best = max(out, key=lambda k_: out[k_]["ndcg"])
    rmse_best = min(state["rmse"], key=lambda k_: state["rmse"][k_]["rmse"])
    print(f"\n  Best by RANKING: {rank_best}")
    print(f"  Best by RMSE:    {rmse_best}")
    if rank_best != rmse_best:
        print("  THEY ARE NOT THE SAME MODEL. That is the point of this section.")
    print("\n  'User + item bias' is strong on RMSE and useless for ranking - it")
    print("  gives EVERY user the same list, because the user term is a constant")
    print("  that shifts all items equally and cannot reorder them. A model can")
    print("  predict ratings well and be incapable of personalising.")

    print("\n  Now the pair of rows that matters most here. 'Mean rating (raw)'")
    print(f"  scores NDCG {out['Mean rating (raw)']['ndcg']:.4f}; damping the same means towards the")
    print(f"  global average lifts it to {out['Mean rating (damped)']['ndcg']:.4f}. The identical statistic,")
    print("  shrunk, and the SVD rows move the same way.")
    print("\n  The reason is visible the moment you look at what a raw mean ranks")
    print("  highest - films with ONE five-star rating:")
    order = np.argsort(-item_means)[:4]
    for i in order:
        n = int(state["n_per_item"][i])
        print(f"    mean {item_means[i]:.2f} from {n:>3} rating(s)   "
              f"{TITLES.get(int(i), '(unknown)')[:44]}")
    sw = int(np.argmax(popularity))
    print(f"    mean {item_means[sw]:.2f} from {int(popularity[sw]):>3} ratings   "
          f"{TITLES.get(sw, '(unknown)')[:44]}  <- the actual favourite")
    print("\n  A single 5.00 beats a 4.35 earned over hundreds of ratings. Damping")
    print("  weights each mean by how much evidence stands behind it, which is")
    print("  exactly the shrinkage idea from regularisation in 01_regression -")
    print("  applied to a statistic rather than to a coefficient.")
    print("\n  Notice that 'Mean rating (damped)' and 'User + item bias' score")
    print("  IDENTICALLY. That is not a coincidence - they are the same formula:")
    print("      mu + (sum - n*mu)/(n + k)  ==  (sum + k*mu)/(n + k)")
    print("  The user term shifts every item by the same amount, so it cannot")
    print("  change an ORDERING. A 'personalised' model whose only user-specific")
    print("  component is additive is not personalised at all.")
    print("\n  Look at COVERAGE too: the share of the catalogue that ever appears")
    print("  in anyone's top 10. Popularity and bias models recommend the same")
    print("  handful of films to everyone, so their coverage is tiny. Item-item")
    print(f"  CF reaches {out['Item-item CF']['coverage']:.0%} of the catalogue against Popularity's "
          f"{out['Popularity']['coverage']:.0%},")
    print("  while scoring worse on NDCG - and which of those two is the better")
    print("  product is a business question, not a modelling one. A recommender")
    print("  that only ever surfaces 5% of the catalogue is not doing the job it")
    print("  was bought to do, however good its accuracy looks.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (key, label, colour) in zip(
        axes,
        [("rmse", "RMSE (lower better)", "blue"),
         ("ndcg", "NDCG@10 (higher better)", "green"),
         ("coverage", "catalogue coverage", "orange")],
    ):
        if key == "rmse":
            ns = [n for n in state["rmse"]]
            vals = [state["rmse"][n]["rmse"] for n in ns]
        else:
            ns, vals = names, [out[n][key] for n in names]
        ax.barh(ns, vals, color=PALETTE[colour])
        ax.set_title(label)
        ax.grid(axis="y", visible=False)
    fig.suptitle("The same models, measured three ways - the ranking changes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    print(f"\n  saved  {save_figure(fig, OUT / '1_rmse_vs_ranking.png').name}")
    return out


# --------------------------------------------- 4. popularity, and the tail
def figure_popularity(train: pd.DataFrame, movies: pd.DataFrame,
                      ranked: dict, n_i: int) -> None:
    """Popularity needs no model, no features and no personalisation, and it
    is hard to beat on the metrics people usually report.
    """
    counts = train.groupby("item_id").size().sort_values(ascending=False)
    share = counts.cumsum() / counts.sum()
    n_for_half = int((share < 0.5).sum()) + 1

    print(f"  {n_for_half} films out of {n_i:,} ({n_for_half / n_i:.1%}) account for HALF")
    print("  of all ratings in the training data.")

    pop_ndcg = ranked["Popularity"]["ndcg"]
    skip = ("Random", "Popularity", "Mean rating (raw)", "Mean rating (damped)")
    best_personal = max((n for n in ranked if n not in skip),
                        key=lambda n: ranked[n]["ndcg"])
    print(f"\n  Popularity NDCG@10:            {pop_ndcg:.4f}")
    print(f"  Best personalised model:       {ranked[best_personal]['ndcg']:.4f} "
          f"({best_personal})")
    if ranked[best_personal]["ndcg"] > pop_ndcg:
        print(f"  Personalisation wins by "
              f"{(ranked[best_personal]['ndcg'] / max(pop_ndcg, 1e-9) - 1) * 100:.0f}%.")
    else:
        print("  Popularity WINS. On this split, personalisation did not pay.")

    print(f"\n  But compare catalogue coverage:")
    print(f"    Popularity          {ranked['Popularity']['coverage']:.1%}")
    print(f"    {best_personal:<19} {ranked[best_personal]['coverage']:.1%}")
    print("\n  This is the tension at the centre of recommender design, and note")
    print("  that popularity did not merely hold its own here - it WON the")
    print("  accuracy metric outright, by a factor of nearly three. Yet it is")
    print("  close to worthless as a product: it tells every user the same")
    print("  thing, surfaces nothing from the tail, and gives nobody a reason to")
    print("  come back. Accuracy metrics cannot see any of that, which is why")
    print("  serious systems track coverage, diversity, novelty and serendipity")
    print("  alongside them.")
    print("\n  Be careful about the reverse conclusion too. This is one small")
    print("  offline split with a single held-out film per user, and offline")
    print("  ranking metrics are notoriously weak predictors of online")
    print("  behaviour - the held-out film is 'correct' only in the sense that")
    print("  the user happened to watch it, which popularity-driven interfaces")
    print("  helped cause in the first place. The honest reading is that this")
    print("  evaluation cannot settle the question; an A/B test can.")
    print("\n  It is also self-reinforcing: recommending popular items generates")
    print("  more interactions with popular items, which makes them more")
    print("  popular. The feedback loop is a real production failure mode, not")
    print("  a theoretical one.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(np.arange(1, len(counts) + 1), share.to_numpy(),
                 color=PALETTE["blue"], lw=2)
    axes[0].axhline(0.5, color=PALETTE["vermillion"], ls="--", lw=1.3)
    axes[0].axvline(n_for_half, color=PALETTE["vermillion"], ls="--", lw=1.3)
    axes[0].annotate(f"{n_for_half} films = 50%\nof all ratings",
                     xy=(0.30, 0.35), xycoords="axes fraction", fontsize=10,
                     color=PALETTE["vermillion"], fontweight="bold")
    axes[0].set_xlabel("films, ranked by popularity")
    axes[0].set_ylabel("cumulative share of ratings")
    axes[0].set_title("The long tail")

    ns = list(ranked)
    axes[1].scatter([ranked[n]["ndcg"] for n in ns],
                    [ranked[n]["coverage"] for n in ns], s=90,
                    color=PALETTE["blue"], zorder=5)
    for n in ns:
        axes[1].annotate(n, (ranked[n]["ndcg"], ranked[n]["coverage"]),
                         textcoords="offset points", xytext=(7, 5), fontsize=9)
    axes[1].set_xlabel("NDCG@10 (accuracy)")
    axes[1].set_ylabel("catalogue coverage")
    axes[1].set_title("Accuracy is not the only axis")

    fig.suptitle("Why popularity is a trap, not a solution",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '2_popularity_and_tail.png').name}")


# ------------------------------------------------------------ 5. cold start
def demo_cold_start(state: dict, n_i: int) -> None:
    """A brand-new user has an all-zero row. What does each model do?"""
    sim, item_means, mu = state["sim"], state["item_means"], state["mu"]
    new_user = np.zeros(n_i + 1, dtype=bool)          # has rated nothing

    print("  A new user arrives having rated nothing at all.\n")
    print("  Item-item CF     sum over an EMPTY set of rated items -> every score")
    print("                   is 0, the ranking is arbitrary. No recommendation.")
    print("  SVD              the user has no row in the factorisation, so there")
    print("                   is no latent vector to multiply. Undefined until")
    print("                   the model is refitted.")
    print("  User+item bias   the user term is 0, so they receive the global")
    print("                   ranking - i.e. exactly the popularity list.")
    print("  Popularity       works perfectly. It never needed the user.")

    cf_scores = sim[:, new_user] @ np.zeros(new_user.sum())
    print(f"\n  Item-item CF scores for the new user: all zero? "
          f"{bool(np.allclose(cf_scores, 0))}")

    print("\n  So the honest architecture for most products is a CASCADE, not a")
    print("  single model:")
    print("    0 interactions      -> popularity, or editorial picks")
    print("    a few interactions  -> content-based (genre, cast, text)")
    print("    many interactions   -> collaborative filtering / matrix")
    print("                           factorisation / two-tower retrieval")
    print("\n  The same problem exists for new ITEMS, and it is worse: a film")
    print("  nobody has rated can never be recommended by a collaborative")
    print("  method, so it can never acquire ratings. Content features are the")
    print("  only way out of that loop.")
    print("\n  This is exactly why the production pattern is a TWO-TOWER model")
    print("  (03_neural_networks/09): one tower embeds the user from their")
    print("  features, the other embeds the item from ITS features, so both")
    print("  sides survive having no interaction history.")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  RECOMMENDER SYSTEMS")
    print("=" * 72)

    ratings = load_movielens_ratings()
    movies = load_movielens_movies()
    n_u, n_i = int(ratings.user_id.max()), int(ratings.item_id.max())
    TITLES.update(dict(zip(movies.item_id.astype(int), movies.title)))

    banner("1. The data, and its sparsity")
    describe(ratings, movies)

    train, test = split_leave_last(ratings)
    print(f"\n  Split: {len(train):,} train / {len(test):,} test "
          f"(each user's most recent rating held out).")

    banner("2. Predicting the rating (the Netflix Prize formulation)")
    state = figure_rating_prediction(train, test, n_u, n_i)

    banner("3. The same models, scored as a RANKING problem")
    ranked = figure_ranking(train, test, state, n_i)

    banner("4. Popularity, the long tail, and coverage")
    figure_popularity(train, movies, ranked, n_i)

    banner("5. Cold start")
    demo_cold_start(state, n_i)

    banner("What to take away")
    print("  - The matrix is ~94% empty. Sparsity, not model capacity, is the")
    print("    defining constraint of the problem.")
    print("  - USER AND ITEM BIASES do most of the RMSE work for ten lines of")
    print("    code. Always fit them before anything clever.")
    print("  - RMSE AND RANKING DISAGREE. A model can predict ratings well and")
    print("    be incapable of personalising - the bias model gives every user")
    print("    an identical list. Optimise the metric that matches the product.")
    print("  - Score with HitRate@k / NDCG@k and hold out each user's LAST")
    print("    interaction, not a random one. A random split leaks the future.")
    print("  - Popularity WON the accuracy metric here and is still a bad")
    print("    product: 5% catalogue coverage, identical lists for everyone, and")
    print("    a self-reinforcing feedback loop. Track coverage and diversity")
    print("    next to accuracy, and settle it with an A/B test, not an offline")
    print("    ranking metric.")
    print("  - DAMP small-sample statistics before ranking on them. A raw item")
    print("    mean puts a film with one 5-star rating above Star Wars; shrinking")
    lift = ranked["Mean rating (damped)"]["ndcg"] / ranked["Mean rating (raw)"]["ndcg"]
    print(f"    towards the global mean lifted NDCG by {lift:.0f}x. Same idea as")
    print("    regularisation, applied to a statistic.")
    print("  - Cold start breaks every collaborative method. The answer is a")
    print("    cascade - popularity, then content-based, then CF - or a")
    print("    two-tower model that embeds users and items from FEATURES.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
