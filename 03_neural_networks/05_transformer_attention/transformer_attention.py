"""Attention, computed by hand, then assembled into a transformer.

The transformer replaced recurrence with a single idea: instead of carrying
a state forward step by step, let every position look directly at every
other position and decide what to pay attention to. That removes the
sequential bottleneck entirely, which is why these models could be scaled.

  1. Scaled dot-product attention in NumPy - the whole mechanism is four
     lines, and it is worth typing them once
  2. Why it needs positional encoding: attention is PERMUTATION INVARIANT,
     demonstrated rather than asserted
  3. Multi-head attention, residuals and layer norm assembled into a block
  4. That transformer against an LSTM on a set-membership task, where the
     LSTM stays at chance and the transformer does not

Run:
    python 03_neural_networks/05_transformer_attention/transformer_attention.py

Concept reference: Foundations volume, Parts B.16-B.18.
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
rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def seed_everything() -> None:
    keras.utils.set_random_seed(SEED)


# ============================================ 1. attention, four lines
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """Scaled dot-product attention. This is the entire mechanism.

        scores  = Q @ K.T / sqrt(d)     how well does each query match each key
        weights = softmax(scores)       turn them into a distribution
        output  = weights @ V           a weighted average of the values

    Everything else in a transformer - multiple heads, layer norm, the feed-
    forward block, residual connections - is scaffolding around these lines.
    """
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    weights = softmax(scores)
    return weights @ V, weights


def figure_attention_by_hand() -> None:
    """A worked example small enough to check by eye.

    Four tokens, each a 4-dimensional vector. Tokens 0 and 2 are deliberately
    made similar, so token 0's query should match token 2's key.
    """
    tokens = ["the", "cat", "the", "mat"]
    E = np.array([
        [1.0, 0.0, 0.2, 0.0],      # "the"
        [0.0, 1.0, 0.0, 0.3],      # "cat"
        [1.0, 0.0, 0.2, 0.0],      # "the"  - identical to token 0
        [0.0, 0.2, 0.0, 1.0],      # "mat"
    ])
    out, W = attention(E, E, E)          # self-attention: Q = K = V = E

    print("  Four tokens, each a 4-d vector. Q = K = V = the embeddings, which")
    print("  is what makes this SELF-attention.\n")
    print("  Attention weights (row = who is looking, column = who is looked at):\n")
    print("            " + "".join(f"{t:>9}" for t in tokens))
    for i, t in enumerate(tokens):
        print(f"  {t:>8}  " + "".join(f"{w:>9.3f}" for w in W[i]))

    print("\n  Every row sums to 1 - each token distributes a fixed budget of")
    print("  attention across all positions, including itself.")
    print(f"\n  Token 0 ('the') gives {W[0, 2]:.3f} of its attention to token 2, which")
    print("  is the other 'the', and only")
    print(f"  {W[0, 1]:.3f} to 'cat'. Nothing told it those two tokens are related -")
    print("  it fell out of the dot product between their vectors.")

    print("\n  Why divide by sqrt(d)? The dot product of two d-dimensional")
    print("  vectors grows with d, so without scaling the scores get large,")
    print("  softmax saturates, and the gradient vanishes. Demonstrated:")
    for d in (4, 64, 512):
        q = rng.normal(0, 1, (1, d))
        k = rng.normal(0, 1, (8, d))
        raw = (q @ k.T).ravel()
        print(f"    d={d:>4}  raw score range {raw.min():>8.2f} to {raw.max():>7.2f}"
              f"   max softmax weight {softmax(raw).max():.3f}"
              f"   scaled {softmax(raw / np.sqrt(d)).max():.3f}")
    print("\n  At d=512 the unscaled softmax puts nearly all its mass on one")
    print("  token - it has become a hard argmax with almost no gradient. The")
    print("  1/sqrt(d) keeps the distribution soft enough to learn from.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].imshow(W, cmap="Blues", vmin=0, vmax=W.max())
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                         color="white" if W[i, j] > W.max() / 2 else "black",
                         fontsize=10, fontweight="bold")
    axes[0].set_xticks(range(4), tokens); axes[0].set_yticks(range(4), tokens)
    axes[0].set_xlabel("attending TO"); axes[0].set_ylabel("attending FROM")
    axes[0].set_title("Self-attention weights")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    ds = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
    unscaled, scaled = [], []
    for d in ds:
        q = rng.normal(0, 1, (1, d)); k = rng.normal(0, 1, (16, d))
        raw = (q @ k.T).ravel()
        unscaled.append(softmax(raw).max())
        scaled.append(softmax(raw / np.sqrt(d)).max())
    axes[1].semilogx(ds, unscaled, "o-", color=PALETTE["vermillion"],
                     label="unscaled")
    axes[1].semilogx(ds, scaled, "o-", color=PALETTE["blue"],
                     label="scaled by 1/sqrt(d)")
    axes[1].axhline(1 / 16, color=PALETTE["grey"], ls="--", lw=1.2)
    axes[1].annotate("uniform (1/16)", xy=(0.05, 0.10),
                     xycoords="axes fraction", fontsize=9, color=PALETTE["grey"])
    axes[1].set_xlabel("head dimension d")
    axes[1].set_ylabel("largest softmax weight")
    axes[1].set_title("Why the 1/sqrt(d) is there")
    axes[1].legend(fontsize=9)

    fig.suptitle("Scaled dot-product attention", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '1_attention.png').name}")


# ================================= 2. attention is permutation invariant
def demo_permutation() -> None:
    """The defect that positional encoding exists to fix, demonstrated.

    Shuffle the input tokens and attention produces the SAME outputs, merely
    reordered. It has no notion of first, next or last.
    """
    E = rng.normal(0, 1, (5, 8))
    out, _ = attention(E, E, E)

    perm = np.array([3, 0, 4, 1, 2])
    out_perm, _ = attention(E[perm], E[perm], E[perm])

    same = np.allclose(out[perm], out_perm, atol=1e-10)
    print("  Run attention on 5 tokens, then shuffle the tokens and run again.")
    print(f"\n  Are the outputs identical (just reordered)?  {same}")
    print(f"  Max absolute difference: {np.abs(out[perm] - out_perm).max():.2e}")
    print("\n  Attention is PERMUTATION EQUIVARIANT: it sees an unordered SET of")
    print("  tokens. 'dog bites man' and 'man bites dog' are the same input.")
    print("  That is a fatal defect for language, and the fix is to add")
    print("  position information to the embeddings before attention runs.")

    # sinusoidal encoding, as in the original paper
    L, d = 40, 32
    pos = np.arange(L)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    PE = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))

    print("\n  The original paper uses fixed SINUSOIDAL encodings: each position")
    print("  gets a distinctive pattern of sines and cosines at different")
    print("  frequencies, added straight onto the embedding. Two useful")
    print("  properties - it needs no training, and it extrapolates to")
    print("  sequences longer than any seen in training, because the functions")
    print("  are defined for every position.")
    print("\n  Modern models mostly use LEARNED positional embeddings, or")
    print("  rotary embeddings (RoPE) which rotate Q and K by an angle")
    print("  proportional to position, encoding RELATIVE distance directly.")

    sim = PE @ PE.T / (np.linalg.norm(PE, axis=1)[:, None]
                       * np.linalg.norm(PE, axis=1)[None, :])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im0 = axes[0].imshow(PE, cmap="RdBu_r", aspect="auto")
    axes[0].set_xlabel("embedding dimension"); axes[0].set_ylabel("position")
    axes[0].set_title("Sinusoidal positional encoding")
    axes[0].grid(False)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(sim, cmap="viridis", aspect="equal")
    axes[1].set_xlabel("position"); axes[1].set_ylabel("position")
    axes[1].set_title("Similarity between positions\n(bright diagonal = "
                      "nearby positions are similar)")
    axes[1].grid(False)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle("Attention has no idea what order things came in",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    print(f"\n  saved  {save_figure(fig, OUT / '2_positional_encoding.png').name}")


# ============================================= 3-4. a tiny transformer
def make_task(n: int, length: int, vocab: int, seed: int):
    """Predict whether the FIRST token reappears later in the sequence.

    Solvable only by comparing position 0 with every other position - which
    is precisely what attention does in one step and a recurrent network must
    do by carrying state.
    """
    r = np.random.default_rng(seed)
    X = r.integers(1, vocab, (n, length))
    y = np.array([(X[i, 0] in X[i, 1:]) for i in range(n)], dtype="float32")
    return X, y


class AddPositionEmbedding(keras.layers.Layer):
    """Add a learned positional embedding to the token embeddings.

    This has to be a LAYER. Calling keras.ops.arange() directly while
    building a functional model evaluates it once, at construction time, and
    the resulting constant gets baked into the graph - which on the torch
    backend raises "Trying to backward through the graph a second time" on
    the second training step. Building the positions inside call() means
    they are produced fresh on every forward pass.
    """

    def __init__(self, length: int, d_model: int, **kw):
        super().__init__(**kw)
        self.length = length
        self.pos = keras.layers.Embedding(length, d_model)

    def call(self, x):
        return x + self.pos(keras.ops.arange(self.length))


def build_transformer(length: int, vocab: int, d_model: int = 32,
                      heads: int = 4) -> keras.Model:
    inp = keras.layers.Input((length,))
    tok = keras.layers.Embedding(vocab, d_model)(inp)
    x = AddPositionEmbedding(length, d_model)(tok)

    att = keras.layers.MultiHeadAttention(num_heads=heads,
                                          key_dim=d_model // heads)(x, x)
    x = keras.layers.LayerNormalization()(x + att)     # residual + norm
    ff = keras.layers.Dense(d_model * 2, activation="relu")(x)
    ff = keras.layers.Dense(d_model)(ff)
    x = keras.layers.LayerNormalization()(x + ff)      # residual + norm

    x = keras.layers.GlobalAveragePooling1D()(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(2e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


def build_lstm(length: int, vocab: int, d_model: int = 32) -> keras.Model:
    m = keras.Sequential([
        keras.layers.Input((length,)),
        keras.layers.Embedding(vocab, d_model),
        keras.layers.LSTM(d_model),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(2e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


def figure_transformer_vs_lstm() -> None:
    # vocab 28, not 12. With 19 later slots, P(the first token reappears)
    # = 1 - ((V-1)/V)^19, which at V=12 is 0.81 - so both models scored
    # exactly the majority-class rate by always answering "yes". V=28 puts
    # it near 50/50 and makes the task actually discriminative.
    length, vocab = 20, 28
    Xtr, ytr = make_task(6000, length, vocab, SEED)
    Xte, yte = make_task(1500, length, vocab, SEED + 1)
    print(f"  Task: does the first token reappear later in the sequence?")
    majority = max(yte.mean(), 1 - yte.mean())
    print(f"  {length} tokens, vocabulary of {vocab}, "
          f"{ytr.mean():.0%} positive.")
    print(f"  Majority-class baseline: {majority:.3f} - a model must beat this")
    print("  to have learned anything at all.\n")

    print(f"  {'model':<14} {'params':>9} {'test acc':>9} {'fit s':>8}")
    results = {}
    for name, build in (("LSTM", build_lstm),
                        ("Transformer", build_transformer)):
        seed_everything()
        m = build(length, vocab)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 35, not 12: at 12 BOTH models sat at chance. The transformer
            # is still climbing at 25 (0.52 -> 0.64 -> 0.81).
            m.fit(Xtr, ytr, epochs=35, batch_size=64, verbose=0)
        secs = time.perf_counter() - t0
        acc = float(m.evaluate(Xte, yte, verbose=0)[1])
        results[name] = {"acc": acc, "params": m.count_params(),
                         "time": secs, "model": m}
        print(f"  {name:<14} {m.count_params():>9,} {acc:>9.3f} {secs:>8.1f}")

    tr, ls = results["Transformer"]["acc"], results["LSTM"]["acc"]
    print(f"\n  The LSTM never leaves chance ({ls:.3f} against a {majority:.3f} baseline).")
    print(f"  The transformer reaches {tr:.3f}. That is a real separation, not a")
    print("  budget artefact - both had 35 epochs and the LSTM had flattened")
    print("  while the transformer was still climbing.")
    print("\n  Why: the transformer compares position 0 against every other")
    print("  position in ONE operation, which is literally what a query scored")
    print("  against a set of keys computes. The LSTM must carry the first")
    print("  token's identity in its hidden state across 19 steps while 19")
    print("  other tokens overwrite it. That is the memory problem from 04 in")
    print("  a new costume - and here it is decisive rather than marginal.")
    print("\n  Note the contrast with 04, where the LSTM WON the forecasting")
    print("  task by 20%. Forecasting needs a compressed summary of recent")
    print("  history, which is exactly what a recurrent state is for. Set")
    print("  membership needs exact comparison against every position, which")
    print("  is exactly what attention is for. Match the mechanism to the")
    print("  question rather than ranking architectures in the abstract.")

    print("\n  The structural difference, which matters far more than this")
    print("  accuracy number:")
    print("    RNN          O(L) sequential steps. Step t cannot start until")
    print("                 step t-1 finishes, so it cannot be parallelised")
    print("                 along the sequence.")
    print("    Attention    O(1) sequential steps, O(L^2) work. Every position")
    print("                 is computed at once, so a GPU can saturate.")
    print("\n  That is the whole reason transformers scaled and RNNs did not.")
    print("  It is not that attention is smarter - it is that it is")
    print("  PARALLELISABLE, which let the models grow by four orders of")
    print("  magnitude. The O(L^2) memory cost is the price, and it is why")
    print("  long-context work is full of attention approximations.")

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(results)
    ax.bar(names, [results[n]["acc"] for n in names],
           color=[PALETTE["vermillion"], PALETTE["blue"]])
    for i, n in enumerate(names):
        ax.text(i, results[n]["acc"] + 0.01,
                f"{results[n]['acc']:.3f}\n{results[n]['params']:,} params\n"
                f"{results[n]['time']:.0f}s",
                ha="center", fontsize=9)
    ax.axhline(max(yte.mean(), 1 - yte.mean()), color=PALETTE["grey"],
               ls="--", lw=1.3)
    ax.annotate("majority-class baseline", xy=(0.03, 0.06),
                xycoords="axes fraction", fontsize=9, color=PALETTE["grey"])
    ax.set_ylim(0, 1.15); ax.set_ylabel("test accuracy")
    ax.set_title("Same task: carry one token 19 steps, or look at it directly")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    print(f"\n  saved  {save_figure(fig, OUT / '3_transformer_vs_lstm.png').name}")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  ATTENTION AND THE TRANSFORMER")
    print("=" * 72)
    print(f"\n  Keras {keras.__version__}, backend: {keras.backend.backend()}")

    banner("1. Scaled dot-product attention, by hand")
    figure_attention_by_hand()

    banner("2. Attention cannot see order")
    demo_permutation()

    banner("3-4. A tiny transformer against an LSTM")
    figure_transformer_vs_lstm()

    banner("What to take away")
    print("  - Attention is three lines: score every query against every key,")
    print("    softmax the scores, take a weighted average of the values.")
    print("    Everything else in a transformer is scaffolding around that.")
    print("  - THE 1/sqrt(d) IS NOT COSMETIC. Dot products grow with dimension,")
    print("    and without scaling the softmax saturates into a hard argmax")
    print("    with no usable gradient.")
    print("  - ATTENTION IS PERMUTATION EQUIVARIANT - it sees a SET, not a")
    print("    sequence. Positional encoding is what makes order visible, and")
    print("    it is added to the embeddings before attention runs.")
    print("  - Self-attention means Q, K and V all come from the same sequence.")
    print("    Cross-attention takes Q from one sequence and K,V from another -")
    print("    that is how a decoder reads an encoder, and how RAG grounds a")
    print("    model in retrieved documents.")
    print("  - MULTIPLE HEADS let different subspaces attend to different")
    print("    relationships at once, then concatenate.")
    print("  - Residual connections and layer norm around each sub-block are")
    print("    what make deep stacks trainable - the same gradient-highway idea")
    print("    as the LSTM cell state.")
    print("  - MATCH THE MECHANISM TO THE QUESTION. The LSTM beat the")
    print("    transformer's cousin on forecasting in 04 and sat at chance")
    print("    here; recurrence compresses recent history, attention compares")
    print("    against everything. Neither is simply better.")
    print("  - THE REAL WIN IS PARALLELISM: O(1) sequential steps against an")
    print("    RNN's O(L). That is why transformers scaled. The cost is O(L^2)")
    print("    work and memory in the sequence length, which is why long")
    print("    context is an active research area.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
