"""The perceptron, and backpropagation, written out by hand in NumPy.

No TensorFlow, no PyTorch, no autograd. Every gradient here is derived on
paper and typed in, because the single most useful thing you can do before
using a framework is to have written the thing the framework is doing.

  1. A single neuron - the 1958 perceptron learning rule on a problem it
     can solve, and the same rule failing forever on XOR
  2. Why it fails, and why one hidden layer fixes it
  3. Backpropagation derived and implemented - four lines of chain rule
  4. Gradient checking - proving the hand-derived gradients are correct
     against a numerical estimate, which is how you debug a real network
  5. Activation functions, and why nobody uses sigmoid in hidden layers
     any more - the vanishing gradient, measured

Run:
    python 03_neural_networks/01_perceptron_from_scratch/perceptron.py

Concept reference: Foundations volume, Parts B.1-B.5.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
rng = np.random.default_rng(SEED)


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ============================================================ 1. one neuron
def perceptron_train(X: np.ndarray, y: np.ndarray, epochs: int = 20,
                     lr: float = 0.1) -> tuple[np.ndarray, float, list[int]]:
    """Rosenblatt's 1958 rule, unchanged.

        prediction = step(w . x + b)
        on a mistake:  w += lr * (target - prediction) * x

    There is no loss function and no gradient here - it predates both. The
    update simply nudges the boundary towards any point it got wrong.
    """
    w, b = np.zeros(X.shape[1]), 0.0
    errors = []
    for _ in range(epochs):
        wrong = 0
        for xi, target in zip(X, y):
            pred = 1 if (xi @ w + b) > 0 else 0
            update = lr * (target - pred)
            if update != 0:
                w += update * xi
                b += update
                wrong += 1
        errors.append(wrong)
    return w, b, errors


def figure_perceptron() -> None:
    # AND is linearly separable; XOR is the famous counter-example.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    problems = {
        "AND": np.array([0, 0, 0, 1]),
        "OR": np.array([0, 1, 1, 1]),
        "XOR": np.array([0, 1, 1, 0]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, (name, y) in enumerate(problems.items()):
        w, b, errors = perceptron_train(X, y, epochs=20)
        preds = ((X @ w + b) > 0).astype(int)
        solved = bool((preds == y).all())
        print(f"  {name:<4} solved: {str(solved):<5}  "
              f"misclassified in final epoch: {errors[-1]}")

        ax = axes[0, col]
        gx, gy = np.meshgrid(np.linspace(-0.4, 1.4, 200),
                             np.linspace(-0.4, 1.4, 200))
        zz = ((np.c_[gx.ravel(), gy.ravel()] @ w + b) > 0).reshape(gx.shape)
        ax.contourf(gx, gy, zz, alpha=0.2, cmap="coolwarm", levels=1)
        for cls, colour in ((0, "blue"), (1, "vermillion")):
            m = y == cls
            ax.scatter(X[m, 0], X[m, 1], s=180, color=PALETTE[colour],
                       edgecolors="k", zorder=5, label=f"class {cls}")
        ax.set_title(f"{name}  -  {'SOLVED' if solved else 'IMPOSSIBLE'}",
                     fontsize=12)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        if col == 0:
            ax.legend(fontsize=9, loc="center left")

        axes[1, col].plot(range(1, len(errors) + 1), errors, "o-",
                          color=PALETTE["blue"])
        axes[1, col].set_xlabel("epoch")
        axes[1, col].set_ylabel("points misclassified")
        axes[1, col].set_ylim(-0.2, 4.2)

    fig.suptitle("One neuron draws one straight line. XOR needs two.",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    print("\n  AND and OR are LINEARLY SEPARABLE - one straight line divides")
    print("  the classes - so the perceptron finds a solution and the error")
    print("  count drops to zero and stays there.")
    print("\n  XOR is not. Look at the bottom-right panel: the error count never")
    print("  settles. The rule keeps moving the line, forever, because no line")
    print("  exists. This is not slow convergence, it is IMPOSSIBILITY, and")
    print("  Minsky and Papert's 1969 proof of it stopped neural network")
    print("  research for most of a decade.")
    print(f"\n  saved  {save_figure(fig, OUT / '1_perceptron_limits.png').name}")


# ================================================== 2-3. backprop, by hand
class TinyNet:
    """A two-layer network with every derivative written out explicitly.

        z1 = X W1 + b1        h  = tanh(z1)
        z2 = h W2 + b2        y^ = sigmoid(z2)
        L  = binary cross-entropy(y^, y)

    The backward pass below is the chain rule applied four times. If you can
    follow it, you can read any framework's autograd output.
    """

    def __init__(self, n_in: int, n_hidden: int, seed: int = SEED) -> None:
        r = np.random.default_rng(seed)
        # Xavier/Glorot scaling: variance 1/n_in keeps activations from
        # saturating or vanishing as they pass through the layer.
        self.W1 = r.normal(0, np.sqrt(1 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = r.normal(0, np.sqrt(1 / n_hidden), (n_hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h = np.tanh(self.z1)
        self.z2 = self.h @ self.W2 + self.b2
        self.out = 1 / (1 + np.exp(-self.z2))
        return self.out.ravel()

    def loss(self, y: np.ndarray) -> float:
        p = np.clip(self.out.ravel(), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def backward(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """The chain rule, four steps, no magic.

        The first line is the one worth memorising: for a sigmoid output with
        binary cross-entropy loss, dL/dz2 is just (prediction - target). The
        sigmoid derivative and the log in the loss cancel exactly. That
        cancellation is WHY these two are always paired.
        """
        n = len(y)
        dz2 = (self.out.ravel() - y).reshape(-1, 1) / n     # sigmoid + BCE cancel
        dW2 = self.h.T @ dz2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ self.W2.T
        dz1 = dh * (1 - self.h ** 2)                        # d/dx tanh = 1 - tanh^2
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads: dict[str, np.ndarray], lr: float) -> None:
        for k, g in grads.items():
            setattr(self, k, getattr(self, k) - lr * g)


def figure_xor_solved() -> TinyNet:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)

    net = TinyNet(2, 4)
    losses = []
    for _ in range(3000):
        net.forward(X)
        losses.append(net.loss(y))
        net.step(net.backward(y), lr=0.5)

    preds = net.forward(X)
    print("  Two inputs -> 4 hidden units (tanh) -> 1 output (sigmoid).")
    print(f"  Trainable parameters: "
          f"{net.W1.size + net.b1.size + net.W2.size + net.b2.size}")
    print(f"\n  Loss: {losses[0]:.4f} at the start -> {losses[-1]:.6f} after 3000 steps\n")
    print(f"  {'input':<10} {'target':>7} {'predicted':>11}")
    for xi, t, p in zip(X, y, preds):
        print(f"  {str(xi.astype(int)):<10} {t:>7.0f} {p:>11.4f}")
    print("\n  XOR solved. The hidden layer is what changed: each hidden unit")
    print("  draws its own line, and the output layer combines them. Two lines")
    print("  can carve out the XOR region; one cannot.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(losses, color=PALETTE["blue"], lw=1.8)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("binary cross-entropy")
    axes[0].set_title("Training loss")
    axes[0].set_yscale("log")

    gx, gy = np.meshgrid(np.linspace(-0.3, 1.3, 300), np.linspace(-0.3, 1.3, 300))
    grid = np.c_[gx.ravel(), gy.ravel()]
    zz = net.forward(grid).reshape(gx.shape)
    c = axes[1].contourf(gx, gy, zz, levels=20, cmap="coolwarm", alpha=0.85)
    for cls, colour in ((0, "blue"), (1, "vermillion")):
        m = y == cls
        axes[1].scatter(X[m, 0], X[m, 1], s=200, color=PALETTE[colour],
                        edgecolors="k", zorder=5)
    axes[1].set_title("The learned function - curved, not linear")
    fig.colorbar(c, ax=axes[1], shrink=0.85)

    # what each hidden unit is doing on its own
    h = np.tanh(grid @ net.W1 + net.b1)
    for j in range(net.W1.shape[1]):
        axes[2].contour(gx, gy, h[:, j].reshape(gx.shape), levels=[0],
                        colors=[PALETTE[["blue", "vermillion", "green",
                                         "orange"][j]]], linewidths=2)
    for cls in (0, 1):
        m = y == cls
        axes[2].scatter(X[m, 0], X[m, 1], s=200,
                        color=PALETTE["blue" if cls == 0 else "vermillion"],
                        edgecolors="k", zorder=5)
    axes[2].set_title("Each hidden unit's own dividing line")

    fig.suptitle("One hidden layer solves XOR", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print("\n  The right-hand panel shows each hidden unit's zero-crossing - its")
    print("  personal straight line. The network's curved boundary is built")
    print("  entirely out of those straight pieces. That is the whole idea of")
    print("  depth: compose simple functions into a complicated one.")
    print(f"\n  saved  {save_figure(fig, OUT / '2_xor_solved.png').name}")
    return net


# ==================================================== 4. gradient checking
def gradient_check(net: TinyNet, X: np.ndarray, y: np.ndarray,
                   eps: float = 1e-6) -> None:
    """Compare every hand-derived gradient against a numerical estimate.

        numerical  =  (L(w + eps) - L(w - eps)) / (2 * eps)

    This is slow - two forward passes per parameter - so it is a DEBUGGING
    tool, never part of training. But it is the only way to be sure a
    hand-written backward pass is right, and it catches sign errors and
    transposed matrices instantly.
    """
    net.forward(X)
    analytic = net.backward(y)

    print("  Comparing hand-derived gradients with a numerical estimate:\n")
    print(f"  {'parameter':<10} {'max abs diff':>14} {'relative error':>16}  verdict")
    all_ok = True
    for name in ("W1", "b1", "W2", "b2"):
        P = getattr(net, name)
        num = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            original = P[i]
            P[i] = original + eps
            net.forward(X); lp = net.loss(y)
            P[i] = original - eps
            net.forward(X); lm = net.loss(y)
            P[i] = original
            num[i] = (lp - lm) / (2 * eps)
            it.iternext()

        a = analytic[name]
        diff = np.abs(a - num).max()
        denom = max(np.abs(a).max() + np.abs(num).max(), 1e-12)
        rel = diff / denom
        ok = rel < 1e-6
        all_ok &= ok
        print(f"  {name:<10} {diff:>14.3e} {rel:>16.3e}  "
              f"{'PASS' if ok else 'FAIL'}")

    net.forward(X)          # restore state after the perturbations
    print(f"\n  All gradients correct: {all_ok}")
    print("\n  A relative error below ~1e-6 means the analytic gradient and the")
    print("  numerical one agree to the limits of floating-point precision.")
    print("  If you ever write a backward pass by hand, write this check first -")
    print("  a transposed matrix or a dropped minus sign still trains, just")
    print("  badly, and you will blame the learning rate for a week.")
    print("\n  Note the cost: two full forward passes PER PARAMETER. For the")
    print(f"  {sum(getattr(net, n).size for n in ('W1', 'b1', 'W2', 'b2'))} "
          f"parameters here that is fine. For a network with 100 million it is")
    print("  impossible, which is exactly why autograd exists - it computes")
    print("  every gradient in ONE backward pass, at the cost of one forward.")


# ================================================ 5. activation functions
def figure_activations() -> None:
    """Why the field abandoned sigmoid for hidden layers.

    The gradient of a sigmoid is at most 0.25. Multiply a few of those
    together during backpropagation and the signal reaching the early layers
    is effectively zero - the vanishing gradient problem, which is measurable
    rather than theoretical.
    """
    z = np.linspace(-6, 6, 400)
    acts = {
        "sigmoid": (1 / (1 + np.exp(-z)),
                    (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
        "tanh": (np.tanh(z), 1 - np.tanh(z) ** 2),
        "ReLU": (np.maximum(0, z), (z > 0).astype(float)),
        "Leaky ReLU": (np.where(z > 0, z, 0.01 * z),
                       np.where(z > 0, 1.0, 0.01)),
    }

    print(f"  {'activation':<12} {'max gradient':>13} {'gradient at z=4':>17} "
          f"{'after 10 layers':>17}")
    for name, (_, d) in acts.items():
        mx = d.max()
        at4 = d[np.argmin(np.abs(z - 4))]
        print(f"  {name:<12} {mx:>13.4f} {at4:>17.6f} {mx ** 10:>17.2e}")

    print("\n  Read the last column. Backpropagation MULTIPLIES gradients layer")
    print("  by layer, so the BEST case for a 10-layer sigmoid network is")
    print("  0.25^10 - about one part in a million. The early layers receive")
    print("  almost nothing and never learn.")
    print("\n  Two honest caveats about that column. It uses each activation's")
    print("  MAXIMUM gradient, which every unit attains only at exactly z=0, so")
    print("  real networks do considerably worse. And it makes tanh look as safe")
    print("  as ReLU, which it is not - tanh saturates at both ends just like")
    print("  sigmoid, it merely happens to have slope 1 at the origin. The")
    print("  measured experiment below is the trustworthy version.")
    print("\n  ReLU's gradient is exactly 1 wherever the unit is active, so it")
    print("  multiplies out to 1 no matter how deep the network. That single")
    print("  property is most of why deep learning started working in 2012.")
    print("\n  ReLU's own failure mode is the DEAD UNIT: gradient is exactly 0")
    print("  for negative inputs, so a unit pushed firmly negative stops")
    print("  updating permanently. Leaky ReLU's 0.01 slope exists to prevent")
    print("  exactly that.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (name, (a, d)), colour in zip(acts.items(),
                                      ("blue", "vermillion", "green", "orange")):
        axes[0].plot(z, a, color=PALETTE[colour], lw=2, label=name)
        axes[1].plot(z, d, color=PALETTE[colour], lw=2, label=name)
    axes[0].set_title("The activation"); axes[0].set_xlabel("z")
    axes[1].set_title("Its gradient - this is what backprop multiplies")
    axes[1].set_xlabel("z")
    axes[1].axhline(0.25, color=PALETTE["grey"], ls="--", lw=1.2)
    axes[1].annotate("sigmoid's maximum is 0.25", xy=(0.04, 0.30),
                     xycoords="axes fraction", fontsize=9, color=PALETTE["grey"])
    for ax in axes:
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", lw=0.8)

    fig.suptitle("Why hidden layers stopped using sigmoid",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '3_activations.png').name}")

    # ---- demonstrate it, rather than asserting it
    print("\n  Measuring it on a real network: gradient magnitude reaching each")
    print("  layer of a 6-layer net, sigmoid vs ReLU.\n")
    depth = 6
    Xd = rng.normal(size=(256, 20))
    yd = (Xd[:, 0] + Xd[:, 1] > 0).astype(float)
    results = {}
    for act in ("sigmoid", "relu"):
        r = np.random.default_rng(SEED)
        Ws = [r.normal(0, np.sqrt(1 / 20), (20, 20)) for _ in range(depth)]
        Wout = r.normal(0, np.sqrt(1 / 20), (20, 1))
        hs, h = [Xd], Xd
        for W in Ws:
            zl = h @ W
            h = 1 / (1 + np.exp(-zl)) if act == "sigmoid" else np.maximum(0, zl)
            hs.append(h)
        out = 1 / (1 + np.exp(-(h @ Wout)))
        delta = (out.ravel() - yd).reshape(-1, 1) / len(yd)
        norms = []
        d = delta @ Wout.T
        for li in range(depth - 1, -1, -1):
            hl = hs[li + 1]
            d = d * (hl * (1 - hl) if act == "sigmoid" else (hl > 0).astype(float))
            norms.append(float(np.abs(hs[li].T @ d).mean()))
            d = d @ Ws[li].T
        results[act] = norms[::-1]

    print(f"  {'layer':>6} {'sigmoid':>14} {'ReLU':>14}   ReLU/sigmoid")
    for i, (s, r_) in enumerate(zip(results["sigmoid"], results["relu"]), start=1):
        print(f"  {i:>6} {s:>14.3e} {r_:>14.3e}   {r_ / max(s, 1e-30):>11.1f}x")

    # The honest measure is each network's OWN attenuation from the layer
    # nearest the loss to the layer furthest from it.
    sig_att = results["sigmoid"][-1] / max(results["sigmoid"][0], 1e-30)
    relu_att = results["relu"][-1] / max(results["relu"][0], 1e-30)
    print("\n  Layer 6 sits next to the loss; layer 1 is the input layer, furthest")
    print("  from it. Read each column top to bottom - that is the gradient")
    print("  DECAYING as it propagates backwards:")
    print(f"    sigmoid  attenuates by {sig_att:>8.0f}x from layer 6 to layer 1")
    print(f"    ReLU     attenuates by {relu_att:>8.1f}x over the same distance")
    print("\n  The sigmoid network's input layer receives a gradient roughly")
    print(f"  {sig_att:.0f} times smaller than its output layer, so it is for practical")
    print("  purposes not being trained at all. ReLU's is essentially flat.")
    print("\n  (The ReLU/sigmoid ratio column falls below 1 in the last two rows")
    print("  only because sigmoid's gradient is LARGEST next to the loss - it")
    print("  simply never reaches the front of the network.)")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  THE PERCEPTRON, AND BACKPROPAGATION BY HAND")
    print("=" * 72)
    print("\n  No framework. Every gradient below is derived on paper and typed")
    print("  in, then verified numerically in section 4.")

    banner("1. One neuron, and the thing it cannot do")
    figure_perceptron()

    banner("2-3. One hidden layer, and backpropagation")
    net = figure_xor_solved()

    banner("4. Gradient checking: proving the derivatives are right")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)
    gradient_check(net, X, y)

    banner("5. Activation functions and the vanishing gradient")
    figure_activations()

    banner("What to take away")
    print("  - A single neuron computes step(w.x + b): ONE straight boundary.")
    print("    XOR is not linearly separable, so no amount of training helps.")
    print("    That is a proof, not a tuning problem.")
    print("  - A HIDDEN LAYER composes straight lines into a curve. Each unit")
    print("    contributes one boundary; the output layer combines them.")
    print("  - Backpropagation is the chain rule applied layer by layer. For")
    print("    sigmoid output + binary cross-entropy the first term collapses")
    print("    to (prediction - target), which is why that pairing is standard.")
    print("  - GRADIENT CHECK any hand-written backward pass. A wrong gradient")
    print("    still trains, just badly, and you will blame the learning rate.")
    print("  - Autograd exists because numerical gradients need two forward")
    print("    passes PER PARAMETER, while backprop gets them all in one.")
    print("  - Sigmoid's gradient peaks at 0.25, so depth multiplies it away.")
    print("    ReLU's is exactly 1 when active, which is most of why deep")
    print("    networks became trainable. Its cost is the dead unit.")
    print("  - Weight INITIALISATION is not cosmetic: Xavier/Glorot scaling by")
    print("    1/sqrt(n_in) keeps activations in the useful range of their")
    print("    nonlinearity. All-zero weights make every unit identical and the")
    print("    network can never break symmetry.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
