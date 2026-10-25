"""Making a model cheaper to serve: quantization, pruning, and batching.

Once a model works, serving it is an engineering problem with four levers.
Three of them trade accuracy for speed or size. The fourth - batching - trades
LATENCY for THROUGHPUT, and the exam asks about that trade specifically,
because the two sound like the same thing and are opposites.

  1. A baseline model, trained and measured
  2. Quantization FP32 -> INT8: size, accuracy, latency
  3. Pruning: how much can you delete before it hurts?
  4. Batching: throughput up, per-request latency up too
  5. The table, and which lever a scenario is asking for

EXECUTED IN FULL. Every number below was measured on this machine, on CPU.

Run:
    python 07_serving/03_model_optimization/model_optimization.py

Concept reference: Success Guide, Part 4.6. Blueprint section 4 (~20%).
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from loaders import load_adult  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)          # a serving replica is not your whole laptop


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


class MLP(nn.Module):
    """Deliberately over-wide: optimisation is only interesting with slack."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


def model_size_mb(model: nn.Module) -> float:
    path = OUT / "_tmp_size.pt"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    mb = path.stat().st_size / 1e6
    path.unlink()
    return mb


def accuracy(model: nn.Module, X: torch.Tensor, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X).squeeze(1)
    return float(((logits > 0).numpy().astype(int) == y).mean())


def latency_ms(model: nn.Module, X: torch.Tensor, batch: int = 1,
               reps: int = 200) -> float:
    """Median per-REQUEST latency at a given batch size."""
    model.eval()
    times = []
    with torch.no_grad():
        for i in range(reps):
            xb = X[i * batch:(i + 1) * batch]
            if len(xb) < batch:
                break
            t0 = time.perf_counter()
            model(xb)
            times.append((time.perf_counter() - t0) * 1000 / batch)
    return float(np.median(times))


# ------------------------------------------------------------- 1. baseline
def train_baseline():
    rule("1. A BASELINE MODEL")
    df = load_adult().dropna()
    y = (df["income"].astype(str).str.contains(">50K")).astype(int).to_numpy()
    X = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)
    Xtr_t, Xte_t = torch.from_numpy(Xtr), torch.from_numpy(Xte)
    ytr_t = torch.from_numpy(ytr.astype(np.float32))

    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(12):
        for i in range(0, len(Xtr_t), 512):
            opt.zero_grad()
            out = model(Xtr_t[i:i + 512]).squeeze(1)
            loss = lossf(out, ytr_t[i:i + 512])
            loss.backward()
            opt.step()

    acc = accuracy(model, Xte_t, yte)
    size = model_size_mb(model)
    lat = latency_ms(model, Xte_t, batch=1)
    params = sum(p.numel() for p in model.parameters())

    print(f"  Adult income, {X.shape[1]} numeric features, {len(Xtr):,} train rows")
    print(f"  Parameters   {params:,}")
    print(f"  Size         {size:.3f} MB")
    print(f"  Accuracy     {acc:.4f}")
    print(f"  Latency      {lat:.3f} ms/request (batch=1, 1 thread)")
    return model, Xte_t, yte, dict(size=size, acc=acc, lat=lat)


# --------------------------------------------------------- 2. quantization
def section_quantize(model, Xte_t, yte, base):
    rule("2. QUANTIZATION - FP32 to INT8")

    qmodel = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8)

    size = model_size_mb(qmodel)
    acc = accuracy(qmodel, Xte_t, yte)
    lat = latency_ms(qmodel, Xte_t, batch=1)

    print(f"  {'':<12} {'size MB':>9} {'accuracy':>10} {'ms/req':>9}")
    print("  " + "-" * 44)
    print(f"  {'FP32':<12} {base['size']:>9.3f} {base['acc']:>10.4f} {base['lat']:>9.3f}")
    print(f"  {'INT8':<12} {size:>9.3f} {acc:>10.4f} {lat:>9.3f}")
    print(f"  {'change':<12} {size / base['size']:>8.2f}x "
          f"{acc - base['acc']:>+10.4f} {lat / base['lat']:>8.2f}x")

    print(f"\n  {base['size'] / size:.1f}x smaller, and accuracy moved by "
          f"{abs(acc - base['acc']):.4f}.")
    print("\n  Weights are stored as 8-bit integers with a scale factor instead of")
    print("  32-bit floats. DYNAMIC quantization does this to the weights only and")
    print("  quantizes activations on the fly - no calibration data needed, which")
    print("  is why it is the cheap first thing to try.")

    # Report what was measured, not what the textbook says.
    if lat > base["lat"]:
        print(f"\n  HONEST NEGATIVE: it got {lat / base['lat']:.1f}x SLOWER here, not faster.")
        print("  Quantization is usually sold as a speed win. On a model this small")
        print("  (67k parameters, 1 CPU thread) the per-op quantize/dequantize")
        print("  overhead costs more than the cheaper integer maths saves.")
        print("  The SIZE win is real and immediate; the SPEED win needs a big model,")
        print("  or hardware with genuine INT8 kernels. Measure, do not assume.")
    else:
        print(f"\n  And {base['lat'] / lat:.1f}x faster, which is the usual reason to do it.")
    return dict(size=size, acc=acc, lat=lat)


# -------------------------------------------------------------- 3. pruning
def section_prune(model, Xte_t, yte, base):
    rule("3. PRUNING - how much can you delete?")

    print("  Zero the smallest-magnitude weights and re-measure. No retraining,")
    print("  so this is the pessimistic case - real pruning fine-tunes after.\n")
    print(f"  {'sparsity':>9} {'accuracy':>10} {'vs baseline':>13}")
    print("  " + "-" * 36)

    results = []
    for sparsity in (0.0, 0.5, 0.8, 0.9, 0.95):
        pruned = copy.deepcopy(model)
        with torch.no_grad():
            for mod in pruned.modules():
                if isinstance(mod, nn.Linear):
                    w = mod.weight.abs().flatten()
                    if sparsity > 0:
                        k = int(len(w) * sparsity)
                        thresh = torch.kthvalue(w, k).values
                        mod.weight[mod.weight.abs() <= thresh] = 0.0
        acc = accuracy(pruned, Xte_t, yte)
        results.append((sparsity, acc))
        print(f"  {sparsity:>8.0%} {acc:>10.4f} {acc - base['acc']:>+13.4f}")

    print("\n  Most of the weights are doing very little - which is the same fact")
    print("  the over-wide layer sizes were hiding. That is what makes pruning and")
    print("  distillation work at all.")
    return results


# ------------------------------------------------------------- 4. batching
def section_batching(model, Xte_t):
    rule("4. BATCHING - the trade the exam asks about")

    # Amortised compute is only half the story. A request also WAITS for the
    # batch to fill. At Poisson arrivals of lambda/sec, the mean wait for a
    # batch of B is (B-1)/(2*lambda). Compute is measured; the wait is modelled
    # and labelled as such - but it is the part the user actually feels.
    arrival_rate = 500.0     # requests/sec arriving at this replica

    print(f"  Arrivals: {arrival_rate:,.0f} req/s. Compute is MEASURED; the")
    print("  queue wait is modelled as (B-1)/(2*lambda) and marked [model].\n")
    print(f"  {'batch':>6} {'compute ms':>12} {'wait ms':>10} {'USER ms':>10} {'req/sec':>12}")
    print("  " + "-" * 56)

    results = []
    for batch in (1, 8, 32, 128):
        compute = latency_ms(model, Xte_t, batch=batch, reps=60)
        wait = (batch - 1) / (2 * arrival_rate) * 1000.0      # [model]
        user = compute + wait
        qps = 1000.0 / compute
        results.append((batch, compute, user, qps))
        print(f"  {batch:>6} {compute:>12.4f} {wait:>10.2f} {user:>10.2f} {qps:>12,.0f}")

    b1, b_big = results[0], results[-1]
    print(f"\n  From batch 1 to batch {b_big[0]}:")
    print(f"    throughput          {b_big[3] / b1[3]:>6.1f}x   (better)")
    print(f"    compute per request {b_big[1] / b1[1]:>6.2f}x   (better - amortised)")
    print(f"    LATENCY THE USER SEES {b_big[2] / b1[2]:>5.0f}x   (worse)")

    print("\n  The middle row is why this is confusing: per-request COMPUTE falls")
    print("  with batching. The user still waits longer, because most of their")
    print("  latency is now queueing for the batch to fill, not computing.")
    print("\n  THE RULE: batching buys throughput and costs latency.")
    print("    'we need more QPS from the same hardware' -> batch")
    print("    'we need lower latency on a single request' -> do NOT batch")
    return results


# ---------------------------------------------------------------- figure
def figure(prune_results, batch_results, base):
    use_house_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    sp = [s * 100 for s, _ in prune_results]
    ac = [a for _, a in prune_results]
    ax1.plot(sp, ac, "o-", color=PALETTE["blue"], lw=2)
    ax1.axhline(base["acc"], color=PALETTE["vermillion"], ls="--", lw=1.2,
                label="unpruned")
    ax1.set_xlabel("weights zeroed (%)")
    ax1.set_ylabel("accuracy")
    ax1.set_title("Pruning, without retraining")
    ax1.legend(frameon=False, fontsize=9)

    bs = [b for b, _, _, _ in batch_results]
    qps = [q for _, _, _, q in batch_results]
    lat = [u for _, _, u, _ in batch_results]
    ax2.plot(bs, qps, "o-", color=PALETTE["green"], lw=2, label="requests/sec")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("batch size")
    ax2.set_ylabel("requests/sec", color=PALETTE["green"])
    ax2b = ax2.twinx()
    ax2b.plot(bs, lat, "s--", color=PALETTE["vermillion"], lw=2, label="user latency")
    ax2b.set_ylabel("latency the user sees (ms)", color=PALETTE["vermillion"])
    ax2.set_title("Batching: throughput up, latency up")

    save_figure(fig, OUT / "model_optimization.png")
    print(f"\n  figure -> {OUT / 'model_optimization.png'}")


def main() -> None:
    print("MAKING A MODEL CHEAPER TO SERVE")
    model, Xte_t, yte, base = train_baseline()
    q = section_quantize(model, Xte_t, yte, base)
    prune_results = section_prune(model, Xte_t, yte, base)
    batch_results = section_batching(model, Xte_t)
    figure(prune_results, batch_results, base)

    rule("WHICH LEVER IS THE SCENARIO ASKING FOR?")
    print("""  technique      effect                              cost
  ------------------------------------------------------------------------
  Quantization   much smaller; faster on BIG models   small accuracy loss
                 (measured 3.7x smaller and SLOWER on this tiny one)
  Pruning        removes low-magnitude weights       small accuracy loss
  Distillation   small student mimics big teacher    a whole training cycle
  Batching       higher THROUGHPUT                   higher per-request latency
  Optimized
    runtime      faster, no retraining               framework constraints
  Caching        removes the work entirely           staleness risk

  On-device is its own chain, and the exam expects it in order:
      quantize -> TF Lite -> Edge TPU / Coral
  A bigger cloud accelerator is irrelevant to a device constraint.""")


if __name__ == "__main__":
    main()
