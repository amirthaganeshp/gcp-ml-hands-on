# Making a model cheaper to serve

> **Concept reference:** Success Guide, Part 4.6 · Blueprint section 4 (~20%)
> **Datasets:** Adult income (6 numeric features, 22,621 train rows)
> **Runtime:** ~25 s on a laptop CPU

```bash
python 07_serving/03_model_optimization/model_optimization.py
```

---

## The idea

Once a model works, serving it is an engineering problem with four levers. Three trade accuracy
for size or speed. The fourth — **batching** — trades **latency for throughput**, and the exam asks
about that one specifically, because the two sound alike and are opposites.

Baseline: a deliberately over-wide MLP, 67,841 parameters, **0.274 MB**, **0.8204** accuracy,
**0.030 ms/request** on one CPU thread.

## 1. Quantization — and an honest negative

| | size MB | accuracy | ms/req |
|---|---|---|---|
| FP32 | 0.274 | 0.8204 | 0.030 |
| **INT8** | **0.074** | 0.8189 | **0.224** |
| change | **0.27×** | −0.0016 | **7.6× slower** |

**3.7× smaller, accuracy essentially unchanged — and it got *slower*.**

> Quantization is usually sold as a speed win. On a model this small (67k parameters, one CPU
> thread) the per-op quantize/dequantize overhead costs more than the cheaper integer arithmetic
> saves. The **size** win is real and immediate; the **speed** win needs a big model, or hardware
> with genuine INT8 kernels.
>
> The slowdown varies run to run (5–8×), which is itself the lesson: **measure, don't assume.** The
> script derives that sentence from what it just measured rather than asserting the textbook claim.

Dynamic quantization stores weights as 8-bit integers with a scale factor and quantizes activations
on the fly — **no calibration data needed**, which is why it's the cheap first thing to try.

## 2. Pruning — most weights are doing very little

Smallest-magnitude weights zeroed, **no retraining**, so this is the pessimistic case:

| sparsity | accuracy | vs baseline |
|---|---|---|
| 0% | 0.8204 | — |
| **50%** | **0.8163** | −0.0041 |
| 80% | 0.7946 | −0.0259 |
| 90% | 0.7825 | −0.0379 |
| 95% | 0.7881 | −0.0324 |

**Half the weights can go for 0.4% accuracy**, without retraining at all. Real pruning fine-tunes
afterwards and does better.

That is the same fact the over-wide layers were hiding, and it is *why* pruning and distillation
work at all.

## 3. Batching — the trade, measured properly

Compute is **measured**; the queue wait is **modelled** as `(B−1)/(2λ)` at 500 req/s and labelled
as such — but it's the part a user actually feels.

| batch | compute ms | wait ms | **user ms** | req/sec |
|---|---|---|---|---|
| 1 | 0.0296 | 0.00 | **0.03** | 33,784 |
| 8 | 0.0061 | 7.00 | **7.01** | 162,767 |
| 32 | 0.0023 | 31.00 | **31.00** | 440,468 |
| 128 | 0.0028 | 127.00 | **127.00** | 351,070 |

From batch 1 → 128:

| | |
|---|---|
| throughput | **10.4× better** |
| compute per request | 0.10× — *better*, amortised |
| **latency the user sees** | **4291× worse** |

> **The middle row is why this is confusing.** Per-request *compute* falls with batching. The user
> still waits far longer, because most of their latency is now queueing for the batch to fill.
>
> An earlier version of this script measured only the compute column and claimed batching "costs
> latency" — the output said the opposite. Measuring the wait fixed the contradiction.

**THE RULE:** more QPS from the same hardware → batch. Lower latency on a single request → **do
not** batch.

## The exam angle

| technique | effect | cost |
|---|---|---|
| **Quantization** | much smaller; faster on *big* models | small accuracy loss |
| **Pruning** | removes low-magnitude weights | small accuracy loss |
| **Distillation** | small student mimics big teacher | a whole training cycle |
| **Batching** | higher **throughput** | higher per-request **latency** |
| Optimized runtime / TensorRT | faster, no retraining | framework constraints |
| Caching frequent predictions | removes the work entirely | staleness risk |

**On-device is its own chain, and the exam expects it in order:**
`quantize → TF Lite → Edge TPU / Coral`. A bigger cloud accelerator is irrelevant to a device
constraint.

## Things to try

- Widen the MLP to 1024 units and re-run. The quantization slowdown should shrink or reverse — the
  overhead is fixed, the savings scale.
- Change `arrival_rate` in section 4. At 5,000 req/s the batch fills 10× faster and the latency
  cost of batching largely disappears — which is why batching suits *busy* endpoints.
