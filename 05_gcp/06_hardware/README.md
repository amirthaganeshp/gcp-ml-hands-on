# Hardware and scaling

> **Concept reference:** Success Guide, Part 3.3 · Blueprint section 3 (~21%)
> **Runtime:** ~2 s · **Executes nothing against Google Cloud** — this is arithmetic

```bash
python 05_gcp/06_hardware/hardware_and_scaling.py
```

---

## The idea

Two questions: **which accelerator**, and **when one is not enough, which distribution strategy**.

Both are decided by arithmetic and by the **shape of the model** — not by which chip is fastest in
isolation. The fastest chip is irrelevant if your input pipeline cannot feed it.

## 1. The three

| | good for | bad for |
|---|---|---|
| **CPU** | small data, classical ML, prototyping, small-model inference | anything with big dense matmuls |
| **GPU** | most deep learning; flexible, every framework, custom ops | when a TPU would be cheaper at scale |
| **TPU** | very large dense matmuls — transformers, big CNNs, huge batches | custom ops, dynamic shapes, tiny models |

> **TPUs are not "faster GPUs".** They are systolic arrays built for large dense matrix
> multiplication. A model full of custom ops, control flow or dynamic shapes may not compile for
> one at all — and a small model will not fill one, so you pay for silicon you never use.

## 2. Cost per epoch

A model taking 40 CPU-hours per epoch:

| hardware | $/hour | speedup | hours | **$/epoch** |
|---|---|---|---|---|
| CPU (n1-8) | 0.38 | 1× | 40.00 | 15.20 |
| GPU (T4) | 0.35 | 12× | 3.33 | **1.17** |
| GPU (A100) | 2.93 | 75× | 0.53 | 1.56 |
| TPU v5e | 1.20 | 95× | **0.42** | **0.51** |

Note the **T4 beats the A100 on cost** while being six times slower — and that ordering flips the
moment your deadline matters more than your bill.

> **"Fastest" and "cheapest" are different questions**, and the scenario picks one. Here TPU
> happens to win both; frequently nothing wins both.

## 3. The utilisation trap

An accelerator that can do a step in 8 ms, fed at different rates:

| input pipeline | step ms | accel utilisation | wasted $ |
|---|---|---|---|
| fast (cached) | 8 | **100%** | 0.00/h |
| ok | 8 | 100% | 0.00/h |
| slow (decode/step) | 30 | **27%** | 2.15/h |
| very slow | 80 | **10%** | 2.64/h |

**Low accelerator utilisation is an input problem, not a compute one.** Buying a bigger
accelerator makes the utilisation *worse* and the bill larger.

The fix is the data pipeline: prefetch, cache, parallel interleave, TFRecord or Parquet instead of
per-file decode.

> Exam shape: *"GPU utilisation is 15% and training is slow"* → **fix the input pipeline.** An
> option offering a larger machine type is the trap.

## 4. Distribution strategies

| strategy | topology | use when |
|---|---|---|
| **MirroredStrategy** | several GPUs, **one** machine | the usual first step beyond one GPU |
| **MultiWorkerMirroredStrategy** | several GPUs, **several** machines | one machine is no longer enough |
| **TPUStrategy** | TPU cores / pods | you have committed to TPU |
| **ParameterServerStrategy** | many machines, **asynchronous** | very large sparse models, embeddings |

**The giveaway is the topology in the question.** "Multiple GPUs on a single VM" is Mirrored;
"across multiple VMs" is MultiWorker. If the scenario says TPU, the strategy is decided for you.

## 5. Data vs model parallelism

| | what is split | when |
|---|---|---|
| **Data parallelism** | the **batch** — every worker holds the whole model | the default; training is too slow |
| **Model parallelism** | the **model** — it does not fit on one device | the model itself is too big |

> *"The model does not fit in GPU memory"* → **model** parallelism (or gradient checkpointing, or a
> smaller batch). *"Training is too slow and the model fits fine"* → **data** parallelism.

**Scaling data parallelism also scales the effective batch size**, which changes the optimisation
problem — the learning rate usually has to rise with it. **More workers is not a free speedup.**

## On the numbers

Throughput figures are representative published rates and **the prices move**. The script prints
them so a stale figure is visible rather than hidden. The **ratios and the crossovers** are the
durable part.

## Things to try

- Set the T4 speedup to 20× and re-run. The TPU is **still** cheapest: the T4 falls to $0.70/epoch
  but only beats the TPU's $0.51 above **27.7×** (0.35 × 40 ÷ 0.505). Try 30× and the T4 becomes
  cheapest while the TPU stays fastest — which is the point: these orderings are not facts to
  memorise, they are arithmetic to redo.
- Make the input pipeline 4 ms. Utilisation pins at 100% and the wasted column goes to zero, which
  is what a well-built `tf.data` pipeline buys you.
