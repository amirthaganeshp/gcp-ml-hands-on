# JAX

> **Runtime:** ~45 s · **Result:** 0.8800 test accuracy in 11.6 s of training

```bash
python 04_frameworks/03_jax/jax_version.py
```

JAX is **not a deep learning framework**. It is NumPy with four function transformations — `grad`,
`jit`, `vmap`, `pmap` — and everything else is built on top. The consequence is that there are no
objects with hidden state: the parameters are an ordinary nested dict you pass in and get back.

That sounds like a small difference and it changes everything about how the code reads.

> **No Flax.** It will not install on this machine — its `orbax-checkpoint` dependency ships test
> fixtures ~260 characters deep, past the Windows path limit even from a drive root. Writing the
> parameters out by hand turns out to show the functional style far more clearly than a Module
> system that looks like Keras anyway.

## ⚠ This script crashes on Windows about 1 run in 12

Measured, not assumed — 1 failure in 12 consecutive runs:

```
Windows fatal exception: code 0x80000001
```

**JAX has no official Windows support.** Google ships Linux and macOS; the Windows wheels are
community-maintained. The crash is a hard fault inside jaxlib's native layer, with no Python
traceback and no stderr unless you run with `-X faulthandler`.

Two changes cut the rate from roughly **1 in 6** to **1 in 12**, and both are better JAX
regardless of the platform:

| change | why it helps | why it is right anyway |
|---|---|---|
| accumulate the loss **on device**, convert once per epoch | ~624 device→host syncs per run → 8 | a `float()` per batch stalls the async dispatch that makes `jit` worth using |
| batch in **NumPy**, not by indexing a device array | removes a device-side gather per batch | the data pipeline belongs on the host; only compute needs to cross |

**The residual crash happens inside the jit-compiled `train_step` itself** and cannot be fixed at
this layer. It is reported here rather than wrapped in a retry, because hiding a real platform
limitation behind a green tick is worse than the limitation.

### Things that were tried and did NOT work

Recorded so nobody repeats them. Pooled rate after the two fixes above:
**4 failures in 70 runs (~6%)**, from ~17% before.

| hypothesis | test | result |
|---|---|---|
| The heavy `jit` benchmark in section 1 is to blame, not training | full script vs a training-only probe, 16 runs each | **disproven** — 1/16 *both*. The crash is spread evenly across XLA work. |
| A guard-page fault means thread-stack exhaustion, so force XLA single-threaded | `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, `OMP_NUM_THREADS=1`, 16 runs | **no effect** — 1/16, and identical runtime, so the flag likely changed nothing |

A statistical note, because it matters for reading these: at a ~6% rate, 16 clean runs would occur
by chance about 37% of the time even with no improvement at all. None of these tests had the power
to *confirm* a fix — but two of them produced a failure outright, which is enough to reject.

### Tested on Linux: 50 runs, 0 failures — and that proves less than it looks

JAX is officially supported on Linux and macOS, so the obvious move is to run it there. Under
**WSL2** (Ubuntu 24.04, a separate Linux venv, same jax 0.11.2 / optax 0.2.8):

| | Windows | WSL2 Linux |
|---|---|---|
| failures | **4 in 70** (5.7%) | **0 in 50** |
| test accuracy | 0.8800 | **0.8800**, all 50 runs |
| wall clock | ~45 s | **19 s** |

The accuracy is **identical to four decimal places across platforms and across all 50 runs**, which
is a stronger reproducibility result than the crash question — same seed, same arithmetic, two
operating systems.

**But 0 in 50 does not establish that the defect is gone**, and it is worth being exact about why:

- By the **rule of three**, zero failures in 50 runs puts the 95% upper bound on the true rate at
  `3/50` = **6.0%** — which is the Windows rate. This data cannot tell the two apart.
- **Fisher's exact test** on 0/50 against 4/70 gives **p = 0.14** two-sided (0.11 one-sided). Not
  significant by any conventional threshold.
- The tempting number is `P(0 failures in 50 | Windows rate) = 0.053`, which reads like
  significance and is not the right test — it conditions on the Windows rate being exactly right
  rather than comparing two observed samples.

So the honest statement is: **the 50 runs are consistent with the defect being absent on Linux, and
equally consistent with no change.** The real reason to trust Linux is not this experiment — it is
that JAX is officially supported there and the crash is a known Windows-jaxlib defect. These runs
are weak confirmation of a conclusion reached on other grounds.

Ruling out a rate below 1% would need ~300 clean runs. That was not done.

> This is the same trap as the "things that did not work" table above, one level up. A clean result
> feels like proof and usually is not — **the question is always how many failures you would have
> expected to see anyway.**

Note what WSL2 does and does not do: installing it **does not repair the Windows venv**. It gives
you a separate Linux venv to run the repository from. For a CPU-only example that takes 12 seconds,
a ~6% crash rate is an annoyance rather than a blocker, which is why the Windows path stays and the
rate is documented rather than hidden.

**Running it on Linux**, for reference — the one non-obvious part is the Keras backend:

```bash
python3 -m venv ~/venvs/ml
~/venvs/ml/bin/pip install jax optax keras truststore numpy
KERAS_BACKEND=jax ~/venvs/ml/bin/python 04_frameworks/03_jax/jax_version.py
```

`common.py` loads Fashion-MNIST through `keras.datasets` and sets the backend with
`os.environ.setdefault(..., "torch")`, so an outer `KERAS_BACKEND=jax` wins. Without it Keras
defaults to the TensorFlow backend and `import keras` fails with `ModuleNotFoundError: No module
named 'tensorflow'`. Setting it to `jax` means **neither TensorFlow nor torch is needed** — the
Linux venv is jax, optax and keras alone.

---

## 1. The transformations

### `grad` — differentiate a *function*, get back another function

```
f(x)         = sum(3x²),  f([1,2,3])   = 42.0
grad f       = 6x,        df([1,2,3])  = [ 6. 12. 18.]
grad(grad f) = 6,         d2f([1,2,3]) = [ 6.  6.  6.]
```

It is a **function transformation**, not a method on a tensor. There is no `.backward()` and no
`.grad` attribute, because there is no object to hang them on. And it composes with itself —
second derivatives are free.

### `vmap` — write for one example, run on a batch

```python
def single(v):
    return jnp.dot(v, v)          # takes a vector, not a batch

single(batch[0])        -> 5.0
jax.vmap(single)(batch) -> [  5.  50. 149. 302.]
```

No reshaping, no `keepdims`, no guessing which axis is the batch. **Write the maths for one example
and let `vmap` add the dimension** — which also means the maths stays readable.

### `jit` — compile with XLA

| | time |
|---|---|
| without `jit` | 4.8 ms |
| with `jit` | 3.8 ms (**1.3×**) |

A modest speedup here, and worth reporting as such: this is a CPU with a small workload. `jit`
traces the function once, hands the whole graph to XLA, and XLA fuses the operations. The first
call pays the compile; later calls with the same input **shapes** are free of it.

**Change a shape and it recompiles**, which is why JAX code avoids variable-length batches and
usually drops the last partial one.

The fourth transformation is `pmap`, for multiple devices — not demonstrated, since there is one
CPU here.

## 2. Parameters as a pytree

```
conv1    w(3, 3, 1, 16),   b(16,)
conv2    w(3, 3, 16, 32),  b(32,)
fc1      w(1568, 64),      b(64,)
fc2      w(64, 10),        b(10,)
```

**This dict *is* the model.** No object, no `self`, nothing hidden — which is why `forward(params,
x)` can be a pure function.

JAX calls any such nested structure a **pytree**, and every transformation understands them.
`jax.grad` on a function of this dict returns a dict *of the same shape* full of gradients
(verified in the script). That is why the optimiser update is one `tree_map` and needs no knowledge
of the architecture at all.

## 3. The training step

```python
@partial(jax.jit, static_argnums=3)
def train_step(params, opt_state, batch, optimizer):
    x, y = batch
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

Four lines, compiled once and reused. Everything is a **pure function of its inputs returning new
values** — nothing is mutated in place, which is precisely what lets `jit` compile it.

There is no early stopping or checkpointing. Like PyTorch, JAX gives you none of that; *unlike*
PyTorch, there is not really a community-standard place to get it either.

## 4. Randomness is a value

```
Same key, twice: identical?  True
```

JAX has **no global seed**, deliberately. A global RNG makes a function impure — call it twice, get
different answers — which breaks `jit`, breaks `vmap`, and makes reproducibility depend on
execution order.

A key is a **value**. Reuse it and you get the same numbers, which is a feature and also a bug
waiting to happen. To get fresh randomness you split explicitly:

```python
k1, k2 = jax.random.split(key)
```

Verbose next to `torch.manual_seed(42)`, and the payoff is that a JAX function's output depends
only on its arguments. That is what makes `jit`, `vmap` and `pmap` safe to apply.

---

## What it gives you, and what it costs

**+** **Transformations compose.** `jit(vmap(grad(f)))` is a normal thing to write and means
exactly what it reads.
**+** Pure functions: params in, predictions out, no hidden state.
**+** Parameters are ordinary dicts — inspect, save, map over them.
**+** XLA is genuinely fast, and the same code targets CPU, GPU and TPU. **JAX is the native
language of TPUs.**

**−** You write the model, the loop, the init and the bookkeeping. This example needed convolution
and max-pool from `jax.lax` by hand.
**−** Explicit PRNG keys are correct and verbose; forgetting to split gives silently repeated
randomness.
**−** `jit` recompiles on every new input **shape**.
**−** The ecosystem is thinner — Flax or Haiku for layers, optax for optimisers, orbax for
checkpoints: four libraries to assemble what Keras ships in one.

---

## The exam angle

- **JAX is the TPU-native framework.** If a question is about TPU training at scale, JAX is a
  plausible answer.
- **`jit` / `grad` / `vmap` / `pmap`** — know what each does.
- **Pure functions and explicit PRNG** are the defining constraint, and the reason the
  transformations are safe.
- JAX has **no managed Vertex AI serving story** comparable to SavedModel — that is a real
  deployment consideration, not a detail.
- Most GCP exam content assumes TensorFlow; JAX appears mainly around TPUs and research workloads.
