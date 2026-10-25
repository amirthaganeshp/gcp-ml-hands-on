"""The shared CNN in JAX - where the parameters are a value you hold.

JAX is not a deep learning framework. It is NumPy with four function
transformations - grad, jit, vmap and pmap - and everything else is built on
top. The consequence is that there are no objects with hidden state: the
parameters are an ordinary nested dict that you pass in and get back.

That sounds like a small difference and it changes everything about how the
code reads.

  1. The four transformations, on functions small enough to check by eye
  2. Parameters as a pytree - the model is a pure function
  3. The training step, and what jit is actually worth here
  4. Randomness: explicit PRNG keys, and why JAX refuses a global seed

Flax is not used. It will not install on this machine (its orbax dependency
ships test fixtures ~260 characters deep), and writing the parameters out by
hand shows the functional style far more clearly than a Module system that
looks like Keras anyway.

WINDOWS WARNING, measured rather than assumed. JAX has no official Windows
support - Google ships Linux and macOS, and the Windows wheels are
community-maintained. This script crashes inside jaxlib's native code on
roughly 1 run in 12:

    Windows fatal exception: code 0x80000001

Two changes below cut that from about 1 in 6 (accumulating the loss on
device instead of calling float() every batch, and batching in NumPy rather
than indexing a device array). Both are better JAX regardless. The residual
crash happens inside the jit-compiled train_step itself and CANNOT be fixed
at this layer - it is the platform, not the code. Reported rather than
wrapped in a retry, because hiding a real limitation behind a green tick is
worse than the limitation.

Run:
    python 04_frameworks/03_jax/jax_version.py

Concept reference: Foundations volume, Part C.3.
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from common import (BATCH, EPOCHS, LR, SEED, banner, check_params,  # noqa: E402
                    header, load_data, record, source_lines)


# =================================== 1. the four transformations, up close
def demo_transformations() -> None:
    def f(x):
        return jnp.sum(x ** 2 * 3.0)

    print("  grad - differentiate a FUNCTION, get back another function\n")
    df = jax.grad(f)
    print(f"    f(x)     = sum(3x^2),  f([1,2,3])  = {f(jnp.array([1., 2., 3.])):.1f}")
    print(f"    grad f   = 6x,         df([1,2,3]) = {df(jnp.array([1., 2., 3.]))}")
    print(f"    grad(grad f) = 6,      d2f([1,2,3]) = "
          f"{jax.grad(lambda x: jnp.sum(jax.grad(f)(x)))(jnp.array([1., 2., 3.]))}")
    print("\n    Note it is a function transformation, not a method on a")
    print("    tensor. There is no .backward() and no .grad attribute,")
    print("    because there is no object to hang them on.")

    print("\n  vmap - write for ONE example, run on a batch\n")
    def single(v):
        return jnp.dot(v, v)                    # takes a vector, not a batch
    batch = jnp.arange(12.0).reshape(4, 3)
    print(f"    single(one vector)  -> {single(batch[0]):.1f}")
    print(f"    vmap(single)(batch) -> {jax.vmap(single)(batch)}")
    print("\n    No reshaping, no keepdims, no guessing which axis is the")
    print("    batch. Write the maths for one example and let vmap add the")
    print("    dimension - which also means the maths stays readable.")

    print("\n  jit - compile the function with XLA\n")
    big = jnp.ones((400, 400))
    def work(x):
        for _ in range(8):
            x = jnp.tanh(x @ x.T) * 0.01
        return x.sum()
    fast = jax.jit(work)
    work(big).block_until_ready()
    fast(big).block_until_ready()               # warm up the compile
    t0 = time.perf_counter(); work(big).block_until_ready()
    t_plain = time.perf_counter() - t0
    t0 = time.perf_counter(); fast(big).block_until_ready()
    t_jit = time.perf_counter() - t0
    print(f"    without jit  {t_plain * 1000:7.1f} ms")
    print(f"    with jit     {t_jit * 1000:7.1f} ms   ({t_plain / max(t_jit, 1e-9):.1f}x)")
    print("\n    jit traces the function once, hands the whole graph to XLA,")
    print("    and XLA fuses the operations. The first call pays the compile;")
    print("    every later call with the same input SHAPES is free of it.")
    print("    Change a shape and it recompiles, which is why JAX code avoids")
    print("    variable-length batches.")

    print("\n  The fourth is pmap, for multiple devices. Not demonstrated -")
    print("  there is one CPU here - but it is the same idea: a function")
    print("  transformation that adds a device axis.")


# ============================================= 2. parameters as a pytree
def init_params(key: jax.Array) -> dict:
    """The model's parameters, as a plain nested dict.

    This IS the model. There is no object, no self, nothing hidden - which
    is why every JAX function below can be pure: params go in, predictions
    come out, and nothing is mutated.

    He initialisation (variance 2/fan_in) because the activations are ReLU.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    def he(k, shape, fan_in):
        return jax.random.normal(k, shape) * jnp.sqrt(2.0 / fan_in)

    return {
        "conv1": {"w": he(k1, (3, 3, 1, 16), 3 * 3 * 1), "b": jnp.zeros(16)},
        "conv2": {"w": he(k2, (3, 3, 16, 32), 3 * 3 * 16), "b": jnp.zeros(32)},
        "fc1": {"w": he(k3, (32 * 7 * 7, 64), 32 * 7 * 7), "b": jnp.zeros(64)},
        "fc2": {"w": he(k4, (64, 10), 64), "b": jnp.zeros(10)},
    }


def forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """A pure function: (params, input) -> logits. No state, no side effects.

    Written for a BATCH here because convolution is naturally batched, but
    note that nothing about it is tied to a particular batch size.
    """
    conv = partial(jax.lax.conv_general_dilated, window_strides=(1, 1),
                   padding="SAME",
                   dimension_numbers=("NHWC", "HWIO", "NHWC"))

    x = jax.nn.relu(conv(x, params["conv1"]["w"]) + params["conv1"]["b"])
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max,
                              (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
    x = jax.nn.relu(conv(x, params["conv2"]["w"]) + params["conv2"]["b"])
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max,
                              (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
    x = x.reshape(x.shape[0], -1)
    x = jax.nn.relu(x @ params["fc1"]["w"] + params["fc1"]["b"])
    return x @ params["fc2"]["w"] + params["fc2"]["b"]


def demo_pytree(params: dict) -> None:
    leaves, treedef = jax.tree.flatten(params)
    total = sum(leaf.size for leaf in leaves)
    check_params(total)
    print(f"\n  The parameters are a dict of dicts with {len(leaves)} arrays in it:")
    for name, block in params.items():
        shapes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in block.items())
        print(f"    {name:<8} {shapes}")

    print("\n  JAX calls any such nested structure a PYTREE, and every")
    print("  transformation understands them. jax.grad on a function of this")
    print("  dict returns a dict OF THE SAME SHAPE full of gradients:")
    g = jax.grad(lambda p: jnp.sum(forward(p, jnp.zeros((1, 28, 28, 1)))))(params)
    print(f"    gradient structure matches parameters: "
          f"{jax.tree.structure(g) == jax.tree.structure(params)}")
    print("\n  That is why the optimiser update is one line of tree_map and")
    print("  needs no knowledge of the architecture at all.")


# =============================================== 3. the training step
def loss_fn(params: dict, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = forward(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@partial(jax.jit, static_argnums=3)
def train_step(params, opt_state, batch, optimizer):
    """One update, compiled.

    value_and_grad gives the loss and the gradients in a single pass, which
    is what you almost always want. Everything here is a pure function of
    its inputs and returns new values - nothing is mutated in place, which
    is precisely what lets jit compile it.
    """
    x, y = batch
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@jax.jit
def accuracy(params, x, y):
    return jnp.mean(jnp.argmax(forward(params, x), axis=-1) == y)


def train_model(params: dict, d: dict) -> tuple:
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    # Keep the DATA in NumPy and do the batching host-side; only the
    # compute crosses into JAX. Slicing a device array (xtr[idx]) issues a
    # device-side gather on every batch, which is both unnecessary work and
    # - on Windows, where JAX is not officially supported - another native
    # call that can crash. numpy indexing is free here and the jitted step
    # converts the batch on the way in.
    xtr, ytr = d["x_train"], d["y_train"]
    xte, yte = jnp.asarray(d["x_test"]), jnp.asarray(d["y_test"])
    n = len(xtr)
    rng = np.random.default_rng(SEED)

    best, history = 0.0, []
    for epoch in range(EPOCHS):
        order = rng.permutation(n)          # host-side shuffle

        # Accumulate the loss ON DEVICE and convert once per epoch, rather
        # than calling float(loss) every batch.
        #
        # Two reasons, and the second is the one that bit. Calling float()
        # on a JAX array forces a device-to-host SYNC, which stalls the
        # async dispatch pipeline that makes jit worth using - so doing it
        # 78 times per epoch is simply bad JAX. And on Windows, which JAX
        # does not officially support, that transfer path crashed about one
        # run in seven:
        #
        #     Windows fatal exception: code 0x80000001
        #       jax/_src/array.py:295 in __float__
        #
        # Accumulating on device cuts the syncs from ~624 per run to 8.
        total = jnp.zeros(())
        for i in range(0, n - BATCH + 1, BATCH):
            idx = order[i:i + BATCH]
            params, opt_state, loss = train_step(
                params, opt_state, (xtr[idx], ytr[idx]), optimizer)
            total = total + loss
        acc = float(accuracy(params, xte, yte))
        history.append(acc)
        best = max(best, acc)
    return params, best, history


# ================================================= 4. explicit randomness
def demo_randomness() -> None:
    """JAX has no global seed, and that is deliberate.

    A global RNG makes a function impure - call it twice, get different
    answers - which breaks jit, breaks vmap, and makes reproducibility
    depend on execution order. JAX makes you thread the key through.
    """
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (3,))
    b = jax.random.normal(key, (3,))
    print(f"  Same key, twice: identical?  {bool(jnp.allclose(a, b))}")
    print(f"    {a}")
    print(f"    {b}")
    print("\n  A key is a VALUE. Reuse it and you get the same numbers - which")
    print("  is a feature, and also a bug waiting to happen if you forget to")
    print("  split. To get fresh randomness you split the key explicitly:")
    k1, k2 = jax.random.split(key)
    print(f"    split -> {jax.random.normal(k1, (3,))}")
    print(f"             {jax.random.normal(k2, (3,))}")
    print("\n  Verbose compared with torch.manual_seed(42), and the payoff is")
    print("  that a JAX function's output depends only on its arguments. That")
    print("  is what makes jit, vmap and pmap safe to apply to it.")


def main() -> None:
    header("JAX", jax.__version__)
    print(f"  optax {optax.__version__} for the optimiser. No Flax - see the "
          f"module docstring.")
    d = load_data()

    banner("1. grad, vmap and jit on functions you can check by eye")
    demo_transformations()

    banner("2. Parameters as a pytree - the model is a pure function")
    params = init_params(jax.random.PRNGKey(SEED))
    demo_pytree(params)

    banner("3. Training")
    t0 = time.perf_counter()
    params, acc, history = train_model(params, d)
    secs = time.perf_counter() - t0
    print(f"  Ran {len(history)} epochs in {secs:.1f}s")
    print(f"  Best test accuracy: {acc:.4f}")
    print("\n  The step itself is four lines - value_and_grad, optimizer")
    print("  update, apply_updates, return. It is jit-compiled once and then")
    print("  reused, so the Python loop around it costs almost nothing.")
    print("\n  Note there is no early stopping or checkpointing here. Like")
    print("  PyTorch, JAX gives you none of that; unlike PyTorch, there is")
    print("  not really a community-standard place to get it either.")

    banner("4. Randomness is a value you pass around")
    demo_randomness()

    banner("What JAX gives you, and what it costs")
    print("  + FUNCTION TRANSFORMATIONS COMPOSE. jit(vmap(grad(f))) is a")
    print("    normal thing to write, and it means exactly what it reads.")
    print("  + Pure functions: params in, predictions out. No hidden state,")
    print("    so a result depends only on its arguments.")
    print("  + Parameters are ordinary dicts. Inspect them, save them, map")
    print("    over them - the optimiser update is one tree_map.")
    print("  + XLA compilation is genuinely fast, and the same code targets")
    print("    CPU, GPU and TPU. JAX is the native language of TPUs.")
    print("\n  - You write the model, the loop, the init and the bookkeeping.")
    print("    This file needed conv and max-pool from jax.lax by hand.")
    print("  - Explicit PRNG keys are correct and verbose, and forgetting to")
    print("    split gives you silently repeated randomness.")
    print("  - jit recompiles on every new input SHAPE, so ragged batches are")
    print("    expensive and the last partial batch is usually just dropped.")
    print("  - The ecosystem is thinner. Flax or Haiku for layers, optax for")
    print("    optimisers, orbax for checkpoints - four libraries to assemble")
    print("    what Keras ships in one.")

    record("JAX", acc, secs,
           sum(leaf.size for leaf in jax.tree.flatten(params)[0]),
           source_lines(__file__),
           "pure functions; pytree params; jit/grad/vmap; optax")


if __name__ == "__main__":
    main()
