# Frameworks: the Same Model, Three Ways

One convolutional network, one dataset, one set of hyperparameters — written three times. The
comparison is only worth anything if everything else is held constant, so the data, architecture
and hyperparameters live in [`common.py`](common.py) and **each script asserts its parameter count
against a shared spec** and exits if it drifts.

```bash
python 04_frameworks/01_tensorflow/tf_version.py
python 04_frameworks/02_pytorch/torch_version.py
python 04_frameworks/03_jax/jax_version.py
python 04_frameworks/00_same_model_three_ways/compare.py
```

**Dataset:** Fashion-MNIST, 10,000 train / 3,000 test · **8 epochs, batch 128, Adam lr=1e-3**

---

## The result

| framework | test acc | train s | params | lines |
|---|---|---|---|---|
| TensorFlow 2.21 | 0.8767 | **5.4** | 105,866 | 197 |
| PyTorch 2.14 | 0.8550 | 8.7 | 105,866 | 198 |
| JAX 0.11 | **0.8800** | 11.6 | 105,866 | 192 |

All three report **exactly 105,866 parameters** — the architectures really are identical.

## So why do the accuracies differ by 2.5 points?

Same architecture, same optimiser, same learning rate, same data, same seed. Most of the difference
comes from **a default nobody chose**:

| framework | initial weight std | default scheme | test acc |
|---|---|---|---|
| JAX (written by hand) | **0.1177** | `he_normal` | **0.8800** |
| Keras Conv2D | 0.0676 | `glorot_uniform` | 0.8767 |
| PyTorch Conv2d | **0.0478** | `kaiming_uniform(a=√5)` | **0.8550** |

*(He target for this layer: 0.1179. Glorot target: 0.0680.)*

**The accuracy ordering matches the initialisation-scale ordering exactly**, and a controlled run
confirms the effect is real: the same PyTorch model over five seeds averages **He 0.8783 / Glorot
0.8677 / default 0.8603**. It is not the *only* effect, though. Seed and data-order noise is about a
point by itself (PyTorch's default spans 0.8463–0.8693 over those five seeds), so the 0.3-point
JAX/TensorFlow gap is inside the noise. Details in
[`00_same_model_three_ways`](00_same_model_three_ways/).

PyTorch's default is the notable one: `kaiming_uniform_(a=math.sqrt(5))` produces a gain suited to
a leaky ReLU of slope √5, which is nobody's network. It is a historical accident rather than a
recommendation, and for ReLU you have to ask for He initialisation explicitly.

> **The lesson for porting a model between frameworks:** matching the *layers* is the easy part.
> Matching the **defaults** — initialisers, optimiser epsilon, whether the loss wants logits or
> probabilities, weight decay — is where a port silently loses two points and nobody can say why.

## What each framework asks of you

| | TensorFlow | PyTorch | JAX |
|---|---|---|---|
| Training loop | `fit()` | you write it | you write it |
| Early stopping | callback | you write it | you write it |
| Checkpointing | callback | you write it | orbax |
| LR scheduling | callback | `lr_scheduler` | optax chain |
| Input pipeline | `tf.data` | `DataLoader` | you write it |
| Shape inference | automatic | you declare | you declare |
| Layers provided | `keras.layers` | `torch.nn` | `jax.lax` (raw) |
| Model state | in the object | in the object | **a value you hold** |
| Randomness | global seed | global seed | **explicit keys** |
| Serving artefact | SavedModel | TorchScript / ONNX | you decide |
| Channel order | NHWC | **NCHW** | either |
| Loss expects | probabilities | **logits** | logits |

Read the last row twice. Keras' `sparse_categorical_crossentropy` defaults to probabilities, so the
Keras model ends in softmax. PyTorch's `CrossEntropyLoss` applies `log_softmax` itself, so the
PyTorch model must **not**. Add a softmax to the PyTorch model and it still trains — just worse.
**That is the single most common bug when porting between them.**

### On lines of code

197 / 198 / 192 looks like a wash, and it is misleading. These files are mostly narration and
demonstrations. The *modelling* code differs far more: Keras trains in one `fit()` call, PyTorch
needs ~35 lines of loop, and JAX needs the loop plus the convolution and pooling written against
`jax.lax`.

---

## Contents

| folder | what it covers |
|---|---|
| [`01_tensorflow/`](01_tensorflow/) | `fit` + callbacks, **tf.data**, **TFRecord**, **SavedModel** export and serving signature |
| [`02_pytorch/`](02_pytorch/) | `Dataset`/`DataLoader`, **the explicit loop**, `train()` vs `eval()`, `state_dict` |
| [`03_jax/`](03_jax/) | **grad / jit / vmap**, parameters as a pytree, explicit PRNG keys, optax — **and a measured Windows stability caveat** |
| [`00_same_model_three_ways/`](00_same_model_three_ways/) | the table above, and the initialisation investigation |

---

## Which one?

**For the GCP ML Engineer exam: TensorFlow/Keras.** Vertex AI's prebuilt containers, SavedModel
serving, TFRecord, TFX and BigQuery ML all assume it, and the exam is written around that stack.

In practice, independent of the exam:

- **Keras** when the model is standard and you want it working today. The callbacks alone justify it.
- **PyTorch** when you need to change the training loop, or when reading research code — almost all
  of it is PyTorch now.
- **JAX** when you need composable transformations or TPUs. The smallest idea of the three, and the
  most demanding to build on.

**Keras 3 runs on all three backends.** Everything in [`03_neural_networks`](../03_neural_networks/)
was written against the Keras API and runs on PyTorch, selected by one environment variable. The
API and the backend are now separable choices.

> And the honest summary: all three trained the same network to within 2.5 points, in under 13
> seconds, in about the same number of lines. **The framework is rarely the thing that decides
> whether a project works.**

---

## Setup notes

`requirements-dl.txt` installs all three. Two Windows-specific traps, both documented in
[`SETUP.md`](../SETUP.md):

- **TensorFlow** will not install if the repo sits on a long path — its bundled headers blow the
  260-character limit. Mapping the repo to a short virtual drive (`subst M: <repo>`) and installing
  through that works, and needs no admin rights or registry change.
- **Flax** cannot be installed here at all: its `orbax-checkpoint` dependency ships test fixtures
  ~260 characters deep *even from a drive root*. The JAX example therefore uses **pure JAX + optax**
  and writes its own layers — which shows the functional style more clearly than Flax's Module
  system would anyway.
- **JAX itself is unstable on Windows.** It has no official Windows support, and `jax_version.py`
  crashes inside jaxlib's native code on roughly **1 run in 12**
  (`Windows fatal exception: code 0x80000001`). Two fixes cut that from ~1 in 6; the residual crash
  is inside the jit-compiled step and cannot be fixed at this layer. See
  [`03_jax/README.md`](03_jax/README.md).
