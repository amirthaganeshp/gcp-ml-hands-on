# PyTorch

> **Runtime:** ~20 s · **Result:** 0.8550 test accuracy in 6.4 s of training

```bash
python 04_frameworks/02_pytorch/torch_version.py
```

PyTorch's defining choice is that **there is no `fit`**. You write the loop, and in exchange you
can see and change every step of it.

---

## 1. The model as a class

Two things Keras inferred that are explicit here:

- **Every layer's input size.** `Conv2d(1, 16, ...)` then `Conv2d(16, 32, ...)` then
  `Linear(32*7*7, 64)`. PyTorch builds the module eagerly, before seeing any data, so it cannot
  work those out for you. Get the arithmetic wrong and it fails at the first forward pass.
- **The output is logits, not softmax.** `CrossEntropyLoss` applies `log_softmax` internally, so
  adding a softmax layer as well is a real and common bug — it trains, badly, and silently.

In exchange the model is an ordinary Python object: printable, subclassable, and you can set a
breakpoint inside `forward()`.

## 2. Dataset and DataLoader

```
One batch: x (128, 1, 28, 28), y (128,)
```

**Channels first** — `(N, C, H, W)` — against TensorFlow's channels-last `(N, H, W, C)`. The
transpose in `build_loaders` does that conversion, and forgetting it is the classic porting error.

`Dataset` says how to get *one* example; `DataLoader` handles batching, shuffling and parallel
workers. A clean split: implement `__len__` and `__getitem__` and everything else is reused.

## 3. The loop

```python
optimizer.zero_grad()      # gradients ACCUMULATE - forget this and you sum every batch so far
loss = criterion(model(xb), yb)
loss.backward()            # autograd fills every .grad
optimizer.step()           # apply the update
```

Around those four lines sit the epoch loop, the batch loop, the evaluation pass, early stopping and
restore-best-weights — **about 35 lines against `fit(...)` plus three callbacks.**

The upside is that all of it is yours. A custom loss on intermediate activations, two optimisers on
different parameter groups, gradient surgery between `backward` and `step` — all of it is just
editing the loop, with no framework to fight.

## 4. `train()` vs `eval()`

Same input, three forward passes through a net with dropout:

| mode | identical every time? |
|---|---|
| `.train()` | **False** — dropout active and random |
| `.eval()` | **True** — dropout off, deterministic |

**Nothing switches these for you.** Evaluating without `.eval()` is one of the most common PyTorch
bugs, and it does not raise — it quietly reports worse numbers than the model deserves. Batch norm
does the same thing differently: batch statistics in train, running averages in eval.

`torch.no_grad()` is the *separate* concern — it stops building the autograd graph, saving memory
and time. You want both.

## 5. `state_dict`

PyTorch saves **weights, not models**. A `state_dict` is an ordinary dict of tensors (8 entries
here), and loading it requires constructing the model class first.

> **The class definition is part of the artefact.** Deploy the weights without the matching source
> and you have nothing. For serving, PyTorch's answers are TorchScript or ONNX export — an extra
> step TensorFlow does not need.

Note `weights_only=True` on load: `torch.load` used to unpickle arbitrary objects, which is remote
code execution if the checkpoint is untrusted.

---

## What it gives you, and what it costs

**+** The loop is explicit, so nothing is hidden and anything can be changed. Debugging is ordinary
Python debugging.
**+** Models are plain objects — printable, subclassable, steppable.
**+** `Dataset`/`DataLoader` is a clean split.

**−** You write the bookkeeping, and it is easy to get subtly wrong.
**−** `zero_grad()` is mandatory and silent if forgotten.
**−** `train()`/`eval()` will not switch themselves, and forgetting costs accuracy without raising.
**−** Channels-first, against everyone else's channels-last.
**−** Serving needs TorchScript or ONNX.

---

## The exam angle

- Vertex AI has **prebuilt PyTorch containers** for training; serving usually means TorchServe or a
  custom container.
- **`CrossEntropyLoss` expects logits.** Expect a softmax-plus-CrossEntropy distractor.
- **`zero_grad()` / `train()` / `eval()`** are the three things a question about a broken PyTorch
  loop is usually testing.
- **ONNX** is the interchange format when you need to leave the PyTorch ecosystem.
