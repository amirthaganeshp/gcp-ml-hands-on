# TensorFlow / Keras

> **Runtime:** ~30 s · **Result:** 0.8767 test accuracy in 5.0 s of training

```bash
python 04_frameworks/01_tensorflow/tf_version.py
```

Keras gives the shortest path to a trained model of the three. What is worth studying is not
`model.fit` — it is the ecosystem around it.

---

## 1. The model

Keras infers every shape from the `Input` spec, initialises the weights and builds the backward
pass. You describe the model; it fills in the rest. That is exactly the trade the other two
frameworks make differently.

## 2. `tf.data`

The chain reads as a description rather than a loop:

```python
ds = tf.data.Dataset.from_tensor_slices((x, y))
ds = ds.cache().shuffle(len(x), seed=SEED)
ds = ds.batch(BATCH)
return ds.prefetch(tf.data.AUTOTUNE)
```

| | one epoch of batches |
|---|---|
| without prefetch | 0.026 s |
| with prefetch | 0.024 s |

**Be honest about that measurement.** The gap is negligible because the data is already in memory
and there is no accelerator to keep fed. On a real pipeline — reading from GCS, decoding JPEGs, on
a GPU — `prefetch` and parallel `map` are frequently the difference between 30% and 95%
utilisation. This is a demonstration of the API, not evidence that it is fast.

Order matters: `batch()` before `map()` when the map is vectorisable, so the work happens once per
batch rather than once per element.

## 3. TFRecord

500 examples → 1,558 KB (raw arrays: 1,531 KB), read back byte-identical.

Worth knowing not because it is pleasant — it is not — but because **Vertex AI training, Dataflow
and TFX all speak it**, and because one large sequential file streams far better from object
storage than 60,000 small ones.

> **The trap:** a TFRecord stores bytes and a schema, *not* an array shape. The reader has to
> supply `(28, 28, 1)` by hand. Forget it and you get a reshape error at read time, usually a long
> way from the code that wrote the file.

## 4. SavedModel

| artefact | purpose |
|---|---|
| `.keras` file | weights + config — for **resuming training** |
| `SavedModel/` directory | graph + weights — for **serving** |

```
Serving signature: takes ['keras_tensor'], returns 'output_0' (4, 10)
```

A SavedModel is a **directory containing the graph**, so the serving process needs no Python model
definition at all. That is why Vertex AI, TF Serving and Cloud Run consume it directly — and it is
the sharpest contrast with PyTorch's `state_dict`.

> Check the serving signature explicitly before deploying. A model that trains perfectly can still
> export a signature nobody can call.

## 5. Callbacks

```python
EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
ModelCheckpoint(path, monitor="val_accuracy", save_best_only=True)
ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
```

Three lines. Every one of these is something the PyTorch and JAX versions write by hand, and it is
the single biggest ergonomic difference between them.

---

## What it gives you, and what it costs

**+** `fit` with callbacks is the shortest path to a trained model, and the callbacks cover most of
what production needs.
**+** `tf.data` is a genuinely good input API — declarative, parallelises and prefetches without
you managing threads.
**+** SavedModel is a self-contained serving artefact needing no Python.
**+** TFRecord is what the rest of the Google Cloud data stack speaks.

**−** The abstraction is hard to see through. When `fit` surprises you, you are debugging a
framework rather than a loop you wrote.
**−** TFRecord is verbose and does not store shapes, so writer and reader can disagree silently.
**−** Two ways to save (`.keras` and SavedModel) for different jobs, easy to confuse.

---

## The exam angle

- **SavedModel is the deployment artefact.** Vertex AI prebuilt containers expect it.
- **TFRecord + tf.data** is the assumed input path for large-scale training.
- **`prefetch(AUTOTUNE)`** is the standard answer to "the accelerator is underutilised".
- **Callbacks** — know `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard`.
- `.keras` for checkpoints, **SavedModel for serving** — expect this distinction.
