"""The shared CNN in TensorFlow/Keras, plus the parts that are TF's own.

Keras gives you the shortest path to a trained model of the three. What is
worth studying here is not `model.fit` - it is the ecosystem around it:
tf.data for input, TFRecord for storage, SavedModel for serving, and
callbacks for everything you would otherwise write by hand.

  1. The model, and what `fit` is doing for you
  2. tf.data - the input pipeline, and why prefetch matters
  3. TFRecord - the storage format Vertex AI and Dataflow expect
  4. SavedModel - export, reload, and the serving signature
  5. Callbacks - early stopping, checkpointing, LR scheduling

Run:
    python 04_frameworks/tensorflow/tf_version.py

Concept reference: Foundations volume, Part C.1.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from common import (BATCH, EPOCHS, LR, SEED, banner, check_params,  # noqa: E402
                    header, load_data, record, source_lines)

ARTIFACTS = Path(__file__).parent / "artifacts"


def build_model() -> tf.keras.Model:
    """The shared architecture, Keras style.

    Note how much is implicit: Keras infers every shape from the input spec,
    initialises the weights, and wires the backward pass. Compare with the
    PyTorch version, where shapes are declared, and the JAX version, where
    the parameters are a value you hold yourself.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input((28, 28, 1)),
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])


# ============================================================ 2. tf.data
def build_pipeline(x: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
    """tf.data is TensorFlow's answer to 'the GPU is idle waiting for data'.

    The chain reads as a description of the pipeline rather than a loop:

        cache()     keep the decoded tensors in memory after the first epoch
        shuffle()   with a buffer - NOT a full shuffle, which matters if the
                    file is ordered by class
        batch()     before map() when the map is vectorisable, so the work
                    happens once per batch instead of once per element
        prefetch()  overlap the CPU preparing batch n+1 with the accelerator
                    computing batch n. This is the line that usually matters.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.cache().shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH)
    return ds.prefetch(tf.data.AUTOTUNE)


def demo_pipeline(d: dict) -> None:
    ds = build_pipeline(d["x_train"], d["y_train"], training=True)
    xb, yb = next(iter(ds))
    print(f"  One batch: x {tuple(xb.shape)} {xb.dtype.name}, "
          f"y {tuple(yb.shape)} {yb.dtype.name}")
    print(f"  Batches per epoch: {len(ds)}")

    # measure what prefetch is worth on this machine
    def drain(dataset) -> float:
        t0 = time.perf_counter()
        for _ in dataset:
            pass
        return time.perf_counter() - t0

    plain = tf.data.Dataset.from_tensor_slices(
        (d["x_train"], d["y_train"])).batch(BATCH)
    t_plain = drain(plain)
    t_pref = drain(plain.prefetch(tf.data.AUTOTUNE))
    print(f"\n  Draining one epoch of batches:")
    print(f"    without prefetch  {t_plain:.3f}s")
    print(f"    with prefetch     {t_pref:.3f}s")
    print("\n  The gap is small here because the data is already in memory and")
    print("  there is no accelerator to keep fed. On a real pipeline - reading")
    print("  from GCS, decoding JPEGs, on a GPU - prefetch and parallel map are")
    print("  frequently the difference between 30% and 95% utilisation.")
    print("  Quote that honestly: this is a demonstration of the API, not")
    print("  evidence that it is fast.")


# =========================================================== 3. TFRecord
def demo_tfrecord(d: dict) -> None:
    """The serialised format TensorFlow tooling expects.

    Worth knowing not because it is pleasant - it is not - but because
    Vertex AI training, Dataflow and TFX all speak it, and because it solves
    a real problem: one large sequential file streams far better from object
    storage than 60,000 small ones.
    """
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / "sample.tfrecord"
    n = 500

    def to_example(img: np.ndarray, label: int) -> bytes:
        feature = {
            # images go in as raw bytes; the shape is NOT stored, so the
            # reader has to know it. That asymmetry is the usual source of
            # TFRecord bugs.
            "image": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[img.tobytes()])),
            "label": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(label)])),
        }
        return tf.train.Example(
            features=tf.train.Features(feature=feature)).SerializeToString()

    with tf.io.TFRecordWriter(str(path)) as w:
        for img, lab in zip(d["x_train"][:n], d["y_train"][:n]):
            w.write(to_example(img, lab))

    size_kb = path.stat().st_size / 1024
    raw_kb = d["x_train"][:n].nbytes / 1024
    print(f"  Wrote {n} examples to {path.name}: {size_kb:,.0f} KB "
          f"(raw arrays: {raw_kb:,.0f} KB)")

    schema = {"image": tf.io.FixedLenFeature([], tf.string),
              "label": tf.io.FixedLenFeature([], tf.int64)}

    def parse(record):
        ex = tf.io.parse_single_example(record, schema)
        img = tf.io.decode_raw(ex["image"], tf.float32)
        return tf.reshape(img, (28, 28, 1)), ex["label"]

    ds = tf.data.TFRecordDataset(str(path)).map(parse).batch(32)
    xb, yb = next(iter(ds))
    same = bool(np.allclose(xb[0].numpy(), d["x_train"][0]))
    print(f"  Read back: {tuple(xb.shape)}, first image identical: {same}")
    print("\n  Note the round trip needs the shape (28, 28, 1) supplied by")
    print("  hand - a TFRecord stores bytes and a schema, not an array shape.")
    print("  Forget it and you get a silent reshape error at read time, often")
    print("  a long way from the code that wrote the file.")


def remove_dir(path: Path, attempts: int = 3) -> Path:
    """Delete a directory tree, working around Windows' read-only attribute.

    A plain shutil.rmtree here failed reproducibly on the SECOND run:

        PermissionError: [WinError 5] Access is denied:
          '...\artifacts\saved_model\assets'

    The giveaway was that `rm -rf` removed the same directory without
    complaint. It is not a lock - it is the Windows READ-ONLY ATTRIBUTE.
    shutil.rmtree calls os.rmdir/os.unlink, which refuse on a read-only
    entry and raise WinError 5; rm -rf clears the attribute first.

    The fix is the standard one: an onexc handler that clears the attribute
    and retries the failed operation. A short sleep-and-retry is kept as a
    second line of defence for genuine transient locks, which a sync client
    or an indexer can briefly hold.
    """
    def clear_readonly_and_retry(func, target, _exc_info):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    for i in range(attempts):
        if not path.exists():
            return path
        try:
            shutil.rmtree(path, onexc=clear_readonly_and_retry)
            return path
        except (PermissionError, OSError):
            time.sleep(0.4 * (i + 1))

    # Genuinely stuck. Do not fail an entire training run over a stale
    # artefact directory - export beside it and say so plainly.
    alt = path.with_name(path.name + "_new")
    print(f"  NOTE: {path.name} could not be removed; "
          f"exporting to {alt.name} instead.")
    return alt


# ========================================================== 4. SavedModel
def demo_savedmodel(model: tf.keras.Model, d: dict) -> None:
    """SavedModel is the deployment artefact - the thing Vertex AI, TF Serving
    and Cloud Run actually consume. It is a directory, not a file, and it
    contains the graph as well as the weights, so the serving process needs
    no Python model definition at all.
    """
    ARTIFACTS.mkdir(exist_ok=True)
    keras_path = ARTIFACTS / "model.keras"
    saved_path = remove_dir(ARTIFACTS / "saved_model")

    model.save(keras_path)
    model.export(saved_path)          # SavedModel, for serving

    reloaded = tf.keras.models.load_model(keras_path)
    a = model.predict(d["x_test"][:64], verbose=0)
    b = reloaded.predict(d["x_test"][:64], verbose=0)
    print(f"  .keras file      {keras_path.stat().st_size / 1024:,.0f} KB "
          f"(weights + config, for resuming training)")
    total = sum(f.stat().st_size for f in saved_path.rglob("*") if f.is_file())
    print(f"  SavedModel dir   {total / 1024:,.0f} KB "
          f"(graph + weights, for serving)")
    print(f"  Predictions identical after reload: "
          f"{bool(np.allclose(a, b, atol=1e-6))}")

    served = tf.saved_model.load(str(saved_path))
    fn = served.signatures["serving_default"]
    out = fn(tf.constant(d["x_test"][:4]))
    key = list(out.keys())[0]
    print(f"\n  Serving signature: takes {list(fn.structured_input_signature[1])}, "
          f"returns '{key}' {tuple(out[key].shape)}")
    print("  That signature is the contract a serving container exposes. It is")
    print("  worth checking it explicitly before deploying - a model that")
    print("  trains perfectly can still export a signature nobody can call.")


# =========================================================== 5. callbacks
def demo_callbacks() -> list:
    """Callbacks are the framework writing your training loop's bookkeeping.

    Everything here is something the PyTorch and JAX versions have to do by
    hand - which is the single biggest ergonomic difference between them.
    """
    ARTIFACTS.mkdir(exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            ARTIFACTS / "best.keras", monitor="val_accuracy",
            save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]


def main() -> None:
    header("TensorFlow", tf.__version__)
    tf.keras.utils.set_random_seed(SEED)
    d = load_data()

    banner("1. The model")
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    check_params(model.count_params())
    print("\n  Keras infers every shape from the Input spec, initialises the")
    print("  weights, and builds the backward pass. You describe the model and")
    print("  it fills in the rest - which is exactly the trade the other two")
    print("  frameworks make differently.")

    banner("2. tf.data - the input pipeline")
    demo_pipeline(d)

    banner("3. TFRecord - the storage format")
    demo_tfrecord(d)

    banner("4. Training")
    train_ds = build_pipeline(d["x_train"], d["y_train"], training=True)
    test_ds = build_pipeline(d["x_test"], d["y_test"], training=False)
    t0 = time.perf_counter()
    # shuffle=False because the tf.data pipeline already shuffles. Keras'
    # fit defaults to shuffle=True and warns that it is ignoring it when the
    # input is a Dataset - harmless, but the warning is worth not emitting.
    hist = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS,
                     verbose=0, shuffle=False, callbacks=demo_callbacks())
    secs = time.perf_counter() - t0
    acc = float(model.evaluate(test_ds, verbose=0)[1])
    print(f"  Trained {len(hist.history['loss'])} of {EPOCHS} epochs "
          f"in {secs:.1f}s")
    print(f"  Test accuracy: {acc:.4f}")
    print("\n  The whole training loop is one call. Early stopping,")
    print("  checkpointing and LR scheduling are three more lines. Hold that")
    print("  thought until you read the PyTorch and JAX versions.")

    banner("5. SavedModel - the deployment artefact")
    demo_savedmodel(model, d)

    banner("What TensorFlow gives you, and what it costs")
    print("  + `fit` with callbacks is the shortest path to a trained model,")
    print("    and the callbacks cover most of what production needs.")
    print("  + tf.data is a genuinely good input API - declarative, and it")
    print("    parallelises and prefetches without you managing threads.")
    print("  + SavedModel is a self-contained serving artefact. The serving")
    print("    process needs no Python model code, which is why Vertex AI,")
    print("    TF Serving and Cloud Run all consume it directly.")
    print("  + TFRecord is what the rest of the Google Cloud data stack speaks.")
    print("\n  - The abstraction is hard to see through. When `fit` does")
    print("    something you did not expect, you are debugging a framework")
    print("    rather than a loop you wrote.")
    print("  - TFRecord is verbose and does not store shapes, so a writer and")
    print("    a reader can disagree silently.")
    print("  - Two ways to save (.keras and SavedModel) that are for different")
    print("    jobs and are easy to confuse.")

    record("TensorFlow", acc, secs, model.count_params(),
           source_lines(__file__),
           "fit + callbacks; tf.data; TFRecord; SavedModel export")


if __name__ == "__main__":
    main()
