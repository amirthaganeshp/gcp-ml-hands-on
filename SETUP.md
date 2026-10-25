# Setup

## Python version

**Use Python 3.12.** This is not arbitrary: TensorFlow does not publish wheels for the newest
Python release for six to twelve months after it lands. At the time of writing there is **no
TensorFlow wheel for Python 3.14** — `pip install tensorflow` fails with
*"No matching distribution found"*.

Parts `01`–`02` need only scikit-learn and NumPy and will run on any modern Python. Part `03`
needs Keras with a backend, and parts `04` and `05` need TensorFlow, so the repository targets 3.12
throughout to keep one environment.

Check what you have:

```bash
py -0p            # Windows: lists every installed Python
python --version
```

If you need 3.12 on Windows, it installs per-user with no admin rights:

```powershell
winget install --id Python.Python.3.12 --source winget --scope user
```

Installing 3.12 does **not** change your default `python`. The launcher keeps them separate and
`py -3.12` selects it explicitly.

---

## Virtual environment

```powershell
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate
```

```bash
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
```

Confirm you are in it — the prompt shows `(.venv)` and:

```bash
python --version        # Python 3.12.x
```

---

## Dependencies

Two files, because the deep-learning stack is large (~2 GB) and parts 1–2 do not need it.

```bash
pip install -r requirements.txt        # core: parts 01-02. ~150 MB
pip install -r requirements-dl.txt     # Keras, PyTorch, TensorFlow, JAX, KFP: part 03 onward. ~2 GB
```

Install the core set first and confirm it works before pulling the deep-learning extras.

---

## Verify

```bash
python 00_setup/check_environment.py
```

This prints your Python version, the version of every package it can find, flags anything missing,
and then actually trains a small model to prove the stack works rather than merely importing.

---

## Datasets

```bash
python 00_setup/download_datasets.py
```

Fetches every dataset that is not built into scikit-learn or Keras and caches it under `data/`.
It is **idempotent** — re-running skips anything already present. Nothing in `data/` is committed.

Datasets built into a library (Iris, Wine, California Housing, Digits, MNIST, Fashion-MNIST)
download themselves on first use and cache in your home directory; the downloader does not
duplicate them.

To re-fetch everything from scratch:

```bash
python 00_setup/download_datasets.py --force
```

---

## Running an example

Every script is standalone and run from the repository root:

```bash
python 01_foundations/02_bias_variance_tradeoff/bias_variance.py
python 02_traditional_ml/02_classification/classification.py
```

Figures are written to that example's own `images/` folder. They are committed, so each README
renders on GitHub; re-running a script overwrites them (see the note on byte-identical figures in
the top-level README). Metrics are printed to the terminal; the one exception is `04_frameworks`,
whose three scripts write a gitignored `04_frameworks/results.json` for the comparison to read.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No matching distribution found for tensorflow` | You are on Python 3.13+. Use 3.12 (above) |
| `ModuleNotFoundError` after installing | The virtual environment is not active. Re-activate it |
| Downloads fail with `SSLError`, but the same URL opens in your browser | A corporate proxy inspecting TLS. Python does not trust the intercepting certificate; your browser does. Fix: `pip install truststore` (already in `requirements.txt`) — the downloader picks it up automatically |
| A download fails or times out | A firewall or a moved URL. Try again, or fetch the URL by hand into `data/` — the script prints the exact path it expects |
| Matplotlib opens no window | Expected. Scripts save figures to the example's `images/` folder rather than displaying them |
| A script is slow | Everything is sized for CPU. If one is genuinely slow, check you have not passed `--full` |

---

## Deep learning extras (part 03 onward)

```bash
pip install -r requirements-dl.txt
```

### Windows long-path warning

TensorFlow ships a deeply nested `include/` tree. Its longest bundled header is about 150
characters on its own, so if this repository sits more than ~110 characters deep, pip aborts
part-way through with:

```
OSError: [Errno 2] No such file or directory:
  '...	ensorflow\include\external\envoy_api\...\client_side_weighted_round_robin.upb.h'
```

It leaves a **broken** `tensorflow` package behind - `import tensorflow` then fails with
`No module named 'tensorflow.python'`. Delete `.venv/Lib/site-packages/tensorflow` if that
happens.

Two fixes: enable Windows long paths (a registry change requiring admin), or clone somewhere
short such as `C:\dev\machine-learning`.

### Keras 3 is backend-agnostic

Part 03 only needs the **Keras API**, and Keras 3 runs on TensorFlow, PyTorch or JAX. Every
script in part 03 sets:

```python
os.environ.setdefault("KERAS_BACKEND", "torch")
```

so **PyTorch alone is enough** for the whole of part 03. Parts 04 and 05 need all three backends,
because comparing them is the point.

### Dataset downloads behind a TLS-inspecting proxy

`keras.datasets.*` downloads fail with an SSL error on a corporate network. The scripts that need
them call `truststore.inject_into_ssl()` first, which makes Python use the OS certificate store.

### Working around the long-path limit without admin rights

Map the repository to a short virtual drive and install through that. The files land in the same
place; only the path pip *writes* gets shorter.

```
subst M: "C:\path	o\machine-learning"
M:\.venv\Scripts\python.exe -m pip install tensorflow
subst M: /D
```

That is a per-session drive mapping, not a system setting. TensorFlow installs cleanly this way
and imports fine afterwards from the normal long path, because only the install needed the
shorter one.

**Flax still will not install**, even from a drive root: its `orbax-checkpoint` dependency ships
test fixtures around 260 characters deep on their own. The JAX example therefore uses pure
JAX + optax.
