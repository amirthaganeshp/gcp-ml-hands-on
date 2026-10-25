# Checks

Scripts that verify the repository still works. **You never have to run them** — they exist so a
mistake shows up on purpose rather than by accident.

```bash
bash .github/scripts/run_list.sh fast     # parts 00-02, core stack only
bash .github/scripts/run_list.sh full     # all 41 runnable examples
python .github/scripts/check_links.py     # every link between the READMEs works
```

**Measured on a laptop CPU:** `fast` **17/17 in ~4 min** · `full` **41/41 in 1,231 s (~20 min)** · links
**263/263**.

`rc=0` means that example was fine. Anything else is a failure, and the runner prints the last 25
lines of its output so you can see why.

> Run these from **the repository root** — the paths are relative to it. `No such file or
> directory` means you are somewhere else; `pwd` tells you where.
>
> The runner finds the project virtualenv by itself (`.venv/Scripts/python.exe` on Windows,
> `.venv/bin/python` elsewhere) and prints which interpreter it chose. Override with
> `PYTHON=/path/to/python`.

---

## What gets run, and what does not

| | |
|---|---|
| **fast** | parts 00–02 — foundations and traditional ML. 17 scripts, core stack only. |
| **full** | everything runnable — adds parts 03–10: neural networks, the framework comparison, the ADK agent and hardware examples, the KFP pipeline, serving, monitoring, the capstone and the summary sheet. 41 scripts. |

**Deliberately not run:** `loaders.py`, `plotting.py`, `04_frameworks/common.py` and
`trainer/__init__.py` are imported by other scripts rather than executed. The other four `05_gcp`
examples (pre-trained APIs, BigQuery ML, AutoML, custom training) need a real Google Cloud project,
so they are documented rather than run; the ADK and hardware examples run locally.

---

## Recipe: run these automatically on your own GitHub

*("Automatically" = every time you push, GitHub runs all this for you and tells you if something
broke. That is all CI means.)*

**This will not work on an enterprise-managed account** whose policy blocks hosted runners — see
the last section. On a personal account it does.

### Step 1 — make an empty repo on your personal account

[github.com/new](https://github.com/new), signed in as **your personal account**. **Choose
Private.** Do not tick "Add a README".

> Private repos get **2,000 free Actions minutes a month**. The `fast` job is a few minutes, so
> everyday use costs nothing. Note the `full` job is ~20 minutes *plus* a multi-gigabyte
> TensorFlow/PyTorch/JAX install — which is exactly why it is on a schedule rather than every push.

### Step 2 — point a copy of this repo at it

**Option A — keep the history:**

```bash
git clone <URL-of-this-repo> mle-personal
cd mle-personal
git remote set-url origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
```

**Option B — start with a clean history** (one fresh commit, nothing carried over):

```bash
git clone <URL-of-this-repo> mle-personal
cd mle-personal
rm -rf .git
git init -b main
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
```

### Step 3 — turn the triggers on

In `.github/workflows/run-list.yml`, **delete the `#`** from the six marked lines:

```yaml
on:
  workflow_dispatch:
  push:                       # <- uncommented
    branches: [main]
  pull_request:               # <- uncommented
    branches: [main]
  schedule:                   # <- uncommented
    - cron: "0 6 * * 1"
```

Indentation matters in YAML — keep the spacing exactly as shown.

### Step 4 — push, then watch

```bash
git add .github/workflows/run-list.yml
git commit -m "Enable CI"
git push -u origin main
```

Open your repo on github.com → **Actions** tab. **Success** is a green tick with a results table
listing every script. **Failure** is a red X naming the failing script, with its output in the log.

---

## Why this is switched off by default

Pushing a workflow from an enterprise-managed account can fail immediately with:

> **GitHub Actions hosted runners are disabled for this repository. For more information please
> contact your GitHub Enterprise Administrator.**

"Actions enabled" on the repository settings is **not** the check that matters — runner
availability is. A repo can report Actions as enabled and still have no machine allowed to run
them, and no self-hosted runner registered.

Nothing in this repo can change that; it needs whoever administers GitHub for the organisation. So
the workflow is **manual-only** by default.

**That is why the logic lives in `scripts/` rather than inside the workflow file.** All of it runs
on your laptop, unchanged, with no GitHub at all — and a CI-only script is one you can never test.

### Optional: run the fast check before every push

Not installed by default:

```bash
cp .github/scripts/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

Undo with `rm .git/hooks/pre-push`. Skip once with `git push --no-verify`.
