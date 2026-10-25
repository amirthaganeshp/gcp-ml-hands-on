# Reference

Three documents that pull the repository together. Nothing here introduces new material — it
indexes and summarises what the examples already demonstrated.

| file | what it is |
|---|---|
| [FINDINGS.md](FINDINGS.md) | **every measured number in one place**, contradictions first |
| [DECISIONS.md](DECISIONS.md) | which model, metric, split, cloud service, framework — and a debugging order |
| [GLOSSARY.md](GLOSSARY.md) | terms, with a pointer to where each one is demonstrated |

---

## If you only read one page

**[FINDINGS.md](FINDINGS.md).** It opens with twelve results that contradicted the expected answer
— `class_weight` scoring the *worst* F1, popularity beating every personalised recommender, both
self-supervised pretext tasks losing to raw pixels, ε=0 winning at reinforcement learning.

They are kept because the explanations are the actual lessons, and because they are a standing
reminder that **a comparison you have not run is a comparison you do not know the answer to.**

## The discipline that mattered most

Four separate conclusions in this repository reversed once the undertrained side of a comparison
was given enough epochs. The CNN looked 0.005 better than a dense net at 6 epochs and 0.025 better
at 15. The LSTM looked broken at 12 epochs and was perfect at 40.

**If a comparison surprises you, check convergence before you believe it.**

## What was measured and what was not

The repository is explicit about this in each section, and it matters:

| | |
|---|---|
| **Phases 0–4** | fully executed. Every number is measured on this machine. |
| **Phase 5 (`05_gcp`)** | **write-only** against the cloud by choice — no project touched, no billing. Only `trainer/task.py` genuinely runs (0.8000 accuracy, 2.6 s locally). |
| **Phase 6 (`06_pipelines`)** | the KFP **compile is real** and the spec was structurally validated; the pipeline was never submitted. |

Anywhere a cloud response or a price appears, it is labelled as illustrative rather than captured.

## Running everything

```bash
python 00_setup/check_environment.py
python 00_setup/download_datasets.py

for f in 0*/*/[a-z]*.py; do python "$f"; done
```

About 25 minutes end to end on a laptop CPU. One known flake: the JAX example crashes roughly 1 run
in 14 on Windows — a platform limitation documented in
[04_frameworks/03_jax](../04_frameworks/03_jax/), not a bug in the example.
