# Decision Tables

What to reach for, and why. Every row points at a folder where the claim is demonstrated or
measured.

---

## Which model?

| the problem | reach for | not |
|---|---|---|
| predict a number, tabular | gradient-boosted trees | a deep network |
| predict a class, tabular | gradient-boosted trees, or logistic regression first | a deep network |
| images | CNN, or transfer learning from a pretrained backbone | a flattened dense net |
| sequences, need recent history compressed | LSTM / GRU | a SimpleRNN |
| sequences, need to compare against everything | transformer | recurrence |
| no labels, want groups | k-means (scaled) | anything, unscaled |
| no labels, arbitrary shapes or outliers | DBSCAN | k-means |
| no labels, overlapping groups | Gaussian mixture | k-means |
| find rare unusual rows | Isolation Forest, then LOF | accuracy as the metric |
| recommend items | matrix factorisation → two-tower | RMSE as the objective |
| forecast a seasonal series | `ARIMA_PLUS`, or lag features + GBM | an LSTM, usually |
| high-cardinality categoricals | embeddings | one-hot |

**The repeated finding:** on small clean tabular data, a gradient-boosted tree you train yourself is
hard to beat and the cheapest thing to operate. See [FINDINGS](FINDINGS.md).

## Which metric?

| situation | use | because |
|---|---|---|
| balanced classes | accuracy, F1 | fine |
| **imbalanced classes** | **PR-AUC**, precision@k | ROC-AUC flatters a rare positive class — it fell 0.159 where PR-AUC fell 0.829 |
| you need probabilities, not ranks | log loss, Brier, a calibration curve | AUC is ranking-only |
| a few big errors are acceptable | MAE | linear in the error |
| big errors are disproportionately costly | RMSE | squares first |
| clustering, no labels | silhouette, Davies-Bouldin | measures *separation*, not truth |
| clustering, labels available | **ARI, NMI** | cluster IDs are arbitrary; accuracy is meaningless |
| recommendation | **HitRate@k, NDCG@k** + coverage | RMSE and ranking disagree |
| anomaly detection | PR-AUC, precision@k | at a 4% base rate, flagging nothing scores 96% |
| forecasting | MAE/MAPE **vs seasonal naive** | a raw error means nothing alone |

## Which validation split?

| data | split | trap avoided |
|---|---|---|
| i.i.d. rows | stratified K-fold | the split lottery — 0.087 accuracy spread across seeds |
| **time-ordered** | **chronological / `TimeSeriesSplit`** | leaking the future; made error look 2× better |
| grouped (patients, users, stores) | `GroupKFold` | the same entity in train and test |
| recommendation | hold out each user's **last** interaction | same leak, different clothes |
| very small data | repeated CV, report the spread | one split is one sample |

## Which Google Cloud service?

| if... | then |
|---|---|
| a common task on common data | **pre-trained API** |
| the data is already in BigQuery and tabular | **BigQuery ML** |
| images / video / text, your labels, no ML staff | **AutoML** |
| custom architecture, custom loss, control of the loop | **custom training** |
| predictions needed once a day, or written to a warehouse | **batch prediction** |
| a user is waiting, sub-second | **online endpoint** |
| extract structured fields from documents | **Document AI**, not Vision OCR |
| high-volume plain text from images | **Vision OCR**, ~10× cheaper per page |
| univariate seasonal forecasting | **`ARIMA_PLUS`** |
| it must not ship unless it clears a metric | **a pipeline with a `dsl.If` gate** |

**Pick the least custom option that fits.** "No ML expertise" or "as quickly as possible" in a
question steers away from custom training; a novel architecture or custom loss steers towards it.

**Cost asymmetry that decides architectures:** a pre-trained API and BigQuery ML cost **nothing at
rest**. A Vertex endpoint bills per node-hour whether or not it is called, and cannot scale to zero.

## Which agent type? (ADK)

| the situation | reach for |
|---|---|
| the next step depends on what the user said | `LlmAgent` |
| a fixed pipeline — draft, review, format | `SequentialAgent` |
| N **independent** lookups, then combine | `ParallelAgent` |
| refine until a quality bar is met | `LoopAgent` (always set `max_iterations`) |
| route on a field you can read yourself | a workflow agent, not a model |
| genuinely irregular control flow | custom `BaseAgent` |

**The question is who decides control flow — the model, or you.** Every `LlmAgent` decision is a
model call. Workflow agents have **no `model` field at all**, which is why they are deterministic,
auditable and free of inference cost.

## Where does the agent run?

| if... | then |
|---|---|
| developing | `adk web` / `adk run` locally |
| production on GCP, want it managed | **Agent Runtime** — `adk deploy agent_engine` |
| custom dependencies, or called rarely | **Cloud Run** — `adk deploy cloud_run`, **scales to zero** |
| you already operate a cluster | **GKE** — `adk deploy gke` |

**Swap `InMemorySessionService` for `VertexAiSessionService` when you deploy.** In-memory state is
lost on restart and not shared across replicas; the failure is intermittent and raises nothing.

## Which framework?

| | choose when |
|---|---|
| **Keras / TensorFlow** | standard model, want it working today; and it is what the GCP exam assumes |
| **PyTorch** | you need to change the training loop, or you are reading research code |
| **JAX** | composable transformations, or TPUs |

**Keras 3 runs on all three backends**, set by one environment variable — so API and backend are
separable choices.

## Debugging a model that will not train

In order. The earlier items are far more common than the later ones.

1. **Learning rate.** A loss that rises, plateaus high, or goes NaN is almost always this. The
   healthy band spanned three orders of magnitude; outside it, nothing works.
2. **Scaling.** An unscaled neural net scored *exactly* 0.5000 — chance.
3. **Are both arms converged?** Four comparisons in this repo reversed on epoch count alone.
4. **Leakage.** If results are too good, look for a column derived from the target.
5. **The loss/output pairing.** PyTorch's `CrossEntropyLoss` wants logits; Keras'
   `sparse_categorical_crossentropy` defaults to probabilities. A double softmax trains, badly.
6. **`zero_grad()` / `train()` / `eval()`** in PyTorch — all silent when forgotten.
7. **Initialisation.** A 2.5-point spread across frameworks came from defaults alone.
8. **Capacity**, last. It is rarely the problem.
