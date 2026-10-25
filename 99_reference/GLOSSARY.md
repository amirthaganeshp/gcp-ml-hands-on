# Glossary

Terms that appear across the repository, with where each one is actually demonstrated rather than
merely defined.

---

## Validation and metrics

**Bias / variance** — error from wrong assumptions vs error from sensitivity to the training
sample. → [01/02](../01_foundations/02_bias_variance_tradeoff/)

**One-standard-error rule** — choose the simplest model within one standard error of the best. It
picked degree 2 over the degree 8 that minimised validation error. → [01/02](../01_foundations/02_bias_variance_tradeoff/)

**Leakage** — any information in training that would not exist at prediction time. Five kinds, plus
a real one in a public dataset (`casual + registered == cnt`). → [01/05](../01_foundations/05_data_leakage/)

**ROC-AUC vs PR-AUC** — ranking quality over all thresholds. ROC flatters a rare positive class
because true negatives are plentiful; PR does not. → [01/04](../01_foundations/04_roc_vs_pr_curves/)

**Calibration** — do predicted probabilities match observed frequencies? Distinct from ranking: a
model can rank perfectly and be badly calibrated. → [01/04](../01_foundations/04_roc_vs_pr_curves/)

**Precision@k** — of the top k you would actually review, how many are real. The practical metric
for anomaly detection and retrieval. → [02/05](../02_traditional_ml/05_anomaly_detection/)

**ARI / NMI** — clustering agreement, invariant to relabelling and adjusted for chance. Accuracy is
meaningless for clustering because cluster IDs are arbitrary. → [02/03](../02_traditional_ml/03_clustering/)

**NDCG@k** — ranking quality with position discounting. → [02/07](../02_traditional_ml/07_recommenders/)

## Traditional ML

**L1 / Lasso** — penalty that drives coefficients to *exactly* zero: feature selection.
**L2 / Ridge** — smooth shrinkage, nothing reaches zero. → [02/01](../02_traditional_ml/01_regression/)

**Voronoi cell** — the convex region k-means assigns to a centroid. Why k-means cannot produce a
crescent at any *k*. → [02/03](../02_traditional_ml/03_clustering/)

**Equivariance vs invariance** — convolution is *equivariant* (shift the input, the feature map
shifts); pooling buys a little true invariance. → [03/03](../03_neural_networks/03_cnn_images/)

**Shrinkage / damping** — pulling a small-sample statistic towards a global mean. A raw item mean
ranked a film with one 5-star rating above Star Wars. → [03/06](../03_neural_networks/06_embeddings/)

**Optimistic initialisation** — initial value estimates set high enough that unvisited options look
attractive. Zero-init Q with a −1 step cost *is* this, which is why ε=0 worked.
→ [02/09](../02_traditional_ml/09_reinforcement_learning/)

**On-policy vs off-policy** — SARSA learns the value of what it actually does (exploration
included); Q-learning learns the optimal policy's value while behaving exploratorily. One line of
difference. → [02/09](../02_traditional_ml/09_reinforcement_learning/)

**Pretext task** — an invented supervised task whose labels come free from the data, used to learn
a representation. The model is discarded; the encoder is kept. → [02/08](../02_traditional_ml/08_self_supervised/)

**Cold start** — no interaction history for a new user or item, so a collaborative model has no
vector for it. → [02/07](../02_traditional_ml/07_recommenders/), fixed in [03/09](../03_neural_networks/09_two_tower_retrieval/)

## Neural networks

**Backpropagation** — the chain rule applied layer by layer. For sigmoid output with binary
cross-entropy the first term collapses to `(prediction − target)`. → [03/01](../03_neural_networks/01_perceptron_from_scratch/)

**Gradient checking** — comparing an analytic gradient against `(L(w+ε) − L(w−ε)) / 2ε`. Two
forward passes *per parameter*, which is why autograd exists. → [03/01](../03_neural_networks/01_perceptron_from_scratch/)

**Vanishing gradient** — gradients multiplied layer by layer shrink towards zero. Measured at
**5,206×** attenuation over six sigmoid layers. → [03/01](../03_neural_networks/01_perceptron_from_scratch/)

**Xavier/Glorot vs He initialisation** — variance scaled by `1/n_in` vs `2/n_in`; He suits ReLU.
The source of a 2.5-point accuracy spread across frameworks. → [04](../04_frameworks/)

**Dropout** — zeroes activations *during training only*. Off at inference. → [03/02](../03_neural_networks/02_mlp_tabular/)

**Batch normalisation** — standardises pre-activations using *batch* statistics in training and
running averages at inference — a classic train/serve skew. → [03/02](../03_neural_networks/02_mlp_tabular/)

**Cell state / gates** — the LSTM's near-linear path through time; when the forget gate is ~1 and
the input gate ~0, state is copied forward undamped. → [03/04](../03_neural_networks/04_rnn_lstm_gru/)

**Attention** — `softmax(QKᵀ/√d)V`. The `√d` is not cosmetic: without it the softmax saturates into
a hard argmax with no gradient. → [03/05](../03_neural_networks/05_transformer_attention/)

**Permutation equivariance** — attention sees an unordered *set*, which is why positional encoding
exists. → [03/05](../03_neural_networks/05_transformer_attention/)

**Wide and deep** — a linear model over crossed categoricals (memorisation) plus embeddings and an
MLP (generalisation), trained jointly. → [03/08](../03_neural_networks/08_wide_and_deep/)

**Two-tower** — separate user and item encoders joined only by a dot product, so item vectors can
be precomputed and indexed. → [03/09](../03_neural_networks/09_two_tower_retrieval/)

## Frameworks

**Pytree** — any nested structure of arrays; every JAX transformation understands them, which is
why an optimiser update is one `tree_map`. → [04/03](../04_frameworks/03_jax/)

**`jit` / `grad` / `vmap` / `pmap`** — JAX's four function transformations. They compose:
`jit(vmap(grad(f)))` means what it reads. → [04/03](../04_frameworks/03_jax/)

**SavedModel** — a directory containing the graph *and* weights, so serving needs no Python.
Contrast `state_dict`, which is weights alone and needs the class definition.
→ [04/01](../04_frameworks/01_tensorflow/), [04/02](../04_frameworks/02_pytorch/)

**`tf.data` / `prefetch`** — declarative input pipeline; `prefetch` overlaps CPU preparation with
accelerator compute. → [04/01](../04_frameworks/01_tensorflow/)

**TFRecord** — TensorFlow's serialised record format. Stores bytes and a schema, **not** array
shapes. → [04/01](../04_frameworks/01_tensorflow/)

## Google Cloud and MLOps

**Application Default Credentials (ADC)** — implicit auth from the environment; explicit key files
are the discouraged path. → [05/01](../05_gcp/01_pretrained_apis/)

**Trainer package** — a Python package whose entry point takes hyperparameters as *command-line
flags* and writes to `AIP_MODEL_DIR`. Anything hardcoded cannot be tuned.
→ [05/04](../05_gcp/04_custom_training/)

**Prebuilt vs custom container** — use prebuilt unless you need a system package, a CUDA version or
a private wheel. → [05/04](../05_gcp/04_custom_training/)

**Batch vs online prediction** — an endpoint bills per node-hour at rest and cannot scale to zero;
batch has no idle cost. → [05/04](../05_gcp/04_custom_training/)

**Component (KFP)** — a function *plus a container*. Its body is isolated: every import goes
inside. → [06](../06_pipelines/)

**Artifact vs parameter** — parameters are small values passed by value and **can** drive control
flow; artifacts are paths tracked with lineage and **cannot**, because the orchestrator never opens
them. → [06](../06_pipelines/)

**Deployment gate** — a `dsl.If` on an evaluation metric, so a model cannot reach an endpoint
without clearing a threshold. → [06](../06_pipelines/)

## Agents

**ADK vs Agent Runtime** — ADK is the *framework* you write an agent in; Agent Runtime (formerly
Agent Engine) is one of four *hosts* you can run it on. Independent choices — the managed host also
takes LangChain and LangGraph. → [05/05](../05_gcp/05_adk_agents/)

**Workflow agent** — `SequentialAgent`, `ParallelAgent`, `LoopAgent`. Deterministic orchestration
with **no `model` field**, so no LLM can sit in the control flow. → [05/05](../05_gcp/05_adk_agents/)

**`output_key`** — the field that writes an agent's result into session state, read by a later
agent's `instruction` via `{key}` templating. The entire data path between agents — and the reason
parallel branches must not read each other's keys. → [05/05](../05_gcp/05_adk_agents/)

**Session vs memory** — a session is *this* conversation; memory is what survives across them.
Wanting an agent to remember someone next week is memory, and no session service delivers it.
→ [05/05](../05_gcp/05_adk_agents/)

**Escalation** — a tool setting `tool_context.actions.escalate = True` to break out of a
`LoopAgent` before `max_iterations`. → [05/05](../05_gcp/05_adk_agents/)
