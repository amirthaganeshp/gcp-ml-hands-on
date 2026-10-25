# Neural Networks

All nine run clean end to end in **15 min 47 s** on a laptop CPU, producing 23
committed figures.

Nine topics, built in order so each one answers a question the previous one raised. The first is
pure NumPy with every gradient derived by hand; the rest use the **Keras 3 API**, which is what the
GCP ML Engineer exam tests.

---

## Contents

| # | Topic | Runtime | The one thing to take from it |
|---|---|---|---|
| [01](01_perceptron_from_scratch/) | **Perceptron & backprop by hand** | ~4 s | Sigmoid attenuates the gradient **5,206×** from output to input; ReLU is flat |
| [02](02_mlp_tabular/) | **MLP on tabular data** | ~2 m | Logistic regression won. The MLP took **268× longer** for −0.0004 AUC |
| [03](03_cnn_images/) | **CNNs** | ~1 m 50 s | At *matched parameters*, the CNN cut error 17% — the prior, not the capacity |
| [04](04_rnn_lstm_gru/) | **RNN / LSTM / GRU** | ~3 m | The textbook LSTM-beats-RNN result **did not reproduce** on the toy task |
| [05](05_transformer_attention/) | **Attention & transformers** | ~2 m 10 s | LSTM stuck at chance (0.513), transformer 0.765 — on the *opposite* task to 04 |
| [06](06_embeddings/) | **Embeddings** | ~1 m 10 s | Star Wars → its two sequels at 0.92, learned from ratings alone |
| [07](07_autoencoders/) | **Autoencoders** | ~2 m | A linear autoencoder **is** PCA; Isolation Forest scored *random* on 784 pixels |
| [08](08_wide_and_deep/) | **Wide & Deep** | ~1 m 30 s | The joint model lost to its own deep half, and XGBoost beat everything |
| [09](09_two_tower_retrieval/) | **Two-tower retrieval** | ~2 m 10 s | A film that does not exist got sensible neighbours with no id and no retraining |

```bash
python 03_neural_networks/01_perceptron_from_scratch/perceptron.py
```

---

## Setup

Topics 02–09 need Keras 3 and a backend:

```bash
pip install -r requirements-dl.txt
```

**Windows long-path warning.** TensorFlow ships a deeply nested `include/` tree whose longest
bundled header is ~150 characters on its own. If this repository sits more than ~110 characters
deep, pip aborts mid-install and leaves a **broken** `tensorflow` package behind (`import
tensorflow` then fails with *"No module named 'tensorflow.python'"* — delete
`site-packages/tensorflow` if that happens).

**Keras 3 is backend-agnostic**, so every example here runs on **PyTorch** instead:

```python
os.environ.setdefault("KERAS_BACKEND", "torch")
```

That is the first thing each script does. Nothing in this section depends on TensorFlow
specifically — the *Keras API* is what matters, and it is identical across backends.

---

## The through-line

**1. The architectural prior is worth more than the parameters.** The CNN and the dense network in
[03](03_cnn_images/) had the *same* capacity and the same data; the CNN cut the error rate 18%
purely because it was told the input is a grid. Capacity is cheap. Knowing something about your
data is not.

**2. Match the mechanism to the question.** The LSTM beat the transformer's cousin on forecasting
in [04](04_rnn_lstm_gru/) and sat at chance in [05](05_transformer_attention/); the transformer did
the reverse. Recurrence compresses recent history, attention compares against everything. **Neither
is simply better**, and any ranking of architectures that ignores the task is folklore.

**3. Undertraining is the most common way to fake a result.** Four separate comparisons in this
section reversed when given a fair epoch budget — the CNN at 6 epochs (+0.005 → +0.024), the LSTM
at 12 epochs (0.512 → 0.822 at length 40), the transformer at 12 (chance → 0.765), the two-tower at 12. **Both
arms of a comparison must be trained to convergence** or the conclusion is about the budget.

**4. Neural networks are data-hungry, and small classical problems are still classical.** Logistic
regression beat the MLP; XGBoost beat Wide & Deep; seasonal naive was competitive with an LSTM;
popularity beat two-tower retrieval. **Run the baseline every time** — on this scale of data it
frequently wins, and reporting a deep model without one is unevaluable.

**5. Everything is the vanishing gradient, wearing different clothes.** Depth in
[01](01_perceptron_from_scratch/), time in [04](04_rnn_lstm_gru/). The LSTM cell state, residual
connections and ReLU are three answers to one problem: keep a path along which the gradient is
multiplied by something close to 1.

---

## Honest results kept rather than tuned away

| topic | expected | measured |
|---|---|---|
| [04](04_rnn_lstm_gru/) | LSTM remembers longer than SimpleRNN | SimpleRNN **best at every length** (1.000 / 1.000 / 0.955); LSTM erratic |
| [03](03_cnn_images/) | augmentation improves accuracy | −0.064 test accuracy — though the overfitting gap shrank (+0.021 → +0.008) |
| [06](06_embeddings/) | embeddings save parameters | at dim 32 they cost **more** than one-hot |
| [06](06_embeddings/) | a U-shaped dimension curve | error rose **monotonically**; the smallest size won |
| [08](08_wide_and_deep/) | joint beats both halves | it lost to its own deep half, and to XGBoost |
| [09](09_two_tower_retrieval/) | learned retrieval beats popularity | popularity won on recall@10 |

Each is explained in place. In most cases the explanation is the actual lesson of the section.

---

Next: [`04_frameworks`](../04_frameworks/) — the same model written three ways, in TensorFlow,
PyTorch and JAX.
