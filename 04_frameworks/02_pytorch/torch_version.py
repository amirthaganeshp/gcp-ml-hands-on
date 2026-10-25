"""The shared CNN in PyTorch - where you write the training loop yourself.

PyTorch's defining choice is that there is no `fit`. You write the loop, and
in exchange you can see and change every step of it. Everything Keras did
implicitly in 04_frameworks/01_tensorflow is explicit here, and the value of
this example is in noticing exactly which things those are.

  1. The model as a class - shapes declared, not inferred
  2. Dataset and DataLoader - the input pipeline
  3. THE LOOP: zero_grad, forward, loss, backward, step - and what each
     line is actually for
  4. train() vs eval(), and the bug that mode-switching prevents
  5. state_dict - saving weights rather than models, and why

Run:
    python 04_frameworks/02_pytorch/torch_version.py

Concept reference: Foundations volume, Part C.2.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from common import (BATCH, EPOCHS, LR, SEED, banner, check_params,  # noqa: E402
                    header, load_data, record, source_lines)
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

ARTIFACTS = Path(__file__).parent / "artifacts"


# ============================================================= 1. the model
class CNN(nn.Module):
    """The shared architecture, PyTorch style.

    Note what has to be written down that Keras inferred: the number of
    INPUT channels to every layer, and the 32*7*7 flattened size. PyTorch
    builds the module eagerly at construction, before it has seen any data,
    so it cannot work those out for you.

    That is the trade. You do the shape arithmetic; in return the module is
    an ordinary Python object you can print, subclass, and step through in a
    debugger.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),   # 32*7*7 = 1568, by hand
            nn.Linear(64, 10),                       # logits, NOT softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def demo_model(model: nn.Module) -> None:
    check_params(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("\n  Two things Keras did for you that are explicit here:")
    print("    - every layer's INPUT size: Conv2d(1, 16, ...) then")
    print("      Conv2d(16, 32, ...) then Linear(32*7*7, 64). Get the")
    print("      arithmetic wrong and it fails at the first forward pass.")
    print("    - the output is LOGITS, not softmax. CrossEntropyLoss applies")
    print("      log_softmax internally, so adding a softmax layer as well is")
    print("      a real and common bug: it trains, badly, and silently.")
    print("\n  In exchange the model is an ordinary Python object. You can")
    print("  print it, subclass it, and set a breakpoint inside forward().")


# ============================================== 2. Dataset and DataLoader
def build_loaders(d: dict) -> tuple[DataLoader, DataLoader]:
    """PyTorch expects channels FIRST - (N, C, H, W) - where TensorFlow and
    the shared arrays use channels last, (N, H, W, C). Transposing here is
    the single most common source of confusion when moving between them.
    """
    def to_tensor(x: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(x.transpose(0, 3, 1, 2)).float(),   # NHWC -> NCHW
            torch.from_numpy(y).long(),
        )

    g = torch.Generator().manual_seed(SEED)
    train = DataLoader(to_tensor(d["x_train"], d["y_train"]),
                       batch_size=BATCH, shuffle=True, generator=g)
    test = DataLoader(to_tensor(d["x_test"], d["y_test"]), batch_size=BATCH)
    return train, test


def demo_loaders(train: DataLoader, test: DataLoader) -> None:
    xb, yb = next(iter(train))
    print(f"  One batch: x {tuple(xb.shape)} {xb.dtype}, "
          f"y {tuple(yb.shape)} {yb.dtype}")
    print(f"  Batches per epoch: {len(train)} train, {len(test)} test")
    print("\n  Note the shape: (N, C, H, W), channels FIRST. TensorFlow and")
    print("  the shared arrays are channels LAST, (N, H, W, C). The transpose")
    print("  in build_loaders is doing that conversion, and forgetting it is")
    print("  the classic error when porting a model between the two.")
    print("\n  Dataset says how to get ONE example; DataLoader handles")
    print("  batching, shuffling and (with num_workers) parallel loading.")
    print("  The split is clean: to use your own data you implement two")
    print("  methods, __len__ and __getitem__, and everything else is reused.")


# ==================================================== 3. the training loop
def train_model(model: nn.Module, train: DataLoader, test: DataLoader) -> tuple:
    """The loop Keras writes for you.

    Five lines do the work, and each one is worth being able to recite:

        optimizer.zero_grad()   gradients ACCUMULATE in PyTorch. Forget this
                                and you are summing every batch seen so far.
        out = model(xb)         forward pass
        loss = criterion(...)   scalar to differentiate
        loss.backward()         autograd fills every .grad
        optimizer.step()        apply the update

    Early stopping, checkpointing and LR scheduling are things you now write.
    That is the cost side of the trade.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc, best_state, patience, bad = 0.0, None, 3, 0
    history = []

    for epoch in range(EPOCHS):
        model.train()                      # dropout/batchnorm into train mode
        running = 0.0
        for xb, yb in train:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        train_loss = running / len(train.dataset)

        model.eval()                       # and back into eval mode
        correct, total, vloss = 0, 0, 0.0
        with torch.no_grad():              # no graph needed for evaluation
            for xb, yb in test:
                out = model(xb)
                vloss += criterion(out, yb).item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(yb)
        acc, vloss = correct / total, vloss / total
        history.append((train_loss, vloss, acc))

        # early stopping and "restore best weights", by hand
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  Early stop at epoch {epoch + 1} "
                      f"(no improvement for {patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc, history


# ================================================== 4. train() vs eval()
def demo_modes(model: nn.Module, test: DataLoader) -> None:
    """A model with dropout or batch norm behaves DIFFERENTLY in the two
    modes, and PyTorch will not switch for you.
    """
    probe = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))
    x = torch.ones(1, 4)

    torch.manual_seed(SEED)
    probe.train()
    train_runs = [probe(x).detach().numpy().copy() for _ in range(3)]
    probe.eval()
    eval_runs = [probe(x).detach().numpy().copy() for _ in range(3)]

    tr_same = bool(np.allclose(train_runs[0], train_runs[1]))
    ev_same = bool(np.allclose(eval_runs[0], eval_runs[1]))
    print(f"  Same input, three forward passes through a net with dropout:")
    print(f"    .train() mode - identical every time?  {tr_same}")
    print(f"    .eval()  mode - identical every time?  {ev_same}")
    print("\n  In train mode dropout is active and random, so the output")
    print("  changes run to run. In eval mode it is off and deterministic.")
    print("  Batch norm does the same thing differently: batch statistics in")
    print("  train, running averages in eval.")
    print("\n  Nothing switches these for you. Evaluating without .eval() is")
    print("  one of the most common PyTorch bugs, and it does not raise - it")
    print("  just quietly reports worse numbers than the model deserves.")
    print("  torch.no_grad() is the separate concern: it stops building the")
    print("  autograd graph, saving memory and time. You want both.")


# ========================================================= 5. state_dict
def demo_saving(model: nn.Module, d: dict) -> None:
    """PyTorch saves WEIGHTS, not models.

    A state_dict is an ordinary dict of tensors. To load it you must first
    construct the model class - so the Python source is part of the artefact,
    which is the sharpest contrast with TensorFlow's SavedModel.
    """
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / "model.pt"
    torch.save(model.state_dict(), path)

    fresh = CNN()                          # the class definition is REQUIRED
    fresh.load_state_dict(torch.load(path, weights_only=True))
    fresh.eval()
    model.eval()

    x = torch.from_numpy(d["x_test"][:64].transpose(0, 3, 1, 2)).float()
    with torch.no_grad():
        same = bool(torch.allclose(model(x), fresh(x), atol=1e-6))
    print(f"  Saved {path.name}: {path.stat().st_size / 1024:,.0f} KB "
          f"(weights only)")
    print(f"  Reloaded into a fresh CNN(); predictions identical: {same}")

    keys = list(model.state_dict().keys())
    print(f"\n  A state_dict is just a dict of tensors - {len(keys)} entries here:")
    for k in keys[:4]:
        print(f"    {k:<28} {tuple(model.state_dict()[k].shape)}")
    print(f"    ... and {len(keys) - 4} more")

    print("\n  This is the sharpest contrast with TensorFlow. A SavedModel")
    print("  carries the graph, so a serving process needs no Python. A")
    print("  state_dict is weights alone, so the CLASS DEFINITION IS PART OF")
    print("  THE ARTEFACT - deploy the weights without the matching source and")
    print("  you have nothing. For serving, PyTorch's answers are TorchScript")
    print("  or ONNX export, which is an extra step TensorFlow does not need.")
    print("\n  Note `weights_only=True` on load. torch.load used to unpickle")
    print("  arbitrary objects, which is remote code execution if you load an")
    print("  untrusted checkpoint. It is the default in recent versions and")
    print("  worth setting explicitly regardless.")


def main() -> None:
    header("PyTorch", torch.__version__)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    d = load_data()

    banner("1. The model as a class")
    model = CNN()
    demo_model(model)

    banner("2. Dataset and DataLoader")
    train, test = build_loaders(d)
    demo_loaders(train, test)

    banner("3. The training loop, written out")
    t0 = time.perf_counter()
    acc, history = train_model(model, train, test)
    secs = time.perf_counter() - t0
    print(f"  Ran {len(history)} of {EPOCHS} epochs in {secs:.1f}s")
    print(f"  Best test accuracy: {acc:.4f}")
    print("\n  Everything in that function is something Keras did in one call:")
    print("  the epoch loop, the batch loop, zero_grad, backward, step, the")
    print("  evaluation pass, early stopping and restore-best-weights. About")
    print("  35 lines against `fit(...)` plus three callbacks.")
    print("\n  The upside is that all of it is yours. A custom loss on")
    print("  intermediate activations, two optimisers on different parameter")
    print("  groups, gradient surgery between backward and step - all of it is")
    print("  just editing the loop, with no framework to fight.")

    banner("4. train() vs eval()")
    demo_modes(model, test)

    banner("5. state_dict - saving weights, not models")
    demo_saving(model, d)

    banner("What PyTorch gives you, and what it costs")
    print("  + The loop is explicit, so nothing is hidden and anything can be")
    print("    changed. Debugging is ordinary Python debugging.")
    print("  + Models are plain objects: printable, subclassable, steppable.")
    print("  + Dataset/DataLoader is a clean split - implement __len__ and")
    print("    __getitem__ and get batching, shuffling and workers free.")
    print("\n  - You write the bookkeeping. Early stopping, checkpointing and")
    print("    LR scheduling are yours, and they are easy to get subtly wrong.")
    print("  - zero_grad() is mandatory and silent if forgotten: gradients")
    print("    accumulate, so the model trains on a growing sum.")
    print("  - train()/eval() will not switch themselves, and forgetting")
    print("    costs accuracy without raising anything.")
    print("  - Channels-first, against everyone else's channels-last.")
    print("  - Serving needs TorchScript or ONNX - a step TF does not have.")

    record("PyTorch", acc, secs,
           sum(p.numel() for p in model.parameters()),
           source_lines(__file__),
           "explicit loop; Dataset/DataLoader; state_dict")


if __name__ == "__main__":
    main()
