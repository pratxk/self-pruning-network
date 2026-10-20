"""Score saved models on the test set without retraining.
uv run python -m train.run_eval [model.npz ...]"""
import argparse
from pathlib import Path

import numpy as np

from train.data import load_fashion_mnist
from train.loop import load_layers

DEFAULT = ["results/part2/model.npz", "results/part3/model.npz"]


def accuracy(layers: list, X: np.ndarray, y: np.ndarray) -> float:
    h = X
    for i, (W, b, mask) in enumerate(layers):
        h = h @ (W * mask) + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return float((h.argmax(1) == y).mean())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*", default=DEFAULT)
    args = ap.parse_args(argv)
    d = load_fashion_mnist()
    for path in args.models:
        if not Path(path).exists():
            print(f"{path}: not found (models aren't committed; run `make train` / `make prune` first)")
            continue
        layers = load_layers(path)
        active = sum(int(m.sum()) for _, _, m in layers)
        total = sum(m.size for _, _, m in layers)
        print(f"{path}: test_acc={accuracy(layers, d.X_test, d.y_test):.4f}  "
              f"sparsity={1 - active / total:.4f}  active_weights={active:,}/{total:,}")


if __name__ == "__main__":
    main()
