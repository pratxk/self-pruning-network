"""Part 2: dense LeNet-300-100 on Fashion-MNIST with Adam.
uv run python -m train.run_dense --seed 0"""
import argparse
import json
from pathlib import Path

import numpy as np

from engine.functional import softmax_cross_entropy
from engine.tensor import Tensor, no_grad
from nn.layers import MaskedLinear
from train import plots
from train.config import RunConfig
from train.data import load_fashion_mnist
from train.loop import build, fit, write_run
from train.repro import make_rng


def init_stats(cfg: RunConfig, X: np.ndarray, y: np.ndarray) -> dict:
    """Per-layer output std at init (same seed as the run) and the initial loss."""
    model, *_ = build(cfg, len(X), make_rng(cfg.seed))
    stds, rms, h = [], [], Tensor(X)
    with no_grad():
        for layer in model.layers:
            h = layer(h)
            if not isinstance(layer, MaskedLinear) or layer is model.linears()[-1]:
                stds.append(float(h.data.std()))
                rms.append(float(np.sqrt((h.data**2).mean())))  # what He init preserves
        loss = float(softmax_cross_entropy(h, y).data)
    return {"input_std": float(X.std()), "act_std": stds, "act_rms": rms,
            "initial_loss": loss, "ln_10": float(np.log(10))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results/part2")
    args = ap.parse_args(argv)

    cfg = RunConfig(seed=args.seed, epochs=args.epochs)
    out = Path(args.out)
    data = load_fashion_mnist()
    stats = init_stats(cfg, data.X_train[:1000], data.y_train[:1000])
    out.mkdir(parents=True, exist_ok=True)
    (out / "init_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"init: {stats}")

    res = fit(cfg, data, out)
    summary = write_run(res, out)
    plots.learning_curve(res.curve, out / "learning_curve.png")
    plots.init_activations(stats, out / "init_activations.png")
    print(f"test_acc={summary['test_acc']:.4f}  -> {out}")


if __name__ == "__main__":
    main()
