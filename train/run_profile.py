"""DESIGN.md section 3 evidence: cProfile top-10 over one dense epoch plus a
timed Adam step on LeNet-300-100 parameters.
uv run python -m train.run_profile --tag after"""
import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np

from nn.layers import MLP
from nn.optim import Adam, masks_of
from train.config import RunConfig
from train.data import load_fashion_mnist
from train.loop import fit


def adam_step_ms(n: int = 200) -> float:
    rng = np.random.default_rng(0)
    model = MLP(rng=rng)
    opt = Adam(model.parameters(), masks_of(model))
    for p in model.parameters():
        p.grad = rng.standard_normal(p.data.shape).astype(np.float32)
    for _ in range(10):
        opt.step()
    t0 = time.perf_counter()
    for _ in range(n):
        opt.step()
    return (time.perf_counter() - t0) / n * 1e3


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="after")
    ap.add_argument("--out", default="results/profile")
    args = ap.parse_args(argv)

    data = load_fashion_mnist()
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.runcall(fit, RunConfig(epochs=1), data, log=lambda s: None)
    epoch_s = time.perf_counter() - t0
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(10)
    step_ms = adam_step_ms()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = (f"epoch_s(profiled)={epoch_s:.3f}\nadam_step_ms={step_ms:.3f}\n"
              f"numpy={np.__version__}\n\n{buf.getvalue()}")
    (out / f"profile_{args.tag}.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
