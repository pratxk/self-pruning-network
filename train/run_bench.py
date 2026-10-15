"""Part 4 cost benchmark: three inference paths per saved seed-0 model, batch 1
and 256, 20 warmups + 200 timed runs, median and p90. All paths must give the
same logits (rtol 1e-5) or the benchmark aborts.
  dense_masked: pre-masked W (zeros stored), dense matmul. Reference only: NOT a
                real saving, BLAS still multiplies every zero.
  csr:          own CSR forward (engine.sparse.sparse_linear).
  compacted:    dead hidden units removed, smaller dense matrices.
uv run python -m train.run_bench"""
import argparse
import platform
import time
from pathlib import Path

import numpy as np

from engine.sparse import compact_mlp, count_flops, csr_from_dense, sparse_linear
from train.data import load_fashion_mnist
from train.loop import load_layers, write_csv

WARMUP, RUNS, BATCHES = 20, 200, (1, 256)


def _relu(z):
    return np.maximum(z, 0)


def dense_forward(layers, x):
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = _relu(x)
    return x


def make_paths(raw):
    masked = [(W * m, b) for W, b, m in raw]
    csr = [(csr_from_dense(W, m), b) for W, b, m in raw]
    compacted = compact_mlp(masked)

    def dense_masked(x):  # zeros stored densely, mask pre-applied (no per-call multiply)
        return dense_forward(masked, x)

    def csr_path(x):
        for i, (C, b) in enumerate(csr):
            x = sparse_linear(x, C, b)
            if i < len(csr) - 1:
                x = _relu(x)
        return x

    paths = {"dense_masked": dense_masked, "csr": csr_path,
             "compacted": lambda x: dense_forward(compacted, x)}
    flops = {"dense_masked": count_flops([(np.ones_like(W), b) for W, b, _ in raw]),
             "csr": count_flops(masked), "compacted": count_flops(compacted)}
    return paths, flops, sum(int(m.sum()) for *_, m in raw)


def time_ms(fn, x) -> tuple[float, float]:
    for _ in range(WARMUP):
        fn(x)
    t = np.empty(RUNS)
    for i in range(RUNS):
        t0 = time.perf_counter_ns()
        fn(x)
        t[i] = time.perf_counter_ns() - t0
    t /= 1e6
    return float(np.median(t)), float(np.quantile(t, 0.9))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/part4/runs")
    ap.add_argument("--out", default="results/part4/bench.csv")
    args = ap.parse_args(argv)

    X = load_fashion_mnist().X_test
    cpu = platform.processor() or platform.machine()
    rows = []
    for model_path in sorted(Path(args.runs).glob("*/t*/seed0/model.npz")):
        method, target = model_path.parts[-4], float(model_path.parts[-3][1:])
        paths, flops, n_active = make_paths(load_layers(model_path))
        ref = paths["dense_masked"](X[:256])
        for name, fn in paths.items():
            if not np.allclose(fn(X[:256]), ref, rtol=1e-5, atol=1e-5):
                raise AssertionError(f"{name} logits differ for {model_path}")
            for batch in BATCHES:
                med, p90 = time_ms(fn, X[:batch])
                rows.append({"method": method, "target": target, "path": name, "batch": batch,
                             "median_ms": med, "p90_ms": p90, "flops": flops[name],
                             "n_active": n_active, "cpu": cpu, "numpy_version": np.__version__})
        print(f"{method} t={target}: " + ", ".join(
            f"{r['path']}@{r['batch']}={r['median_ms']:.3f}ms" for r in rows[-6:]), flush=True)
    if not rows:
        raise SystemExit(f"no seed-0 models under {args.runs}: run train.run_sweep first")
    write_csv(Path(args.out), rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
