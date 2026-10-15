"""Part 4: sparsity x method x seed sweep -> results/part4/sweep.csv (resumable:
finished (seed, method, target, config_hash) rows are skipped on restart).
uv run python -m train.run_sweep            # full grid, 5-8 h single process
uv run python -m train.run_sweep --quick    # 1 seed, 3 sparsities, ~20 min
uv run python -m train.run_sweep --jobs 4   # parallel processes (BLAS stays 1 thread each)"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from engine.sparse import count_flops
from engine.tensor import Tensor, no_grad
from train.config import RunConfig
from train.data import load_fashion_mnist
from train.loop import fit, save_model

TARGETS = (0.5, 0.75, 0.9, 0.95, 0.98, 0.99)
QUICK_TARGETS = (0.9, 0.95, 0.98)
CLAIM_TARGET = 0.95
SEEDS = range(5)
CLAIM_SEEDS = range(10)  # 5 seeds cannot reach p < 0.05 (see train/stats.py)
# Confirmatory test of the 99% gap seen in the exploratory sweep, on fresh seeds
# (the sweep used 0-4 at 99%), pre-registered in README before running.
CONFIRM_TARGET = 0.99
CONFIRM_METHODS = ("magnitude", "fisher_obd")
CONFIRM_SEEDS = range(5, 15)
# label -> config overrides
METHODS = {
    "magnitude": dict(method="magnitude"),
    "magnitude_layer": dict(method="magnitude_layer"),
    "taylor_fo": dict(method="taylor_fo"),
    "fisher_obd": dict(method="fisher_obd"),
    "fisher_obd_fo": dict(method="fisher_obd_fo"),
    "fisher_obd_regrow": dict(method="fisher_obd", regrow=True),
    "magnitude_oneshot": dict(method="magnitude", schedule="oneshot"),
}
COLUMNS = ["seed", "method", "schedule", "regrow", "target", "achieved", "n_active", "flops",
           "val_acc", "test_acc", "train_time_s", "churn", "config_hash"]


def grid(quick: bool, confirm: bool = False) -> list[tuple[str, RunConfig]]:
    if confirm:
        return [(l, RunConfig(seed=s, target_sparsity=CONFIRM_TARGET, **METHODS[l]))
                for l in CONFIRM_METHODS for s in CONFIRM_SEEDS]
    seeds, targets = (range(1), QUICK_TARGETS) if quick else (SEEDS, TARGETS)
    jobs = [("dense", RunConfig(seed=s)) for s in seeds]
    for t in targets:
        for label, kw in METHODS.items():
            run_seeds = CLAIM_SEEDS if (not quick and t == CLAIM_TARGET
                                        and label in ("magnitude", "fisher_obd")) else seeds
            jobs += [(label, RunConfig(seed=s, target_sparsity=t, **kw)) for s in run_seeds]
    return jobs


def _key(seed, label, target, h) -> tuple:
    return int(seed), label, round(float(target), 6), h


def run_one(label: str, cfg: RunConfig, out: str) -> dict:
    data = load_fashion_mnist()
    res = fit(cfg, data, log=lambda s: None)
    layers = [(l.W.data * l.mask, l.b.data) for l in res.model.linears()]
    run_dir = Path(out) / "runs" / label / f"t{cfg.target_sparsity}" / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with no_grad():
        preds = res.model(Tensor(data.X_test)).data.argmax(axis=1).astype(np.int8)
    np.save(run_dir / "test_preds.npy", preds)  # for McNemar
    if cfg.seed == 0:
        save_model(run_dir / "model.npz", res.model)  # for the cost benchmark
    return {"seed": cfg.seed, "method": label, "schedule": cfg.schedule,
            "regrow": cfg.regrow, "target": cfg.target_sparsity,
            "achieved": res.pruner.sparsity() if res.pruner else 0.0,
            "n_active": res.curve[-1]["n_active"], "flops": count_flops(layers),
            "val_acc": res.curve[-1]["val_acc"], "test_acc": res.test_acc,
            "train_time_s": res.train_time_s,
            "churn": res.regrower.churn if res.regrower else "", "config_hash": cfg.hash()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--confirm", action="store_true",
                    help="confirmatory 99%% claim: fresh seeds 5-14 -> results/part4_confirm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out or ("results/part4_quick" if args.quick else
                            "results/part4_confirm" if args.confirm else "results/part4"))
    out.mkdir(parents=True, exist_ok=True)
    sweep_csv = out / "sweep.csv"

    done = set()
    if sweep_csv.exists():
        with sweep_csv.open() as f:
            done = {_key(r["seed"], r["method"], r["target"], r["config_hash"])
                    for r in csv.DictReader(f)}
    todo = [(l, c) for l, c in grid(args.quick, args.confirm)
            if _key(c.seed, l, c.target_sparsity, c.hash()) not in done]
    print(f"{len(done)} runs done, {len(todo)} to go -> {sweep_csv}")
    load_fashion_mnist()  # download once before workers start

    new_file = not sweep_csv.exists()
    with sweep_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_one, l, c, str(out)): (l, c) for l, c in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                writer.writerow(row)
                f.flush()  # a crash loses nothing that finished
                print(f"[{i}/{len(todo)}] {row['method']} t={row['target']} "
                      f"seed={row['seed']} test_acc={row['test_acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
