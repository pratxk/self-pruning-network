"""Part 4 stats, claim.json and every figure, from committed CSVs only.
uv run python -m train.run_report [--sweep-dir results/part4_quick]"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from train import plots
from train.loop import load_layers, write_csv
from train.run_sweep import CLAIM_TARGET
from train.stats import bootstrap_ci, holm, mcnemar_exact, sign_flip_p

plt = plots.plt
BASELINE = "magnitude"
CLAIM_METHOD = "fisher_obd"


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def acc_by(rows) -> dict:
    """(method, target) -> {seed: test_acc}"""
    out = defaultdict(dict)
    for r in rows:
        out[r["method"], float(r["target"])][int(r["seed"])] = float(r["test_acc"])
    return out


def paired_stats(acc, targets) -> list[dict]:
    rows = []
    for method in sorted({m for m, _ in acc} - {BASELINE, "dense"}):
        family = []
        for t in targets:
            a, b = acc.get((method, t), {}), acc.get((BASELINE, t), {})
            seeds = sorted(a.keys() & b.keys())
            if len(seeds) < 2:
                continue
            d = np.array([a[s] - b[s] for s in seeds])
            lo, hi = bootstrap_ci(d)
            family.append({"target": t, "method_a": method, "method_b": BASELINE,
                           "n_seeds": len(seeds), "mean_a": np.mean([a[s] for s in seeds]),
                           "mean_b": np.mean([b[s] for s in seeds]), "mean_diff": d.mean(),
                           "ci_low": lo, "ci_high": hi, "p_perm": sign_flip_p(d)})
        for row, ph in zip(family, holm([r["p_perm"] for r in family])):
            row["p_holm"] = ph
        rows += family
    return rows


def noise_floor(acc) -> dict:
    dense = list(acc.get(("dense", 0.0), {}).values())
    p = float(np.mean(dense)) if dense else float("nan")
    return {"dense_mean": p, "dense_std": float(np.std(dense, ddof=1)) if len(dense) > 1 else None,
            "binomial_se": float(np.sqrt(p * (1 - p) / 10_000))}


def claim(stats_rows, runs_dir: Path) -> dict:
    row = next((r for r in stats_rows if r["method_a"] == CLAIM_METHOD
                and r["target"] == CLAIM_TARGET), None)
    if row is None:
        return {"supported": None, "reason": f"no paired runs at {CLAIM_TARGET}"}
    x, y = 100 * row["mean_a"], 100 * row["mean_b"]
    d, lo, hi = 100 * row["mean_diff"], 100 * row["ci_low"], 100 * row["ci_high"]
    out = {
        "claim": (f"At {CLAIM_TARGET:.0%} sparsity on Fashion-MNIST with LeNet-300-100, "
                  f"Fisher-OBD pruning retains {x:.2f}% test accuracy versus {y:.2f}% for "
                  f"global magnitude pruning, a paired gain of {d:+.2f} pp "
                  f"(95% CI [{lo:+.2f}, {hi:+.2f}]) over {row['n_seeds']} seeds."),
        "target": CLAIM_TARGET, "fisher_obd_acc": x, "magnitude_acc": y, "delta_pp": d,
        "ci_pp": [lo, hi], "p_perm": row["p_perm"], "p_holm": row["p_holm"],
        "n_seeds": row["n_seeds"], "supported": bool(lo > 0),
    }
    pa = runs_dir / CLAIM_METHOD / f"t{CLAIM_TARGET}" / "seed0" / "test_preds.npy"
    pb = runs_dir / BASELINE / f"t{CLAIM_TARGET}" / "seed0" / "test_preds.npy"
    if pa.exists() and pb.exists():
        from train.data import load_fashion_mnist
        y_test = load_fashion_mnist().y_test
        b, c, p = mcnemar_exact(np.load(pa) == y_test, np.load(pb) == y_test)
        out["mcnemar_seed0"] = {"fisher_only_right": b, "magnitude_only_right": c, "p": p}
    return out


# --- figures ----------------------------------------------------------------

def fig_pareto(acc, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in sorted({m for m, _ in acc} - {"dense"}):
        ts = sorted(t for m, t in acc if m == method)
        mean = [100 * np.mean(list(acc[method, t].values())) for t in ts]
        std = [100 * np.std(list(acc[method, t].values())) for t in ts]
        ax.errorbar([1 - t for t in ts], mean, yerr=std, marker="o", capsize=3, label=method)
    if ("dense", 0.0) in acc:
        ax.axhline(100 * np.mean(list(acc["dense", 0.0].values())), color="k", ls="--",
                   lw=1, label="dense")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set(xlabel="density (1 - sparsity), log scale", ylabel="test accuracy (%)",
           title="Accuracy vs sparsity (mean ± std over seeds)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_latency(bench: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, batch in zip(axes, ("1", "256")):
        for name in ("dense_masked", "csr", "compacted"):
            pts = sorted((float(r["target"]), float(r["median_ms"])) for r in bench
                         if r["batch"] == batch and r["path"] == name and r["method"] in
                         ("fisher_obd", "dense"))
            if pts:
                label = name + (" (not a real saving)" if name == "dense_masked" else "")
                ax.plot(*zip(*pts), marker="o", label=label)
        ax.set(xlabel="sparsity", ylabel="median latency (ms, log)", title=f"batch {batch}",
               yscale="log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Inference latency, Fisher-OBD models, 1 BLAS thread")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_layer_density(runs_dir: Path, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in sorted(runs_dir.glob(f"*/t{CLAIM_TARGET}/seed0/model.npz")):
        dens = [m.mean() for *_, m in load_layers(model)]
        ax.plot(range(1, len(dens) + 1), dens, marker="o", label=model.parts[-4])
    ax.set_xticks([1, 2, 3], ["784→300", "300→100", "100→10"])
    ax.set(xlabel="layer", ylabel="density", yscale="log",
           title=f"Per-layer density at {CLAIM_TARGET:.0%} sparsity (seed 0)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_schedule(events: list[dict], path: Path) -> None:
    steps = [int(e["step"]) for e in events]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [float(e["target"]) for e in events], label="cubic target")
    ax.plot(steps, [float(e["achieved"]) for e in events], ".", ms=3, label="achieved")
    ax.set(xlabel="training step", ylabel="sparsity", title="Pruning schedule (Part 3 run)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", default="results/part4")
    ap.add_argument("--figures", default=None,
                    help="default: results/figures, or <sweep-dir>/figures for other sweeps")
    args = ap.parse_args(argv)
    sweep_dir = Path(args.sweep_dir)
    figs = Path(args.figures or ("results/figures" if sweep_dir == Path("results/part4")
                                 else sweep_dir / "figures"))
    figs.mkdir(parents=True, exist_ok=True)

    part2 = Path("results/part2")
    if (part2 / "curve.csv").exists():
        rows = [{k: float(v) for k, v in r.items()} for r in read_csv(part2 / "curve.csv")]
        plots.learning_curve(rows, figs / "learning_curve.png")
    if (part2 / "init_stats.json").exists():
        plots.init_activations(json.loads((part2 / "init_stats.json").read_text()),
                               figs / "init_activations.png")
    if Path("results/part3/prune_events.csv").exists():
        fig_schedule(read_csv(Path("results/part3/prune_events.csv")),
                     figs / "sparsity_schedule.png")

    if not (sweep_dir / "sweep.csv").exists():
        print(f"no {sweep_dir}/sweep.csv yet: only Part 2/3 figures written")
        return
    acc = acc_by(read_csv(sweep_dir / "sweep.csv"))
    targets = sorted({t for _, t in acc if t > 0})
    stats_rows = paired_stats(acc, targets)
    if stats_rows:
        write_csv(sweep_dir / "stats.csv", stats_rows)
    result = {**claim(stats_rows, sweep_dir / "runs"), "noise_floor": noise_floor(acc)}
    (sweep_dir / "claim.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    fig_pareto(acc, figs / "pareto.png")
    fig_layer_density(sweep_dir / "runs", figs / "layer_density.png")
    if (sweep_dir / "bench.csv").exists():
        fig_latency(read_csv(sweep_dir / "bench.csv"), figs / "latency_vs_sparsity.png")
    print(f"figures -> {figs}")


if __name__ == "__main__":
    main()
