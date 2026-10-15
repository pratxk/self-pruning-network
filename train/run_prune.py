"""Part 3: one self-pruning run.
uv run python -m train.run_prune --method fisher_obd --sparsity 0.90 --seed 0"""
import argparse
from pathlib import Path

from train.config import METHODS, RunConfig
from train.data import load_fashion_mnist
from train.loop import fit, write_run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="fisher_obd", choices=[m for m in METHODS if m != "none"])
    ap.add_argument("--sparsity", type=float, default=0.90)
    ap.add_argument("--schedule", default="cubic", choices=["cubic", "oneshot"])
    ap.add_argument("--regrow", action="store_true", help="RigL-style regrowth (Tier 3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results/part3")
    args = ap.parse_args(argv)

    cfg = RunConfig(seed=args.seed, epochs=args.epochs, method=args.method,
                    schedule=args.schedule, target_sparsity=args.sparsity, regrow=args.regrow)
    out = Path(args.out)
    res = fit(cfg, load_fashion_mnist(), out)
    s = write_run(res, out)
    print(f"target={s['target_sparsity']:.4f} achieved={s['achieved_sparsity']:.6f} "
          f"test_acc={s['test_acc']:.4f}  -> {out}")


if __name__ == "__main__":
    main()
