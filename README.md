# Self-Pruning Network

A from-scratch NumPy autodiff engine trains LeNet-300-100 on Fashion-MNIST and
prunes itself during training. Weights are ranked by Optimal Brain Damage
saliency `½ v̂ w²`, with the diagonal-Fisher curvature read from Adam's second
moment, under a cubic sparsity schedule, with optional RigL-style regrowth. Masking is provably
correct: pruned weights stay exactly zero, and revived weights behave like
fresh Adam parameters.

**Headline result:** _filled from `results/part4/claim.json` after the sweep (pending)._

> scikit-learn is used only to download Fashion-MNIST via `fetch_openml`
> (`train/data.py`). All training, gradients, models and pruning are
> implemented here in NumPy. `tests/test_no_banned_imports.py` enforces this.

## Pre-registered claim

Written and committed before the Part 4 sweep runs. `train/run_report.py`
fills in the numbers and sets `"supported"` in `results/part4/claim.json`.

> At 95% sparsity on Fashion-MNIST with LeNet-300-100, Fisher-OBD pruning
> retains X% test accuracy versus Y% for global magnitude pruning, a paired gain
> of Δ pp (95% CI [a, b]) over 10 seeds.

The claim is supported only if the CI excludes 0 in Fisher-OBD's favour. If not,
the README reports that, and at which sparsity a real gap does appear.

## Install

```bash
uv sync                          # Python >= 3.12, deps pinned in uv.lock
# fallback: pip install -r requirements.txt pytest
```

## Commands

| Purpose | Command | Runtime (est.) | Writes |
| --- | --- | --- | --- |
| Install | `uv sync` | 1 min | `.venv/` |
| Gradient-check tests | `uv run pytest tests/test_gradcheck.py -v` | < 30 s | — |
| All tests | `uv run pytest -v` | < 2 min | — |
| Part 2 dense training | `uv run python -m train.run_dense --seed 0` | ~1 min | `results/part2/` |
| Part 3 self-pruning run | `uv run python -m train.run_prune --method fisher_obd --sparsity 0.90 --seed 0` | ~1 min | `results/part3/` |
| Profile (DESIGN §3) | `uv run python -m train.run_profile` | < 30 s | `results/profile/` |
| Part 4 Pareto sweep | `uv run python -m train.run_sweep [--jobs N]` | ~3 h at 1 job | `results/part4/` |
| Part 4 quick check | `uv run python -m train.run_sweep --quick` | ~20 min | `results/part4_quick/` |
| Cost benchmark | `uv run python -m train.run_bench` | ~5 min | `results/part4/bench.csv` |
| Stats + plots | `uv run python -m train.run_report` | < 1 min | `claim.json`, `results/figures/` |

The dataset downloads on first run to `data/` (git-ignored) and is verified by
SHA-256. Quick mode writes to its own folder, so it never overwrites committed
numbers.

## Results

_Pending the full sweep: learning curve, Pareto plot, cost table (active
params, FLOPs, latency per path) and stats table, each linked to its CSV._

## Honest limitations

- **CSR crossover:** unstructured CSR loses to dense BLAS until high sparsity.
  The dense-masked path is labelled as *not a real speedup* wherever latency is
  reported.
- **Scope:** one dataset and one architecture.
- **Diagonal Fisher:** ignores interactions between weights removed together,
  mitigated by gradual pruning.
- **Timing:** CPU only, single BLAS thread, one machine.
- **Initial loss** at init is about 3.06, not ln 10 = 2.303. With Glorot output
  init, the logit std is about 1.24 (`results/part2/init_stats.json`).
- **The criterion changed from the original spec.** The planned
  `½ v̂ w² − m̂ w` underperformed magnitude by about 5 pp at 90% sparsity. The
  causes are a mis-scaled curvature (Adam's `v` is the batch-mean gradient
  squared, ≈ B× below per-example Fisher) and an invalid first-order
  extrapolation mid-training. It is kept as the ablation `fisher_obd_fo`
  (DESIGN.md §1).

## Repo map

```
engine/  autodiff Tensor, ops, fused softmax-CE, gradcheck, CSR + compaction
nn/      Module, MaskedLinear, MLP, He/Glorot init, mask-aware Adam / SGD
prune/   criteria (magnitude, Taylor-FO, Fisher-OBD, Fisher-OBD+FO), cubic schedule, Pruner, RigL regrowth
train/   data, config, loop, entry points, paired-seed stats
tests/   gradcheck, engine, masked weights, pruning, sparse, stats, repro, import guard
```

Derivations and design answers: [DESIGN.md](DESIGN.md).
