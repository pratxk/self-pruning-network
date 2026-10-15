"""Training loop shared by every entry point: dense and pruned runs are the
same code path (dense = no pruner, all-ones masks)."""
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from engine.functional import accuracy, softmax_cross_entropy
from engine.tensor import Tensor, no_grad
from nn.layers import MLP
from nn.optim import Adam, SGDMomentum, masks_of
from prune.criteria import FisherOBD, FisherOBDFirstOrder, Magnitude, TaylorFirstOrder
from prune.pruner import PruneEvent, Pruner
from prune.regrow import RegrowEvent, Regrower
from prune.schedule import CubicSchedule, OneShotSchedule
from train.config import RunConfig
from train.data import Splits
from train.repro import make_rng

LR_DECAY = 0.1
REGROW_STOP_EPOCH = 15


class NonFiniteError(RuntimeError):
    pass


@dataclass
class FitResult:
    cfg: RunConfig
    model: MLP
    pruner: Pruner | None
    curve: list[dict] = field(default_factory=list)
    events: list[PruneEvent] = field(default_factory=list)
    regrow_events: list[RegrowEvent] = field(default_factory=list)
    regrower: Regrower | None = None
    test_loss: float = float("nan")
    test_acc: float = float("nan")
    train_time_s: float = 0.0


def steps_per_epoch(cfg: RunConfig, n_train: int) -> int:
    return math.ceil(n_train / cfg.batch_size)


def build(cfg: RunConfig, n_train: int, rng: np.random.Generator):
    model = MLP(cfg.sizes, rng=rng, dtype=np.dtype(cfg.dtype).type)
    masks = masks_of(model)
    if cfg.optimizer == "adam":
        opt = Adam(model.parameters(), masks, lr=cfg.lr)
    elif cfg.optimizer == "sgd":
        opt = SGDMomentum(model.parameters(), masks, lr=cfg.lr)
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    if cfg.method == "none":
        return model, opt, None, None
    spe = steps_per_epoch(cfg, n_train)
    criterion = {"magnitude": Magnitude, "magnitude_layer": Magnitude,
                 "taylor_fo": TaylorFirstOrder, "fisher_obd": FisherOBD,
                 "fisher_obd_fo": lambda: FisherOBDFirstOrder(cfg.batch_size)}[cfg.method]()
    if cfg.schedule == "cubic":
        schedule = CubicSchedule(cfg.target_sparsity, cfg.prune_start_epoch * spe,
                                 cfg.prune_end_epoch * spe, cfg.prune_every)
    else:
        schedule = OneShotSchedule(cfg.target_sparsity, cfg.prune_end_epoch * spe)
    scope = "layerwise" if cfg.method == "magnitude_layer" else "global"
    pruner = Pruner(model, opt, criterion, schedule, scope=scope, rng=rng)
    regrower = Regrower(pruner, stop_step=REGROW_STOP_EPOCH * spe) if cfg.regrow else None
    return model, opt, pruner, regrower


def evaluate(model: MLP, X: np.ndarray, y: np.ndarray, batch: int = 1000) -> tuple[float, float]:
    loss = correct = 0.0
    with no_grad():
        for i in range(0, len(X), batch):
            logits = model(Tensor(X[i:i + batch]))
            n = len(logits.data)
            loss += float(softmax_cross_entropy(logits, y[i:i + batch]).data) * n
            correct += accuracy(logits, y[i:i + batch]) * n
    return loss / len(X), correct / len(X)


def _check_finite(loss: Tensor, model: MLP, step: int, out_dir: Path | None) -> None:
    bad = not np.isfinite(loss.data)
    report = {"step": step, "loss": float(loss.data), "layers": []}
    for i, l in enumerate(model.linears()):
        g = l.W.grad if l.W.grad is not None else np.zeros(1)
        finite = bool(np.isfinite(g).all() and np.isfinite(l.W.data).all())
        bad |= not finite
        report["layers"].append({"layer": i, "max_abs_grad": float(np.nanmax(np.abs(g))),
                                 "max_abs_w": float(np.nanmax(np.abs(l.W.data)))})
    if bad:
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "nan_dump.json").write_text(json.dumps(report, indent=2))
        raise NonFiniteError(f"non-finite loss or gradient at step {step}: {report}")


def fit(cfg: RunConfig, data: Splits, out_dir: Path | None = None,
        log=print) -> FitResult:
    rng = make_rng(cfg.seed)
    model, opt, pruner, regrower = build(cfg, len(data.X_train), rng)
    res = FitResult(cfg, model, pruner, regrower=regrower)
    step, t_start = 0, time.perf_counter()
    for epoch in range(cfg.epochs):
        opt.lr = cfg.lr * (LR_DECAY if epoch >= cfg.lr_decay_epoch else 1.0)
        t0 = time.perf_counter()
        perm = rng.permutation(len(data.X_train))
        tot_loss = tot_correct = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = data.X_train[idx], data.y_train[idx]
            opt.zero_grad()
            logits = model(Tensor(xb))
            loss = softmax_cross_entropy(logits, yb)
            loss.backward()
            _check_finite(loss, model, step, out_dir)
            if regrower is not None:
                regrower.accumulate()
            opt.step()
            step += 1
            if pruner is not None and (ev := pruner.maybe_prune(step)) is not None:
                res.events.append(ev)
            if regrower is not None and (rev := regrower.maybe_regrow(step)) is not None:
                res.regrow_events.append(rev)
            tot_loss += float(loss.data) * len(idx)
            tot_correct += accuracy(logits, yb) * len(idx)
        val_loss, val_acc = evaluate(model, data.X_val, data.y_val)
        row = {"epoch": epoch + 1,
               "train_loss": tot_loss / len(perm), "train_acc": tot_correct / len(perm),
               "val_loss": val_loss, "val_acc": val_acc,
               "epoch_time_s": time.perf_counter() - t0,
               "sparsity": pruner.sparsity() if pruner else 0.0,
               "n_active": sum(l.n_active() for l in model.linears())}
        res.curve.append(row)
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in row.items()))
    res.train_time_s = time.perf_counter() - t_start
    if pruner is not None:
        # hard budget: the run fails rather than report a missed target
        assert pruner.sparsity() >= cfg.target_sparsity - 1e-9, (
            f"budget missed: {pruner.sparsity():.6f} < {cfg.target_sparsity}")
    res.test_loss, res.test_acc = evaluate(model, data.X_test, data.y_test)
    return res


# --- outputs ----------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_model(path: Path, model: MLP) -> None:
    arrays = {}
    for i, l in enumerate(model.linears()):
        arrays.update({f"W{i}": l.W.data, f"b{i}": l.b.data, f"mask{i}": l.mask})
    np.savez_compressed(path, **arrays)


def load_layers(path: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """[(W, b, mask), ...] from save_model output."""
    with np.load(path) as f:
        n = sum(1 for k in f.files if k.startswith("W"))
        return [(f[f"W{i}"], f[f"b{i}"], f[f"mask{i}"]) for i in range(n)]


def event_rows(events: list[PruneEvent]) -> list[dict]:
    return [{"step": e.step, "target": e.target, "achieved": e.achieved,
             **{f"density_l{i}": d for i, d in enumerate(e.densities)},
             "dead_units": e.dead_units} for e in events]


def write_run(res: FitResult, out_dir: Path) -> dict:
    """curve.csv, config.json, summary.json, model.npz (+ prune_events.csv)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "curve.csv", res.curve)
    res.cfg.save(out_dir / "config.json")
    if res.events:
        write_csv(out_dir / "prune_events.csv", event_rows(res.events))
    if res.regrow_events:
        write_csv(out_dir / "regrow_events.csv", [vars(e) for e in res.regrow_events])
    save_model(out_dir / "model.npz", res.model)
    summary = {"test_loss": res.test_loss, "test_acc": res.test_acc,
               "val_acc": res.curve[-1]["val_acc"], "train_time_s": res.train_time_s,
               "target_sparsity": res.cfg.target_sparsity,
               "achieved_sparsity": res.pruner.sparsity() if res.pruner else 0.0,
               "n_active": res.curve[-1]["n_active"], "config_hash": res.cfg.hash(),
               "churn": res.regrower.churn if res.regrower else None}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
