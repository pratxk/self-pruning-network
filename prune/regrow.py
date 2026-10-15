"""RigL-style regrowth (Evci et al. 2020) on top of the pruner's schedule.

Every `every` steps until `stop_step`: drop the k lowest-saliency active weights
(via Pruner.compute_masks with an exact keep count, so layer floors still hold)
and grow the k pruned weights with the largest EMA of |dL/dW_eff| (the dense,
counterfactual gradient). Sparsity is unchanged by an event. Grown weights start
at 0 with fresh optimizer state (apply_mask), and are protected from both regrowth
drops and scheduled pruning for `grace` events: at w = 0 their saliency is 0 and
they would otherwise be dropped immediately."""
import math
from dataclasses import dataclass

import numpy as np

from prune.pruner import Pruner


@dataclass(frozen=True)
class RegrowEvent:
    step: int
    alpha: float
    dropped: int
    grown: int
    churn: int  # cumulative: regrown weights that were later dropped again


class Regrower:
    def __init__(self, pruner: Pruner, stop_step: int, every: int = 500,
                 alpha0: float = 0.3, grace: int = 2, ema: float = 0.9):
        self.pruner = pruner
        self.layers = pruner.layers
        self.stop_step, self.every = stop_step, every
        self.alpha0, self.grace, self.ema = alpha0, grace, ema
        self.acc = [np.zeros(l.mask.shape) for l in self.layers]       # EMA |dense grad|
        self.grace_left = [np.zeros(l.mask.shape, np.int32) for l in self.layers]
        self.ever_grown = [np.zeros(l.mask.shape, bool) for l in self.layers]
        self.churn = 0

    def accumulate(self) -> None:
        """Call each step after backward()."""
        for a, l in zip(self.acc, self.layers):
            a *= self.ema
            a += (1 - self.ema) * np.abs(l.dense_grad())

    def alpha(self, step: int) -> float:
        return 0.5 * self.alpha0 * (1 + math.cos(math.pi * min(step / self.stop_step, 1.0)))

    def maybe_regrow(self, step: int) -> RegrowEvent | None:
        if step % self.every or step >= self.stop_step or self.pruner.sparsity() == 0:
            return None
        active = sum(l.n_active() for l in self.layers)
        alpha = self.alpha(step)
        k = int(alpha * active)
        if k == 0:
            return None
        old = [l.mask.copy() for l in self.layers]

        # drop: the pruner's own ranking at an exact keep count, grace protected
        self.pruner.protected = [g > 0 for g in self.grace_left]
        kept = self.pruner.compute_masks(0.0, keep=active - k)
        dropped = sum(int((o & ~m).sum()) for o, m in zip(old, kept))

        # grow: largest accumulated dense gradient among weights pruned BEFORE this event
        flat = np.concatenate([np.where(o, -np.inf, a).ravel() for o, a in zip(old, self.acc)])
        grow_flat = np.zeros(flat.size, bool)
        grow_flat[self.pruner._order(flat)[:dropped]] = True
        bounds = np.cumsum([0] + [o.size for o in old])
        grown = [grow_flat[a:b].reshape(o.shape) for o, a, b in zip(old, bounds[:-1], bounds[1:])]

        for o, m, e in zip(old, kept, self.ever_grown):
            self.churn += int((o & ~m & e).sum())
        self.pruner.apply_masks([m | g for m, g in zip(kept, grown)])
        for gl, eg, g in zip(self.grace_left, self.ever_grown, grown):
            np.maximum(gl - 1, 0, out=gl)
            gl[g] = self.grace
            eg |= g
        self.pruner.protected = [g > 0 for g in self.grace_left]
        return RegrowEvent(step, alpha, dropped, int(grow_flat.sum()), self.churn)
