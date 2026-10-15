import math
from dataclasses import dataclass

import numpy as np

from nn.layers import MaskedLinear
from nn.optim import Optimizer
from prune.criteria import ImportanceCriterion
from prune.schedule import Schedule


def apply_mask(layer: MaskedLinear, optimizer: Optimizer, new_mask: np.ndarray) -> None:
    """The single choke point for mask edits: computes the pruned / revived sets,
    updates the mask in place (optimizer holds it by reference) and lets the
    optimizer zero W and reset per-element state at both sets."""
    new_mask = np.asarray(new_mask, dtype=bool)
    pruned = layer.mask & ~new_mask
    revived = ~layer.mask & new_mask
    layer.mask[...] = new_mask
    optimizer.on_mask_change(layer.W, pruned, revived)


@dataclass(frozen=True)
class PruneEvent:
    step: int
    target: float
    achieved: float
    densities: tuple[float, ...]
    dead_units: int


class Pruner:
    """Keeps the top-k weights by criterion score; the only code that changes masks
    during pruning. Masks only shrink here (regrowth lives in prune.regrow)."""

    def __init__(self, model, optimizer: Optimizer, criterion: ImportanceCriterion,
                 schedule: Schedule, scope: str = "global", min_density: float = 0.01,
                 max_sparsity_out: float = 0.80, rng: np.random.Generator | None = None):
        if scope not in ("global", "layerwise"):
            raise ValueError(f"unknown scope {scope!r}")
        self.layers: list[MaskedLinear] = model.linears()
        self.optimizer = optimizer
        self.criterion = criterion
        self.schedule = schedule
        self.scope = scope
        self.min_density = min_density
        self.max_sparsity_out = max_sparsity_out
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # per-layer bool masks of weights that must be kept (regrowth grace period)
        self.protected: list[np.ndarray] | None = None

    # -- accounting ---------------------------------------------------------
    def total(self) -> int:
        return sum(l.mask.size for l in self.layers)

    def sparsity(self) -> float:
        """Pruned / total weights, biases excluded."""
        return 1 - sum(l.n_active() for l in self.layers) / self.total()

    def densities(self) -> tuple[float, ...]:
        return tuple(l.n_active() / l.mask.size for l in self.layers)

    # -- pruning ------------------------------------------------------------
    def maybe_prune(self, step: int) -> PruneEvent | None:
        if not self.schedule.is_prune_step(step):
            return None
        target = self.schedule.target(step)
        self.apply_masks(self.compute_masks(target))
        dead = self.remove_dead_units()
        return PruneEvent(step, target, self.sparsity(), self.densities(), dead)

    def _floors(self, keep_total: int) -> list[int]:
        floors = []
        for i, l in enumerate(self.layers):
            frac = 1 - self.max_sparsity_out if i == len(self.layers) - 1 else self.min_density
            floors.append(min(math.ceil(frac * l.mask.size), l.n_active()))
        if sum(floors) > keep_total:
            # ponytail: floors yield to the hard budget (e.g. 99% on LeNet-300-100:
            # floors need 2,852 weights, budget keeps 2,662) -> scale them down.
            scale = keep_total / sum(floors)
            floors = [int(f * scale) for f in floors]
        return floors

    def _order(self, scores: np.ndarray) -> np.ndarray:
        """Indices by score descending; ties broken by a seeded random key so the
        kept count is exact and reproducible."""
        return np.lexsort((self.rng.random(scores.size), -scores))

    def compute_masks(self, target: float, keep: int | None = None) -> list[np.ndarray]:
        """Masks keeping round((1 - target) * total) weights, or exactly `keep`."""
        active = sum(l.n_active() for l in self.layers)
        keep_total = min(keep if keep is not None else round((1 - target) * self.total()),
                         active)
        floors = self._floors(keep_total)
        scores = [self.criterion.score(l, self.optimizer.state(l.W)).ravel()
                  for l in self.layers]
        if self.protected is not None:
            scores = [np.where(p.ravel() & l.mask.ravel(), np.inf, s)
                      for p, l, s in zip(self.protected, self.layers, scores)]

        if self.scope == "layerwise":
            frac = keep_total / self.total()
            quotas = [min(max(round(frac * l.mask.size), f), l.n_active())
                      for l, f in zip(self.layers, floors)]
            big = int(np.argmax([l.mask.size for l in self.layers]))
            # ponytail: absorb rounding / floor excess in the largest layer
            quotas[big] = min(max(quotas[big] + keep_total - sum(quotas), floors[big]),
                              self.layers[big].n_active())
        else:
            # reserve each layer's floor, then rank the rest globally
            reserved = []
            for s, f in zip(scores, floors):
                r = np.zeros(s.size, dtype=bool)
                r[self._order(s)[:f]] = True
                reserved.append(r)
            flat = np.concatenate([np.where(r, -np.inf, s) for s, r in zip(scores, reserved)])
            extra = np.zeros(flat.size, dtype=bool)
            extra[self._order(flat)[:keep_total - sum(floors)]] = True
            bounds = np.cumsum([0] + [s.size for s in scores])
            return [(r | extra[a:b]).reshape(l.mask.shape)
                    for l, r, a, b in zip(self.layers, reserved, bounds[:-1], bounds[1:])]

        masks = []
        for l, s, q in zip(self.layers, scores, quotas):
            m = np.zeros(s.size, dtype=bool)
            m[self._order(s)[:q]] = True
            masks.append(m.reshape(l.mask.shape))
        return masks

    def apply_masks(self, new_masks: list[np.ndarray]) -> None:
        for layer, m in zip(self.layers, new_masks):
            apply_mask(layer, self.optimizer, m)

    def remove_dead_units(self) -> int:
        """Prune every weight of hidden units left with no incoming or no outgoing
        weights (dead compute). Returns the number of weights removed."""
        removed, changed = 0, True
        while changed:
            changed = False
            for a, b in zip(self.layers[:-1], self.layers[1:]):
                dead = ~a.mask.any(axis=0) | ~b.mask.any(axis=1)
                n = int(a.mask[:, dead].sum() + b.mask[dead].sum())
                if n:
                    changed = True
                    removed += n
                    ma, mb = a.mask.copy(), b.mask.copy()
                    ma[:, dead] = False
                    mb[dead] = False
                    apply_mask(a, self.optimizer, ma)
                    apply_mask(b, self.optimizer, mb)
        return removed
