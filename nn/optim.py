"""Mask-aware optimizers.

Pruned positions get grad exactly 0 by the chain rule, but a zero gradient is
not a zero update: stale momentum keeps moving a pruned weight (drift) and a
global step count gives a revived weight a ~3.16x too-large first Adam step.
Fix: per-element step counts, state reset on prune AND revive, W *= M after
every step. naive=True is textbook behaviour, kept only so tests prove the bug.
"""
import numpy as np

from engine.tensor import Tensor


class Optimizer:
    def __init__(self, params: list[Tensor], masks: dict[int, np.ndarray] | None = None,
                 lr: float = 1e-3):
        """masks: id(param) -> bool mask array, shared by reference with the
        layer (mask edits are in place). Params without a mask are never pruned."""
        self.params = list(params)
        self.masks = masks or {}
        self.lr = lr
        self._state: dict[int, dict[str, np.ndarray]] = {}

    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                self._update(p, p.grad, self.masks.get(id(p)))

    def _update(self, p: Tensor, g: np.ndarray, mask: np.ndarray | None) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def on_mask_change(self, param: Tensor, pruned: np.ndarray, revived: np.ndarray) -> None:
        raise NotImplementedError

    def state(self, param: Tensor) -> dict[str, np.ndarray]:
        """Read-only view of a param's optimizer state (used by criteria)."""
        return self._state.get(id(param), {})


class Adam(Optimizer):
    def __init__(self, params, masks=None, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, naive=False):
        super().__init__(params, masks, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.naive = naive
        self._t_global = 0
        self._tables: dict[np.dtype, tuple[np.ndarray, np.ndarray]] = {}

    def _bias(self, t: np.ndarray, dtype) -> tuple[np.ndarray, np.ndarray]:
        """(1 - b1^t, 1 - b2^t) per element. Per-element t makes this a pow per
        weight per step (80% of an epoch, results/profile/); a table indexed by t
        turns it into a gather. Entry 0 is set to 1: t = 0 only where pruned, where
        m = v = 0, so 0/1 = 0 instead of 0/0. t <= _t_global always bounds the size."""
        tab = self._tables.get(np.dtype(dtype))
        if tab is None or len(tab[0]) <= self._t_global:
            k = np.arange(max(2 * self._t_global, 1024))
            tab1, tab2 = (np.where(k == 0, 1.0, 1 - np.power(b, k)).astype(dtype)
                          for b in (self.b1, self.b2))
            tab = (tab1, tab2)
            self._tables[np.dtype(dtype)] = tab
        return tab[0][t], tab[1][t]

    def _init_state(self, p: Tensor) -> dict[str, np.ndarray]:
        return self._state.setdefault(id(p), {
            "m": np.zeros_like(p.data),
            "v": np.zeros_like(p.data),
            "t": np.zeros(p.data.shape, dtype=np.int32),
        })

    def step(self) -> None:
        self._t_global += 1
        super().step()

    def state(self, param):
        """m, v, t plus bias-corrected m_hat, v_hat (0 where t = 0)."""
        s = self._state.get(id(param))
        if s is None:
            return {}
        bc1, bc2 = self._bias(s["t"], s["m"].dtype)
        return {**s, "m_hat": s["m"] / bc1, "v_hat": s["v"] / bc2}

    def _update(self, p, g, mask):
        s = self._init_state(p)
        m, v, t = s["m"], s["v"], s["t"]
        # Dense updates are exact at pruned positions: g = m = v = 0 there,
        # so m, v stay 0 and the update is 0/(0+eps) = 0.
        m *= self.b1
        m += (1 - self.b1) * g
        v *= self.b2
        v += (1 - self.b2) * g * g
        if self.naive:
            t[...] = self._t_global
        else:
            t += mask if mask is not None else 1
        bc1, bc2 = self._bias(t, p.data.dtype)
        p.data -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)
        if mask is not None and not self.naive:
            p.data *= mask  # hard-zero invariant

    def on_mask_change(self, param, pruned, revived):
        param.data[pruned | revived] = 0.0
        if self.naive:
            return
        s = self._init_state(param)
        for k in ("m", "v", "t"):
            s[k][pruned | revived] = 0


class SGDMomentum(Optimizer):
    def __init__(self, params, masks=None, lr=1e-2, momentum=0.9, naive=False):
        super().__init__(params, masks, lr)
        self.momentum = momentum
        self.naive = naive

    def _update(self, p, g, mask):
        buf = self._state.setdefault(id(p), {"velocity": np.zeros_like(p.data)})["velocity"]
        buf *= self.momentum
        buf += g
        p.data -= self.lr * buf
        if mask is not None and not self.naive:
            p.data *= mask

    def on_mask_change(self, param, pruned, revived):
        param.data[pruned | revived] = 0.0
        if not self.naive:
            self._state.setdefault(id(param), {"velocity": np.zeros_like(param.data)})[
                "velocity"][pruned | revived] = 0.0


def masks_of(model) -> dict[int, np.ndarray]:
    """id(W) -> mask for every MaskedLinear in a model."""
    return {id(layer.W): layer.mask for layer in model.linears()}
