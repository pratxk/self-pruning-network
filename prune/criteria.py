"""Importance criteria: estimated loss increase from removing each weight
(higher = keep). Pruned positions score -inf so pruning never keeps them."""
from abc import ABC, abstractmethod

import numpy as np

from nn.layers import MaskedLinear


class ImportanceCriterion(ABC):
    name: str

    def score(self, layer: MaskedLinear, opt_state: dict) -> np.ndarray:
        s = self._raw(layer.W.data.astype(np.float64), opt_state)
        return np.where(layer.mask, s, -np.inf)

    @abstractmethod
    def _raw(self, w: np.ndarray, opt_state: dict) -> np.ndarray:
        """Score per weight, shape of W."""


def _moments(opt_state: dict) -> tuple[np.ndarray, np.ndarray]:
    if "m_hat" not in opt_state:
        raise RuntimeError("criterion needs Adam moments: run the dense warmup first")
    return opt_state["m_hat"], opt_state["v_hat"]


class Magnitude(ImportanceCriterion):
    name = "magnitude"

    def _raw(self, w, opt_state):
        return np.abs(w)


class TaylorFirstOrder(ImportanceCriterion):
    """|m_hat * w|: first-order term only, m_hat ~ E[g] (smoothed)."""
    name = "taylor_fo"

    def _raw(self, w, opt_state):
        m_hat, _ = _moments(opt_state)
        return np.abs(m_hat * w)


class FisherOBD(ImportanceCriterion):
    """Optimal Brain Damage with H_ii ~ diag Fisher ~ v_hat: 1/2 v_hat w^2.
    The first-order term is dropped (classic OBD): see FisherOBDFirstOrder and
    DESIGN.md section 1 for the measured reason. Ranking is invariant to the
    scale of v_hat, so the batch-size correction is irrelevant here."""
    name = "fisher_obd"

    def _raw(self, w, opt_state):
        _, v_hat = _moments(opt_state)
        return 0.5 * v_hat * w**2


class FisherOBDFirstOrder(ImportanceCriterion):
    """Ablation, full second-order Taylor: dL ~ -g w + 1/2 H w^2 with g ~ m_hat and
    H_ii ~ B * v_hat. Adam's v is an EMA of the squared mini-batch MEAN gradient,
    E[g_bar^2] = E[g]^2 + Var(g_n)/B, so the per-example Fisher is ~B times larger.
    Without B the curvature term is 128x too small and -m_hat w dominates."""
    name = "fisher_obd_fo"

    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size

    def _raw(self, w, opt_state):
        m_hat, v_hat = _moments(opt_state)
        return 0.5 * self.batch_size * v_hat * w**2 - m_hat * w


CRITERIA = {c.name: c for c in (Magnitude, TaylorFirstOrder, FisherOBD, FisherOBDFirstOrder)}
