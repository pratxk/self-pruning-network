"""Paired-seed statistics, NumPy only (scipy is not a dependency).

Sign-flip permutation test is exact: with n seeds there are 2^n sign patterns,
so the smallest two-sided p is 2 / 2^n (n = 5 -> 0.0625, can never reach 0.05;
n = 10 -> ~0.002). Hence 10 seeds for the headline claim."""
import itertools
import math

import numpy as np


def sign_flip_p(d: np.ndarray) -> float:
    """Exact two-sided permutation p for H0: paired differences symmetric about 0."""
    d = np.asarray(d, dtype=np.float64)
    if len(d) > 20:
        raise ValueError("exact enumeration only up to 20 pairs")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    stats = np.abs((signs * d).mean(axis=1))
    return float(np.mean(stats >= abs(d.mean()) - 1e-12))


def bootstrap_ci(d: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    d = np.asarray(d, dtype=np.float64)
    rng = rng if rng is not None else np.random.default_rng(0)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * float(p[i])))
        adj[i] = running
    return adj.tolist()


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar on per-example correctness: (b, c, p) where
    b = A right & B wrong, c = A wrong & B right."""
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b, c = int((a & ~b_).sum()), int((~a & b_).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return b, c, min(1.0, 2 * tail)
