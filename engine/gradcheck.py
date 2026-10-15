import numpy as np

from engine.tensor import Tensor, no_grad


def check_grad(fn, inputs: list[Tensor], eps: float = 1e-6, seed: int = 0) -> float:
    """Max relative error |a - n| / max(|a| + |n|, 1e-8) between autodiff and
    central differences, over every element of every input. fn(*inputs) is
    reduced to a scalar with a fixed random projection R so every output
    element is tested. Use float64 inputs."""
    out = fn(*inputs)
    R = np.random.default_rng(seed).standard_normal(out.shape)
    for t in inputs:
        t.zero_grad()
    (out * R).sum().backward()

    def scalar():
        with no_grad():
            return float((fn(*inputs).data * R).sum())

    worst = 0.0
    for t in inputs:
        analytic = t.grad if t.grad is not None else np.zeros_like(t.data)
        for idx in np.ndindex(t.data.shape):
            orig = t.data[idx]
            t.data[idx] = orig + eps
            fp = scalar()
            t.data[idx] = orig - eps
            fm = scalar()
            t.data[idx] = orig
            a, n = float(analytic[idx]), (fp - fm) / (2 * eps)
            worst = max(worst, abs(a - n) / max(abs(a) + abs(n), 1e-8))
    return worst
