import numpy as np

from engine.ops import _make
from engine.tensor import Tensor


def softmax_cross_entropy(logits: Tensor, y: np.ndarray) -> Tensor:
    """Fused, stable mean cross-entropy. y: int class labels (not a Tensor).
    dL/dz = (softmax(z) - onehot(y)) / N."""
    z = logits.data
    n = z.shape[0]
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    s = e.sum(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(s[:, 0])
    rows = np.arange(n)
    loss = np.asarray(np.mean(lse - z[rows, y]), dtype=z.dtype)

    def grads(G):
        d = e / s
        d[rows, y] -= 1
        return (G * d / n,)

    return _make(loss, (logits,), "softmax_cross_entropy", grads)


def accuracy(logits, y: np.ndarray) -> float:
    z = logits.data if isinstance(logits, Tensor) else logits
    return float((z.argmax(axis=1) == y).mean())
