"""Array-level reverse-mode autodiff: each Tensor knows its parents and a closure
that adds its gradient into theirs."""
import contextlib

import numpy as np

_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    """Disable graph building (evaluation / inference)."""
    global _grad_enabled
    prev, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled() -> bool:
    return _grad_enabled


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum grad back down to `shape`, undoing numpy broadcasting."""
    if grad.ndim > len(shape):
        grad = grad.sum(axis=tuple(range(grad.ndim - len(shape))))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] > 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


def _noop():
    pass


class Tensor:
    __array_ufunc__ = None  # make ndarray <op> Tensor defer to Tensor's reflected ops

    def __init__(self, data, requires_grad=False, dtype=None, _prev=(), op=""):
        arr = np.asarray(data)
        if dtype is None:
            dtype = arr.dtype if arr.dtype.kind == "f" else np.float32
        self.data = arr.astype(dtype, copy=False)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._prev = _prev
        self._backward = _noop
        self.op = op

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self.op!r}, requires_grad={self.requires_grad})"

    @property
    def shape(self):
        return self.data.shape

    def _accum(self, g: np.ndarray) -> None:
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self) -> None:
        if self.data.size != 1:
            raise RuntimeError("backward() needs a scalar output")
        if not self.requires_grad:
            raise RuntimeError("output has no graph (built under no_grad or from constants)")
        # Iterative post-order DFS: recursion overflows on long graphs.
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._prev if id(p) not in seen)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
        for node in topo:  # free the graph so it does not leak between steps
            node._prev, node._backward = (), _noop

    # operator overloads -> engine.ops
    def __add__(self, o):
        return _ops().add(self, o)

    def __radd__(self, o):
        return _ops().add(o, self)

    def __sub__(self, o):
        return _ops().sub(self, o)

    def __rsub__(self, o):
        return _ops().sub(o, self)

    def __mul__(self, o):
        return _ops().mul(self, o)

    def __rmul__(self, o):
        return _ops().mul(o, self)

    def __truediv__(self, o):
        return _ops().div(self, o)

    def __rtruediv__(self, o):
        return _ops().div(o, self)

    def __matmul__(self, o):
        return _ops().matmul(self, o)

    def __rmatmul__(self, o):
        return _ops().matmul(o, self)

    def __neg__(self):
        return _ops().neg(self)

    def sum(self, axis=None, keepdims=False):
        return _ops().sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return _ops().mean(self, axis, keepdims)

    def relu(self):
        return _ops().relu(self)

    def reshape(self, *shape):
        return _ops().reshape(self, shape[0] if len(shape) == 1 else shape)

    @property
    def T(self):
        return _ops().transpose(self)


def _ops():
    from engine import ops  # ponytail: lazy import breaks the tensor<->ops cycle

    return ops
