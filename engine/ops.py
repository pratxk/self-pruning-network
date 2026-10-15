"""Differentiable ops. Each returns a Tensor whose _backward pushes G (out.grad)
into its parents. Non-Tensor operands become constants with the other operand's dtype."""
import numpy as np

from engine.tensor import Tensor, is_grad_enabled, unbroadcast


def _wrap(x, like: Tensor) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=like.data.dtype))


def _pair(a, b):
    if not isinstance(a, Tensor):
        a = _wrap(a, b)
    return a, _wrap(b, a)


def _make(data, parents, op, grads_fn) -> Tensor:
    """grads_fn(G) -> one gradient (or None) per parent."""
    req = is_grad_enabled() and any(p.requires_grad for p in parents)
    out = Tensor(data, requires_grad=req, _prev=parents if req else (), op=op)
    if req:
        def _backward():
            if out.grad is None:
                return
            for p, g in zip(parents, grads_fn(out.grad)):
                if p.requires_grad and g is not None:
                    p._accum(g)

        out._backward = _backward
    return out


def add(a, b):
    a, b = _pair(a, b)
    return _make(a.data + b.data, (a, b), "add",
                 lambda G: (unbroadcast(G, a.shape), unbroadcast(G, b.shape)))


def sub(a, b):
    a, b = _pair(a, b)
    return _make(a.data - b.data, (a, b), "sub",
                 lambda G: (unbroadcast(G, a.shape), unbroadcast(-G, b.shape)))


def mul(a, b):
    a, b = _pair(a, b)
    return _make(a.data * b.data, (a, b), "mul",
                 lambda G: (unbroadcast(G * b.data, a.shape), unbroadcast(G * a.data, b.shape)))


def div(a, b):
    a, b = _pair(a, b)
    return _make(a.data / b.data, (a, b), "div",
                 lambda G: (unbroadcast(G / b.data, a.shape),
                            unbroadcast(-G * a.data / b.data**2, b.shape)))


def neg(a):
    return _make(-a.data, (a,), "neg", lambda G: (-G,))


def matmul(a, b):
    a, b = _pair(a, b)
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError("matmul supports 2-D operands only")
    return _make(a.data @ b.data, (a, b), "matmul",
                 lambda G: (G @ b.data.T, a.data.T @ G))


def _axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    axis = (axis,) if isinstance(axis, int) else axis
    return tuple(ax % ndim for ax in axis)


def _expand(G, a_shape, axes, keepdims):
    if not keepdims:
        G = np.expand_dims(G, axes)
    return np.broadcast_to(G, a_shape)


def sum(a, axis=None, keepdims=False):
    axes = _axes(axis, a.data.ndim)
    return _make(a.data.sum(axis=axes, keepdims=keepdims), (a,), "sum",
                 lambda G: (_expand(G, a.shape, axes, keepdims),))


def mean(a, axis=None, keepdims=False):
    axes = _axes(axis, a.data.ndim)
    out = a.data.mean(axis=axes, keepdims=keepdims)
    count = a.data.size // max(out.size, 1)
    return _make(out, (a,), "mean",
                 lambda G: (_expand(G, a.shape, axes, keepdims) / count,))


def relu(a):
    return _make(np.maximum(a.data, 0), (a,), "relu", lambda G: (G * (a.data > 0),))


def tanh(a):
    out = np.tanh(a.data)
    return _make(out, (a,), "tanh", lambda G: (G * (1 - out**2),))


def sigmoid(a):
    out = 1 / (1 + np.exp(-a.data))
    return _make(out, (a,), "sigmoid", lambda G: (G * out * (1 - out),))


def exp(a):
    out = np.exp(a.data)
    return _make(out, (a,), "exp", lambda G: (G * out,))


def log(a):
    return _make(np.log(a.data), (a,), "log", lambda G: (G / a.data,))


def reshape(a, shape):
    return _make(a.data.reshape(shape), (a,), "reshape", lambda G: (G.reshape(a.shape),))


def transpose(a):
    return _make(a.data.T, (a,), "transpose", lambda G: (G.T,))
