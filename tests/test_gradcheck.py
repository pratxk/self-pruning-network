import numpy as np
import pytest

from engine import ops
from engine.functional import softmax_cross_entropy
from engine.gradcheck import check_grad
from engine.tensor import Tensor

TOL = 1e-6


def t(*shape, seed=0, low=None):
    """float64 leaf; `low` keeps values away from kinks / zero (ReLU, div, log)."""
    x = np.random.default_rng(seed).standard_normal(shape)
    if low is not None:
        x = np.sign(x) * (np.abs(x) + low)
    return Tensor(x, requires_grad=True)


def pos(*shape, seed=0):
    return Tensor(np.random.default_rng(seed).uniform(0.5, 2.0, shape), requires_grad=True)


UNARY = {
    "neg": (ops.neg, lambda: [t(3, 4)]),
    "relu": (ops.relu, lambda: [t(3, 4, low=1e-3)]),
    "tanh": (ops.tanh, lambda: [t(3, 4)]),
    "sigmoid": (ops.sigmoid, lambda: [t(3, 4)]),
    "exp": (ops.exp, lambda: [t(3, 4)]),
    "log": (ops.log, lambda: [pos(3, 4)]),
    "sum_all": (lambda a: ops.sum(a), lambda: [t(3, 4)]),
    "sum_axis0": (lambda a: ops.sum(a, axis=0), lambda: [t(3, 4)]),
    "sum_axis1_keep": (lambda a: ops.sum(a, axis=1, keepdims=True), lambda: [t(3, 4)]),
    "mean_all": (lambda a: ops.mean(a), lambda: [t(3, 4)]),
    "mean_axis_neg1": (lambda a: ops.mean(a, axis=-1), lambda: [t(3, 4)]),
    "reshape": (lambda a: ops.reshape(a, (4, 3)), lambda: [t(3, 4)]),
    "transpose": (ops.transpose, lambda: [t(3, 4)]),
}

BINARY = {
    "add": (ops.add, lambda: [t(3, 4), t(3, 4, seed=1)]),
    "sub": (ops.sub, lambda: [t(3, 4), t(3, 4, seed=1)]),
    "mul": (ops.mul, lambda: [t(3, 4), t(3, 4, seed=1)]),
    "div": (ops.div, lambda: [t(3, 4), pos(3, 4, seed=1)]),
    "matmul": (ops.matmul, lambda: [t(3, 5), t(5, 4, seed=1)]),
}

BROADCAST = {
    "(N,D)+(D,)": (ops.add, lambda: [t(5, 3), t(3, seed=1)]),
    "(N,1)*(1,D)": (ops.mul, lambda: [t(5, 1), t(1, 3, seed=1)]),
    "scalar*(N,D)": (ops.mul, lambda: [t(seed=2), t(5, 3, seed=1)]),
    "(N,D)/(D,)": (ops.div, lambda: [t(5, 3), pos(3, seed=1)]),
    "(N,D)-(1,D)": (ops.sub, lambda: [t(5, 3), t(1, 3, seed=1)]),
}


@pytest.mark.parametrize("name", sorted({**UNARY, **BINARY}))
def test_op_gradcheck(name):
    fn, make = {**UNARY, **BINARY}[name]
    assert check_grad(fn, make()) < TOL


@pytest.mark.parametrize("name", sorted(BROADCAST))
def test_broadcast_gradcheck(name):
    fn, make = BROADCAST[name]
    inputs = make()
    assert check_grad(fn, inputs) < TOL
    for x in inputs:
        assert x.grad.shape == x.data.shape


def test_softmax_cross_entropy():
    y = np.array([0, 2, 1, 2, 0])
    logits = t(5, 3)
    assert check_grad(lambda z: softmax_cross_entropy(z, y), [logits]) < TOL

    shifted = Tensor(logits.data + 1000.0, requires_grad=True)
    loss = softmax_cross_entropy(shifted, y)
    assert np.isfinite(loss.data)
    assert check_grad(lambda z: softmax_cross_entropy(z, y), [shifted]) < TOL

    z = logits.data
    naive = -np.mean(np.log(np.exp(z) / np.exp(z).sum(1, keepdims=True))[np.arange(5), y])
    assert np.isclose(softmax_cross_entropy(logits, y).data, naive)
    assert np.isclose(loss.data, naive)  # shift-invariant


def test_node_reused_twice():
    assert check_grad(lambda x: x * x + x, [t(3, 4)]) < TOL


def test_two_layer_mlp_end_to_end():
    y = np.array([0, 1, 2, 1])
    x = t(4, 6, seed=3)
    params = [t(6, 5, seed=4), t(5, seed=5), t(5, 3, seed=6), t(3, seed=7)]

    def f(x, W1, b1, W2, b2):
        return softmax_cross_entropy(ops.tanh(x @ W1 + b1) @ W2 + b2, y)

    assert check_grad(f, [x, *params]) < TOL
