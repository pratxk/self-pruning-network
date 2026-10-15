import numpy as np
import pytest

from engine.tensor import Tensor, no_grad, unbroadcast


def test_deep_chain_no_recursion_error():
    x = Tensor(np.ones(3), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.sum().backward()
    assert np.array_equal(x.grad, np.ones(3))


def test_no_grad_records_nothing():
    x = Tensor(np.ones(3), requires_grad=True)
    with no_grad():
        y = (x * 2.0).sum()
    assert y._prev == () and not y.requires_grad
    with pytest.raises(RuntimeError):
        y.backward()


def test_backward_needs_scalar():
    x = Tensor(np.ones(3), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2.0).backward()


def test_gradients_accumulate_across_backward_calls():
    x = Tensor(np.array([2.0]), requires_grad=True)
    (x * 3.0).sum().backward()
    (x * 3.0).sum().backward()
    assert x.grad[0] == 6.0


def test_graph_freed_after_backward():
    x = Tensor(np.ones(2), requires_grad=True)
    y = x * 2.0
    loss = y.sum()
    loss.backward()
    assert loss._prev == () and y._prev == ()


def test_constants_keep_tensor_dtype():
    x = Tensor(np.ones(2, dtype=np.float32), requires_grad=True)
    assert (x * 2.0 + 1.0).data.dtype == np.float32
    assert (np.ones(2) * x).data.dtype == np.float32


def test_unbroadcast_shapes():
    g = np.ones((4, 5, 3))
    assert unbroadcast(g, (3,)).shape == (3,)
    assert unbroadcast(g, (5, 1)).tolist() == [[12.0]] * 5
    assert unbroadcast(g, ()).shape == ()
