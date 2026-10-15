"""Masked-weight correctness: forward equivalence, exact-zero parameter grads,
the dense (counterfactual) grad, and mask-aware optimizer state."""
import numpy as np
import pytest

from engine.functional import softmax_cross_entropy
from engine.gradcheck import check_grad
from engine.sparse import csr_from_dense, sparse_linear
from engine.tensor import Tensor
from nn.layers import MLP, MaskedLinear
from nn.optim import Adam, SGDMomentum, masks_of
from prune.pruner import apply_mask

F64 = np.float64


def layer_with_mask(density=0.3, seed=0, shape=(6, 4)):
    rng = np.random.default_rng(seed)
    layer = MaskedLinear(*shape, rng, dtype=F64)
    layer.b.data[:] = rng.standard_normal(shape[1])
    layer.mask[...] = rng.random(shape) < density
    layer.W.data *= layer.mask
    return layer


def test_forward_equivalence():
    layer = layer_with_mask()
    x = np.random.default_rng(1).standard_normal((5, 6))
    masked = layer(Tensor(x)).data
    W0 = np.where(layer.mask, layer.W.data, 0.0)
    dense = x @ W0 + layer.b.data
    csr = sparse_linear(x, csr_from_dense(layer.W.data, layer.mask), layer.b.data)
    assert np.allclose(masked, dense, atol=1e-6)
    assert np.allclose(masked, csr, atol=1e-6)


def test_param_grad_exactly_zero_at_masked_positions():
    layer = layer_with_mask()
    x = Tensor(np.random.default_rng(1).standard_normal((5, 6)))
    softmax_cross_entropy(layer(x), np.array([0, 1, 2, 3, 0])).backward()
    assert np.all(layer.W.grad[~layer.mask] == 0)


def test_dense_grad_is_the_counterfactual():
    """W_eff.grad at pruned positions == d loss / d w with the mask lifted."""
    layer = layer_with_mask()
    x = Tensor(np.random.default_rng(1).standard_normal((5, 6)))
    y = np.array([0, 1, 2, 3, 0])
    softmax_cross_entropy(layer(x), y).backward()
    dense = layer.dense_grad()

    lifted = MaskedLinear(6, 4, np.random.default_rng(0), dtype=F64)
    lifted.W.data[...] = layer.W.data  # pruned positions are 0 here too
    lifted.b.data[...] = layer.b.data

    def loss():
        return float(softmax_cross_entropy(lifted(x), y).data)

    eps = 1e-6
    for idx in zip(*np.nonzero(~layer.mask)):
        lifted.W.data[idx] += eps
        fp = loss()
        lifted.W.data[idx] -= 2 * eps
        fm = loss()
        lifted.W.data[idx] += eps
        assert np.isclose(dense[idx], (fp - fm) / (2 * eps), rtol=1e-5, atol=1e-9)


def test_masked_linear_gradcheck_70pct_mask():
    layer = layer_with_mask(density=0.3)
    x = Tensor(np.random.default_rng(2).standard_normal((5, 6)), requires_grad=True)
    assert check_grad(lambda W, b, x: layer(x), [layer.W, layer.b, x]) < 1e-6


# --- optimizer state -------------------------------------------------------

def _task(seed=0):
    rng = np.random.default_rng(seed)
    model = MLP((8, 16, 4), rng=rng, dtype=F64)
    X = rng.standard_normal((64, 8))
    y = rng.integers(0, 4, 64)
    return model, X, y


def _train(model, opt, X, y, steps, rng):
    for _ in range(steps):
        idx = rng.integers(0, len(X), 16)
        opt.zero_grad()
        softmax_cross_entropy(model(Tensor(X[idx])), y[idx]).backward()
        opt.step()


def _prune_half(model, opt, rng):
    for layer in model.linears():
        new = layer.mask & (rng.random(layer.mask.shape) < 0.5)
        apply_mask(layer, opt, new)


def _drift(make_opt, state_keys):
    model, X, y = _task()
    opt = make_opt(model)
    rng = np.random.default_rng(7)
    _train(model, opt, X, y, 200, rng)
    _prune_half(model, opt, rng)
    _train(model, opt, X, y, 500, rng)
    for layer in model.linears():
        dead = ~layer.mask
        assert np.all(layer.W.data[dead] == 0.0)
        st = opt.state(layer.W)
        for k in state_keys:
            assert np.all(st[k][dead] == 0)


def test_adam_no_drift():
    _drift(lambda m: Adam(m.parameters(), masks_of(m), lr=1e-2), ("m", "v"))


@pytest.mark.xfail(strict=True, reason="naive Adam drifts: proves the test has teeth")
def test_naive_adam_drifts():
    _drift(lambda m: Adam(m.parameters(), masks_of(m), lr=1e-2, naive=True), ("m", "v"))


def test_sgd_momentum_no_drift():
    _drift(lambda m: SGDMomentum(m.parameters(), masks_of(m), lr=1e-2), ("velocity",))


@pytest.mark.xfail(strict=True, reason="naive SGD momentum drifts")
def test_naive_sgd_momentum_drifts():
    _drift(lambda m: SGDMomentum(m.parameters(), masks_of(m), lr=1e-2, naive=True),
           ("velocity",))


def _revival(make_opt):
    """Element 0 pruned from the start, revived at step 1000, then fed a fixed
    gradient sequence. Returns (revived trajectory, fresh 1-element trajectory)."""
    rng = np.random.default_rng(3)
    p = Tensor(rng.standard_normal(4), requires_grad=True)
    mask = np.array([False, True, True, True])
    p.data *= mask
    opt = make_opt(p, mask)
    layer = type("L", (), {"W": p, "mask": mask})()  # apply_mask duck type
    for _ in range(1000):
        p.grad = rng.standard_normal(4) * mask  # chain rule: 0 at pruned
        opt.step()
    apply_mask(layer, opt, np.ones(4, dtype=bool))

    seq = rng.standard_normal(50)
    fresh = Tensor(np.zeros(1), requires_grad=True)
    fresh_opt = make_opt(fresh, None)
    revived, ref = [], []
    for g in seq:
        p.grad = np.r_[g, rng.standard_normal(3)]
        opt.step()
        fresh.grad = np.array([g])
        fresh_opt.step()
        revived.append(p.data[0])
        ref.append(fresh.data[0])
    return np.array(revived), np.array(ref)


def test_adam_revival_matches_fresh_adam_bit_for_bit():
    revived, ref = _revival(lambda p, m: Adam([p], {id(p): m} if m is not None else {}))
    assert np.array_equal(revived, ref)


def test_naive_adam_revival_first_step_too_big():
    """Fresh Adam's first step is lr (m_hat = g, v_hat = g^2). Naive global t
    gives ((1-b1)/(1-b1^t)) / sqrt((1-b2)/(1-b2^t)) * lr: 2.515x at t = 1001,
    tending to 0.1/sqrt(0.001) = 3.16x once b2^t ~ 0 (t >~ 5000)."""
    revived, ref = _revival(
        lambda p, m: Adam([p], {id(p): m} if m is not None else {}, naive=True))
    t = 1001
    expected = (0.1 / (1 - 0.9**t)) / np.sqrt(0.001 / (1 - 0.999**t))
    assert abs(revived[0]) / abs(ref[0]) == pytest.approx(expected, rel=1e-4)
    assert expected > 2.5


def test_sgd_revival_matches_fresh_sgd_bit_for_bit():
    revived, ref = _revival(
        lambda p, m: SGDMomentum([p], {id(p): m} if m is not None else {}))
    assert np.array_equal(revived, ref)


def test_mask_event_isolation():
    """Pruning entries leaves every other entry's W, m, v, t bit-identical to a run
    with no mask event."""
    def run(event):
        model, X, y = _task()
        opt = Adam(model.parameters(), masks_of(model), lr=1e-2)
        rng = np.random.default_rng(5)
        _train(model, opt, X, y, 50, rng)
        layer = model.linears()[0]
        touched = np.zeros_like(layer.mask)
        touched[0, :3] = True
        if event:
            apply_mask(layer, opt, layer.mask & ~touched)
        # identical, gradient-free steps so only optimizer state propagates
        for _ in range(20):
            for prm in model.parameters():
                prm.grad = np.zeros_like(prm.data)
            opt.step()
        st = opt.state(layer.W)
        return layer.W.data[~touched], {k: st[k][~touched] for k in ("m", "v", "t")}

    W_a, s_a = run(event=False)
    W_b, s_b = run(event=True)
    assert np.array_equal(W_a, W_b)
    for k in s_a:
        assert np.array_equal(s_a[k], s_b[k])
