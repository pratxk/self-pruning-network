import numpy as np
import pytest

from engine.functional import softmax_cross_entropy
from engine.tensor import Tensor
from nn.layers import MLP
from nn.optim import Adam, masks_of
from prune.criteria import FisherOBD, FisherOBDFirstOrder, Magnitude, TaylorFirstOrder
from prune.pruner import Pruner
from prune.schedule import ConstantSchedule, CubicSchedule, OneShotSchedule


def test_cubic_schedule():
    s = CubicSchedule(0.9, start_step=100, end_step=1100, every=100)
    assert s.target(0) == 0 and s.target(100) == 0
    assert s.target(1100) == pytest.approx(0.9) and s.target(5000) == pytest.approx(0.9)
    t = [s.target(k) for k in range(0, 1300, 7)]
    assert all(b >= a for a, b in zip(t, t[1:]))
    assert s.is_prune_step(100) and s.is_prune_step(1100) and not s.is_prune_step(150)


def test_cubic_schedule_lands_on_end_step():
    s = CubicSchedule(0.5, 0, 250, every=100)
    assert s.is_prune_step(250)


def test_other_schedules():
    o = OneShotSchedule(0.9, at_step=50)
    assert o.target(49) == 0 and o.target(50) == 0.9 and o.is_prune_step(50)
    assert not ConstantSchedule().is_prune_step(0)


def _trained(seed=0, steps=30):
    rng = np.random.default_rng(seed)
    model = MLP((20, 30, 15, 4), rng=rng, dtype=np.float64)
    opt = Adam(model.parameters(), masks_of(model), lr=1e-2)
    X, y = rng.standard_normal((64, 20)), rng.integers(0, 4, 64)
    for _ in range(steps):
        opt.zero_grad()
        softmax_cross_entropy(model(Tensor(X)), y).backward()
        opt.step()
    return model, opt


@pytest.mark.parametrize("scope", ["global", "layerwise"])
@pytest.mark.parametrize("crit", [Magnitude(), TaylorFirstOrder(), FisherOBD(),
                                  FisherOBDFirstOrder()])
def test_exact_budget(scope, crit):
    model, opt = _trained()
    p = Pruner(model, opt, crit, ConstantSchedule(), scope=scope)
    masks = p.compute_masks(0.9)
    assert sum(m.sum() for m in masks) == round(0.1 * p.total())  # kept count exact
    p.apply_masks(masks)
    p.remove_dead_units()
    assert p.sparsity() >= 0.9 - 1e-9


def test_masks_only_shrink_and_weights_zeroed():
    model, opt = _trained()
    p = Pruner(model, opt, Magnitude(), ConstantSchedule())
    p.apply_masks(p.compute_masks(0.5))
    before = [l.mask.copy() for l in model.linears()]
    p.apply_masks(p.compute_masks(0.8))
    for l, b in zip(model.linears(), before):
        assert not (l.mask & ~b).any()
        assert np.all(l.W.data[~l.mask] == 0)


def test_floors():
    model, opt = _trained()
    p = Pruner(model, opt, Magnitude(), ConstantSchedule())
    p.apply_masks(p.compute_masks(0.97))
    dens = p.densities()
    assert min(dens) >= 0.01 - 1e-12
    assert dens[-1] >= 0.2 - 1e-12  # output layer <= 80% sparsity


def test_fisher_score_formula():
    model, opt = _trained()
    layer = model.linears()[0]
    st = opt.state(layer.W)
    w = layer.W.data
    assert np.allclose(FisherOBD().score(layer, st), 0.5 * st["v_hat"] * w**2)
    layer.mask[0, 0] = False
    assert FisherOBD().score(layer, st)[0, 0] == -np.inf


def test_fisher_scores_hand_built_state():
    model, _ = _trained()
    layer = model.linears()[2]
    layer.W.data[...] = 2.0
    st = {"m_hat": np.full(layer.W.shape, 0.5), "v_hat": np.full(layer.W.shape, 3.0)}
    assert np.all(FisherOBD().score(layer, st) == 0.5 * 3.0 * 4.0)
    fo = FisherOBDFirstOrder(batch_size=128).score(layer, st)
    assert np.all(fo == 0.5 * 128 * 3.0 * 4.0 - 0.5 * 2.0)


def test_dead_units_removed():
    model, opt = _trained()
    p = Pruner(model, opt, Magnitude(), ConstantSchedule())
    a, b = model.linears()[:2]
    m = a.mask.copy()
    m[:, 3] = False  # hidden unit 3: all incoming pruned
    p.apply_masks([m, b.mask.copy(), model.linears()[2].mask.copy()])
    removed = p.remove_dead_units()
    assert removed == b.mask.shape[1]
    assert not b.mask[3].any()


def test_maybe_prune_event():
    model, opt = _trained()
    p = Pruner(model, opt, FisherOBD(), CubicSchedule(0.9, 0, 10, every=5))
    assert p.maybe_prune(3) is None
    ev = p.maybe_prune(10)
    assert ev is not None and ev.achieved >= 0.9 - 1e-9 and ev.target == pytest.approx(0.9)


def test_99pct_budget_beats_floors():
    model, opt = _trained()
    p = Pruner(model, opt, Magnitude(), ConstantSchedule())
    p.apply_masks(p.compute_masks(0.99))
    p.remove_dead_units()
    assert p.sparsity() >= 0.99 - 1e-9
