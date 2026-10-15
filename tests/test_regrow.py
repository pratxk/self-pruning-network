import numpy as np

from engine.functional import softmax_cross_entropy
from engine.tensor import Tensor
from nn.layers import MLP
from nn.optim import Adam, masks_of
from prune.criteria import FisherOBD
from prune.pruner import Pruner
from prune.regrow import Regrower
from prune.schedule import ConstantSchedule


def _setup(sparsity=0.8, seed=0):
    rng = np.random.default_rng(seed)
    model = MLP((20, 30, 15, 4), rng=rng, dtype=np.float64)
    opt = Adam(model.parameters(), masks_of(model), lr=1e-2)
    X, y = rng.standard_normal((64, 20)), rng.integers(0, 4, 64)
    pruner = Pruner(model, opt, FisherOBD(), ConstantSchedule(), rng=rng)
    reg = Regrower(pruner, stop_step=1000, every=10, alpha0=0.3, grace=2)

    def step():
        opt.zero_grad()
        softmax_cross_entropy(model(Tensor(X)), y).backward()
        reg.accumulate()
        opt.step()

    for _ in range(20):
        step()
    pruner.apply_masks(pruner.compute_masks(sparsity))
    for _ in range(10):
        step()
    return model, opt, pruner, reg, step


def test_regrow_keeps_budget_and_grows_fresh():
    model, opt, pruner, reg, _ = _setup()
    before_active = sum(l.n_active() for l in model.linears())
    old = [l.mask.copy() for l in model.linears()]
    ev = reg.maybe_regrow(10)
    assert ev is not None and ev.dropped == ev.grown > 0
    assert sum(l.n_active() for l in model.linears()) == before_active
    for l, o in zip(model.linears(), old):
        grown = l.mask & ~o
        assert np.all(l.W.data[grown] == 0.0)  # starts at zero: function unchanged
        st = opt.state(l.W)
        assert np.all(st["m"][grown] == 0) and np.all(st["t"][grown] == 0)
        assert np.all(l.W.data[~l.mask] == 0.0)


def test_grown_weights_protected_during_grace():
    model, _, pruner, reg, step = _setup()
    old = [l.mask.copy() for l in model.linears()]
    reg.maybe_regrow(10)
    grown = [l.mask & ~o for l, o in zip(model.linears(), old)]
    step()
    # scheduled pruning to higher sparsity must not remove protected weights
    pruner.apply_masks(pruner.compute_masks(0.9))
    for l, g in zip(model.linears(), grown):
        assert np.all(l.mask[g])


def test_alpha_cosine_annealed():
    *_, reg, _ = _setup()
    assert reg.alpha(0) == 0.3 and abs(reg.alpha(1000)) < 1e-12
    assert reg.maybe_regrow(1000) is None and reg.maybe_regrow(15) is None


def test_churn_counts_regrown_then_dropped():
    model, _, pruner, reg, step = _setup()
    grown_total = 0
    for s in range(10, 200, 10):
        step()
        if (ev := reg.maybe_regrow(s)) is not None:
            grown_total += ev.grown
    # churn counts drop events of previously grown weights (a weight may cycle twice)
    assert 0 < reg.churn <= grown_total
