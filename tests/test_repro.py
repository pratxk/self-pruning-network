import numpy as np

from train.config import RunConfig
from train.data import Splits
from train.loop import fit


def _toy_splits(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((600, 20)).astype(np.float32)
    y = rng.integers(0, 4, 600)
    return Splits(X[:400], y[:400], X[400:500], y[400:500], X[500:], y[500:])


def _run(seed, **kw):
    cfg = RunConfig(seed=seed, sizes=(20, 16, 8, 4), epochs=1, batch_size=32, **kw)
    return fit(cfg, _toy_splits(), log=lambda s: None)


def test_same_seed_identical_weights():
    a, b = _run(0), _run(0)
    for la, lb in zip(a.model.linears(), b.model.linears()):
        assert np.array_equal(la.W.data, lb.W.data)
        assert np.array_equal(la.b.data, lb.b.data)


def test_different_seed_differs():
    a, b = _run(0), _run(1)
    assert not np.array_equal(a.model.linears()[0].W.data, b.model.linears()[0].W.data)


def test_pruned_run_identical_and_on_budget():
    kw = dict(method="fisher_obd", target_sparsity=0.8, epochs=3,
              prune_start_epoch=1, prune_end_epoch=2, prune_every=5)
    cfg = RunConfig(seed=0, sizes=(20, 16, 8, 4), batch_size=32, **kw)
    a = fit(cfg, _toy_splits(), log=lambda s: None)
    b = fit(cfg, _toy_splits(), log=lambda s: None)
    assert a.pruner.sparsity() >= 0.8 - 1e-9
    for la, lb in zip(a.model.linears(), b.model.linears()):
        assert np.array_equal(la.mask, lb.mask) and np.array_equal(la.W.data, lb.W.data)
