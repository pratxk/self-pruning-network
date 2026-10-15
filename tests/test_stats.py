import numpy as np
import pytest

from train.stats import bootstrap_ci, holm, mcnemar_exact, sign_flip_p


def test_sign_flip_floor_with_5_seeds():
    assert sign_flip_p(np.array([1.0, 2, 3, 4, 5])) == pytest.approx(2 / 32)


def test_sign_flip_10_seeds_can_reach_005():
    assert sign_flip_p(np.arange(1.0, 11)) == pytest.approx(2 / 1024)


def test_sign_flip_null():
    assert sign_flip_p(np.array([1.0, -1, 1, -1])) == 1.0


def test_bootstrap_constant_and_coverage():
    assert bootstrap_ci(np.full(5, 0.7)) == pytest.approx((0.7, 0.7))
    lo, hi = bootstrap_ci(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert 0.1 <= lo < 0.3 < hi <= 0.5


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_mcnemar():
    a = np.array([1] * 10 + [0] * 5, bool)
    b = np.array([0] * 10 + [0] * 5, bool)
    assert mcnemar_exact(a, b) == (10, 0, pytest.approx(2 / 1024))
    assert mcnemar_exact(a, a)[2] == 1.0
