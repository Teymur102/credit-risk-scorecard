import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scorecard.metrics import ks_statistic, psi  # noqa: E402
from scorecard.points import (ScaleSpec, attribute_points, pd_from_score,  # noqa: E402
                              score_from_pd)
from scorecard.woe import SMOOTH, fit_binning, monotonic_edges, woe_iv_table  # noqa: E402


# ---------------- WoE / IV ----------------
def test_woe_iv_hand_computed():
    # bin 0: 80 goods, 20 bads; bin 1: 20 goods, 80 bads
    idx = np.array([0] * 100 + [1] * 100)
    y = np.array([0] * 80 + [1] * 20 + [0] * 20 + [1] * 80)
    t = woe_iv_table(idx, y, 2)
    g = (np.array([80, 20]) + SMOOTH) / (100 + 2 * SMOOTH)
    b = (np.array([20, 80]) + SMOOTH) / (100 + 2 * SMOOTH)
    np.testing.assert_allclose(t.woe, np.log(g / b))
    np.testing.assert_allclose(t.iv.sum(), np.sum((g - b) * np.log(g / b)))
    assert t.woe[0] > 0 > t.woe[1]  # safer bin has positive WoE
    assert t.iv.sum() == pytest.approx(1.2 * np.log(4), rel=0.03)  # unsmoothed value 1.663


def test_iv_zero_when_feature_uninformative():
    idx = np.tile([0, 1], 500)
    y = np.repeat([0, 1], 500)  # same bad rate in both bins
    assert woe_iv_table(idx, y, 2).iv.sum() == pytest.approx(0.0, abs=1e-12)


def test_monotonic_binning_is_monotonic_and_respects_min_size():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    p = 1 / (1 + np.exp(-(1.5 * x + 0.8 * np.sin(4 * x))))  # monotone trend + wiggle
    y = rng.binomial(1, p)
    edges = monotonic_edges(x, y, max_bins=20, min_frac=0.05)
    fb = fit_binning(x, y, "x", max_bins=20, min_frac=0.05)
    rates = fb.table.bad_rate.to_numpy()
    assert np.all(np.diff(rates) > 0)
    assert fb.table.pct_pop.min() >= 0.05
    assert len(edges) == len(fb.edges)


def test_categorical_binning_maps_unseen_levels():
    x = np.array(["a"] * 600 + ["b"] * 390 + ["c"] * 10)
    y = np.array([0] * 500 + [1] * 100 + [0] * 200 + [1] * 190 + [1] * 10)
    fb = fit_binning(x, y, "cat", kind="categorical", min_frac=0.05)
    assert np.isfinite(fb.transform(np.array(["zzz"]))).all()


# ---------------- points formula ----------------
def test_scaling_constants():
    s = ScaleSpec(pdo=20, base_score=600, base_odds=50)
    assert s.factor == pytest.approx(20 / np.log(2))
    assert s.offset == pytest.approx(600 - 20 / np.log(2) * np.log(50))


def test_base_odds_map_to_base_score_and_pdo_doubles_odds():
    s = ScaleSpec(pdo=20, base_score=600, base_odds=50)
    pd_base = 1 / 51  # odds 50:1 good:bad
    assert score_from_pd(pd_base, s) == pytest.approx(600)
    pd_double = 1 / 101  # odds 100:1
    assert score_from_pd(pd_double, s) == pytest.approx(620)
    assert pd_from_score(620, s) == pytest.approx(pd_double)


def test_attribute_points_sum_to_score():
    s = ScaleSpec()
    b0, coefs, woes = -1.2, np.array([-0.6, -0.3, -0.9]), np.array([0.4, -1.1, 0.2])
    total = sum(attribute_points(w, c, b0, 3, s) for w, c in zip(woes, coefs))
    logit_bad = b0 + coefs @ woes
    pd_bad = 1 / (1 + np.exp(-logit_bad))
    assert total == pytest.approx(score_from_pd(pd_bad, s))


# ---------------- KS ----------------
def test_ks_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert ks_statistic(y, np.array([1, 2, 3, 4, 5, 6])) == pytest.approx(1.0)
    assert ks_statistic(y, np.ones(6)) == pytest.approx(0.0)


def test_ks_hand_example_with_ties():
    # risk ordering: 0.9(bad) 0.8(good) 0.8(bad) 0.3(good) 0.1(good) 0.1(bad)
    y = np.array([1, 0, 1, 0, 0, 1])
    r = np.array([0.9, 0.8, 0.8, 0.3, 0.1, 0.1])
    # at r>=0.9: bad 1/3 good 0 -> 0.333; r>=0.8: bad 2/3 good 1/3 -> 0.333;
    # r>=0.3: 2/3 vs 2/3 -> 0; all: 1 vs 1 -> 0
    assert ks_statistic(y, r) == pytest.approx(1 / 3)


def test_ks_matches_roc_definition():
    from sklearn.metrics import roc_curve
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    r = y * 0.5 + rng.normal(size=5000)
    fpr, tpr, _ = roc_curve(y, r)
    assert ks_statistic(y, r) == pytest.approx(np.max(tpr - fpr))


def test_psi_zero_for_identical_and_positive_for_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(size=10000)
    assert psi(a, a) == pytest.approx(0.0, abs=1e-12)
    assert psi(a, a + 0.5) > 0.1
