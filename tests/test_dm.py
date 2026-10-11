"""Diebold-Mariano sanity checks and loss-function tests."""
from __future__ import annotations

import numpy as np

from src import evaluate as ev


def test_identical_forecasts_give_zero_dm():
    rng = np.random.default_rng(0)
    a = rng.uniform(10, 30, 300)
    res = ev.diebold_mariano(a, a.copy(), a.copy(), h=5)
    assert abs(res["dm"]) < 1e-8
    assert 0.0 <= res["p"] <= 1.0


def test_better_model_has_positive_dm_and_small_p():
    rng = np.random.default_rng(1)
    n = 500
    a = rng.uniform(12, 35, n)
    # benchmark is noisy, model is close to the truth
    bench = a + rng.normal(0, 4.0, n)
    model = a + rng.normal(0, 0.5, n)
    res = ev.diebold_mariano(a, model, bench, h=5)
    assert res["dm"] > 0            # model lower loss -> positive DM
    assert res["p"] < 0.05          # and clearly significant
    # symmetry: swapping gives a negative DM
    rev = ev.diebold_mariano(a, bench, model, h=5)
    assert rev["dm"] < 0


def test_pvalue_in_range():
    rng = np.random.default_rng(2)
    a = rng.uniform(10, 30, 200)
    f1 = a + rng.normal(0, 2, 200)
    f2 = a + rng.normal(0, 2, 200)
    res = ev.diebold_mariano(a, f1, f2, h=21)
    assert 0.0 <= res["p"] <= 1.0


def test_qlike_minimized_by_truth():
    a = np.full(200, 20.0)
    # QLIKE is minimised when the variance forecast equals the true variance,
    # so a perfect forecast must beat a badly biased one.
    perfect = ev.qlike(a, np.full(200, 20.0))
    biased = ev.qlike(a, np.full(200, 5.0))
    assert perfect < biased


def test_r2_oos_zero_for_benchmark():
    rng = np.random.default_rng(3)
    a = rng.uniform(10, 30, 100)
    bench = np.full(100, a.mean())
    assert abs(ev.r2_oos(a, bench, bench)) < 1e-9
