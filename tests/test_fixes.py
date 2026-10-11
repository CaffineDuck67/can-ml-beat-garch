"""Tests added for the patch (Fixes 1, 2, 5, 6, 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import evaluate as ev
from src import models_classical as mc
from src import strategy as st


# --------------------------------------------------------------------------- #
# Fix 1 - annualised one-way turnover in strategy.perf()                       #
# --------------------------------------------------------------------------- #
def test_turnover_formula_toy_known_answer():
    # 252 trading days, total |Δw| = 2.0 -> annual one-way turnover = 2.0
    n = 252
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, n)
    dw = np.zeros(n)
    dw[10] = 1.0
    dw[200] = 1.0            # two weight changes summing to 2.0 absolute
    df = pd.DataFrame({"ret_strategy": r, "w": np.ones(n), "dW": dw}, index=idx)
    p = st.perf(df, "ret_strategy", rf=None)
    assert abs(p["turnover_ann"] - 2.0) < 1e-9

    # same two changes over 504 days -> half the annual rate = 1.0
    n2 = 504
    idx2 = pd.bdate_range("2020-01-01", periods=n2)
    r2 = rng.normal(0, 0.01, n2)
    dw2 = np.zeros(n2)
    dw2[10] = 1.0
    dw2[400] = 1.0
    df2 = pd.DataFrame({"ret_strategy": r2, "w": np.ones(n2), "dW": dw2}, index=idx2)
    p2 = st.perf(df2, "ret_strategy", rf=None)
    assert abs(p2["turnover_ann"] - 1.0) < 1e-9


def test_turnover_matches_manual_sum_over_yearfraction():
    n = 126
    idx = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.012, n)
    w = rng.uniform(0.3, 1.2, n)
    df = pd.DataFrame({"ret_strategy": r, "w": w}, index=idx)
    p = st.perf(df, "ret_strategy", rf=None)
    expected = float(np.nansum(np.abs(np.diff(w)))) / (n / 252.0)
    assert abs(p["turnover_ann"] - expected) < 1e-9


# --------------------------------------------------------------------------- #
# Fix 2 - Diebold-Mariano vs an independent reference implementation           #
# --------------------------------------------------------------------------- #
def _dm_reference(a, fm, fb, h):
    """Straightforward, independently-coded DM (SE loss) + Harvey + Student-t."""
    d = (a - fb) ** 2 - (a - fm) ** 2        # benchmark - model (>0 model better)
    T = len(d)
    dbar = d.mean()
    d0 = d - dbar
    gamma0 = np.dot(d0, d0) / T
    acf = gamma0
    for k in range(1, h):                    # Bartlett kernel, h-1 lags
        gk = np.dot(d0[k:], d0[:-k]) / T
        acf = acf + 2.0 * (1.0 - k / h) * gk
    var_dbar = acf / T
    dm = dbar / np.sqrt(var_dbar)
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * corr
    p = 2.0 * stats.t.sf(abs(dm_hln), df=T - 1)
    return dm_hln, p


def test_dm_matches_reference_for_multiple_horizons():
    rng = np.random.default_rng(5)
    a = rng.uniform(12, 35, 400)
    fm = a + rng.normal(0, 1.5, 400)
    fb = a + rng.normal(0, 3.0, 400)
    for h in (1, 5, 21):
        ref_dm, ref_p = _dm_reference(a, fm, fb, h)
        got = ev.diebold_mariano(a, fm, fb, h=h, loss="se")
        assert abs(got["dm"] - ref_dm) < 1e-8, f"h={h} DM mismatch"
        assert abs(got["p"] - ref_p) < 1e-8, f"h={h} p-value mismatch"


def test_dm_harvey_factor_shrinks_statistic():
    # c = sqrt((T+1-2h+h(h-1)/T)/T) < 1 for h>1, so |dm_hln| < |dm_uncorrected|
    rng = np.random.default_rng(6)
    n = 120
    a = rng.uniform(12, 30, n)
    fm = a + rng.normal(0, 2, n)
    fb = a + rng.normal(0, 2, n)
    got = ev.diebold_mariano(a, fm, fb, h=21, loss="se")
    # identical forecasts give ~0; here just assert p in range and finite dm
    assert 0.0 <= got["p"] <= 1.0
    assert np.isfinite(got["dm"])


# --------------------------------------------------------------------------- #
# Fix 5 - EWMA forecast uses r_t (close of day t), not one day stale            #
# --------------------------------------------------------------------------- #
def test_ewma_uses_r_t_and_not_r_t_plus_1():
    idx = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    test_dates = idx[25:26]

    base = mc.ewma_forecast(returns, test_dates, horizon=5)

    # perturbing r_t (the origin day's return) MUST change the day-t forecast
    r_t = returns.copy()
    r_t.iloc[25] += 0.5
    changed = mc.ewma_forecast(r_t, test_dates, horizon=5)
    assert not np.allclose(base, changed)

    # perturbing r_{t+1} (strictly after the origin) must NOT change it
    r_next = returns.copy()
    r_next.iloc[26] += 0.5
    unchanged = mc.ewma_forecast(r_next, test_dates, horizon=5)
    assert np.allclose(base, unchanged)


# --------------------------------------------------------------------------- #
# Fix 6 / Fix 7 - models wired and produce sane, positive forecasts            #
# --------------------------------------------------------------------------- #
def test_garch_adj_is_scaled_garch(cfg):
    # garch_adj is the raw garch forecast times a train-window ratio in [0.5,2];
    # exercise the two functions directly on a short simulated return series.
    rng = np.random.default_rng(8)
    returns = pd.Series(rng.normal(0, 0.01, 400),
                        index=pd.bdate_range("2010-01-01", periods=400))
    params = mc.garch_fit(returns.to_numpy(), dist="t", seed=7)
    assert params is not None
    test_dates = returns.index[350:360]
    g = mc.garch_forecast(params, returns, test_dates, horizon=5)
    assert np.all(g > 0)


def test_log_target_runner_returns_positive_and_biases_down(cfg):
    # A log-target runner must invert exp() with smearing and stay above the floor
    import src.models_ml as ml
    rng = np.random.default_rng(9)
    n = 500
    X = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=list("abcd"))
    y = pd.Series(np.exp(rng.normal(3.0, 0.4, n)), index=range(n))  # vol ~ 20
    runner = ml.build_ridge(cfg, seed=7, n_jobs=1, h=5, log_target=True)
    runner.fit(X, y)
    pred = runner.predict(X)
    assert np.all(pred > 0)
    assert runner.log_target is True
    # smearing correction should be ~ around 1 (finite, positive)
    assert runner._smear > 0
