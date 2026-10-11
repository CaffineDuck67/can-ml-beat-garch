"""Step 5 - classical volatility-forecast baselines.

All forecasts for horizon ``h`` target annualized realized vol in percent:

    RV_h(t) = sqrt( (252/h) * sum_{i=1..h} r_{t+i}^2 ) * 100

and must use ONLY information available up to (and including) day t.

Models implemented
  * Historical-mean volatility   : expanding mean of the (past) realized target.
  * EWMA                          : RiskMetrics, lambda = 0.94.
  * GARCH(1,1) with Student-t     : `arch` fit on an expanding window; multi-step
                                    forecast obtained by continuing the variance
                                    recursion with FIXED parameters (so no future
                                    returns enter a day-t forecast).
  * HAR-RV (OLS)                  : daily / weekly / monthly RV components.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TD = 252  # trading days per year

# feature column names produced by features.py (HAR components)
HAR_COMPONENTS = ["rv_1", "rv_5", "rv_21"]


def _annualized_vol_from_var(var_daily: np.ndarray, h: int) -> float:
    """Turn a per-day variance forecast into the h-day realized-vol target (%)."""
    return float(np.sqrt((TD / h) * float(np.sum(var_daily))) * 100.0)


# --------------------------------------------------------------------------- #
# Historical mean                                                              #
# --------------------------------------------------------------------------- #
def historical_mean(train_target: pd.Series) -> float:
    """Expanding-mean volatility forecast (constant across the test block)."""
    return float(np.nanmean(train_target.to_numpy()))


# --------------------------------------------------------------------------- #
# EWMA (RiskMetrics)                                                           #
# --------------------------------------------------------------------------- #
def ewma_forecast(returns: pd.Series, test_dates: pd.DatetimeIndex,
                  horizon: int, lam: float = 0.94) -> np.ndarray:
    """EWMA variance forecast evaluated at each test day using returns <= t.

    sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r_t^2

    The forecast origin t uses the return realised *at the close of day t*
    (r_t), which is information available at t. This is deliberately made
    consistent with the GARCH conditional   variance (which uses r_t) and with
    HAR-RV (whose day-t regressors rv_1..rv_21 also embed r_t). It previously
    carried a spurious one-day handicap via a ``.shift(1)``.
    """
    r2 = returns.pow(2).reindex(test_dates.union(returns.index)).sort_index()
    # recursive EWMA with adjust=False on r^2; NO shift -> sigma^2_t uses r_t.
    ewma = r2.ewm(alpha=1.0 - lam, adjust=False).mean()
    sigma2 = ewma.reindex(test_dates).to_numpy()
    return np.sqrt(TD * sigma2) * 100.0


# --------------------------------------------------------------------------- #
# GARCH(1,1)                                                                   #
# --------------------------------------------------------------------------- #
def garch_fit(train_returns: np.ndarray, dist: str = "t",
              seed: int = 7) -> Optional[Dict[str, float]]:
    """Fit GARCH(1,1) and return the (omega, alpha, beta) variance parameters."""
    from arch import arch_model

    raw = np.asarray(train_returns, dtype=float)
    raw = raw[~np.isnan(raw)]
    # arch's optimizer is far better behaved when the return unit is percent,
    # not decimal; alpha/beta are scale-invariant and omega is rescaled back.
    clean = raw * 100.0
    am = arch_model(clean, vol="GARCH", p=1, q=1, mean="Zero", dist=dist)
    try:
        res = am.fit(disp="off", show_warning=False)
    except Exception:
        try:
            res = am.fit(disp="off", show_warning=False,
                         options={"maxiter": 200})
        except Exception:
            return None
    p = res.params
    return {
        "omega": float(p.get("omega", 1e-6)) / 1e4,   # back to decimal-variance units
        "alpha": float(p.get("alpha[1]", 0.05)),
        "beta": float(p.get("beta[1]", 0.90)),
        "uncond": None,
    }


def garch_forecast(params: Dict[str, float], returns: pd.Series,
                   test_dates: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """Multi-step h-day RV forecast at each test date, using only r <= t.

    The variance recursion is run forward over the whole (aligned) return
    history with the FIXED fitted parameters; then, at each origin t, we know
    sigma^2_t and r_t and roll the h-step conditional-variance expectation
    forward analytically. No future return is ever used for a day-t forecast.
    """
    omega, alpha, beta = params["omega"], params["alpha"], params["beta"]
    phi = min(alpha + beta, 0.999)
    uncond = omega / (1.0 - phi)

    r = returns.reindex(test_dates.union(returns.index)).sort_index()
    r_full = r.to_numpy(dtype=float)
    n = len(r_full)

    sigma2 = np.full(n, uncond)
    for t in range(1, n):
        prev_r = r_full[t - 1] if np.isfinite(r_full[t - 1]) else 0.0
        sigma2[t] = omega + alpha * prev_r ** 2 + beta * sigma2[t - 1]

    # map each test date -> its position in r_full
    pos = {d: i for i, d in enumerate(r.index)}
    out = np.empty(len(test_dates))
    for j, d in enumerate(test_dates):
        i = pos[d]
        r_t = r_full[i] if np.isfinite(r_full[i]) else 0.0
        s_t = sigma2[i]
        v1 = omega + alpha * r_t ** 2 + beta * s_t          # E_t[sigma^2_{t+1}]
        k = np.arange(horizon)
        v = uncond + (phi ** k) * (v1 - uncond)             # v_{t+1..t+h}
        out[j] = _annualized_vol_from_var(v, horizon)
    return out


# --------------------------------------------------------------------------- #
# HAR-RV (OLS)                                                                 #
# --------------------------------------------------------------------------- #
def har_fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    target_col: str, lags: list[str] | None = None) -> np.ndarray:
    """Fit HAR-RV OLS on the training rows and predict on the test rows.

    Uses only past realized-vol components (rv_1, rv_5, rv_21) as regressors.
    """
    import statsmodels.api as sm

    lags = lags or HAR_COMPONENTS
    X_tr = sm.add_constant(train_df[lags].to_numpy(dtype=float))
    y_tr = train_df[target_col].to_numpy(dtype=float)
    mask = ~np.isnan(y_tr) & ~np.isnan(X_tr).any(axis=1)
    model = sm.OLS(y_tr[mask], X_tr[mask]).fit()

    X_te = sm.add_constant(test_df[lags].to_numpy(dtype=float), has_constant="add")
    pred = model.predict(X_te)
    # vol cannot be negative
    return np.clip(np.asarray(pred, dtype=float), 0.0, None)
