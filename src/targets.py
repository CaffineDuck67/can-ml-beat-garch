"""Step 3 - forecast targets.

Realized-volatility proxy (annualized, expressed in percent like the VIX):

    RV_h(t) = sqrt( (252 / h) * sum_{i=1..h} r_{t+i}^2 ) * 100

This is *forward*-looking: it uses returns from t+1 through t+h and is the
quantity a model must predict using only information available up to (and
including) day t. It is therefore the target, never a feature.

Bonus classification target:
    regime_flag(t) = 1 if the maximum drawdown of the price path over the next
    ``window`` trading days (t .. t+window) exceeds the configured threshold.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import utils

log = utils.get_logger("targets")


def forward_realized_vol(returns: pd.Series, h: int, days_per_year: int) -> pd.Series:
    """Annualized forward realized vol (in %) over the next ``h`` days.

    Uses only r_{t+1}..r_{t+h}. The last ``h`` rows are NaN by construction
    (the forward window is incomplete).
    """
    r2 = returns.pow(2)
    back_sum = r2.rolling(h).sum()          # sum of r2[t-h+1 .. t]
    forward_sum = back_sum.shift(-h)        # value at t -> sum r2[t+1 .. t+h]
    rv = np.sqrt((days_per_year / h) * forward_sum) * 100.0
    rv.name = f"rv_h{h}_raw"
    return rv


def forward_max_drawdown(prices: pd.Series, window: int) -> np.ndarray:
    """Max drawdown (as a positive fraction) within the forward window t..t+w.

    For every origin t we walk the price path over the next ``window`` days,
    track the running peak and record the largest peak-to-trough decline.
    """
    p = prices.to_numpy(dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    for t in range(n - window):
        seg = p[t: t + window + 1]
        peak = np.maximum.accumulate(seg)
        dd = 1.0 - seg / peak
        out[t] = dd.max()
    return out


def build_targets(cfg: dict) -> pd.DataFrame:
    clean = utils.read_table(utils.CLEAN_DAILY)
    ret = clean[utils.SP500_RET]
    px = clean[utils.SP500_CLOSE]
    dpy = cfg["targets"]["trading_days_per_year"]

    out = pd.DataFrame(index=clean.index)
    for h in cfg["targets"]["horizons"]:
        col = utils.rv_target_col(h)
        out[col] = forward_realized_vol(ret, h, dpy)
        log.info("Target %s: %d non-NaN, mean %.2f%%", col,
                 out[col].notna().sum(), out[col].mean())

    reg = cfg["targets"]["regime"]
    mdd = forward_max_drawdown(px, int(reg["window"]))
    out["regime_flag"] = np.where(
        np.isnan(mdd), np.nan,
        (mdd > reg["max_drawdown_threshold"]).astype(float))
    out["max_drawdown_fwd"] = mdd
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build targets (Step 3)")
    utils.add_common_args(parser)
    args = parser.parse_args()
    cfg = utils.config_from_args(args)

    if utils.TARGETS.exists() and not args.force:
        log.info("targets already exist -> skipping (use --force)")
        return
    tg = build_targets(cfg)
    tg.to_csv(utils.TARGETS)
    log.info("Wrote %s shape=%s", utils.TARGETS.name, tg.shape)


if __name__ == "__main__":
    main()
