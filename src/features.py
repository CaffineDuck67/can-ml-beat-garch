"""Step 4 - feature engineering (strictly information-up-to-t).

Every feature at time ``t`` may only use observations dated ``<= t``. We rely
on backward-looking ``rolling`` windows and ``diff``/``shift`` operators, which
never touch the future. A dedicated pytest (tests/test_no_leakage.py) proves
this by perturbing post-t data and asserting features at t are unchanged.

Feature groups
  * Realized-vol components (HAR): 1d, 5d, 21d, 63d annualized vol (%).
  * Return dynamics: 1/5/21d cumulative returns, 63d skew & kurtosis, downside
    semi-variance.
  * VIX: level, 5d change, and the variance-risk-premium proxy (VIX - RV21).
  * Rates: 10Y-2Y slope, 21d change in 10Y yield, 21d change in fed funds.
  * Credit: high-yield OAS level and 21d change.
  * Commodities / macro: 21d oil return, unemployment change, CPI YoY.

Monthly macro columns in the clean panel are ALREADY publication-lag shifted
(see data_clean.py), so differencing them here stays leak-safe.
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd

from src import utils

log = utils.get_logger("features")

TD = 252  # trading days per year (annualization)


def _rolling_vol(returns: pd.Series, w: int) -> pd.Series:
    """Annualized realized vol (%) over the *trailing* ``w`` days ending t."""
    r2 = returns.pow(2)
    return np.sqrt((TD / w) * r2.rolling(w).sum()) * 100.0


def build_features(cfg: dict) -> pd.DataFrame:
    clean = utils.read_table(utils.CLEAN_DAILY)
    r = clean[utils.SP500_RET]
    f = pd.DataFrame(index=clean.index)

    # ---- HAR realized-vol components (trailing, end at t) ---- #
    for w in cfg["features"]["vol_windows"]:
        f[f"rv_{w}"] = _rolling_vol(r, w)

    # ---- return dynamics ---- #
    for lag in cfg["features"]["return_lags"]:
        f[f"ret_{lag}"] = r.rolling(lag).sum()          # cumulative past return
    win = cfg["features"]["skew_kurt_window"]
    f["ret_skew_63"] = r.rolling(win).skew()
    f["ret_kurt_63"] = r.rolling(win).kurt()
    downside = r.clip(upper=0.0).pow(2)
    f["downside_var_63"] = np.sqrt((TD / win) * downside.rolling(win).sum()) * 100.0

    # ---- VIX ---- #
    f["vix_level"] = clean[utils.VIX]
    vc = cfg["features"]["vix_change_window"]
    f["vix_chg_5"] = clean[utils.VIX].diff(vc)
    f["vrp_21"] = clean[utils.VIX] - f["rv_21"]          # variance risk premium

    # ---- rates ---- #
    f["yc_slope"] = clean["T10Y2Y"]
    f["dgs10_chg_21"] = clean["DGS10"].diff(cfg["features"]["yield_change_window"])
    f["dff_chg_21"] = clean["DFF"].diff(cfg["features"]["rate_change_window"])

    # ---- credit / financial stress ---- #
    # BAMLH0A0HYM2 (ICE BofA high-yield OAS) was truncated by FRED in 2024 and
    # now only serves ~2023+, so it cannot cover the 2010+ OOS window. We use it
    # only when it has enough coverage; otherwise we substitute the Chicago Fed
    # NFCI financial-conditions index (full weekly history). The choice is logged
    # and reported honestly in the paper.
    sw = cfg["features"]["spread_change_window"]
    hy = clean.get("BAMLH0A0HYM2")
    notes: dict = {}
    hy_cov = float(hy.notna().mean()) if hy is not None else 0.0
    if hy is not None and hy_cov >= float(cfg["data"].get("hy_min_coverage", 0.8)):
        f["stress_level"] = hy
        f["stress_chg"] = hy.diff(sw)
        notes["credit_proxy"] = "BAMLH0A0HYM2"
        log.info("Credit feature uses BAMLH0A0HYM2 (coverage %.1f%%)", 100 * hy_cov)
    else:
        f["stress_level"] = clean["NFCI"]
        f["stress_chg"] = clean["NFCI"].diff(sw)
        notes["credit_proxy"] = "NFCI (substitute: BAMLH0A0HYM2 coverage " \
                               f"only {100*hy_cov:.1f}% after FRED truncation)"
        log.warning("BAMLH0A0HYM2 coverage %.1f%% -> substituting NFCI for the "
                    "credit-stress feature", 100 * hy_cov)
    notes["hy_coverage_pct"] = round(100 * hy_cov, 2)
    utils.save_json(notes, utils.PROCESSED_DIR / "feature_notes.json")

    # ---- commodities / macro (all backward-looking) ---- #
    ow = cfg["features"]["oil_return_window"]
    # WTI printed negative once (2020-04-20); mask non-positive prices so the
    # log-return is NaN there (dropped later) instead of -inf.
    oil_px = clean["DCOILWTICO"].where(clean["DCOILWTICO"] > 0)
    f["oil_ret_21"] = np.log(oil_px / oil_px.shift(ow)) * 100

    uw = cfg["features"]["unemployment_change_window"]      # months
    f["unrate_chg_3m"] = clean["UNRATE"].diff(uw * 21)       # ~uw months of trading days

    cw = cfg["features"]["cpi_yoy_window"]                  # months
    f["cpi_yoy"] = (clean["CPIAUCSL"] / clean["CPIAUCSL"].shift(cw * 21) - 1.0) * 100

    return f


def feature_columns(cfg: dict, feats: pd.DataFrame) -> List[str]:
    """Ordered, stable list of ML feature columns."""
    return list(feats.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build features (Step 4)")
    utils.add_common_args(parser)
    args = parser.parse_args()
    cfg = utils.config_from_args(args)

    feats = build_features(cfg)
    # Drop rows still containing NaN (start-up warm-up windows). This is done
    # only now, AFTER feature construction, as required by the spec.
    n0 = len(feats)
    feats = feats.dropna()
    log.info("Dropped %d warm-up/NaN rows -> %d kept", n0 - len(feats), len(feats))
    feats.to_csv(utils.FEATURES)
    log.info("Wrote %s shape=%s (%d features)", utils.FEATURES.name,
             feats.shape, feats.shape[1])


if __name__ == "__main__":
    main()
