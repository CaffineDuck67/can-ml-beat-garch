"""Step 2 - cleaning and leak-safe alignment.

Everything is aligned to S&P 500 trading days (the master calendar).

Leakage rules implemented here (documented again in the paper):
  * Daily macro series (yields, VIX, oil, HY spread): forward-filled only.
  * Monthly series (UNRATE, CPIAUCSL) are NOT contemporaneous - they are
    published with a lag. We shift each monthly observation forward by its
    publication lag (default ~40 days) BEFORE forward-filling, so a value for
    reference month m can only be used as a feature on/after m + lag. This
    guarantees the model only sees data that was genuinely public on date t.

The cleaned panel still contains NaNs at the very start (e.g. before oil or
spread data begin); those rows are dropped only AFTER feature construction.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import utils

log = utils.get_logger("data_clean")


def _load_raw_csv(name: str, col: str) -> pd.Series:
    path = utils.RAW_DIR / f"{name}.csv.gz"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if col in df.columns:
        s = df[col]
    else:  # FRED files have the series id as the only data column
        s = df.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return pd.to_numeric(s, errors="coerce").sort_index()


def _load_fred(series_id: str) -> pd.Series:
    return _load_raw_csv(f"FRED_{series_id}", series_id)


def build_clean_panel(cfg: dict) -> pd.DataFrame:
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"])

    # ---- master calendar & price ---- #
    sp = _load_raw_csv("SP500", "Close").rename(utils.SP500_CLOSE)
    sp = sp[(sp.index >= start) & (sp.index <= end)]
    master = sp.index
    log.info("Master trading calendar: %d rows (%s -> %s)",
             len(master), master.min().date(), master.max().date())

    frame = pd.DataFrame(index=master)
    frame[utils.SP500_CLOSE] = sp
    # log returns r_t = ln(P_t / P_{t-1}) - uses only t and t-1 (no lookahead)
    frame[utils.SP500_RET] = np.log(frame[utils.SP500_CLOSE]).diff()

    # ---- VIX ---- #
    vix = _load_raw_csv("VIX", "Close").rename(utils.VIX)
    frame[utils.VIX] = vix.reindex(master).ffill()

    # ---- daily macro (forward-fill only) ---- #
    for sid in cfg["data"]["daily_series"]:
        s = _load_fred(sid)
        frame[sid] = s.reindex(master).ffill()

    # ---- weekly macro (e.g. NFCI): shift by reporting lag, then forward-fill ---- #
    lag_map = cfg["data"]["publication_lag_days"]
    for sid in cfg["data"].get("weekly_series", []):
        s = _load_fred(sid)
        lag = int(lag_map.get(sid, 5))
        shifted = s.copy()
        shifted.index = shifted.index + pd.Timedelta(days=lag)
        frame[sid] = shifted.reindex(master).ffill()
        log.info("Weekly %s shifted by %d reporting-lag days", sid, lag)

    # ---- monthly macro (shift by publication lag, then forward-fill) ---- #
    for sid in cfg["data"]["monthly_series"]:
        s = _load_fred(sid)
        lag = int(lag_map.get(sid, 40))
        # A monthly obs stamped at month-start becomes usable ``lag`` days later.
        shifted = s.copy()
        shifted.index = shifted.index + pd.Timedelta(days=lag)
        frame[sid] = shifted.reindex(master).ffill()
        log.info("Monthly %s shifted by %d publication-lag days", sid, lag)

    # keep the FRED raw levels that later modules need for YoY / changes
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean & align data (Step 2)")
    utils.add_common_args(parser)
    args = parser.parse_args()
    cfg = utils.config_from_args(args)

    if utils.CLEAN_DAILY.exists() and not args.force:
        log.info("clean_daily already exists -> skipping (use --force)")
        return

    frame = build_clean_panel(cfg)
    frame.to_csv(utils.CLEAN_DAILY)
    log.info("Wrote %s  shape=%s", utils.CLEAN_DAILY.name, frame.shape)
    # report coverage of each column
    cov = frame.notna().mean().mul(100).round(2)
    log.info("Column coverage (%% non-NaN):\n%s", cov.to_string())


if __name__ == "__main__":
    main()
