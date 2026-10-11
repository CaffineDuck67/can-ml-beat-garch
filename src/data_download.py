"""Step 1 - data acquisition.

Downloads:
  * S&P 500 (^GSPC) and VIX (^VIX) daily prices via ``yfinance``.
  * Eight macro series from FRED through the *public* CSV endpoint
    (``https://fred.stlouisfed.org/graph/fredgraph.csv``) - no API key needed.

Robustness:
  * retry + exponential backoff + timeout on every HTTP request;
  * ``pandas_datareader`` fallback if the fredgraph endpoint fails;
  * optional ``fredapi`` path when ``FRED_API_KEY`` is present in the env;
  * local caching: existing raw files are reused unless ``--force`` is passed.

A machine-readable audit trail (row counts, date ranges, missing values,
which source was used) is written to ``data/raw/download_log.json``.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from src import utils

log = utils.get_logger("data_download")

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


# --------------------------------------------------------------------------- #
# Low-level HTTP with retry                                                   #
# --------------------------------------------------------------------------- #
def _get_with_retry(url: str, params: Dict[str, str], timeout: int,
                    max_retries: int, backoff_factor: float) -> Optional[bytes]:
    """GET a URL, retrying with exponential backoff. Returns bytes or None."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    delay = backoff_factor
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                return resp.content
            log.warning("Attempt %d/%d got status %s for %s",
                        attempt, max_retries, resp.status_code, params)
        except requests.RequestException as exc:  # network errors
            log.warning("Attempt %d/%d raised %s", attempt, max_retries, exc)
        if attempt < max_retries:
            import time
            time.sleep(delay)
            delay *= 2
    return None


# --------------------------------------------------------------------------- #
# FRED                                                                         #
# --------------------------------------------------------------------------- #
def _parse_fred_csv(content: bytes, series_id: str) -> pd.Series:
    """Parse fredgraph.csv content into a clean date-indexed Series."""
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    date_col = df.columns[0]  # "observation_date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    s = pd.to_numeric(df[series_id], errors="coerce")  # '.' -> NaN
    s.name = series_id
    return s.sort_index()


def download_fred(series_id: str, start: str, end: str, cfg: dict,
                  force: bool) -> tuple[pd.Series, str]:
    """Return (series, source_label). Tries fredapi -> fredgraph -> datareader."""
    dest = utils.RAW_DIR / f"FRED_{series_id}.csv.gz"
    if dest.exists() and not force:
        s = pd.read_csv(dest, index_col=0, parse_dates=True)[series_id]
        log.info("  [cache] %s (%d rows)", series_id, len(s))
        return s, "cache"

    req = cfg["data"]["request"]

    # (a) fredapi if an API key is present
    import os
    api_key = os.environ.get("FRED_API_KEY")
    if api_key:
        try:
            from fredapi import Fred  # type: ignore
            fred = Fred(api_key=api_key)
            s = fred.get_series(series_id, observation_start=start,
                                observation_end=end)
            s = pd.to_numeric(s, errors="coerce").dropna()
            s.name = series_id
            log.info("  [fredapi] %s (%d rows)", series_id, len(s))
            s.to_frame().to_csv(dest)
            return s, "fredapi"
        except Exception as exc:  # pragma: no cover
            log.warning("  fredapi failed for %s (%s); falling back", series_id, exc)

    # (b) public fredgraph.csv endpoint (no key)
    content = _get_with_retry(
        FRED_URL, {"id": series_id, "cosd": start, "coed": end},
        req["timeout"], req["max_retries"], req["backoff_factor"])
    if content is not None:
        s = _parse_fred_csv(content, series_id)
        s.to_frame().to_csv(dest)
        log.info("  [fredgraph] %s (%d rows)", series_id, len(s))
        return s, "fredgraph"

    # (c) pandas_datareader fallback
    try:
        import pandas_datareader.data as web
        s = web.DataReader(series_id, "fred", start=start, end=end)[series_id]
        s = pd.to_numeric(s, errors="coerce")
        s.name = series_id
        s.to_frame().to_csv(dest)
        log.info("  [pandas_datareader] %s (%d rows)", series_id, len(s))
        return s, "pandas_datareader"
    except Exception as exc:
        log.error("  ALL FRED sources failed for %s: %s", series_id, exc)
        raise RuntimeError(f"Could not download FRED series {series_id}") from exc


# --------------------------------------------------------------------------- #
# Prices (yfinance)                                                            #
# --------------------------------------------------------------------------- #
def download_price(ticker: str, name: str, start: str, end: str,
                   cfg: dict, force: bool) -> pd.DataFrame:
    dest = utils.RAW_DIR / f"{name}.csv.gz"
    if dest.exists() and not force:
        df = pd.read_csv(dest, index_col=0, parse_dates=True)
        log.info("  [cache] %s (%d rows)", ticker, len(df))
        return df

    req = cfg["data"]["request"]
    last_err: Optional[Exception] = None
    for attempt in range(1, req["max_retries"] + 1):
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end, progress=False,
                             auto_adjust=True)
            if df is not None and len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                df.index.name = "date"
                df.to_csv(dest)
                log.info("  [yfinance] %s (%d rows)", ticker, len(df))
                return df
            last_err = RuntimeError("empty dataframe")
        except Exception as exc:
            last_err = exc
        import time
        time.sleep(req["backoff_factor"] * (2 ** (attempt - 1)))

    # Stooq fallback via pandas_datareader (daily, adjusted close available)
    log.warning("  yfinance failed for %s (%s); trying stooq fallback", ticker, last_err)
    stooq_map = {"^GSPC": "^spx", "^VIX": "^vix"}
    try:
        import pandas_datareader.data as web
        df = web.DataReader(stooq_map.get(ticker, ticker), "stooq",
                            start=start, end=end)
        df = df.sort_index()
        df.index = pd.to_datetime(df.index)
        df.to_csv(dest)
        log.info("  [stooq] %s (%d rows)", ticker, len(df))
        return df
    except Exception as exc:
        raise RuntimeError(f"Could not download price data for {ticker}") from exc


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def _audit(name: str, s: pd.Series | pd.DataFrame) -> Dict[str, object]:
    idx = s.index
    n = len(s)
    if isinstance(s, pd.Series):
        missing = int(s.isna().sum())
    else:
        missing = int(s.isna().sum().sum())
    return {
        "rows": int(n),
        "start": str(pd.Timestamp(idx.min()).date()) if n else None,
        "end": str(pd.Timestamp(idx.max()).date()) if n else None,
        "missing_values": missing,
        "missing_pct": round(100.0 * missing / max(n, 1), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw data (Step 1)")
    utils.add_common_args(parser)
    args = parser.parse_args()
    cfg = utils.config_from_args(args)
    start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]
    force = args.force
    log.info("Downloading data %s -> %s (force=%s)", start, end, force)

    audit: Dict[str, Dict[str, object]] = {}

    # Prices
    for name, ticker in cfg["data"]["tickers"].items():
        log.info("Price: %s (%s)", name, ticker)
        df = download_price(ticker, name.upper(), start, end, cfg, force)
        audit[f"price:{name}:{ticker}"] = _audit(name, df["Close"]
                                                 if "Close" in df else df)

    # Macro
    for sid in cfg["data"]["fred_series"]:
        log.info("FRED: %s", sid)
        s, source = download_fred(sid, start, end, cfg, force)
        a = _audit(sid, s)
        a["source"] = source
        audit[f"fred:{sid}"] = a

    utils.save_json(audit, utils.DOWNLOAD_LOG)
    log.info("Wrote download log with %d entries -> %s", len(audit),
             utils.DOWNLOAD_LOG.name)
    # Surface any 0-row download immediately
    for k, v in audit.items():
        if v["rows"] == 0:
            log.error("ZERO-ROW download: %s -> pipeline may fail", k)


if __name__ == "__main__":
    main()
