"""Step 9 - economic value: a volatility-targeting backtest on the S&P 500.

Position (leverage) at each rebalance date t:

    w_t = clip( target_vol / forecast_vol_t , leverage_floor, leverage_cap )

forecast_vol_t is the model's annualized h-day realized-vol forecast made using
information up to t (purely out-of-sample). The position is held for
``rebalance_every`` days and applied to the NEXT day's return (no look-ahead).
Transaction cost = transaction_cost_bps * |w_t - w_{t-1}| charged on rebalance
days. We compare buy-and-hold, GARCH-forecast targeting, and an ML-forecast
targeting.

Turnover (Fix 1): annualised ONE-WAY turnover is
    turnover_ann = sum(|Δw_t|) / (n_days / 252)
i.e. the total absolute weight change over the sample scaled to a yearly rate,
NOT the previous ``mean(|Δw|) * (252/reb)`` which understated it.

Sharpe (Fix 9): reported BOTH raw (mean r / std r) and EXCESS over the risk-free
rate (mean (r - rf) / std r). rf is the FRED DFF effective federal-funds rate
from the clean panel: daily rf = DFF/100/252, lagged one day so a position held
on t earns the rf known at t-1.

Best-ML selection (Fix 9): the ML model used for the *honest* second-half
backtest is chosen on the FIRST half of the OOS window only, so its selection
never sees the evaluation half. A full-window table is also produced but is
clearly labelled as containing selection bias.

DISCLAIMER: research only, not investment advice; past results do not guarantee
future results.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import utils

DISCLAIMER = ("Research only - NOT investment advice. Past performance does not "
              "guarantee future results.")

log = utils.get_logger("strategy")

# models eligible to be labelled "best ML" for volatility targeting
ML_MODELS = {"ridge", "rf", "lightgbm", "xgboost", "lstm",
             "ridge_log", "rf_log", "lightgbm_log", "xgboost_log"}


def _risk_free(returns: pd.Series, cfg: dict) -> pd.Series:
    """Daily risk-free rate (decimal) from FRED DFF, lagged one day."""
    clean = utils.read_table(utils.CLEAN_DAILY)
    dff = pd.to_numeric(clean["DFF"], errors="coerce")           # in % p.a.
    rf = (dff / 100.0) / cfg["targets"]["trading_days_per_year"]  # daily decimal
    rf = rf.reindex(returns.index)
    return rf.shift(1).fillna(0.0)


def backtest(returns: pd.Series, forecast_vol_pct: pd.Series, cfg: dict) -> pd.DataFrame:
    st = cfg["strategy"]
    target = st["target_vol"]
    cap, floor = st["leverage_cap"], st["leverage_floor"]
    reb = int(st["rebalance_every"])
    cost = st["transaction_cost_bps"] / 1e4

    fv = (forecast_vol_pct.reindex(returns.index) / 100.0).clip(lower=1e-3)
    raw_w = (target / fv).clip(lower=floor, upper=cap)

    weights = np.full(len(returns), np.nan)
    prev_w = 0.0
    for i, _date in enumerate(returns.index):
        if i % reb == 0:
            prev_w = raw_w.iloc[i]
        weights[i] = prev_w
    weights = pd.Series(weights, index=returns.index)

    # apply weight from the PREVIOUS day to today's return (no look-ahead)
    w_lag = weights.shift(1).fillna(0.0)
    r = returns.fillna(0.0)
    gross = w_lag * r
    # transaction costs on the day the target weight actually changes
    delta = weights.diff().abs()
    delta.iloc[0] = abs(weights.iloc[0])          # initial position build
    cost_vec = delta * cost
    net = gross - cost_vec
    return pd.DataFrame({"ret_bhold": r, "w": weights,
                         "ret_strategy": net, "dW": delta}, index=returns.index)


def perf(df: pd.DataFrame, ret_col: str, rf: pd.Series | None = None) -> dict:
    r = df[ret_col].dropna()
    n = len(r)
    if n < 2:
        return {}
    ann_ret = float(np.mean(r) * 252)
    ann_vol = float(np.std(r) * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    if rf is not None:
        ex = (r - rf.reindex(r.index).fillna(0.0))
        sharpe_excess = float(np.mean(ex) * 252) / ann_vol if ann_vol > 0 else np.nan
    else:
        sharpe_excess = np.nan
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1.0)
    # annualised ONE-WAY turnover (Fix 1): total |Δw| scaled to a full year
    dw = df["dW"].to_numpy(dtype=float) if "dW" in df else df["w"].diff().abs().to_numpy(float)
    turnover = float(np.nansum(dw)) / (n / 252.0)
    return {
        "ann_return": ann_ret, "ann_vol": ann_vol,
        "sharpe": sharpe, "sharpe_excess": sharpe_excess,
        "max_drawdown": float(dd.min()),
        "total_return": float(cum.iloc[-1] - 1.0),
        "turnover_ann": turnover,
    }


def _pick_best_ml_on_window(cfg: dict, preds: pd.DataFrame, horizon: int,
                            mask: pd.Series) -> str:
    """Choose the ML model with the lowest RMSE on ``preds`` rows in ``mask``."""
    target = utils.rv_target_col(horizon)
    backend = cfg["models"]["ml"]["boosting"].get("backend", "lightgbm")
    ml_cols = [c for c in preds.columns
               if c.endswith(f"__h{horizon}")
               and c[:-len(f'__h{horizon}')] in ML_MODELS]
    best, best_rmse = None, np.inf
    a = preds.loc[mask, target].to_numpy(dtype=float)
    for c in ml_cols:
        f = preds.loc[mask, c].to_numpy(dtype=float)
        ok = ~np.isnan(a) & ~np.isnan(f)
        if ok.sum() < 20:
            continue
        r = float(np.sqrt(np.mean((a[ok] - f[ok]) ** 2)))
        if r < best_rmse:
            best_rmse, best = r, c[: -len(f"__h{horizon}")]
    return best or backend


def _table(rows, note=None):
    cols = ["strategy", "ann_return", "ann_vol", "sharpe", "sharpe_excess",
            "max_drawdown", "total_return", "turnover_ann"]
    t = pd.DataFrame(rows)
    t = t[[c for c in cols if c in t.columns]]
    return t


def run_all(cfg: dict, verbose: bool = True) -> dict:
    global _CFG
    _CFG = cfg
    preds = utils.read_table(utils.OOS_PREDICTIONS)
    horizon = cfg["strategy"].get("horizon", 21)
    if horizon not in cfg["targets"]["horizons"]:
        horizon = cfg["targets"]["horizons"][-1]
    returns = preds["actual_return"]
    price = preds["price"]
    rf = _risk_free(returns, cfg)

    # ---- first-half / second-half split of the OOS window (Fix 9) ---- #
    idx = preds.index
    mid = idx[len(idx) // 2]
    first_half = idx < mid
    second_half = idx >= mid
    best_ml_sel = _pick_best_ml_on_window(cfg, preds, horizon, first_half)
    log.info("Best ML chosen on FIRST half (h=%d): %s ; evaluating on SECOND half",
             horizon, best_ml_sel)

    variants = {
        "buy_and_hold": None,
        "vol_target_garch": preds[f"garch__h{horizon}"],
        f"vol_target_{best_ml_sel}": preds[f"{best_ml_sel}__h{horizon}"],
    }

    def _build(sub_mask):
        curves, rows = {}, []
        for name, fv in variants.items():
            sub_returns = returns[sub_mask]
            sub_fv = None if fv is None else fv[sub_mask]
            if fv is None:
                r = sub_returns.fillna(0.0)
                df = pd.DataFrame({"ret_bhold": r, "ret_strategy": r, "w": 1.0},
                                  index=r.index)
                df["dW"] = 0.0
            else:
                df = backtest(sub_returns, sub_fv, cfg)
            curves[name] = (1 + df["ret_strategy"].fillna(0)).cumprod()
            p = perf(df, "ret_strategy", rf[sub_mask])
            p["strategy"] = name
            rows.append(p)
            df.to_csv(utils.RESULTS_DIR / f"bt_{name}.csv")
        return curves, rows

    # honest second-half table (primary) + full-window table (selection-biased)
    curves_h, rows_h = _build(second_half)
    _, rows_full = _build(pd.Series(True, index=idx))

    table_h = _table(rows_h)
    table_h.to_csv(utils.TABLES_DIR / "strategy_performance.csv", index=False)
    _to_md(table_h, utils.TABLES_DIR / "strategy_performance.md")

    table_full = _table(rows_full)
    table_full.to_csv(utils.TABLES_DIR / "strategy_performance_full.csv", index=False)
    _to_md(table_full, utils.TABLES_DIR / "strategy_performance_full.md")

    # plot uses the honest (second-half) curves
    _plot(curves_h, price[second_half])

    utils.save_json({
        "best_ml_selection_half": best_ml_sel, "horizon": horizon,
        "eval_window": "second half of OOS (model selected on first half)",
        "rows": rows_h, "rows_full": rows_full,
        "selection_bias_note": ("strategy_performance_full.csv selects the ML "
                                "model on the SAME window it is evaluated on -> "
                                "optimistic; use strategy_performance.csv."),
        "disclaimer": DISCLAIMER,
    }, utils.TABLES_DIR / "strategy_summary.json")
    if verbose:
        log.info("\n== Honest (2nd-half, model chosen on 1st half) ==\n%s",
                 table_h.to_string(index=False))
        log.info("\n== Full-window (SELECTION-BIASED) ==\n%s",
                 table_full.to_string(index=False))
        log.info("DISCLAIMER: %s", DISCLAIMER)
    return {"honest": table_h, "full": table_full,
            "best_ml": best_ml_sel, "horizon": horizon}


_CFG = None  # set by run_all for _run_variant convenience


def _plot(curves: dict, price: pd.Series) -> None:
    import matplotlib.pyplot as plt
    utils.configure_matplotlib()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    for name, c in curves.items():
        ax[0].plot(c.index, c.values, label=name)
    bh = (1 + (price.pct_change().fillna(0))).cumprod()
    if len(curves):
        ax[0].plot(bh.index, (bh / bh.iloc[0] * curves[list(curves)[0]].iloc[0]).values,
                   "--", color="gray", alpha=0.6, label="index (scaled)")
    ax[0].set_ylabel("Growth of $")
    ax[0].set_title("Volatility-targeting equity curves (OOS, 2nd half)")
    ax[0].legend()
    for name, c in curves.items():
        dd = c / c.cummax() - 1.0
        ax[1].fill_between(dd.index, dd.values, 0, alpha=0.3, label=name)
    ax[1].set_ylabel("Drawdown")
    ax[1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(utils.FIGURES_DIR / "strategy_equity_drawdown.png")
    plt.close(fig)


def _to_md(df, path):
    try:
        df.to_markdown(path, index=False, floatfmt=".4f")
    except Exception:
        df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Economic value (Step 9)")
    utils.add_common_args(parser)
    args = parser.parse_args()
    cfg = utils.config_from_args(args)
    if not utils.OOS_PREDICTIONS.exists():
        log.error("OOS predictions missing; run walkforward first")
        return
    run_all(cfg)


if __name__ == "__main__":
    main()
