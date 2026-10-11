"""Step 6 - expanding-window walk-forward validation engine.

Design (the crux of a leakage-free benchmark):

  * Initial training window of ~``train_initial_years`` years, then a purely
    out-of-sample forecast is produced for every trading day from ``oos_start``.
  * Models are re-fit on a rolling cadence:
        - ML / HAR / hist-mean : every ``ml_refit_every`` trading days
        - GARCH                : every ``garch.refit_every`` (default 63)
        - LSTM                 : every ``lstm.refit_every`` (default 63)
    Between re-fits the last fitted parameters are reused; forecasts still only
    use information up to each origin day t.
  * PURGE / EMBARGO: for an h-day target we cut the last ``h`` training rows so
    no forward-looking target window in the tail of the training set overlaps
    the test block (this is the classic overlapping-target leak).
  * NO random splits anywhere in the main benchmark.

Output: ``results/oos_predictions.csv.gz`` with one column of forecasts per
(model x horizon) plus the realized target and the day-t features needed later.
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import utils, models_classical as mc, models_ml as ml

log = utils.get_logger("walkforward")


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    feats = utils.read_table(utils.FEATURES)
    targets = utils.read_table(utils.TARGETS)
    clean = utils.read_table(utils.CLEAN_DAILY)
    returns = clean[utils.SP500_RET]
    target_cols = [utils.rv_target_col(h) for h in cfg["targets"]["horizons"]]
    ds = feats.join(targets[target_cols + ["regime_flag"]], how="inner")
    return ds, feats, returns, target_cols


def _block_positions(n: int, start_pos: int, step: int) -> List[tuple[int, int]]:
    blocks = []
    i = start_pos
    while i < n:
        blocks.append((i, min(i + step, n)))
        i += step
    return blocks


def run_walkforward(cfg: dict, dataset: pd.DataFrame, returns: pd.Series,
                    feature_cols: List[str], include_lstm: bool = True) -> pd.DataFrame:
    seed = utils.primary_seed(cfg)
    horizons: List[int] = cfg["targets"]["horizons"]
    lam = cfg["models"]["ewma_lambda"]
    garch_refit = int(cfg["models"]["garch"]["refit_every"])
    lstm_refit = int(cfg["models"]["lstm"]["refit_every"])
    refit_every = int(cfg["validation"]["ml_refit_every"])

    index = dataset.index
    oos_start = pd.Timestamp(cfg["validation"]["oos_start"])
    start_pos = int(np.searchsorted(index, oos_start))
    if start_pos >= len(index):
        raise RuntimeError("oos_start after end of data; nothing to forecast")

    # ---- train_initial_years: require a genuine in-sample burn-in ---- #
    init_years = cfg["validation"].get("train_initial_years")
    if init_years:
        days_per_year = int(cfg["targets"]["trading_days_per_year"])
        required = int(float(init_years) * days_per_year)
        have = int(dataset[utils.rv_target_col(horizons[0])].iloc[:start_pos]
                   .notna().sum())
        if start_pos < required:
            raise RuntimeError(
                f"train_initial_years={init_years} needs >= {required} trading "
                f"days before oos_start={cfg['validation']['oos_start']}, but "
                f"only {start_pos} positions exist. Extend the data or lower "
                f"train_initial_years / move oos_start later.")
        log.info("Initial-window check OK: %d positions (%d valid targets) >= "
                 "%d required for %s years", start_pos, have, required,
                 init_years)
    log.info("OOS test starts at %s (position %d of %d)",
             index[start_pos].date(), start_pos, len(index))

    n = len(index)
    blocks = _block_positions(n, start_pos, refit_every)

    # containers
    preds: Dict[str, np.ndarray] = {}
    dates_all = index[start_pos:]
    for h in horizons:
        for m in _model_names(cfg, include_lstm):
            preds[f"{m}__h{h}"] = np.full(len(dates_all), np.nan)

    garch_params = None
    lstm_model = None
    tuned_params: Dict[int, Dict[str, dict]] = {}
    for h in horizons:
        target_col = utils.rv_target_col(h)
        target_arr = dataset[target_col].to_numpy(dtype=float)
        garch_params = None
        garch_full = None          # cached full-index GARCH forecast per refit
        lstm_model = None
        # persistent runners: hyper-parameters are tuned once (oldest window)
        runners = _make_runners(cfg, seed, h)
        for bi, (b0, b1) in enumerate(tqdm(blocks, desc=f"h={h}", leave=False)):
            test_dates = index[b0:b1]
            # ---- purge/embargo: drop last h rows before training cut ---- #
            train_end = b0 - h
            if train_end < 30:  # need a minimal training window
                continue
            train = dataset.iloc[:train_end]
            y_train = train[target_col]
            valid = y_train.notna()
            train = train[valid]
            X_train = train[feature_cols]
            test = dataset.iloc[b0:b1]
            X_test = test[feature_cols]

            # ---------------- classical ---------------- #
            preds[f"hist_mean__h{h}"][b0 - start_pos:b1 - start_pos] = \
                mc.historical_mean(train[target_col])
            preds[f"ewma__h{h}"][b0 - start_pos:b1 - start_pos] = \
                mc.ewma_forecast(returns, test_dates, h, lam)

            # GARCH: refit on its own cadence; cache the FULL-index forecast so
            # the per-block work is only a slice (avoids an O(N) recursion/block).
            refit_garch = (bi % max(1, garch_refit // refit_every) == 0
                           or garch_params is None)
            if refit_garch:
                garch_params = mc.garch_fit(
                    returns.reindex(train.index).dropna().to_numpy(),
                    dist=cfg["models"]["garch"]["dist"], seed=seed)
                if garch_params is not None:
                    # causal forecast over every date once (value at t uses r<=t)
                    garch_full = mc.garch_forecast(garch_params, returns, index, h)
            if garch_params is not None:
                g = garch_full[b0:b1]
                preds[f"garch__h{h}"][b0 - start_pos:b1 - start_pos] = g
                # ---- garch_adj: Jensen/bias correction estimated on the TRAIN
                # window only (never on the test block). GARCH forecasts a
                # conditional *variance*; sqrt makes E[sigma] a downward-biased
                # predictor of E[RVol]. Rescale by mean(actual RV_h)/mean(garch)
                # over the training rows up to the embargo, with a guardrail.
                g_tr = garch_full[:train_end]
                a_tr = target_arr[:train_end]
                ok = ~np.isnan(g_tr) & ~np.isnan(a_tr)
                if ok.sum() > 10 and np.mean(g_tr[ok]) > 1e-8:
                    ratio = float(np.clip(np.mean(a_tr[ok]) / np.mean(g_tr[ok]),
                                          0.5, 2.0))
                else:
                    ratio = 1.0
                preds[f"garch_adj__h{h}"][b0 - start_pos:b1 - start_pos] = g * ratio

            # HAR-RV: refit every block
            try:
                preds[f"har__h{h}"][b0 - start_pos:b1 - start_pos] = \
                    mc.har_fit_predict(train, test, target_col)
            except Exception as exc:
                log.warning("HAR failed (h=%d, block %d): %s", h, bi, exc)

            # ---------------- ML ---------------- #
            for name, runner in runners.items():
                try:
                    runner.fit(X_train, train[target_col])
                    preds[f"{name}__h{h}"][b0 - start_pos:b1 - start_pos] = \
                        runner.predict(X_test)
                    tuned_params.setdefault(h, {})[name] = dict(runner.best_params or {})
                except Exception as exc:
                    log.warning("ML %s failed (h=%d blk %d): %s", name, h, bi, exc)

            # ---------------- LSTM ---------------- #
            if include_lstm:
                if lstm_model is None or bi % max(1, lstm_refit // refit_every) == 0:
                    try:
                        lstm_model = ml.LSTMModel(cfg, seed)
                        lstm_model.fit(X_train, train[target_col])
                    except Exception as exc:
                        log.warning("LSTM build failed: %s", exc)
                        lstm_model = None
                if lstm_model is not None:
                    try:
                        # Predict over a CONTIGUOUS window that starts inside the
                        # training region so every test day gets a full seq_len
                        # lookback using only information up to that day.
                        offset = lstm_model.seq_len - 1
                        lo = max(0, b0 - offset)
                        X_ext = dataset[feature_cols].iloc[lo:b1]
                        p = lstm_model.predict(X_ext)
                        first_pos = lo + offset  # dataset position of 1st pred
                        pos = np.arange(first_pos, first_pos + len(p))
                        keep = pos >= b0
                        idx = pos[keep] - start_pos
                        preds[f"lstm__h{h}"][idx] = p[keep]
                    except Exception as exc:
                        log.warning("LSTM predict failed: %s", exc)

    # assemble output
    out = pd.DataFrame(index=dates_all)
    for k, v in preds.items():
        out[k] = v
    for h in horizons:
        out[utils.rv_target_col(h)] = dataset[utils.rv_target_col(h)].reindex(dates_all)
    # keep day-t features + VIX for later (SHAP / crisis splits)
    feat_keep = [c for c in ["vix_level", "rv_21", "rv_1", "rv_5"] if c in dataset.columns]
    for c in feat_keep:
        out[c] = dataset[c].reindex(dates_all)
    out["actual_return"] = returns.reindex(dates_all)
    out["price"] = utils.read_table(utils.CLEAN_DAILY)[utils.SP500_CLOSE].reindex(dates_all)
    out.attrs["feature_cols"] = feature_cols
    out.attrs["tuned_params"] = {int(k): v for k, v in tuned_params.items()}
    return out


def _model_names(cfg: dict, include_lstm: bool) -> List[str]:
    backend = cfg["models"]["ml"]["boosting"].get("backend", "lightgbm")
    names = ["hist_mean", "ewma", "garch", "garch_adj", "har", "ridge", "rf",
             backend, "ridge_log", "rf_log", f"{backend}_log"]
    if include_lstm:
        names.append("lstm")
    return names


def _make_runners(cfg: dict, seed: int, h: int) -> Dict[str, ml.MLRunner]:
    """Fresh runners per horizon (tuning state is never shared across horizons).

    Includes the level-target models and the log-target (+ Duan smearing)
    variants required by the patch. Each runner is constructed with ``h`` so its
    tuning ``TimeSeriesSplit`` uses ``gap=h`` (Fix 8).
    """
    n_jobs = cfg["runtime"].get("n_jobs", 1)
    backend = cfg["models"]["ml"]["boosting"].get("backend", "lightgbm")
    return {
        "ridge": ml.build_ridge(cfg, seed, n_jobs, h=h),
        "rf": ml.build_rf(cfg, seed, n_jobs, h=h),
        backend: ml.build_boosting(cfg, seed, n_jobs, h=h),
        "ridge_log": ml.build_ridge(cfg, seed, n_jobs, h=h, log_target=True),
        "rf_log": ml.build_rf(cfg, seed, n_jobs, h=h, log_target=True),
        f"{backend}_log": ml.build_boosting(cfg, seed, n_jobs, h=h, log_target=True),
    }


def lstm_available(cfg: dict) -> bool:
    if not cfg["models"]["lstm"].get("enabled", False):
        return False
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        log.warning("torch not installed -> skipping LSTM")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward engine (Step 6)")
    utils.add_common_args(parser)
    args = parser.parse_args()
    cfg = utils.config_from_args(args)

    dataset, feats, returns, _ = load_dataset(cfg)
    feature_cols = list(feats.columns)
    log.info("Dataset: %d rows, %d features", len(dataset), len(feature_cols))
    include_lstm = lstm_available(cfg)
    log.info("LSTM included: %s", include_lstm)

    preds = run_walkforward(cfg, dataset, returns, feature_cols, include_lstm)
    utils.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(utils.OOS_PREDICTIONS)
    log.info("Wrote %s shape=%s", utils.OOS_PREDICTIONS.name, preds.shape)
    utils.save_json({"feature_cols": feature_cols,
                     "models": _model_names(cfg, include_lstm),
                     "horizons": cfg["targets"]["horizons"],
                     "boosting_backend": cfg["models"]["ml"]["boosting"].get(
                         "backend", "lightgbm"),
                     "tuned_params": preds.attrs.get("tuned_params", {})},
                    utils.RESULTS_DIR / "run_meta.json")


if __name__ == "__main__":
    main()
