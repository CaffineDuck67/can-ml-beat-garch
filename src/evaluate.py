"""Step 7 - forecast evaluation.

Metrics (per model, per horizon, purely out-of-sample):
  * RMSE, MAE on annualized realized-vol forecasts (%).
  * QLIKE - a robust volatility loss that, unlike RMSE, is not minimized by the
    conditional mean and is scale-invariant:
        QLIKE(a, f) = ln(f^2) - ln(a^2) + a^2 / f^2
  * Out-of-sample R2 vs the historical-mean benchmark:
        R2_oos = 1 - sum((a - f)^2) / sum((a - f_histmean)^2)

Statistical significance:
  * Diebold-Mariano test of the loss differential between each candidate and a
    benchmark (GARCH and HAR-RV) under two losses -- squared error ("se") and
    QLIKE ("qlike") -- with a Newey-West Bartlett HAC variance using (h-1) lags
    to absorb the overlapping-target autocorrelation, and the Harvey-Leybourne-
    Newbold (1997) small-sample correction applied as
        dm_hln = dm * sqrt((T + 1 - 2h + h(h-1)/T) / T)
    with the p-value read from a Student-t distribution with (T-1) degrees of
    freedom (not the normal).
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from src import utils

EPS = 1e-6
log = utils.get_logger("evaluate")


# --------------------------------------------------------------------------- #
# Loss functions                                                               #
# --------------------------------------------------------------------------- #
def rmse(a, f) -> float:
    a, f = _pair(a, f)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def mae(a, f) -> float:
    a, f = _pair(a, f)
    return float(np.mean(np.abs(a - f)))


def qlike(a, f) -> float:
    a, f = _pair(a, f)
    a2 = np.clip(a, EPS, None) ** 2
    f2 = np.clip(f, EPS, None) ** 2
    return float(np.mean(np.log(f2) - np.log(a2) + a2 / f2))


def r2_oos(a, f, bench) -> float:
    a, f, bench = _pair3(a, f, bench)
    sse = np.sum((a - f) ** 2)
    sse_b = np.sum((a - bench) ** 2)
    return float(1.0 - sse / sse_b)


def _pair(a, f):
    a = np.asarray(a, float); f = np.asarray(f, float)
    m = ~(np.isnan(a) | np.isnan(f))
    return a[m], f[m]


def _pair3(a, f, b):
    a = np.asarray(a, float); f = np.asarray(f, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(f) | np.isnan(b))
    return a[m], f[m], b[m]


# --------------------------------------------------------------------------- #
# Loss functions used by the Diebold-Mariano test                             #
# --------------------------------------------------------------------------- #
def _se_loss(a, f) -> np.ndarray:
    """Per-observation squared-error loss (a - f)^2."""
    a, f = np.asarray(a, float), np.asarray(f, float)
    return (a - f) ** 2


def _qlike_loss(a, f) -> np.ndarray:
    """Per-observation QLIKE loss: ln(f^2) - ln(a^2) + a^2/f^2 (>=0, robust)."""
    a, f = np.asarray(a, float), np.asarray(f, float)
    a2 = np.clip(a, EPS, None) ** 2
    f2 = np.clip(f, EPS, None) ** 2
    return np.log(f2) - np.log(a2) + a2 / f2


def _loss_vec(kind: str, a, f) -> np.ndarray:
    if kind == "se":
        return _se_loss(a, f)
    if kind == "qlike":
        return _qlike_loss(a, f)
    raise ValueError(f"unknown DM loss '{kind}' (use 'se' or 'qlike')")


# --------------------------------------------------------------------------- #
# Diebold-Mariano                                                              #
# --------------------------------------------------------------------------- #
def diebold_mariano(a, f_model, f_bench, h: int,
                    loss: str = "se") -> Dict[str, float]:
    """Return the (Harvey-corrected) DM statistic and Student-t p-value.

    ``loss`` is "se" (squared error) or "qlike". Positive DM => the model has
    LOWER loss than the benchmark (i.e. is better). The multi-step correction is
    applied to the statistic (not inside the variance) and inference uses a
    Student-t with T-1 degrees of freedom (Harvey, Leybourne & Newbold 1997).
    """
    a = np.asarray(a, float)
    fm = np.asarray(f_model, float)
    fb = np.asarray(f_bench, float)
    ok = ~(np.isnan(a) | np.isnan(fm) | np.isnan(fb))
    a, fm, fb = a[ok], fm[ok], fb[ok]
    # loss differential: benchmark minus model (>0 means the model is better)
    d = _loss_vec(loss, a, fb) - _loss_vec(loss, a, fm)
    T = len(d)
    if T < h + 5:
        return {"dm": np.nan, "p": np.nan, "n": int(T), "loss": loss}
    dbar = d.mean()
    d0 = d - dbar
    # Newey-West Bartlett HAC long-run variance with (h-1) lags
    gamma0 = float(np.dot(d0, d0) / T)
    acf = gamma0
    for k in range(1, h):
        gk = float(np.dot(d0[k:], d0[:-k]) / T)
        acf += 2.0 * (1.0 - k / (h)) * gk
    var_dbar = max(acf, EPS) / T
    dm = dbar / np.sqrt(var_dbar)
    # Harvey-Leybourne-Newbold small-sample correction (applied to the statistic)
    corr = np.sqrt(max((T + 1.0 - 2.0 * h + h * (h - 1) / T) / T, 1e-12))
    dm_hln = dm * corr
    p = 2.0 * stats.t.sf(abs(dm_hln), df=T - 1)
    return {"dm": float(dm_hln), "p": float(p), "n": int(T), "loss": loss}


# --------------------------------------------------------------------------- #
# Model registry from predictions                                            #
# --------------------------------------------------------------------------- #
def _model_names(preds: pd.DataFrame, h: int) -> List[str]:
    names = []
    suffix = f"__h{h}"
    for c in preds.columns:
        if c.endswith(suffix) and not c.startswith("hist_mean"):
            names.append(c[: -len(suffix)])
    if any(c.endswith(suffix) for c in preds.columns) and \
            f"hist_mean__h{h}" in preds.columns:
        names.insert(0, "hist_mean")
    # dedup, keep order
    seen = set(); out = []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def build_metrics(preds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for h in cfg["targets"]["horizons"]:
        target = utils.rv_target_col(h)
        bench = preds[f"hist_mean__h{h}"]
        for m in _model_names(preds, h):
            f = preds[f"{m}__h{h}"]
            a = preds[target]
            rows.append({
                "horizon": h, "model": m,
                "n": int((~np.isnan(a) & ~np.isnan(f)).sum()),
                "RMSE": rmse(a, f), "MAE": mae(a, f), "QLIKE": qlike(a, f),
                "R2_oos": r2_oos(a, f, bench),
            })
    return pd.DataFrame(rows)


def build_dm_table(preds: pd.DataFrame, cfg: dict,
                   benchmarks=("garch", "har")) -> pd.DataFrame:
    """DM tests for BOTH losses (qlike first, then se) vs each benchmark.

    Output columns: horizon, model, benchmark, loss, DM, p_value, n.
    """
    rows = []
    for h in cfg["targets"]["horizons"]:
        target = utils.rv_target_col(h)
        a = preds[target]
        for m in _model_names(preds, h):
            for b in benchmarks:
                col_b = f"{b}__h{h}"
                if col_b not in preds.columns:
                    continue
                # QLIKE listed first (robust, preferred vol loss), then SE
                for loss in ("qlike", "se"):
                    res = diebold_mariano(a, preds[f"{m}__h{h}"], preds[col_b],
                                          h, loss=loss)
                    rows.append({"horizon": h, "model": m, "benchmark": b,
                                 "loss": loss, "DM": res["dm"],
                                 "p_value": res["p"], "n": res["n"]})
    return pd.DataFrame(rows)


def build_subperiod_table(preds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sp = cfg["evaluation"]["subperiods"]
    vix_thr = cfg["evaluation"]["vix_crisis_threshold"]
    rows = []
    for h in cfg["targets"]["horizons"]:
        target = utils.rv_target_col(h)
        bench = preds[f"hist_mean__h{h}"]
        periods = dict(sp)
        # crisis / non-crisis by VIX
        if "vix_level" in preds.columns:
            periods["crisis_vix_gt_%d" % vix_thr] = ("2010-01-01", None, "crisis")
            periods["noncrisis_vix_lte_%d" % vix_thr] = ("2010-01-01", None, "noncrisis")
        for pname, rng in periods.items():
            if len(rng) == 3:  # vix-based
                mode = rng[2]
                mask = (preds["vix_level"] > vix_thr) if mode == "crisis" \
                    else (preds["vix_level"] <= vix_thr)
                sub = preds[mask]
            else:
                s0 = pd.Timestamp(rng[0])
                s1 = pd.Timestamp(rng[1]) if rng[1] else None
                mask = preds.index >= s0
                if s1 is not None:
                    mask = mask & (preds.index <= s1)
                sub = preds[mask]
            if len(sub) < 20:
                continue
            for m in _model_names(preds, h):
                rows.append({"horizon": h, "period": pname, "model": m,
                             "RMSE": rmse(sub[target], sub[f"{m}__h{h}"]),
                             "QLIKE": qlike(sub[target], sub[f"{m}__h{h}"]),
                             "R2_oos": r2_oos(sub[target], sub[f"{m}__h{h}"],
                                              sub["hist_mean__h{0}".format(h)]),
                             "n": int(sub[target].notna().sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bonus module: high-risk regime classification                               #
# --------------------------------------------------------------------------- #
def run_regime_classifier(cfg: dict) -> Dict[str, float]:
    """Expanding-window logistic + GBM classifier for the next-21d drawdown flag.

    Evaluated out-of-sample (from oos_start) with PR-AUC and Brier score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, brier_score_loss
    from lightgbm import LGBMClassifier

    feats = utils.read_table(utils.FEATURES)
    targets = utils.read_table(utils.TARGETS)
    ds = feats.join(targets[["regime_flag"]], how="inner").dropna(subset=["regime_flag"])
    feature_cols = list(feats.columns)
    oos_start = pd.Timestamp(cfg["validation"]["oos_start"])
    idx = ds.index
    start = int(np.searchsorted(idx, oos_start))
    h = cfg["targets"]["regime"]["window"]
    y = ds["regime_flag"].to_numpy()

    proba_log = np.full(len(ds), np.nan)
    proba_gbm = np.full(len(ds), np.nan)
    y = ds["regime_flag"].to_numpy()
    # block-based expanding window (re-fit every 21 days) to keep it fast while
    # remaining purely out-of-sample (each day's score uses only data <= t).
    cadence = 21
    Xall = ds[feature_cols].to_numpy()
    for b0 in range(start, len(ds), cadence):
        b1 = min(b0 + cadence, len(ds))
        tr_end = b0 - h
        if tr_end < 50:
            continue
        Xtr = Xall[:tr_end]
        ytr = y[:tr_end]
        if len(np.unique(ytr)) < 2:
            continue
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=1000, C=0.1,
                                random_state=utils.primary_seed(cfg))
        lr.fit(sc.transform(Xtr), ytr)
        gbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                             random_state=utils.primary_seed(cfg), verbosity=-1)
        gbm.fit(Xtr, ytr)
        Xte = sc.transform(Xall[b0:b1])
        proba_log[b0:b1] = lr.predict_proba(Xte)[:, 1]
        proba_gbm[b0:b1] = gbm.predict_proba(Xte)[:, 1]

    out = {}
    # Score only the out-of-sample region: build a full-length mask that is False
    # before ``start`` and False where the probability is still NaN.
    for name, pr in [("logistic", proba_log), ("lightgbm", proba_gbm)]:
        m = ~np.isnan(pr)
        m[:start] = False
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        out[f"{name}_pr_auc"] = float(average_precision_score(y[m], pr[m]))
        out[f"{name}_brier"] = float(brier_score_loss(y[m], pr[m]))
        out[f"{name}_base_rate"] = float(y[m].mean())
        out[f"{name}_n"] = int(m.sum())
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forecasts (Step 7)")
    utils.add_common_args(parser)
    args = parser.parse_args()
    cfg = utils.config_from_args(args)

    preds = utils.read_table(utils.OOS_PREDICTIONS)
    log.info("Loaded OOS predictions %s", preds.shape)

    metrics = build_metrics(preds, cfg)
    dm = build_dm_table(preds, cfg)
    sub = build_subperiod_table(preds, cfg)

    for name, table in [("main_metrics", metrics), ("dm_tests", dm),
                        ("subperiod_metrics", sub)]:
        table.to_csv(utils.TABLES_DIR / f"{name}.csv", index=False)
        _to_markdown(table, utils.TABLES_DIR / f"{name}.md")
        log.info("Saved %s (%d rows)", name, len(table))

    regime = run_regime_classifier(cfg)
    if regime:
        utils.save_json(regime, utils.TABLES_DIR / "regime_classification.json")
        pd.DataFrame([regime]).to_csv(
            utils.TABLES_DIR / "regime_classification.csv", index=False)
        log.info("Regime classifier metrics: %s", regime)

    # machine-readable summary for the report step
    summary = {
        "metrics": metrics.to_dict("records"),
        "dm": dm.to_dict("records"),
        "subperiods": sub.to_dict("records"),
        "regime": regime,
    }
    utils.save_json(summary, utils.METRICS_JSON)
    # print headline
    log.info("\n%s", metrics.to_string(index=False))


def _to_markdown(df: pd.DataFrame, path) -> None:
    try:
        df.to_markdown(path, index=False, floatfmt=".4f")
    except Exception:
        df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
