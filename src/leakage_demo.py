"""Step 6b - leakage demonstration (apples-to-apples).

Shows how a naive RANDOM K-fold CV inflates apparent forecast skill relative to
the honest expanding-window walk-forward benchmark, because overlapping,
serially-correlated realized-vol targets let a model "see the future" through
its neighbours.

The comparison is now apples-to-apples (Fix 4):
  * the random-K-fold out-of-fold (OOF) RMSE is measured on EXACTLY the same
    rows as the walk-forward out-of-sample window (dates >= ``oos_start``);
  * the SAME model and the SAME tuned hyper-parameters as the walk-forward
    LightGBM (read from ``run_meta.json``) are used - not the first grid value;
  * a second column reports the FULL-SAMPLE random-CV RMSE (all rows), which is
    even more optimistic because the leaky folds also reuse the earliest data;
  * the headline metric is renamed "RMSE reduction vs walk-forward (%)":
        reduction = 100 * (walkforward_RMSE - random_RMSE) / walkforward_RMSE
    i.e. how much the naive split *shrinks* the reported error versus honest
    walk-forward. A large positive number means the leaky evaluation flattered
    the model a lot.

The main benchmark NEVER uses random splits; this module exists only to
quantify the size of that error.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import utils

log = utils.get_logger("leakage_demo")


def _clone_boost(cfg: dict, params: dict, seed: int):
    backend = cfg["models"]["ml"]["boosting"].get("backend", "lightgbm")
    if backend == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params, random_state=seed, n_jobs=1,
                            tree_method="hist")
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**params, random_state=seed, n_jobs=1, verbosity=-1)


def random_cv_rmse(cfg: dict, X: pd.DataFrame, y: pd.Series, seed: int,
                   params: dict, n_splits: int = 5) -> float:
    """Pooled out-of-fold RMSE from a random K-fold split (the WRONG way).

    ``params`` are the *same* tuned hyper-parameters the walk-forward LightGBM
    used, so the only difference versus walk-forward is the (leaky) split.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    Xn = X.to_numpy(dtype=float)
    yn = y.to_numpy(dtype=float)
    oof = np.full(len(Xn), np.nan)
    for tr, te in kf.split(Xn):
        m = _clone_boost(cfg, params, seed)
        m.fit(Xn[tr], yn[tr])
        oof[te] = m.predict(Xn[te])
    return float(np.sqrt(np.nanmean((yn - oof) ** 2)))


def _default_params(cfg: dict) -> dict:
    mlcfg = cfg["models"]["ml"]["boosting"]
    return {k: (v[0] if isinstance(v, list) else v) for k, v in {
        "n_estimators": mlcfg["n_estimators"],
        "learning_rate": mlcfg["learning_rate"],
        "num_leaves": mlcfg["num_leaves"],
        "min_child_samples": mlcfg["min_child_samples"]}.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage demo (Step 6b)")
    utils.add_common_args(parser)
    args = parser.parse_args()
    cfg = utils.config_from_args(args)
    utils.configure_matplotlib()
    import matplotlib.pyplot as plt

    feats = utils.read_table(utils.FEATURES)
    targets = utils.read_table(utils.TARGETS)
    feature_cols = list(feats.columns)
    seed = utils.primary_seed(cfg)
    backend = cfg["models"]["ml"]["boosting"].get("backend", "lightgbm")
    oos_start = pd.Timestamp(cfg["validation"]["oos_start"])

    # tuned hyper-parameters persisted by the walk-forward engine (per horizon)
    tuned = {}
    if utils.RESULTS_DIR.joinpath("run_meta.json").exists():
        meta = utils.load_json(utils.RESULTS_DIR / "run_meta.json")
        tuned = {int(k): v for k, v in meta.get("tuned_params", {}).items()}

    # honest walk-forward RMSE on exactly the OOS rows (from the predictions)
    wf_rmse = {}
    if utils.OOS_PREDICTIONS.exists():
        preds = utils.read_table(utils.OOS_PREDICTIONS)
        for h in cfg["targets"]["horizons"]:
            tcol = utils.rv_target_col(h)
            pc = f"{backend}__h{h}"
            if tcol in preds.columns and pc in preds.columns:
                a = preds[tcol].to_numpy(dtype=float)
                f = preds[pc].to_numpy(dtype=float)
                ok = ~np.isnan(a) & ~np.isnan(f)
                wf_rmse[h] = float(np.sqrt(np.mean((a[ok] - f[ok]) ** 2)))

    rows = []
    for h in cfg["targets"]["horizons"]:
        tcol = utils.rv_target_col(h)
        ds = feats.join(targets[[tcol]], how="inner").dropna(subset=[tcol])
        X, y = ds[feature_cols], ds[tcol]
        params = tuned.get(h, {}).get(backend) or _default_params(cfg)

        # (a) apples-to-apples: leaky OOF evaluated on the SAME OOS rows
        oos = ds.index >= oos_start
        X_oos, y_oos = ds.loc[oos, feature_cols], ds.loc[oos, tcol]
        # random-K-fold on the OOS block only
        rmse_oos = random_cv_rmse(cfg, X_oos, y_oos, seed, params)
        # (b) full-sample random CV (all rows) - second, clearly-labelled column
        rmse_full = random_cv_rmse(cfg, X, y, seed, params)

        wf = wf_rmse.get(h, np.nan)
        rows.append({
            "horizon": h, "model": backend,
            "walkforward_RMSE": wf,
            "random_kfold_RMSE_oos_rows": rmse_oos,
            "random_kfold_RMSE_full_sample": rmse_full,
            "rmse_reduction_vs_walkforward_pct":
                (100 * (wf - rmse_oos) / wf) if wf and wf == wf else np.nan,
        })

    table = pd.DataFrame(rows)
    table.to_csv(utils.TABLES_DIR / "leakage_comparison.csv", index=False)
    try:
        table.to_markdown(utils.TABLES_DIR / "leakage_comparison.md",
                          index=False, floatfmt=".4f")
    except Exception:
        pass
    log.info("\n%s", table.to_string(index=False))

    # figure: walk-forward vs the two leaky evaluations
    x = np.arange(len(table))
    w = 0.27
    plt.figure(figsize=(9, 5))
    plt.bar(x - w, table["walkforward_RMSE"], width=w,
            label="Walk-forward (honest OOS)")
    plt.bar(x, table["random_kfold_RMSE_oos_rows"], width=w,
            label="Random K-fold, OOS rows (leaky)")
    plt.bar(x + w, table["random_kfold_RMSE_full_sample"], width=w,
            label="Random K-fold, full sample (leakier)")
    plt.xticks(x, [f"h={h}d" for h in table["horizon"]])
    plt.ylabel("RMSE (annualized vol %)")
    plt.title("Random-split CV shrinks reported RMSE vs walk-forward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(utils.FIGURES_DIR / "leakage_random_vs_walkforward.png")
    plt.close()


if __name__ == "__main__":
    main()
