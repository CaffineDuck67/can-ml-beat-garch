"""Step 8 - SHAP explainability for the best tree model.

We re-fit a single boosted tree (the config backend) on the full pre-OOS feature
panel to explain *which features the model relies on* and how that reliance
differs between calm and high-VIX (crisis) regimes. Figures go to
results/figures/. SHAP here explains model structure; it is not an OOS
performance claim (that lives in evaluate.py).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import utils
import src.features as FE
import src.models_ml as ml

log = utils.get_logger("explain")


def _best_tree_horizon(cfg: dict) -> int:
    """Pick the horizon where a tree model has the lowest OOS RMSE (if known)."""
    import json
    path = utils.METRICS_JSON
    tree_names = {"lightgbm", "xgboost", "rf"}
    if path.exists():
        data = utils.load_json(path)
        rows = [r for r in data["metrics"] if r["model"] in tree_names]
        if rows:
            best = min(rows, key=lambda r: r["RMSE"])
            return int(best["horizon"])
    return cfg["targets"]["horizons"][-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP explainability (Step 8)")
    utils.add_common_args(parser)
    args = parser.parse_args()
    cfg = utils.config_from_args(args)
    utils.configure_matplotlib()
    import matplotlib.pyplot as plt
    import shap

    feats = utils.read_table(utils.FEATURES)
    targets = utils.read_table(utils.TARGETS)
    feature_cols = list(feats.columns)
    horizon = _best_tree_horizon(cfg)
    target_col = utils.rv_target_col(horizon)
    ds = feats.join(targets[[target_col]], how="inner").dropna(subset=[target_col])
    X = ds[feature_cols]
    y = ds[target_col]

    seed = utils.primary_seed(cfg)
    runner = ml.build_boosting(cfg, seed, cfg["runtime"].get("n_jobs", 1))
    # tune once on the earliest window, then we only need the fitted estimator
    runner.fit(X, y)
    model = runner._est
    log.info("Explaining boosted tree for horizon h=%d (best_params=%s)",
             horizon, runner.best_params)

    # SHAP on a subsample for speed
    samp = X.sample(min(2000, len(X)), random_state=seed).sort_index()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(samp.to_numpy())
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)

    # 1) global importance bar
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, samp.to_numpy(), feature_names=feature_cols,
                      show=False, max_display=15)
    plt.title(f"SHAP global feature importance (best tree, h={horizon})")
    plt.tight_layout()
    plt.savefig(utils.FIGURES_DIR / "shap_importance.png")
    plt.close()

    # 2) beeswarm (same call produces it; save distinct file)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, samp.to_numpy(), feature_names=feature_cols,
                      show=False, max_display=15)
    plt.title("SHAP beeswarm")
    plt.tight_layout()
    plt.savefig(utils.FIGURES_DIR / "shap_beeswarm.png")
    plt.close()

    # 3) calm vs high-VIX comparison (mean |SHAP| per feature, grouped)
    vix_thr = cfg["evaluation"]["vix_crisis_threshold"]
    crisis = (samp["vix_level"] > vix_thr).to_numpy()
    calm = ~crisis
    ma_crisis = np.abs(shap_values[crisis]).mean(axis=0) if crisis.any() else mean_abs * 0
    ma_calm = np.abs(shap_values[calm]).mean(axis=0) if calm.any() else mean_abs * 0
    # normalize each regime to sum 1 for a like-for-like share comparison
    n_crisis = ma_crisis / (ma_crisis.sum() + 1e-9)
    n_calm = ma_calm / (ma_calm.sum() + 1e-9)
    top = np.argsort(n_crisis + n_calm)[-12:]
    x = np.arange(len(top))
    plt.figure(figsize=(10, 6))
    plt.barh(x + 0.2, n_calm[top], height=0.38, label="Calm (VIX<=30)")
    plt.barh(x - 0.2, n_crisis[top], height=0.38, label=f"Crisis (VIX>{vix_thr})")
    plt.yticks(x, [feature_cols[i] for i in top])
    plt.legend()
    plt.title("Feature reliance: calm vs high-VIX (normalized mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(utils.FIGURES_DIR / "shap_crisis_vs_calm.png")
    plt.close()

    # save a small table of global importances
    imp = pd.DataFrame({"feature": feature_cols,
                        "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False)
    imp.to_csv(utils.TABLES_DIR / "shap_global_importance.csv", index=False)
    log.info("Saved 3 SHAP figures + importance table")


if __name__ == "__main__":
    main()
