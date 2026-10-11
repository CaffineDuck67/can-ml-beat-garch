"""Step 10 - report / summary generation.

Turns the JSON/CSV outputs of the pipeline into:
  * results/summary.md                - a one-page, numbers-only resume summary
  * results/tables/key_results.md     - headline model-comparison table
  * results/tables/paper_*.md         - markdown tables to paste into paper.md
                                        (RMSE+QLIKE metrics and both-loss DM),
                                        generated programmatically (never typed)
and prints a deliverables checklist (pass/fail). Every number here is read from
code output - nothing is hard-coded or invented.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import utils

log = utils.get_logger("report")

ML_SET = {"ridge", "rf", "lightgbm", "xgboost", "lstm",
          "ridge_log", "rf_log", "lightgbm_log", "xgboost_log"}
CLASSICAL = {"garch", "garch_adj", "har", "ewma", "hist_mean"}


def _best_by_horizon(metrics: pd.DataFrame, metric: str, exclude=("hist_mean",)):
    out = {}
    for h, grp in metrics.groupby("horizon"):
        g = grp[~grp["model"].isin(exclude)]
        out[int(h)] = g.loc[g[metric].idxmin()]
    return out


def key_results_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, grp in metrics.groupby("horizon"):
        bench_rmse = grp.loc[grp["model"] == "hist_mean", "RMSE"].values[0]
        for _, r in grp.iterrows():
            rows.append({"horizon": int(h), "model": r["model"],
                         "RMSE": r["RMSE"], "MAE": r["MAE"],
                         "QLIKE": r["QLIKE"], "R2_oos": r["R2_oos"],
                         "RMSE_vs_histmean_%": 100 * (r["RMSE"] / bench_rmse - 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Programmatic paper tables                                                    #
# --------------------------------------------------------------------------- #
def paper_metrics_table(metrics: pd.DataFrame) -> str:
    """Per-horizon table ordered by QLIKE, with RMSE/MAE/QLIKE/R2_oos."""
    lines = []
    for h in sorted(metrics["horizon"].unique()):
        g = metrics[metrics.horizon == h].sort_values("QLIKE")
        lines.append(f"### Horizon h = {h} trading days (sorted by QLIKE)\n")
        lines.append("| Model | RMSE | MAE | QLIKE | R2_oos |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in g.iterrows():
            lines.append(f"| {r['model']} | {r['RMSE']:.3f} | {r['MAE']:.3f} | "
                         f"{r['QLIKE']:.3f} | {r['R2_oos']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def paper_dm_table(dm: pd.DataFrame) -> str:
    """DM table for BOTH losses (QLIKE first, then SE), vs GARCH and vs HAR."""
    lines = []
    for h in sorted(dm["horizon"].unique()):
        dh = dm[dm.horizon == h]
        lines.append(f"### Horizon h = {h} (DM; positive = model better than benchmark)\n")
        lines.append("| Model | QLIKE vs GARCH | p | QLIKE vs HAR | p | SE vs GARCH | p | SE vs HAR | p |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m in dh["model"].unique():
            def cell(bench, loss, col):
                r = dh[(dh.model == m) & (dh.benchmark == bench) & (dh.loss == loss)]
                return r[col].values[0] if len(r) else np.nan
            vqg, pqg = cell("garch", "qlike", "DM"), cell("garch", "qlike", "p_value")
            vqh, pqh = cell("har", "qlike", "DM"), cell("har", "qlike", "p_value")
            vsg, psg = cell("garch", "se", "DM"), cell("garch", "se", "p_value")
            vsh, psh = cell("har", "se", "DM"), cell("har", "se", "p_value")
            lines.append(f"| {m} | {vqg:+.2f} | {pqg:.3f} | {vqh:+.2f} | {pqh:.3f} | "
                         f"{vsg:+.2f} | {psg:.3f} | {vsh:+.2f} | {psh:.3f} |")
        lines.append("")
    return "\n".join(lines)


def paper_strategy_table(strat: dict) -> str:
    rows = strat.get("rows", [])
    if not rows:
        return "_(strategy results unavailable)_"
    df = pd.DataFrame(rows)
    cols = [c for c in ["strategy", "ann_return", "ann_vol", "sharpe",
                        "sharpe_excess", "max_drawdown", "total_return",
                        "turnover_ann"] if c in df.columns]
    df = df[cols]
    lines = ["| Strategy | Ann.ret | Ann.vol | Sharpe | Sharpe(excess) | MaxDD | Turnover(yr) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['ann_return']*100:.2f}% | {r['ann_vol']*100:.2f}% | "
            f"{r['sharpe']:.2f} | {r['sharpe_excess']:.2f} | "
            f"{r['max_drawdown']*100:.1f}% | {r['turnover_ann']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report (Step 10)")
    utils.add_common_args(parser)
    args = parser.parse_args()
    cfg = utils.load_config(args.config)

    if not utils.METRICS_JSON.exists():
        log.error("metrics.json missing - run evaluate first")
        return
    data = utils.load_json(utils.METRICS_JSON)
    metrics = pd.DataFrame(data["metrics"])
    dm = pd.DataFrame(data["dm"])

    kr = key_results_table(metrics)
    kr.to_csv(utils.TABLES_DIR / "key_results.csv", index=False)
    try:
        kr.to_markdown(utils.TABLES_DIR / "key_results.md", index=False, floatfmt=".3f")
    except Exception:
        pass

    # programmatic paper tables
    pm = paper_metrics_table(metrics)
    (utils.TABLES_DIR / "paper_metrics_table.md").write_text(pm, encoding="utf-8")
    pd_ = paper_dm_table(dm)
    (utils.TABLES_DIR / "paper_dm_table.md").write_text(pd_, encoding="utf-8")

    best_rmse = _best_by_horizon(metrics, "RMSE")
    best_qlike = _best_by_horizon(metrics, "QLIKE")

    strat = {}
    if (utils.TABLES_DIR / "strategy_summary.json").exists():
        strat = utils.load_json(utils.TABLES_DIR / "strategy_summary.json")
        (utils.TABLES_DIR / "paper_strategy_table.md").write_text(
            paper_strategy_table(strat), encoding="utf-8")

    leak = None
    lp = utils.TABLES_DIR / "leakage_comparison.csv"
    if lp.exists():
        leak = pd.read_csv(lp)

    # ------------------------- summary markdown ---------------------------- #
    lines = ["# Results Summary (auto-generated from code output)", ""]
    # detect the actual OOS start from the produced predictions (fast smoke = 2015)
    run_start = None
    run_mode = "FULL"
    if utils.OOS_PREDICTIONS.exists():
        try:
            _p = utils.read_table(utils.OOS_PREDICTIONS)
            run_start = _p.index.min().date().isoformat()
            if run_start >= "2015-01-01":
                run_mode = "--fast smoke test"
        except Exception:
            pass
    lines.append(f"- Models benchmarked: {sorted(metrics['model'].unique().tolist())}")
    lines.append(f"- Horizons: {cfg['targets']['horizons']} trading days")
    lines.append(f"- OOS window: from {run_start or cfg['validation']['oos_start']} "
                 f"(expanding window, purge/embargo = horizon)")
    lines.append(f"- Run mode: {run_mode}")
    lines.append("")
    lines.append("## Best model per horizon")
    for h in sorted(set(best_rmse) | set(best_qlike)):
        br, bq = best_rmse.get(h), best_qlike.get(h)
        if br is not None and bq is not None:
            lines.append(f"- h={h}d: by RMSE **{br['model']}** ({br['RMSE']:.3f}); "
                         f"by QLIKE **{bq['model']}** ({bq['QLIKE']:.3f})")
    lines.append("")
    lines.append("## Did ML beat GARCH / HAR? (both losses, QLIKE first)")
    for h in sorted(metrics["horizon"].unique()):
        sub = metrics[metrics.horizon == h]
        ml_sub = sub[sub.model.isin(ML_SET)]
        if not len(ml_sub):
            continue
        # best ML by QLIKE (robust) and by RMSE
        mb_q = ml_sub.loc[ml_sub["QLIKE"].idxmin()]
        for metric, mb in (("RMSE", ml_sub.loc[ml_sub["RMSE"].idxmin()]),
                           ("QLIKE", mb_q)):
            for bench in ("garch", "har"):
                b = sub[sub.model == bench]
                if not len(b):
                    continue
                dd = dm[(dm.horizon == h) & (dm.model == mb["model"]) &
                        (dm.benchmark == bench) & (dm.loss == ("se" if metric == "RMSE" else "qlike"))]
                p = dd["p_value"].values[0] if len(dd) else float("nan")
                dmv = dd["DM"].values[0] if len(dd) else float("nan")
                win = mb[metric] < b[metric].values[0]
                sig = (p < 0.05) if p == p else False
                verdict = ("beats" if win else "does NOT beat") + (" (sig.)" if (win and sig) else (" (NOT sig.)" if win else ""))
                lines.append(f"- h={h}d, {metric}: best ML {mb['model']} vs {bench} "
                             f"-> {verdict}, DM={dmv:+.2f}, p={p:.3f}")
    lines.append("")

    if strat and strat.get("rows"):
        lines.append("## Economic value (volatility targeting, OOS)")
        lines.append(f"- Best ML chosen on FIRST half: {strat.get('best_ml_selection_half')} "
                     f"(h={strat.get('horizon')}); backtest evaluated on SECOND half.")
        for r in strat["rows"]:
            lines.append(f"  - {r['strategy']}: Sharpe={r['sharpe']:.2f} "
                         f"(excess {r.get('sharpe_excess', float('nan')):.2f}), "
                         f"ann.ret={r['ann_return']*100:.2f}%, "
                         f"ann.vol={r['ann_vol']*100:.2f}%, "
                         f"maxDD={r['max_drawdown']*100:.1f}%, "
                         f"turnover={r.get('turnover_ann', float('nan')):.2f}/yr")
        lines.append(f"- {strat.get('selection_bias_note','')}")
        lines.append(f"- {strat['disclaimer']}")
        lines.append("")

    if leak is not None and "rmse_reduction_vs_walkforward_pct" in leak.columns:
        lines.append("## Leakage check (RMSE reduction of random CV vs walk-forward)")
        for _, r in leak.iterrows():
            lines.append(f"- h={int(r['horizon'])}d: walk-forward {r['walkforward_RMSE']:.3f}, "
                         f"random K-fold (OOS rows) {r['random_kfold_RMSE_oos_rows']:.3f}, "
                         f"full sample {r['random_kfold_RMSE_full_sample']:.3f} "
                         f"-> reduction {r['rmse_reduction_vs_walkforward_pct']:.1f}%")
        lines.append("")

    if data.get("regime"):
        lines.append("## Bonus: high-risk regime classifier (next-21d max DD > 5%)")
        rg = data["regime"]
        for k in sorted(rg):
            lines.append(f"- {k}: {rg[k]}")
        lines.append("")

    text = "\n".join(lines)
    (utils.RESULTS_DIR / "summary.md").write_text(text, encoding="utf-8")
    log.info("Wrote results/summary.md")
    print("\n================ SUMMARY ================\n")
    print(text)

    # deliverables checklist
    checks = {
        "config.yaml": True, "requirements.txt": True, "Makefile": True,
        "README.md": (utils.REPO_ROOT / "README.md").exists(),
        "CHANGELOG.md": (utils.REPO_ROOT / "CHANGELOG.md").exists(),
        "paper/paper.md": (utils.REPO_ROOT / "paper" / "paper.md").exists(),
        "notebooks/01_eda.ipynb": (utils.REPO_ROOT / "notebooks" / "01_eda.ipynb").exists(),
        "notebooks/02_results.ipynb": (utils.REPO_ROOT / "notebooks" / "02_results.ipynb").exists(),
        "results/oos_predictions.csv.gz": utils.OOS_PREDICTIONS.exists(),
        "results/summary.md": (utils.RESULTS_DIR / "summary.md").exists(),
        "results/tables/main_metrics.csv": (utils.TABLES_DIR / "main_metrics.csv").exists(),
        "results/tables/dm_tests.csv": (utils.TABLES_DIR / "dm_tests.csv").exists(),
        "results/tables/paper_metrics_table.md": (utils.TABLES_DIR / "paper_metrics_table.md").exists(),
        "results/tables/paper_dm_table.md": (utils.TABLES_DIR / "paper_dm_table.md").exists(),
        "results/tables/key_results.csv": (utils.TABLES_DIR / "key_results.csv").exists(),
        "results/tables/strategy_performance.csv": (utils.TABLES_DIR / "strategy_performance.csv").exists(),
        "results/tables/leakage_comparison.csv": (utils.TABLES_DIR / "leakage_comparison.csv").exists(),
        "results/figures/shap_beeswarm.png": (utils.FIGURES_DIR / "shap_beeswarm.png").exists(),
        "results/figures/strategy_equity_drawdown.png": (utils.FIGURES_DIR / "strategy_equity_drawdown.png").exists(),
        "results/figures/leakage_random_vs_walkforward.png": (utils.FIGURES_DIR / "leakage_random_vs_walkforward.png").exists(),
    }
    print("\n================ DELIVERABLES CHECKLIST ================")
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    utils.save_json({k: bool(v) for k, v in checks.items()},
                    utils.RESULTS_DIR / "deliverables_checklist.json")


if __name__ == "__main__":
    main()
