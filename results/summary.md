# Results Summary (auto-generated from code output)

- Models benchmarked: ['ewma', 'garch', 'garch_adj', 'har', 'hist_mean', 'lightgbm', 'lightgbm_log', 'lstm', 'rf', 'rf_log', 'ridge', 'ridge_log']
- Horizons: [5, 21] trading days
- OOS window: from 2015-01-02 (expanding window, purge/embargo = horizon)
- Run mode: --fast smoke test

## Best model per horizon
- h=5d: by RMSE **ridge** (7.569); by QLIKE **rf** (1.406)
- h=21d: by RMSE **rf_log** (7.983); by QLIKE **garch** (1.435)

## Did ML beat GARCH / HAR? (both losses, QLIKE first)
- h=5d, RMSE: best ML ridge vs garch -> beats (sig.), DM=+3.77, p=0.000
- h=5d, RMSE: best ML ridge vs har -> beats (sig.), DM=+3.38, p=0.001
- h=5d, QLIKE: best ML rf vs garch -> beats (sig.), DM=+4.81, p=0.000
- h=5d, QLIKE: best ML rf vs har -> beats (sig.), DM=+4.77, p=0.000
- h=21d, RMSE: best ML rf_log vs garch -> beats (NOT sig.), DM=+1.52, p=0.128
- h=21d, RMSE: best ML rf_log vs har -> beats (NOT sig.), DM=+0.92, p=0.359
- h=21d, QLIKE: best ML rf vs garch -> does NOT beat, DM=-0.40, p=0.692
- h=21d, QLIKE: best ML rf vs har -> beats (NOT sig.), DM=+0.46, p=0.644

## Economic value (volatility targeting, OOS)
- Best ML chosen on FIRST half: lightgbm_log (h=21); backtest evaluated on SECOND half.
  - buy_and_hold: Sharpe=0.80 (excess 0.61), ann.ret=13.15%, ann.vol=16.36%, maxDD=-27.1%, turnover=0.00/yr
  - vol_target_garch: Sharpe=0.82 (excess 0.60), ann.ret=12.23%, ann.vol=14.91%, maxDD=-22.0%, turnover=2.47/yr
  - vol_target_lightgbm_log: Sharpe=1.06 (excess 0.83), ann.ret=15.10%, ann.vol=14.23%, maxDD=-17.8%, turnover=2.59/yr
- strategy_performance_full.csv selects the ML model on the SAME window it is evaluated on -> optimistic; use strategy_performance.csv.
- Research only - NOT investment advice. Past performance does not guarantee future results.

## Leakage check (RMSE reduction of random CV vs walk-forward)
- h=5d: walk-forward 7.895, random K-fold (OOS rows) 5.716, full sample 5.601 -> reduction 27.6%
- h=21d: walk-forward 9.547, random K-fold (OOS rows) 3.152, full sample 3.525 -> reduction 67.0%

## Bonus: high-risk regime classifier (next-21d max DD > 5%)
- lightgbm_base_rate: 0.2223357167177392
- lightgbm_brier: 0.20540820680043817
- lightgbm_n: 2937
- lightgbm_pr_auc: 0.22231351273304506
- logistic_base_rate: 0.2223357167177392
- logistic_brier: 0.1772935378173551
- logistic_n: 2937
- logistic_pr_auc: 0.3479438987225235
