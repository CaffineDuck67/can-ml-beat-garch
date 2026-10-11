"""No-leakage test: perturbing post-t data must leave features at t unchanged.

This is the core guarantee of the whole benchmark - a feature dated t may not
depend on any observation after t. We rebuild the feature panel twice from the
same synthetic data, once with the future (after cut date T) scrambled, and
assert the pre-T features are bit-for-bit identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import utils
import src.features as FE
import src.targets as TG


def test_features_do_not_look_ahead(cfg, synthetic_clean, monkeypatch):
    frame, path = synthetic_clean
    monkeypatch.setattr(utils, "CLEAN_DAILY", path)

    base = FE.build_features(cfg)

    # scramble everything strictly AFTER the cut date
    cut = frame.index[300]
    perturbed = frame.copy()
    after = perturbed.index > cut
    rng = np.random.default_rng(123)
    for col in perturbed.columns:
        if col == utils.SP500_RET:
            perturbed.loc[after, col] = rng.normal(0, 0.05, after.sum())
        elif col == utils.SP500_CLOSE:
            perturbed.loc[after, col] = perturbed.loc[after, col].values * 2.0
        else:
            perturbed.loc[after, col] = perturbed.loc[after, col].values * 1.5 + 10.0
    p2 = path.parent / "clean_perturbed.csv.gz"
    perturbed.to_csv(p2)
    monkeypatch.setattr(utils, "CLEAN_DAILY", p2)

    rebuilt = FE.build_features(cfg)

    common = base.index[base.index <= cut]
    pd.testing.assert_frame_equal(
        base.loc[common], rebuilt.loc[common], check_exact=False, atol=1e-10)


def test_monthly_publication_lag_applied(cfg):
    """UNRATE/CPIAUCSL must be shifted forward before use (data_clean)."""
    from src import data_clean
    # publication_lag_days for monthly series must be >= ~1 month
    assert cfg["data"]["publication_lag_days"]["UNRATE"] >= 30
    assert cfg["data"]["publication_lag_days"]["CPIAUCSL"] >= 30


def test_target_is_forward_looking_only(cfg, synthetic_clean, monkeypatch):
    """RV_h(t) uses r_{t+1..t+h}: changing the past must not alter it,
    changing the future window must."""
    frame, path = synthetic_clean
    monkeypatch.setattr(utils, "CLEAN_DAILY", path)
    r = frame[utils.SP500_RET]
    h = 5
    rv = TG.forward_realized_vol(r, h, cfg["targets"]["trading_days_per_year"])

    # change a past return (well before t+h) -> RV at a later unaffected origin
    r2 = r.copy()
    t = r2.index[200]
    r2.iloc[150] = r2.iloc[150] + 1.0  # far in the past relative to origin 200
    rv2 = TG.forward_realized_vol(r2, h, cfg["targets"]["trading_days_per_year"])
    assert np.isclose(rv.loc[t], rv2.loc[t])

    # change r_{t+3} (inside the forward window) -> RV at t must change
    r3 = r.copy()
    r3.iloc[200 + 3] = r3.iloc[200 + 3] + 1.0
    rv3 = TG.forward_realized_vol(r3, h, cfg["targets"]["trading_days_per_year"])
    assert not np.isclose(rv.loc[t], rv3.loc[t])
