"""Shared pytest fixtures: a small synthetic, fully leak-safe data set."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# make the repository root importable so `from src import ...` works under pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import utils  # noqa: E402


@pytest.fixture()
def cfg():
    return utils.load_config(str(utils.repo_path("config.yaml")))


def _index(n=600):
    return pd.bdate_range("2005-01-03", periods=n)


@pytest.fixture()
def synthetic_clean(tmp_path, cfg):
    """A fake clean_daily panel (no network) written to a temp CSV."""
    n = 600
    idx = _index(n)
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.01, n)
    price = 1000 * np.exp(np.cumsum(ret))
    frame = pd.DataFrame({
        utils.SP500_CLOSE: price,
        utils.SP500_RET: ret,
        utils.VIX: 15 + np.abs(rng.normal(0, 3, n)),
        "DGS10": 3 + rng.normal(0, 0.3, n).cumsum() * 0.01,
        "DGS2": 2 + rng.normal(0, 0.3, n).cumsum() * 0.01,
        "T10Y2Y": rng.normal(1, 0.5, n),
        "DFF": 2 + rng.normal(0, 0.2, n),
        "DCOILWTICO": 60 + np.abs(rng.normal(0, 10, n)),
        "BAMLH0A0HYM2": np.where(np.arange(n) > 500, 4 + rng.normal(0, 0.5, n), np.nan),
        "NFCI": rng.normal(0, 0.5, n),
        "UNRATE": 5 + rng.normal(0, 0.2, n),
        "CPIAUCSL": 250 + np.arange(n) * 0.1,
    }, index=idx)
    path = tmp_path / "clean_daily.csv.gz"
    frame.to_csv(path)
    return frame, path
