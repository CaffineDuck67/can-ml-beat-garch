"""Shared utilities: config loading, logging, seeding and path helpers.

Every pipeline module imports from here so that behaviour (paths, seeds,
trading-day conventions) is defined in exactly one place.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
# The repository root is the parent of the ``src`` package.
REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: str) -> Path:
    """Return an absolute path under the repository root."""
    return REPO_ROOT.joinpath(*parts)


# Canonical intermediate-file locations ------------------------------------- #
RAW_DIR = repo_path("data", "raw")
PROCESSED_DIR = repo_path("data", "processed")
RESULTS_DIR = repo_path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

CLEAN_DAILY = PROCESSED_DIR / "clean_daily.csv.gz"
TARGETS = PROCESSED_DIR / "targets.csv.gz"
FEATURES = PROCESSED_DIR / "features.csv.gz"
DATASET = PROCESSED_DIR / "dataset.csv.gz"
DOWNLOAD_LOG = RAW_DIR / "download_log.json"
OOS_PREDICTIONS = RESULTS_DIR / "oos_predictions.csv.gz"
METRICS_JSON = TABLES_DIR / "metrics.json"

for _d in (RAW_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Column-name conventions (single source of truth)                            #
# --------------------------------------------------------------------------- #
SP500_CLOSE = "sp500_close"
SP500_RET = "sp500_ret"          # simple log return r_t = ln(P_t/P_{t-1})
VIX = "vix"

def rv_target_col(h: int) -> str:
    """Name of the realized-vol target column for horizon ``h`` (in %)."""
    return f"rv_h{h}"


def pred_col(model: str, h: int) -> str:
    """Name of the OOS forecast column produced by ``model`` for horizon h."""
    return f"{model}__h{h}"


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load ``config.yaml`` and resolve runtime defaults.

    ``data.end_date`` of ``null`` is replaced by today's date.
    """
    cfg_path = Path(path) if path else repo_path("config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)

    end = cfg["data"].get("end_date")
    if end in (None, "", "null"):
        cfg["data"]["end_date"] = _dt.date.today().isoformat()

    start = cfg["data"].get("start_date") or "2000-01-01"
    cfg["data"]["start_date"] = start
    return cfg


def apply_fast(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate ``cfg`` for a quick smoke run: fewer refits, later OOS start."""
    cfg["runtime"]["fast"] = True
    v = cfg["validation"]
    v["ml_refit_every"] = 63
    cfg["models"]["garch"]["refit_every"] = 126
    cfg["models"]["lstm"]["refit_every"] = 126
    cfg["models"]["lstm"]["epochs"] = 8
    v["oos_start"] = "2015-01-01"
    v["cv_splits"] = 3
    return cfg


# --------------------------------------------------------------------------- #
# Seeding & logging                                                           #
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    """Seed python, numpy and (if present) torch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch is optional
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def primary_seed(cfg: Dict[str, Any]) -> int:
    seeds = cfg["runtime"]["seeds"]
    return int(seeds[0]) if seeds else 7


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# --------------------------------------------------------------------------- #
# CLI helpers                                                                 #
# --------------------------------------------------------------------------- #
def add_common_args(parser: "argparse.ArgumentParser") -> None:  # noqa: F821
    parser.add_argument("--config", default=str(repo_path("config.yaml")),
                        help="Path to config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Re-download / recompute even if cached")
    parser.add_argument("--fast", action="store_true",
                        help="Quick smoke run (fewer refits, later OOS start)")


def config_from_args(args: "argparse.Namespace"):  # noqa: F821
    cfg = load_config(args.config)
    if getattr(args, "fast", False):
        cfg = apply_fast(cfg)
    seed = primary_seed(cfg)
    set_global_seed(seed)
    return cfg


# --------------------------------------------------------------------------- #
# I/O helpers                                                                 #
# --------------------------------------------------------------------------- #
def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_table(path: Path):
    """Read a saved table (csv or csv.gz) with a parsed DatetimeIndex."""
    import pandas as pd

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def configure_matplotlib() -> None:
    """Non-interactive, reproducible matplotlib defaults."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
