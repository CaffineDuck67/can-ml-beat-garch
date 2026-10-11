"""Step 5 - machine-learning volatility models.

Hyper-parameters are selected with ``TimeSeriesSplit`` cross-validation *inside
the training window only*. To keep the walk-forward affordable we run the grid
search ONCE - on the earliest (oldest) training window - then reuse the winning
parameters for every subsequent re-fit, re-training the estimator on the
expanding data. Because tuning only ever uses the oldest window (strictly in
the past), no test or future information enters the selection. All models share
the same feature matrix and predict the same realized-vol target.

Provided models
  * Ridge (alpha tuned by ts-CV)
  * Random Forest (small ts-CV grid)
  * LightGBM or XGBoost (modest ts-CV grid; backend from config)
  * Optional LSTM (PyTorch) - degrades gracefully if torch is unavailable.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VOL_FLOOR = 0.05  # annualized vol can't realistically be below this (%)


def _clone(kind: str, params: dict, cfg: dict, seed: int, n_jobs: int):
    """Build a fresh estimator of ``kind`` with concrete ``params``."""
    if kind == "ridge":
        return make_pipeline(StandardScaler(),
                             Ridge(alpha=params["alpha"], random_state=seed))
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"], random_state=seed, n_jobs=n_jobs)
    if kind == "boost":
        backend = cfg["models"]["ml"]["boosting"].get("backend", "lightgbm").lower()
        if backend == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=seed, n_jobs=n_jobs, tree_method="hist",
                                objective="reg:squarederror", **params)
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbosity=-1,
                             **params)
    raise ValueError(kind)


class MLRunner:
    """Lazily tuned ML model with a common fit/predict API.

    Parameters
    ----------
    h : int
        Forecast horizon. Used to set a ``gap=h`` on the tuning
        ``TimeSeriesSplit`` so that the (overlapping) h-step target on the last
        validation rows never bleeds into the next training fold.
    log_target : bool
        If True the model is trained on ``log(RV)`` and inverted with a Duan
        (1983) *smearing* correction estimated on the training window only.
    """

    def __init__(self, name: str, kind: str, grid: Dict[str, List],
                 cfg: dict, seed: int, n_jobs: int, h: int = 1,
                 log_target: bool = False):
        self.name = name
        self.kind = kind
        self.grid = grid
        self.cfg = cfg
        self.seed = seed
        self.n_jobs = n_jobs
        self.h = int(h)
        self.log_target = bool(log_target)
        self.best_params: Optional[dict] = None
        self._smear: float = 1.0
        self._est = None

    def _tune(self, X: pd.DataFrame, y: pd.Series) -> dict:
        tscv = TimeSeriesSplit(n_splits=self.cfg["validation"]["cv_splits"],
                               gap=self.h)
        # estimator single-threaded; parallelise across grid points instead
        template = _clone(self.kind, self._default_params(), self.cfg, self.seed, 1)
        search = GridSearchCV(template, self.grid, cv=tscv,
                              scoring="neg_mean_squared_error", n_jobs=-1)
        search.fit(X.to_numpy(), y.to_numpy())
        # strip any pipeline step-prefix (e.g. 'ridge__alpha' -> 'alpha')
        return {k.split("__")[-1]: v for k, v in search.best_params_.items()}

    def _default_params(self) -> dict:
        if self.kind == "ridge":
            return {"alpha": self.cfg["models"]["ml"]["ridge_alphas"][0]}
        if self.kind == "rf":
            return {"n_estimators": 200, "max_depth": None,
                    "min_samples_leaf": 20, "max_features": "sqrt"}
        return {k: v[0] for k, v in self.grid.items()}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLRunner":
        y_fit = y
        if self.log_target:
            y_clip = np.clip(y.to_numpy(dtype=float), VOL_FLOOR, None)
            y_log = np.log(y_clip)
            y_fit = pd.Series(y_log, index=y.index)
            if self.best_params is None:
                self.best_params = self._tune(X, y_fit)  # tune once, oldest window
            est = _clone(self.kind, self.best_params, self.cfg, self.seed, self.n_jobs)
            est.fit(X.to_numpy(), y_fit.to_numpy())
            self._est = est
            # Duan smearing: E[exp(eps)] from in-sample residuals (train window only)
            resid = y_fit.to_numpy() - np.asarray(est.predict(X.to_numpy()), float)
            self._smear = float(np.mean(np.exp(np.clip(resid, -20.0, 20.0))))
            return self
        if self.best_params is None:
            self.best_params = self._tune(X, y)  # tune once, on the oldest window
        est = _clone(self.kind, self.best_params, self.cfg, self.seed, self.n_jobs)
        est.fit(X.to_numpy(), y.to_numpy())
        self._est = est
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = np.asarray(self._est.predict(X.to_numpy()), dtype=float)
        if self.log_target:
            p = self._smear * np.exp(np.clip(p, -20.0, 20.0))
        return np.clip(p, VOL_FLOOR, None)


# --------------------------------------------------------------------------- #
# Builders (fresh instance per horizon so tuning state is not shared)          #
#   A log_target=True variant trains on log(RV) with a smearing back-transform.#
# --------------------------------------------------------------------------- #
def build_ridge(cfg: dict, seed: int, n_jobs: int, h: int = 1,
                log_target: bool = False) -> MLRunner:
    alphas = cfg["models"]["ml"]["ridge_alphas"]
    grid = {"ridge__alpha": alphas}  # pipeline step-prefixed for GridSearchCV
    name = "ridge_log" if log_target else "ridge"
    return MLRunner(name, "ridge", grid, cfg, seed, n_jobs, h=h,
                    log_target=log_target)


def build_rf(cfg: dict, seed: int, n_jobs: int, h: int = 1,
             log_target: bool = False) -> MLRunner:
    rf = cfg["models"]["ml"]["rf"]
    grid = {"n_estimators": _as_list(rf["n_estimators"]),
            "max_depth": rf["max_depth"],
            "min_samples_leaf": rf["min_samples_leaf"],
            "max_features": rf["max_features"]}
    name = "rf_log" if log_target else "rf"
    return MLRunner(name, "rf", grid, cfg, seed, n_jobs, h=h,
                    log_target=log_target)


def build_boosting(cfg: dict, seed: int, n_jobs: int, h: int = 1,
                   log_target: bool = False) -> MLRunner:
    ml = cfg["models"]["ml"]["boosting"]
    base = ml.get("backend", "lightgbm").lower()
    name = f"{base}_log" if log_target else base
    grid = {"n_estimators": _as_list(ml["n_estimators"]),
            "learning_rate": ml["learning_rate"],
            "num_leaves": ml["num_leaves"],
            "min_child_samples": ml["min_child_samples"]}
    if base == "xgboost":
        grid = {"n_estimators": _as_list(ml["n_estimators"]),
                "learning_rate": ml["learning_rate"],
                "max_depth": [3, 5], "subsample": [0.8, 1.0]}
    return MLRunner(name, "boost", grid, cfg, seed, n_jobs, h=h,
                    log_target=log_target)


def _as_list(x):
    return x if isinstance(x, list) else [x]


# --------------------------------------------------------------------------- #
# Optional LSTM                                                                 #
# --------------------------------------------------------------------------- #
class LSTMModel:
    """Small recurrent sequence model. Skipped by callers if torch is absent."""

    name = "lstm"

    def __init__(self, cfg: dict, seed: int):
        import torch  # noqa: F401  (raises -> caller skips gracefully)
        self.cfg = cfg["models"]["lstm"]
        self.seed = seed
        self.seq_len = int(self.cfg["seq_len"])
        self._net = None
        self._mean = None
        self._std = None
        self._y_std = 1.0

    def _make_net(self, n_feat: int):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, in_dim, hidden, layers):
                super().__init__()
                self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers,
                  batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        return _Net(n_feat, int(self.cfg["hidden_size"]),
                    int(self.cfg["num_layers"]))

    def _sequences(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.stack([X[i - self.seq_len + 1: i + 1]
                         for i in range(self.seq_len - 1, n)])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMModel":
        import torch
        torch.manual_seed(self.seed)
        Xn = X.to_numpy(dtype=float)
        self._mean = np.nanmean(Xn, axis=0)
        self._std = np.nanstd(Xn, axis=0) + 1e-8
        Xs = (np.nan_to_num(Xn, nan=0.0) - self._mean) / self._std
        yv = y.to_numpy(dtype=float)
        self._y_std = float(np.std(yv) + 1e-8)
        ys = yv / self._y_std

        seqs = self._sequences(Xs)
        ys = ys[self.seq_len - 1:]
        xs = torch.tensor(seqs, dtype=torch.float32)
        yy = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

        net = self._make_net(X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=float(self.cfg["lr"]))
        loss_fn = torch.nn.MSELoss()
        bs = int(self.cfg["batch_size"])
        patience = int(self.cfg.get("patience", 0))
        n = len(xs)
        # Time-ordered hold-out: last ~10% of the training window validates.
        # Because LSTM samples are ordered by date, this never uses future
        # sequences to train an earlier prediction.
        n_tr = n if n < 20 else int(0.9 * n)
        xs_tr, yy_tr = xs[:n_tr], yy[:n_tr]
        xs_va, yy_va = xs[n_tr:], yy[n_tr:]
        m = len(xs_tr)
        best_val = float("inf")
        best_state = None
        bad = 0
        for _ in range(int(self.cfg["epochs"])):
            net.train()
            perm = torch.randperm(m)
            tot = 0.0
            for i in range(0, m, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(net(xs_tr[b]), yy_tr[b])
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            if tot / max(m, 1) < 1e-4:
                break
            if len(xs_va):
                net.eval()
                with torch.no_grad():
                    val = float(loss_fn(net(xs_va), yy_va))
                if val < best_val - 1e-6:
                    best_val = val
                    best_state = {k: v.detach().clone()
                                  for k, v in net.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if patience and bad >= patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = net
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        if self._net is None:
            raise RuntimeError("LSTM not fitted")
        Xn = X.to_numpy(dtype=float)
        Xs = (np.nan_to_num(Xn, nan=0.0) - self._mean) / self._std
        seqs = self._sequences(Xs)
        with torch.no_grad():
            pred = self._net(torch.tensor(seqs, dtype=torch.float32)).numpy().ravel()
        return np.clip(pred * self._y_std, VOL_FLOOR, None)
