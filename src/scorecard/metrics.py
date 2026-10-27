"""Discrimination, calibration and stability metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score


def ks_statistic(y, risk) -> float:
    """Kolmogorov-Smirnov: max distance between the cumulative distributions of
    bads and goods when accounts are ordered by `risk` (higher = riskier).
    Ties are handled by evaluating only at distinct risk values."""
    y = np.asarray(y, dtype=float)
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(-risk, kind="mergesort")
    r, yy = risk[order], y[order]
    cum_bad = np.cumsum(yy) / yy.sum()
    cum_good = np.cumsum(1 - yy) / (1 - yy).sum()
    last_of_tie = np.r_[r[1:] != r[:-1], True]
    return float(np.max(np.abs(cum_bad - cum_good)[last_of_tie]))


def gini(auc: float) -> float:
    return 2 * auc - 1


def summary(y, pd_hat) -> dict:
    auc = roc_auc_score(y, pd_hat)
    return {"auc": auc, "gini": gini(auc), "ks": ks_statistic(y, pd_hat),
            "brier": brier_score_loss(y, pd_hat),
            "mean_pd": float(np.mean(pd_hat)), "observed_rate": float(np.mean(y))}


def psi(expected, actual, n_bins: int = 10) -> float:
    """Population Stability Index, bins = deciles of the expected (train) distribution.
    < 0.10 stable, 0.10-0.25 monitor, > 0.25 significant shift (industry rule of thumb)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=len(edges) + 1)
    a = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=len(edges) + 1)
    e = np.clip(e / e.sum(), 1e-6, None)
    a = np.clip(a / a.sum(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def calibration_table(y, pd_hat, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pd_hat)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    t = df.groupby("bin").agg(n=("y", "size"), mean_pd=("p", "mean"), observed=("y", "mean"))
    return t.reset_index()


def hosmer_lemeshow(y, pd_hat, n_bins: int = 10) -> dict:
    t = calibration_table(y, pd_hat, n_bins)
    exp_bad = t.n * t.mean_pd
    obs_bad = t.n * t.observed
    chi2 = float(np.sum((obs_bad - exp_bad) ** 2 / (t.n * t.mean_pd * (1 - t.mean_pd))))
    return {"hl_chi2": chi2, "hl_p_value": float(stats.chi2.sf(chi2, n_bins - 2))}
