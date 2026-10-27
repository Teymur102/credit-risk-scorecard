"""Expected loss, cutoff analysis, profit curve and fairness checks.

Every economic parameter below is an ASSUMPTION, not something in the data.
The dataset has no loss, recovery or revenue information.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Economics:
    lgd: float = 0.75          # loss given default, unsecured card (assumed)
    ccf: float = 0.50          # credit conversion factor on the undrawn limit (assumed)
    margin: float = 0.08       # annual net revenue on EAD of a non-defaulting account (assumed)

    def as_dict(self) -> dict:
        return asdict(self)


def exposure_at_default(bill_amt1, limit_bal, econ: Economics) -> np.ndarray:
    """EAD = current drawn balance + CCF x undrawn limit (balances floored at 0)."""
    drawn = np.clip(np.asarray(bill_amt1, dtype=float), 0, None)
    undrawn = np.clip(np.asarray(limit_bal, dtype=float) - drawn, 0, None)
    return drawn + econ.ccf * undrawn


def account_profit(default, ead, econ: Economics) -> np.ndarray:
    """Realised one-year profit if the account is approved: margin on goods, LGD x EAD on bads."""
    d = np.asarray(default, dtype=float)
    return (1 - d) * econ.margin * ead - d * econ.lgd * ead


def expected_profit(pd_hat, ead, econ: Economics) -> np.ndarray:
    p = np.asarray(pd_hat, dtype=float)
    return (1 - p) * econ.margin * ead - p * econ.lgd * ead


def cutoff_table(score, default, pd_hat, ead, econ: Economics, cutoffs) -> pd.DataFrame:
    """Approve accounts with score >= cutoff (higher score = safer)."""
    score = np.asarray(score)
    default = np.asarray(default)
    pd_hat = np.asarray(pd_hat)
    ead = np.asarray(ead)
    rows = []
    for c in cutoffs:
        a = score >= c
        n_a = a.sum()
        rows.append({
            "cutoff": c,
            "approval_rate": a.mean(),
            "bad_rate_approved": default[a].mean() if n_a else np.nan,
            "bad_rate_declined": default[~a].mean() if (~a).sum() else np.nan,
            "expected_loss": float(np.sum(pd_hat[a] * econ.lgd * ead[a])),
            "realised_loss": float(np.sum(default[a] * econ.lgd * ead[a])),
            "profit": float(np.sum(account_profit(default[a], ead[a], econ))),
        })
    return pd.DataFrame(rows)


def best_cutoff(score, default, ead, econ: Economics, cutoffs) -> float:
    prof = [np.sum(account_profit(default[score >= c], ead[score >= c], econ)) for c in cutoffs]
    return float(cutoffs[int(np.argmax(prof))])


def adverse_impact(approved, group, default, pd_hat) -> pd.DataFrame:
    df = pd.DataFrame({"approved": approved, "group": group, "default": default, "pd": pd_hat})
    t = df.groupby("group", observed=True).agg(
        n=("approved", "size"), approval_rate=("approved", "mean"),
        default_rate_all=("default", "mean"), mean_pd_all=("pd", "mean"))
    t["default_rate_approved"] = df[df.approved].groupby("group", observed=True)["default"].mean()
    t["air_vs_highest"] = t["approval_rate"] / t["approval_rate"].max()
    t["four_fifths_flag"] = t["air_vs_highest"] < 0.8
    return t.reset_index()
