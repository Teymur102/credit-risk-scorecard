"""Convert a WoE logistic regression into a points-based scorecard.

    Score = Offset + Factor * ln(odds_good)
    Factor = PDO / ln(2)
    Offset = BaseScore - Factor * ln(BaseOdds)

With logit(P(bad)) = b0 + sum_j b_j * WoE_j, ln(odds_good) = -(b0 + sum_j b_j * WoE_j),
so each attribute i of variable j earns
    points_ij = -(b_j * WoE_ij + b0 / n) * Factor + Offset / n
and an applicant's score is the sum of their attribute points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScaleSpec:
    pdo: float = 20.0         # points to double the odds
    base_score: float = 600.0
    base_odds: float = 50.0   # good:bad odds at the base score

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)


def score_from_pd(pd_bad, spec: ScaleSpec = ScaleSpec()) -> np.ndarray:
    p = np.clip(np.asarray(pd_bad, dtype=float), 1e-9, 1 - 1e-9)
    return spec.offset + spec.factor * np.log((1 - p) / p)


def pd_from_score(score, spec: ScaleSpec = ScaleSpec()) -> np.ndarray:
    odds = np.exp((np.asarray(score, dtype=float) - spec.offset) / spec.factor)
    return 1 / (1 + odds)


def attribute_points(woe: float, coef: float, intercept: float, n_vars: int,
                     spec: ScaleSpec = ScaleSpec()) -> float:
    return -(coef * woe + intercept / n_vars) * spec.factor + spec.offset / n_vars


def build_scorecard(model, spec: ScaleSpec = ScaleSpec()) -> pd.DataFrame:
    """One row per (variable, bin) with WoE, bad rate and integer points."""
    b0 = float(model.lr.intercept_[0])
    n = len(model.features)
    rows = []
    for f, b in zip(model.features, model.lr.coef_[0]):
        fb = model.binnings[f]
        for i, (label, r) in enumerate(zip(fb.labels(), fb.table.itertuples())):
            pts = attribute_points(r.woe, b, b0, n, spec)
            rows.append({"feature": f, "bin": i, "attribute": label, "n_train": r.n,
                         "bad_rate": r.bad_rate, "woe": r.woe, "coef": b,
                         "points_exact": pts, "points": int(round(pts))})
    return pd.DataFrame(rows)


def score_points(model, card: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Integer points per variable for each applicant (columns = variables)."""
    out = {}
    for f in model.features:
        pts = card.loc[card.feature == f, "points"].to_numpy()
        out[f] = pts[model.binnings[f].bin_index(X[f].to_numpy())]
    return pd.DataFrame(out, index=X.index)


def reason_codes(points_row: pd.Series, card: pd.DataFrame, descriptions: dict,
                 top: int = 4) -> pd.DataFrame:
    """Adverse-action reasons: variables where the applicant lost the most points
    relative to the best attainable attribute of that variable."""
    best = card.groupby("feature")["points"].max()
    lost = (best[points_row.index] - points_row).sort_values(ascending=False)
    lost = lost[lost > 0].head(top)
    return pd.DataFrame({"rank": range(1, len(lost) + 1), "feature": lost.index,
                         "reason": [descriptions.get(f, f) for f in lost.index],
                         "points_received": points_row[lost.index].to_numpy(),
                         "best_possible": best[lost.index].to_numpy(),
                         "points_lost": lost.to_numpy()})
