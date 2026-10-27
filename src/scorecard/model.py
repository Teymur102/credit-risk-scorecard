"""Champion (WoE logistic regression) and challenger (gradient boosting) models."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .woe import FeatureBinning, fit_binning

PAY_STATUS = frozenset(f"PAY_{i}" for i in range(1, 7))


@dataclass
class SelectionConfig:
    iv_min: float = 0.02        # below this a variable is "weak" (Siddiqi rule of thumb)
    iv_cv_max: float = 0.25     # max coefficient of variation of IV across 5 folds
    corr_max: float = 0.70      # max |Pearson corr| between WoE-transformed variables
    p_max: float = 0.05         # Wald p-value threshold in the final regression
    max_bins: int = 10
    # variables binned as categories (one bin per level, rare levels pooled) instead of
    # monotonic numeric bins
    categorical: frozenset = PAY_STATUS
    min_bin_frac: float = 0.05
    seed: int = 42


def _logit_inference(X: np.ndarray, coef: np.ndarray, intercept: float):
    """Standard errors and Wald p-values from the inverse Fisher information."""
    Xd = np.column_stack([np.ones(len(X)), X])
    beta = np.concatenate([[intercept], coef])
    p = 1 / (1 + np.exp(-Xd @ beta))
    w = p * (1 - p)
    cov = np.linalg.inv((Xd * w[:, None]).T @ Xd)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pval = 2 * stats.norm.sf(np.abs(z))
    return se, pval


@dataclass
class WoEScorecardModel:
    cfg: SelectionConfig = field(default_factory=SelectionConfig)
    binnings: dict = field(default_factory=dict)
    selection: pd.DataFrame | None = None
    features: list = field(default_factory=list)
    lr: LogisticRegression | None = None
    coef_table: pd.DataFrame | None = None

    def _bin(self, X: pd.DataFrame, y: np.ndarray, cols) -> dict:
        c = self.cfg
        return {f: fit_binning(X[f].to_numpy(), y, f,
                               "categorical" if f in c.categorical else "numeric",
                               c.max_bins, c.min_bin_frac) for f in cols}

    def woe_matrix(self, X: pd.DataFrame, cols=None) -> pd.DataFrame:
        cols = cols or self.features
        return pd.DataFrame({f: self.binnings[f].transform(X[f].to_numpy()) for f in cols},
                            index=X.index)

    def fit(self, X: pd.DataFrame, y, candidates: list[str], stability: bool = True):
        c = self.cfg
        y = np.asarray(y)
        self.binnings = self._bin(X, y, candidates)
        rows = {f: {"feature": f, "iv": b.iv, "n_bins": b.n_bins, "status": "kept", "reason": ""}
                for f, b in self.binnings.items()}

        # 1. stability: refit the binning on 5 folds, look at how much IV moves
        if stability:
            ivs = {f: [] for f in candidates}
            for tr, _ in StratifiedKFold(5, shuffle=True, random_state=c.seed).split(X, y):
                for f, b in self._bin(X.iloc[tr], y[tr], candidates).items():
                    ivs[f].append(b.iv)
            for f in candidates:
                v = np.array(ivs[f])
                rows[f]["iv_fold_cv"] = float(v.std() / v.mean()) if v.mean() > 0 else np.inf

        def drop(f, reason):
            rows[f]["status"] = "dropped"
            rows[f]["reason"] = reason

        # 2. weak variables
        for f in candidates:
            if rows[f]["iv"] < c.iv_min:
                drop(f, f"weak: IV < {c.iv_min}")
            elif stability and rows[f]["iv_fold_cv"] > c.iv_cv_max:
                drop(f, f"unstable: IV fold CV > {c.iv_cv_max}")

        # 3. correlation: greedily keep the higher-IV member of each correlated pair
        alive = sorted([f for f in candidates if rows[f]["status"] == "kept"],
                       key=lambda f: -rows[f]["iv"])
        W = self.woe_matrix(X, alive)
        corr = W.corr().abs()
        kept: list[str] = []
        for f in alive:
            clash = [k for k in kept if corr.loc[f, k] > c.corr_max]
            if clash:
                drop(f, f"correlated with {clash[0]} (|r|={corr.loc[f, clash[0]]:.2f})")
            else:
                kept.append(f)

        # 4. regression: drop wrong-sign, then insignificant, one at a time
        while True:
            Xw = self.woe_matrix(X, kept).to_numpy()
            lr = LogisticRegression(C=np.inf, max_iter=2000).fit(Xw, y)
            coef = lr.coef_[0]
            se, pval = _logit_inference(Xw, coef, lr.intercept_[0])
            wrong = [(rows[f]["iv"], f) for f, b in zip(kept, coef) if b >= 0]
            if wrong:
                f = min(wrong)[1]
                drop(f, "wrong sign in regression (collinearity)")
                kept.remove(f)
                continue
            p_feat = pval[1:]
            if p_feat.max() > c.p_max:
                f = kept[int(np.argmax(p_feat))]
                drop(f, f"not significant (p={p_feat.max():.3f})")
                kept.remove(f)
                continue
            break

        self.features = kept
        self.lr = lr
        self.coef_table = pd.DataFrame({
            "term": ["intercept"] + kept,
            "coef": np.concatenate([lr.intercept_, coef]),
            "std_err": se, "p_value": pval,
        })
        self.selection = pd.DataFrame(rows.values()).sort_values("iv", ascending=False)
        return self

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.woe_matrix(X).to_numpy())[:, 1]


CHALLENGER_GRID = [
    dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100),
    dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50),
    dict(learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=200),
    dict(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=20),
]


def make_challenger(params: dict, seed: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=500, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, random_state=seed, **params)
