"""Monotonic binning, Weight of Evidence (WoE) and Information Value (IV).

Convention (Siddiqi): WoE = ln(%good / %bad). A positive WoE means the bin is
safer than average; the logistic regression coefficients on WoE inputs should
therefore be negative when the model predicts P(bad).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SMOOTH = 0.5  # added to good/bad counts so an all-good or all-bad bin has finite WoE


def woe_iv_table(bin_idx: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Counts, bad rate, WoE and IV contribution for each bin index 0..n_bins-1."""
    bin_idx = np.asarray(bin_idx)
    y = np.asarray(y)
    n = np.bincount(bin_idx, minlength=n_bins).astype(float)
    bad = np.bincount(bin_idx, weights=y, minlength=n_bins).astype(float)
    good = n - bad
    dist_good = (good + SMOOTH) / (good.sum() + SMOOTH * n_bins)
    dist_bad = (bad + SMOOTH) / (bad.sum() + SMOOTH * n_bins)
    woe = np.log(dist_good / dist_bad)
    iv = (dist_good - dist_bad) * woe
    with np.errstate(invalid="ignore", divide="ignore"):
        bad_rate = np.where(n > 0, bad / n, np.nan)
    return pd.DataFrame({"n": n.astype(int), "bads": bad.astype(int), "goods": good.astype(int),
                         "pct_pop": n / n.sum(), "bad_rate": bad_rate, "woe": woe, "iv": iv})


def information_value(bin_idx, y, n_bins) -> float:
    return float(woe_iv_table(bin_idx, y, n_bins)["iv"].sum())


def _assign(x: np.ndarray, edges: list[float]) -> np.ndarray:
    # bin i holds edges[i-1] < x <= edges[i]
    return np.searchsorted(np.asarray(edges, dtype=float), x, side="left")


def monotonic_edges(x: np.ndarray, y: np.ndarray, max_bins: int = 10,
                    min_frac: float = 0.05) -> list[float]:
    """Quantile pre-bins, then merge neighbours until (a) every bin holds at least
    `min_frac` of the sample and (b) the bad rate is strictly monotonic in the
    direction of the overall rank correlation between x and y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cands = np.unique(np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1]))
    edges = [e for e in cands if e < x.max()]
    rx = pd.Series(x).rank().to_numpy()
    corr = np.corrcoef(rx, y)[0, 1] if np.std(rx) > 0 else 0.0
    direction = 1.0 if (np.isnan(corr) or corr >= 0) else -1.0
    min_n = min_frac * len(x)

    while edges:
        idx = _assign(x, edges)
        k = len(edges) + 1
        n = np.bincount(idx, minlength=k).astype(float)
        bad = np.bincount(idx, weights=y, minlength=k)
        rate = bad / np.maximum(n, 1)
        small = np.where(n < min_n)[0]
        if len(small):
            i = small[np.argmin(n[small])]
            if i == 0:
                j = 0
            elif i == k - 1:
                j = k - 2
            else:  # merge with the neighbour whose bad rate is closer
                j = i - 1 if abs(rate[i] - rate[i - 1]) <= abs(rate[i] - rate[i + 1]) else i
            del edges[j]  # removing edges[j] merges bins j and j+1
            continue
        diffs = np.diff(rate) * direction
        bad_steps = np.where(diffs <= 0)[0]
        if len(bad_steps):
            j = bad_steps[np.argmin(np.abs(diffs[bad_steps]))]
            del edges[j]
            continue
        break
    return edges


@dataclass
class FeatureBinning:
    name: str
    kind: str  # "numeric" or "categorical"
    edges: list = field(default_factory=list)
    categories: dict = field(default_factory=dict)  # raw value -> bin index
    table: pd.DataFrame | None = None

    @property
    def n_bins(self) -> int:
        return len(self.table)

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def bin_index(self, x) -> np.ndarray:
        x = np.asarray(x)
        if self.kind == "numeric":
            return _assign(x.astype(float), self.edges)
        other = self.categories.get("__other__", 0)
        return np.array([self.categories.get(v, other) for v in x])

    def transform(self, x) -> np.ndarray:
        return self.table["woe"].to_numpy()[self.bin_index(x)]

    def labels(self) -> list[str]:
        if self.kind == "numeric":
            e = self.edges
            if not e:
                return ["all"]
            out = [f"<= {_fmt(e[0])}"]
            out += [f"({_fmt(a)}, {_fmt(b)}]" for a, b in zip(e[:-1], e[1:])]
            out.append(f"> {_fmt(e[-1])}")
            return out
        groups: dict[int, list] = {}
        for v, i in self.categories.items():
            if v != "__other__":
                groups.setdefault(i, []).append(str(v))
        return [", ".join(sorted(groups.get(i, []))) or "other" for i in range(self.n_bins)]


def _fmt(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.3g}"


def fit_binning(x, y, name: str, kind: str = "numeric", max_bins: int = 10,
                min_frac: float = 0.05) -> FeatureBinning:
    x = np.asarray(x)
    y = np.asarray(y)
    if kind == "numeric":
        edges = monotonic_edges(x, y, max_bins=max_bins, min_frac=min_frac)
        fb = FeatureBinning(name, kind, edges=edges)
    else:
        # categorical: one bin per level, levels below min_frac pooled into "other"
        vc = pd.Series(x).value_counts(normalize=True)
        big = [v for v in vc.index if vc[v] >= min_frac]
        cats = {v: i for i, v in enumerate(sorted(big))}
        other_idx = len(big)
        for v in vc.index:
            if v not in cats:
                cats[v] = other_idx
        cats["__other__"] = other_idx
        if other_idx == len(cats) - 1:  # nothing pooled -> unseen levels go to largest bin
            cats["__other__"] = cats[vc.index[0]]
        fb = FeatureBinning(name, kind, categories=cats)
    idx = fb.bin_index(x)
    n_bins = int(idx.max()) + 1 if kind == "categorical" else len(fb.edges) + 1
    fb.table = woe_iv_table(idx, y, n_bins)
    return fb
