"""Static PNG figures (matplotlib). Palette validated for CVD separation:
champion = blue, challenger = orange; ink and grid are neutral."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .business import account_profit  # noqa: E402

CHAMP, CHAL = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def roc_ks(path, y, p_champ, p_chal, score_champ):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    for p, c, name in [(p_champ, CHAMP, "Champion: WoE scorecard"), (p_chal, CHAL, "Challenger: gradient boosting")]:
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import roc_auc_score
        a.plot(fpr, tpr, color=c, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    a.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
    a.set(title="ROC curve, held-out test set", xlabel="False positive rate (goods declined)",
          ylabel="True positive rate (bads caught)", xlim=(0, 1), ylim=(0, 1.01))
    a.legend(loc="lower right")

    s = np.sort(np.unique(score_champ))
    cum_bad = np.array([(score_champ[y == 1] <= v).mean() for v in s])
    cum_good = np.array([(score_champ[y == 0] <= v).mean() for v in s])
    k = int(np.argmax(cum_bad - cum_good))
    b.plot(s, cum_bad, color=CHAMP, label="Bads (defaulted)")
    b.plot(s, cum_good, color=INK2, label="Goods")
    b.vlines(s[k], cum_good[k], cum_bad[k], color=INK, lw=1.2)
    b.annotate(f"KS = {cum_bad[k] - cum_good[k]:.3f}\nat score {s[k]:.0f}",
               (s[k], (cum_bad[k] + cum_good[k]) / 2), xytext=(10, 0), textcoords="offset points",
               color=INK, va="center")
    b.set(title="KS: cumulative share at or below score (champion)", xlabel="Scorecard points",
          ylabel="Cumulative share", ylim=(0, 1.01))
    b.legend(loc="lower right")
    _save(fig, path)


def calibration(path, cal_champ, cal_chal, score_tr, score_te, psi_val):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    hi = max(cal_champ.observed.max(), cal_chal.observed.max(), cal_champ.mean_pd.max()) * 1.05
    a.plot([0, hi], [0, hi], color=AXIS, lw=1, ls="--", label="Perfect calibration")
    a.plot(cal_champ.mean_pd, cal_champ.observed, color=CHAMP, marker="o", ms=6, label="Champion")
    a.plot(cal_chal.mean_pd, cal_chal.observed, color=CHAL, marker="s", ms=6, label="Challenger")
    a.set(title="Reliability: predicted vs observed default (test deciles)",
          xlabel="Mean predicted PD", ylabel="Observed default rate", xlim=(0, hi), ylim=(0, hi))
    a.legend(loc="upper left")

    bins = np.linspace(min(score_tr.min(), score_te.min()), max(score_tr.max(), score_te.max()), 30)
    b.hist(score_tr, bins=bins, density=True, color=CHAMP, alpha=0.55, label="Train", edgecolor=SURFACE)
    b.hist(score_te, bins=bins, density=True, histtype="step", color=INK, lw=1.5, label="Test")
    b.set(title=f"Champion score distribution (PSI train vs test = {psi_val:.4f})",
          xlabel="Scorecard points", ylabel="Density")
    b.legend(loc="upper left")
    _save(fig, path)


def _curve(score, y, ead, econ):
    order = np.argsort(-score, kind="mergesort")
    yy, ee = y[order], ead[order]
    n = np.arange(1, len(y) + 1)
    appr = n / len(y)
    bad = np.cumsum(yy) / n
    prof = np.cumsum(account_profit(yy, ee, econ)) / 1e6
    return appr, bad, prof


def cutoff_profit(path, y, ead, econ, score_champ, score_chal, appr_champ, appr_chal):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    for s, c, name, ap in [(score_champ, CHAMP, "Champion", appr_champ), (score_chal, CHAL, "Challenger", appr_chal)]:
        x, bad, prof = _curve(s, y, ead, econ)
        m = x >= 0.05
        a.plot(x[m] * 100, bad[m] * 100, color=c, label=name)
        b.plot(x * 100, prof, color=c, label=name)
        i = int(np.searchsorted(x, ap)) - 1
        a.plot(x[i] * 100, bad[i] * 100, "o", color=c, ms=8, mec=SURFACE, mew=2)
        b.plot(x[i] * 100, prof[i], "o", color=c, ms=8, mec=SURFACE, mew=2)
    b.axhline(0, color=AXIS, lw=1)
    a.set(title="Bad rate among approved vs approval rate (test)", xlabel="Approval rate (%)",
          ylabel="Bad rate of approved (%)")
    b.set(title="Profit vs approval rate (test, assumed economics)", xlabel="Approval rate (%)",
          ylabel="Profit (NT$ millions)")
    a.legend(loc="upper left")
    b.legend(loc="lower left")
    b.annotate("dots = cutoff chosen on\nout-of-fold training scores", (0.98, 0.97),
               xycoords="axes fraction", ha="right", va="top", color=INK2, fontsize=9)
    _save(fig, path)
