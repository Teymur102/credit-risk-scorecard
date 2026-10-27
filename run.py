"""Reproduce every number, table and figure in this repo:  python run.py"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from scorecard import business as biz  # noqa: E402
from scorecard import data as D  # noqa: E402
from scorecard import metrics as M  # noqa: E402
from scorecard import plots  # noqa: E402
from scorecard.model import CHALLENGER_GRID, SelectionConfig, WoEScorecardModel, make_challenger  # noqa: E402
from scorecard.points import ScaleSpec, build_scorecard, reason_codes, score_from_pd, score_points  # noqa: E402
from scorecard.woe import fit_binning  # noqa: E402

SEED = 42
TEST_SIZE = 0.30
SPEC = ScaleSpec(pdo=20, base_score=600, base_odds=50)
ECON = biz.Economics(lgd=0.75, ccf=0.50, margin=0.08)
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

DESCRIPTIONS = {
    "PAY_1": "Payment status last month (months past due)",
    "PAY_2": "Payment status 2 months ago", "PAY_3": "Payment status 3 months ago",
    "PAY_4": "Payment status 4 months ago", "PAY_5": "Payment status 5 months ago",
    "PAY_6": "Payment status 6 months ago",
    "MAX_DELINQ_6M": "Worst delinquency in the last 6 months",
    "MONTHS_DELINQ_6M": "Number of delinquent months in the last 6",
    "LIMIT_BAL": "Low credit limit",
    "UTIL_1": "Balance high relative to credit limit",
    "AVG_UTIL_6M": "Average utilisation over 6 months high",
    "PAY_RATIO_1": "Small share of last statement repaid",
    "AVG_PAY_AMT_6M": "Low average payment amount over 6 months",
    "EDUCATION": "Education level",
    **{f"PAY_AMT{i}": f"Low payment amount {i} month(s) ago" for i in range(1, 7)},
    **{f"BILL_AMT{i}": f"Statement balance {i} month(s) ago" for i in range(1, 7)},
}


def fold_points_score(model, X):
    card = build_scorecard(model, SPEC)
    return score_points(model, card, X).sum(axis=1).to_numpy()


def main():
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    out: dict = {"seed": SEED, "test_size": TEST_SIZE,
                 "scaling": {"pdo": SPEC.pdo, "base_score": SPEC.base_score,
                             "base_odds": SPEC.base_odds, "factor": SPEC.factor,
                             "offset": SPEC.offset},
                 "economics_assumptions": ECON.as_dict()}

    # ---------------- data ----------------
    raw = D.load_raw(ROOT / "data")
    df = D.add_features(raw)
    y = df[D.TARGET].to_numpy()
    feats = D.candidate_features(df)
    out["data"] = {"rows": len(df), "columns_raw": raw.shape[1] - 1, "default_rate": float(y.mean()),
                   "n_candidates": len(feats), "raw_sha256": D.sha256(ROOT / "data" / D.RAW_ZIP),
                   "missing_values": int(df.isna().sum().sum())}
    print(f"data: {len(df)} rows, default rate {y.mean():.4f}, {len(feats)} candidate features")

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE,
                                      stratify=y, random_state=SEED)
    tr, te = df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_te].reset_index(drop=True)
    ytr, yte = y[idx_tr], y[idx_te]
    out["split"] = {"train_rows": len(tr), "test_rows": len(te),
                    "train_default_rate": float(ytr.mean()), "test_default_rate": float(yte.mean())}

    # ---------------- 5-fold CV on the training set (whole pipeline refit per fold) ----------------
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(skf.split(tr, ytr))
    oof_champ_pd = np.zeros(len(tr))
    oof_champ_score = np.zeros(len(tr))
    oof_chal = {g: np.zeros(len(tr)) for g in range(len(CHALLENGER_GRID))}
    cv_rows = []
    for k, (a, b) in enumerate(folds):
        m = WoEScorecardModel(SelectionConfig(seed=SEED)).fit(tr.iloc[a], ytr[a], feats)
        p = m.predict_pd(tr.iloc[b])
        oof_champ_pd[b] = p
        oof_champ_score[b] = fold_points_score(m, tr.iloc[b])
        cv_rows.append({"fold": k, "model": "champion", "n_features": len(m.features),
                        **M.summary(ytr[b], p)})
        mv = WoEScorecardModel(SelectionConfig(seed=SEED, categorical=frozenset())).fit(tr.iloc[a], ytr[a], feats)
        cv_rows.append({"fold": k, "model": "variant_monotonic_pay", "n_features": len(mv.features),
                        **M.summary(ytr[b], mv.predict_pd(tr.iloc[b]))})
        for g, params in enumerate(CHALLENGER_GRID):
            h = make_challenger(params, SEED).fit(tr[feats].iloc[a], ytr[a])
            q = h.predict_proba(tr[feats].iloc[b])[:, 1]
            oof_chal[g][b] = q
            cv_rows.append({"fold": k, "model": f"challenger_cfg{g}", **M.summary(ytr[b], q)})
        print(f"  fold {k}: champion AUC {cv_rows[-2 - len(CHALLENGER_GRID)]['auc']:.4f} "
              f"({len(m.features)} vars)")
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(REPORTS / "cv_folds.csv", index=False)
    cv_sum = cv.groupby("model")[["auc", "gini", "ks", "brier"]].agg(["mean", "std"])
    best_g = int(np.argmax([cv_sum.loc[f"challenger_cfg{g}", ("auc", "mean")]
                            for g in range(len(CHALLENGER_GRID))]))
    out["challenger_grid"] = [{"cfg": g, **CHALLENGER_GRID[g],
                               "cv_auc_mean": cv_sum.loc[f"challenger_cfg{g}", ("auc", "mean")],
                               "cv_auc_std": cv_sum.loc[f"challenger_cfg{g}", ("auc", "std")]}
                              for g in range(len(CHALLENGER_GRID))]
    out["challenger_selected_cfg"] = best_g
    out["cv"] = {name: {f"{m_}_{s}": float(cv_sum.loc[key, (m_, s)])
                        for m_ in ["auc", "gini", "ks", "brier"] for s in ["mean", "std"]}
                 for name, key in [("champion", "champion"), ("challenger", f"challenger_cfg{best_g}"),
                                   ("variant_monotonic_pay", "variant_monotonic_pay")]}
    for name in ("champion", "variant_monotonic_pay"):
        out["cv"][name]["n_features_per_fold"] = cv[cv.model == name]["n_features"].astype(int).tolist()
    oof_chal_pd = oof_chal[best_g]

    # ---------------- final fits on the full training set ----------------
    champ = WoEScorecardModel(SelectionConfig(seed=SEED)).fit(tr, ytr, feats)
    champ.selection.to_csv(REPORTS / "iv_table.csv", index=False)
    champ.coef_table.to_csv(REPORTS / "champion_coefficients.csv", index=False)
    # variant kept only for the write-up: PAY_x forced into monotonic numeric bins
    mono = WoEScorecardModel(SelectionConfig(seed=SEED, categorical=frozenset())).fit(tr, ytr, feats)
    mono_pd_te = mono.predict_pd(te)
    out["variant_monotonic_pay"] = {"features": mono.features, **M.summary(yte, mono_pd_te),
                                    "pay_1_bins": mono.binnings["PAY_1"].labels()}
    card = build_scorecard(champ, SPEC)
    card.to_csv(REPORTS / "scorecard_points.csv", index=False)
    pts_tr = score_points(champ, card, tr)
    pts_te = score_points(champ, card, te)
    score_tr, score_te = pts_tr.sum(axis=1).to_numpy(), pts_te.sum(axis=1).to_numpy()
    champ_pd_tr, champ_pd_te = champ.predict_pd(tr), champ.predict_pd(te)
    out["champion_features"] = champ.features
    edu = fit_binning(tr.EDUCATION.to_numpy(), ytr, "EDUCATION", "categorical")
    out["excluded_by_policy_iv"] = {"EDUCATION": edu.iv}
    print(f"champion: {len(champ.features)} variables: {champ.features}")

    chal = make_challenger(CHALLENGER_GRID[best_g], SEED).fit(tr[feats], ytr)
    chal_pd_tr = chal.predict_proba(tr[feats])[:, 1]
    chal_pd_te = chal.predict_proba(te[feats])[:, 1]
    out["challenger_iterations"] = int(chal.n_iter_)

    # ---------------- held-out test evaluation ----------------
    test = {}
    for name, ptr, pte, s_tr, s_te in [
        ("champion", champ_pd_tr, champ_pd_te, score_tr, score_te),
        ("challenger", chal_pd_tr, chal_pd_te, score_from_pd(chal_pd_tr, SPEC), score_from_pd(chal_pd_te, SPEC)),
    ]:
        test[name] = {**M.summary(yte, pte), **M.hosmer_lemeshow(yte, pte),
                      "train_auc": roc_auc_score(ytr, ptr),
                      "psi_train_vs_test_score": M.psi(s_tr, s_te)}
        M.calibration_table(yte, pte).assign(model=name).to_csv(
            REPORTS / f"calibration_{name}.csv", index=False)
    test["champion"]["auc_integer_points_score"] = roc_auc_score(yte, -score_te)
    test["champion"]["ks_integer_points_score"] = M.ks_statistic(yte, -score_te)
    test["auc_lift_challenger_minus_champion"] = test["challenger"]["auc"] - test["champion"]["auc"]
    test["gini_lift_challenger_minus_champion"] = test["challenger"]["gini"] - test["champion"]["gini"]
    # bootstrap CI on the AUC difference (paired, 1000 resamples of the test set)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(1000):
        i = rng.integers(0, len(yte), len(yte))
        if yte[i].min() == yte[i].max():
            continue
        diffs.append(roc_auc_score(yte[i], chal_pd_te[i]) - roc_auc_score(yte[i], champ_pd_te[i]))
    test["auc_lift_bootstrap_95ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    out["test"] = test
    print(f"test AUC champion {test['champion']['auc']:.4f}  challenger {test['challenger']['auc']:.4f}")

    # ---------------- business layer ----------------
    ead_tr = biz.exposure_at_default(tr.BILL_AMT1, tr.LIMIT_BAL, ECON)
    ead_te = biz.exposure_at_default(te.BILL_AMT1, te.LIMIT_BAL, ECON)
    # cutoff chosen on out-of-fold TRAIN scores, then applied unchanged to the test set
    grid = np.arange(np.floor(np.percentile(oof_champ_score, 1)), np.ceil(np.percentile(oof_champ_score, 99)) + 1)
    cut = biz.best_cutoff(oof_champ_score, ytr, ead_tr, ECON, grid)
    chal_oof_score = score_from_pd(oof_chal_pd, SPEC)
    chal_grid = np.unique(np.round(np.percentile(chal_oof_score, np.arange(1, 100, 0.25)), 2))
    chal_cut = biz.best_cutoff(chal_oof_score, ytr, ead_tr, ECON, chal_grid)
    chal_score_te = score_from_pd(chal_pd_te, SPEC)

    approve_te = score_te >= cut
    approve_chal_te = chal_score_te >= chal_cut
    pd_card_te = champ_pd_te  # PD used for expected loss = champion regression PD
    bus = {"champion_cutoff_score": cut, "challenger_cutoff_score": chal_cut,
           "break_even_pd": ECON.margin / (ECON.margin + ECON.lgd)}
    for name, appr, p in [("champion", approve_te, champ_pd_te), ("challenger", approve_chal_te, chal_pd_te)]:
        bus[name] = {
            "approval_rate": float(appr.mean()),
            "bad_rate_approved": float(yte[appr].mean()),
            "bad_rate_declined": float(yte[~appr].mean()),
            "expected_loss_approved": float(np.sum(p[appr] * ECON.lgd * ead_te[appr])),
            "realised_loss_approved": float(np.sum(yte[appr] * ECON.lgd * ead_te[appr])),
            "profit_approved": float(np.sum(biz.account_profit(yte[appr], ead_te[appr], ECON))),
        }
    bus["profit_approve_all"] = float(np.sum(biz.account_profit(yte, ead_te, ECON)))
    bus["expected_loss_all"] = float(np.sum(champ_pd_te * ECON.lgd * ead_te))
    bus["realised_loss_all"] = float(np.sum(yte * ECON.lgd * ead_te))
    bus["total_ead_test"] = float(ead_te.sum())
    out["business"] = bus
    print(f"cutoff {cut:.0f}: approval {bus['champion']['approval_rate']:.3f}, "
          f"bad rate {bus['champion']['bad_rate_approved']:.4f}")

    # paired bootstrap of the test-set profit difference at the two chosen cutoffs
    prof_ch = biz.account_profit(yte, ead_te, ECON) * approve_te
    prof_cl = biz.account_profit(yte, ead_te, ECON) * approve_chal_te
    boots = [np.sum(prof_cl[i] - prof_ch[i]) for i in
             (rng.integers(0, len(yte), len(yte)) for _ in range(1000))]
    bus["profit_diff_challenger_minus_champion"] = float(prof_cl.sum() - prof_ch.sum())
    bus["profit_diff_bootstrap_95ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # like-for-like: both models at the same approval rate on the test set
    matched = []
    prof_all = biz.account_profit(yte, ead_te, ECON)
    for rate in (bus["champion"]["approval_rate"], 0.50, 0.70, 0.90):
        k = int(round(rate * len(yte)))
        row = {"approval_rate": k / len(yte)}
        for name, s in [("champion", champ_pd_te), ("challenger", chal_pd_te)]:
            top = np.argsort(s, kind="mergesort")[:k]  # k lowest-PD accounts
            row[f"{name}_bad_rate"] = float(yte[top].mean())
            row[f"{name}_profit"] = float(prof_all[top].sum())
        matched.append(row)
    pd.DataFrame(matched).to_csv(REPORTS / "matched_approval_comparison.csv", index=False)

    # sensitivity of the profit-maximising cutoff to the economic assumptions
    sens = []
    for lgd in (0.50, 0.75, 0.90):
        for margin in (0.08, 0.12, 0.16):
            e = biz.Economics(lgd=lgd, ccf=ECON.ccf, margin=margin)
            ea_tr = biz.exposure_at_default(tr.BILL_AMT1, tr.LIMIT_BAL, e)
            c_ = biz.best_cutoff(oof_champ_score, ytr, ea_tr, e, grid)
            a_ = score_te >= c_
            sens.append({"lgd": lgd, "margin": margin, "break_even_pd": margin / (margin + lgd),
                         "cutoff": c_, "test_approval_rate": a_.mean(),
                         "test_bad_rate_approved": yte[a_].mean(),
                         "test_profit": float(np.sum(biz.account_profit(yte[a_], ead_te[a_], e))),
                         "test_profit_approve_all": float(np.sum(biz.account_profit(yte, ead_te, e)))})
    pd.DataFrame(sens).to_csv(REPORTS / "economics_sensitivity.csv", index=False)

    cuts = sorted(set(np.percentile(score_te, np.arange(10, 100, 10)).round().tolist() + [cut]))
    ct = biz.cutoff_table(score_te, yte, pd_card_te, ead_te, ECON, cuts)
    ct.to_csv(REPORTS / "cutoff_table_test.csv", index=False)

    # ---------------- explainability ----------------
    pi = permutation_importance(chal, te[feats], yte, scoring="roc_auc", n_repeats=5,
                                random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": feats, "auc_drop_mean": pi.importances_mean,
                        "auc_drop_std": pi.importances_std}).sort_values("auc_drop_mean", ascending=False)
    imp.to_csv(REPORTS / "challenger_permutation_importance.csv", index=False)
    rng_pts = card.groupby("feature")["points"].agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
    rng_pts.rename("points_range").to_csv(REPORTS / "champion_points_range.csv")

    declined = np.where(~approve_te)[0]
    med = np.median(score_te[declined])
    pick = declined[np.argmin(np.abs(score_te[declined] - med))]
    rc = reason_codes(pts_te.iloc[pick], card, DESCRIPTIONS, top=4)
    rc.to_csv(REPORTS / "reason_codes_sample.csv", index=False)
    out["reason_code_sample"] = {"test_row": int(pick), "score": int(score_te[pick]),
                                 "cutoff": cut, "pd": float(champ_pd_te[pick]),
                                 "actual_default": int(yte[pick])}

    # ---------------- fairness ----------------
    sex = te.SEX.map({1: "male", 2: "female"})
    age = pd.cut(te.AGE, [0, 24, 34, 44, 54, 200], labels=["<25", "25-34", "35-44", "45-54", "55+"])
    fair = []
    for model_name, appr, p in [("champion", approve_te, champ_pd_te), ("challenger", approve_chal_te, chal_pd_te)]:
        for attr, g in [("SEX", sex), ("AGE_BAND", age)]:
            t = biz.adverse_impact(appr, g, yte, p)
            t.insert(0, "attribute", attr)
            t.insert(0, "model", model_name)
            fair.append(t)
    fair = pd.concat(fair)
    fair.to_csv(REPORTS / "fairness_test.csv", index=False)
    out["fairness_min_air"] = {f"{m_}_{a}": float(fair[(fair.model == m_) & (fair.attribute == a)].air_vs_highest.min())
                               for m_ in ["champion", "challenger"] for a in ["SEX", "AGE_BAND"]}

    # ---------------- figures ----------------
    plots.roc_ks(FIGURES / "roc_ks.png", yte, champ_pd_te, chal_pd_te, score_te)
    plots.calibration(FIGURES / "calibration.png",
                      M.calibration_table(yte, champ_pd_te), M.calibration_table(yte, chal_pd_te),
                      score_tr, score_te, test["champion"]["psi_train_vs_test_score"])
    plots.cutoff_profit(FIGURES / "cutoff_profit.png", yte, ead_te, ECON,
                        score_te, chal_score_te, approve_te.mean(), approve_chal_te.mean())

    out["runtime_seconds"] = time.time() - t0
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"done in {out['runtime_seconds']:.1f}s -> reports/metrics.json")


if __name__ == "__main__":
    main()
