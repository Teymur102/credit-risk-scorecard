"""Download, cache and load the UCI "Default of Credit Card Clients" dataset (id 350)."""
from __future__ import annotations

import hashlib
import io
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
RAW_ZIP = "default_of_credit_card_clients.zip"
TARGET = "default"

# ECOA / Reg B protected or prohibited-basis attributes present in the data.
# They are never used as model inputs; they are kept only for the fairness check.
PROTECTED = ["SEX", "MARRIAGE", "AGE"]
# Not a protected basis, but excluded by policy: weak signal (its IV is still
# reported by run.py), a plausible proxy for age/national origin, and "education
# level" is not an adverse-action reason a lender wants to send a declined customer.
POLICY_EXCLUDED = ["EDUCATION"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def fetch_raw(data_dir: Path) -> Path:
    """Return the cached raw zip, downloading it from UCI on first use."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / RAW_ZIP
    if not path.exists():
        print(f"Downloading {UCI_URL}")
        with urllib.request.urlopen(UCI_URL, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def load_raw(data_dir: Path) -> pd.DataFrame:
    path = fetch_raw(data_dir)
    with zipfile.ZipFile(path) as z:
        xls = [n for n in z.namelist() if n.endswith(".xls")][0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_excel(io.BytesIO(z.read(xls)), header=1)
    df = df.rename(columns={"default payment next month": TARGET, "PAY_0": "PAY_1"})
    return df.drop(columns=["ID"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple, explainable behavioural features built from the 6-month history."""
    out = df.copy()
    # EDUCATION: 1 graduate school, 2 university, 3 high school; 0/4/5/6 are
    # "others"/undocumented codes -> pooled into 4
    out["EDUCATION"] = out["EDUCATION"].where(out["EDUCATION"].isin([1, 2, 3]), 4)
    pay =out[[f"PAY_{i}" for i in range(1, 7)]]
    bill = out[[f"BILL_AMT{i}" for i in range(1, 7)]]
    paid = out[[f"PAY_AMT{i}" for i in range(1, 7)]]
    limit = out["LIMIT_BAL"].astype(float)

    out["MAX_DELINQ_6M"] = pay.clip(lower=0).max(axis=1)
    out["MONTHS_DELINQ_6M"] = (pay > 0).sum(axis=1)
    out["UTIL_1"] = out["BILL_AMT1"] / limit
    out["AVG_UTIL_6M"] = bill.mean(axis=1) / limit
    # share of the August statement repaid in September (PAY_AMT1 / BILL_AMT2),
    # capped at 2; accounts with no positive August balance are set to 1 (nothing owed)
    out["PAY_RATIO_1"] = np.where(out["BILL_AMT2"] > 0,
                                  (out["PAY_AMT1"] / out["BILL_AMT2"].clip(lower=1)).clip(0, 2),
                                  1.0)
    out["AVG_PAY_AMT_6M"] = paid.mean(axis=1)
    return out


def candidate_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in PROTECTED + POLICY_EXCLUDED + [TARGET]]
