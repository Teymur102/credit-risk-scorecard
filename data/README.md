# Data: Default of Credit Card Clients (UCI id 350)

- **Source:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **File cached here:** `default_of_credit_card_clients.zip`, downloaded from
  `https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip`.
  `run.py` downloads it only if it is missing and records its SHA-256 in `reports/metrics.json`.
- **Citation:** Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the
  predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480.
- **License:** CC BY 4.0 (as listed on the UCI page).

## What is in it

30,000 credit card holders at a Taiwanese bank, observed April-September 2005. One row per client.

| Field | Meaning |
|---|---|
| `LIMIT_BAL` | Credit limit, NT dollars (individual + family/supplementary) |
| `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Demographics |
| `PAY_0`, `PAY_2`..`PAY_6` | Repayment status Sept back to April (renamed `PAY_1`..`PAY_6` in the code). -1 = paid duly, 1..8 = months past due. -2 and 0 occur but are not documented by the source; they are commonly read as "no balance" and "revolving / minimum paid" |
| `BILL_AMT1`..`BILL_AMT6` | Statement balance Sept back to April |
| `PAY_AMT1`..`PAY_AMT6` | Amount paid Sept back to April |
| `default payment next month` | Target: 1 = defaulted on the October 2005 payment |

## Limits that matter for interpreting this project

1. **Existing customers, not applicants.** Every row already holds a card and six months of
   history, so this is a *behavioural* score. "Approve / decline" in the README is a stand-in for an
   account-management action (line renewal, limit increase, authorisation strategy).
2. **Default definition is one missed payment next month**, not the usual 90+ days past due over
   12 months. Many of these "defaults" would cure. That inflates the bad rate (22%) far above a
   typical prime card book.
3. **No dates per row**, so no out-of-time validation is possible. The train/test split is random,
   which makes PSI between train and test near zero by construction.
4. **Taiwan, 2005**, during a domestic card-debt crisis. Not representative of a US portfolio today.
5. **No loss, recovery or revenue fields.** LGD, EAD and margin in the business layer are assumptions.
6. **Undocumented codes:** `EDUCATION` 0/5/6 and `MARRIAGE` 0 appear without definitions;
   `PAY_x` values -2 and 0 likewise.
7. **No race, ethnicity or national origin**, so the fair-lending check covers sex and age only.
