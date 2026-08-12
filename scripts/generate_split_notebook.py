"""Generate notebooks/02_train_test_split.ipynb"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s))


md(
    """# 02 — Leakage audit & stratified train/test split

**Task:** Finalize features (esp. `age at event`) and create an 80/20 stratified split for 4-class ILD classification.

**Constraints:**
- Do **not** train a model
- Do **not** fit scalers/imputers/encoders
- Do **not** modify the raw CSV
- Do **not** change the target definition

**Source paper for Dataset B:** Boros & Martusewicz-Boros, *PLOS Medicine* (Zenodo DOI 10.5281/zenodo.15295274)
"""
)

md("## 1. Imports")
code(
    """from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)
"""
)

md("## 2. Configuration")
code(
    '''PROJECT_ROOT = Path("..").resolve()
if not (PROJECT_ROOT / "database_final_anonymized_for_zenodo.csv").exists():
    PROJECT_ROOT = Path(".").resolve()

RAW_PATH = PROJECT_ROOT / "database_final_anonymized_for_zenodo.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["IPF", "HP", "CTD", "1_SAR"]
TARGET_COL = "diagnosis group"
RANDOM_STATE = 42
TEST_SIZE = 0.20

CANDIDATE_FEATURES = [
    "fev1fvc_abs", "fev1_z", "fev1_pp",
    "fvc_z", "fvc_pp", "fev1fvc_z",
    "tlc_z", "tlc_pp", "tlco_z", "tlco_pp",
    "sex", "age at event", "Height", "Weight",
]

PAPER_BASELINE_AGE = {
    "1_SAR": 43.26, "CTD": 59.56, "HP": 52.34, "i-NSIP": 57.51,
    "IPF": 68.3, "o-ILD": 53.44, "u-ILD": 61.37, "ALL": 51.13,
}

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW exists:", RAW_PATH.exists())
'''
)

md("## 3. Load Dataset B (raw, read-only)")
code(
    '''raw = pd.read_csv(RAW_PATH, sep=";", decimal=",")
raw.columns = [c.strip() for c in raw.columns]
print("Raw shape:", raw.shape)
print("Columns:", list(raw.columns))
raw.head()
'''
)

md(
    """## 4. Investigate `age at event` (temporal leakage audit)

### Question
Does `age at event` mean:
- **A.** age at baseline PFT / `First test`, or
- **B.** age at a later follow-up / death / censoring (`event_date`)?

### Documentation facts (source paper)
- PFTs are from **first presentation / baseline**.
- *“time to event was time from the first accessible pulmonary function test to censoring or death”* (censoring 15 Mar 2023).
- Table 2 reports baseline **Age (years)** by diagnosis group.
"""
)
code(
    '''ft = pd.to_datetime(raw["First test"], dayfirst=True, errors="coerce")
ed = pd.to_datetime(raw["event_date"], dayfirst=True, errors="coerce")
delta_days = (ed - ft).dt.days
st = pd.to_numeric(raw["survival time"], errors="coerce")
age = pd.to_numeric(raw["age at event"], errors="coerce")
implied_baseline = age - (st / 365.25)

print("survival time == (event_date - First test) days:", int((delta_days == st).sum()), "/", len(raw))
print("Overall mean age at event:", round(age.mean(), 2))
print("Overall mean implied baseline age (age_at_event - survival_years):", round(implied_baseline.mean(), 2))
print("Paper Table 2 overall Age:", PAPER_BASELINE_AGE["ALL"])

rows = []
for g, sub in raw.groupby("diagnosis group"):
    a = pd.to_numeric(sub["age at event"], errors="coerce")
    imp = a - pd.to_numeric(sub["survival time"], errors="coerce") / 365.25
    paper = PAPER_BASELINE_AGE[g]
    rows.append({
        "diagnosis group": g,
        "n": len(sub),
        "age_at_event_mean": round(a.mean(), 2),
        "implied_baseline_mean": round(imp.mean(), 2),
        "paper_Age_mean": paper,
        "abs_diff_implied_vs_paper": round(abs(imp.mean() - paper), 3),
    })
age_compare = pd.DataFrame(rows).sort_values("diagnosis group")
age_compare
'''
)

md(
    """### Conclusion on `age at event`

**Verdict: B — age at later event (death or censoring), not baseline PFT age.**

Evidence:
1. Column naming and adjacency to `event_date` / `dead or alive` / `survival time`.
2. `survival time` ≈ days from `First test` to `event_date`.
3. `age_at_event − survival_years` reproduces the paper’s baseline Age means (overall ~51.13).

**Decision: EXCLUDE `age at event` from the primary diagnosis model** to avoid temporal leakage.

> Note: the released CSV does **not** contain a separate baseline-age column, so we cannot safely substitute another age feature.
"""
)

md("## 5. Finalize feature set")
code(
    '''FINAL_FEATURES = [
    "fev1fvc_abs", "fev1_z", "fev1_pp",
    "fvc_z", "fvc_pp", "fev1fvc_z",
    "tlc_z", "tlc_pp", "tlco_z", "tlco_pp",
    "sex", "Height", "Weight",
]

EXCLUSIONS = {
    "Case_number": "Identifier — not a clinical predictor.",
    "First test": "Baseline calendar date — temporal/cohort information; not physiological.",
    "event_date": "Death/censoring date — future relative to baseline PFT.",
    "dead or alive": "Mortality/censoring outcome — outcome leakage.",
    "survival time": "Follow-up duration from First test to event — outcome timing leakage.",
    "age at event": "Age at death/censoring, not baseline — temporal leakage (see Section 4).",
    "diagnosis group": "Target y — must not appear in X.",
}

print("Retained features (n={}):".format(len(FINAL_FEATURES)))
for f in FINAL_FEATURES:
    print(" ", f)
print("\\nExcluded:")
for k, v in EXCLUSIONS.items():
    print(f"  - {k}: {v}")
print("\\nNote: correlated z/%predicted pairs are NOT removed yet.")
'''
)

md("## 6. Full-column leakage audit")
code(
    '''audit = []
for c in raw.columns:
    if c in FINAL_FEATURES:
        audit.append({"column": c, "role": "retained_feature",
                      "recommendation": "Retain in X (baseline PFT / demographics)."})
    elif c == TARGET_COL:
        audit.append({"column": c, "role": "target",
                      "recommendation": "Use as y only."})
    elif c in EXCLUSIONS:
        audit.append({"column": c, "role": "excluded",
                      "recommendation": "Exclude from X: " + EXCLUSIONS[c]})
    else:
        audit.append({"column": c, "role": "unexpected",
                      "recommendation": "Exclude until reviewed."})
audit_df = pd.DataFrame(audit)
audit_df
'''
)

md("## 7. Build modeling table & define X, y")
code(
    '''filtered = raw[raw[TARGET_COL].isin(TARGET_CLASSES)].copy()
modeling = filtered[FINAL_FEATURES + [TARGET_COL]].copy().reset_index(drop=True)
for c in FINAL_FEATURES:
    if c != "sex":
        modeling[c] = pd.to_numeric(modeling[c], errors="coerce")
modeling["sex"] = modeling["sex"].astype(str)

X = modeling[FINAL_FEATURES].copy()
y = modeling[TARGET_COL].copy()

print("Modeling rows:", len(modeling), "(expected 5092)")
print("X shape:", X.shape)
print("y value counts:\\n", y.value_counts().reindex(TARGET_CLASSES))
assert len(modeling) == 5092
assert TARGET_COL not in X.columns
'''
)

md(
    """## 8. Stratified 80/20 train/test split

**Preprocessing leakage rule (documented, not implemented yet):**

> Any preprocessing whose parameters are learned from data must be fitted using training data only and then applied to the test data.

No scaler / imputer / encoder is fitted in this notebook.
"""
)
code(
    '''X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

def dist(ys):
    counts = ys.value_counts().reindex(TARGET_CLASSES)
    pct = (100 * counts / counts.sum()).round(2)
    return pd.DataFrame({"count": counts.astype(int), "percent": pct})

train_dist = dist(y_train)
test_dist = dist(y_test)

print(f"n_train={len(X_train)}, n_test={len(X_test)}, sum={len(X_train)+len(X_test)}")
print("\\nTRAIN:\\n", train_dist)
print("\\nTEST:\\n", test_dist)
print("\\nmax |train%-test%|:", (train_dist["percent"] - test_dist["percent"]).abs().max())
'''
)

md("## 9. Validation assertions")
code(
    '''checks = []

assert TARGET_COL not in X_train.columns and TARGET_COL not in X_test.columns
checks.append("PASS: target not in X")

for col in EXCLUSIONS:
    assert col not in X_train.columns and col not in X_test.columns
checks.append("PASS: excluded leakage/id fields absent from X")

assert len(X_train) + len(X_test) == 5092
checks.append("PASS: train+test = 5092")

assert set(X_train.index).isdisjoint(set(X_test.index))
checks.append("PASS: train/test indices disjoint")

assert set(TARGET_CLASSES).issubset(set(y_train.unique()))
assert set(TARGET_CLASSES).issubset(set(y_test.unique()))
checks.append("PASS: all 4 classes in train and test")

max_gap = float((train_dist["percent"] - test_dist["percent"]).abs().max())
assert max_gap < 1.0
checks.append(f"PASS: class % gap train vs test = {max_gap:.2f} pp")

assert list(X_train.columns) == FINAL_FEATURES
checks.append("PASS: X columns == FINAL_FEATURES")

for c in checks:
    print(c)
'''
)

md("## 10. Save derived split artifacts")
code(
    '''train_out = X_train.copy()
train_out[TARGET_COL] = y_train.values
test_out = X_test.copy()
test_out[TARGET_COL] = y_test.values

train_path = PROCESSED_DIR / "train.csv"
test_path = PROCESSED_DIR / "test.csv"
train_out.to_csv(train_path, index=False)
test_out.to_csv(test_path, index=False)

meta = {
    "target": TARGET_COL,
    "final_features": FINAL_FEATURES,
    "age_at_event": "EXCLUDED — age at death/censoring, not baseline",
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "train_class_counts": {k: int(v) for k, v in train_dist["count"].items()},
    "test_class_counts": {k: int(v) for k, v in test_dist["count"].items()},
    "preprocessing_rule": (
        "Any preprocessing whose parameters are learned from data must be fitted "
        "using training data only and then applied to the test data."
    ),
    "no_model_trained": True,
}
(REPORTS_DIR / "02_train_test_split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
audit_df.to_csv(REPORTS_DIR / "02_leakage_audit.csv", index=False)

print("Saved", train_path)
print("Saved", test_path)
print("No model was trained.")
'''
)

md(
    """## Stop point

Leakage audit complete; `age at event` excluded; stratified split saved.  
**Next stage:** train-only preprocessing + baseline classifier (not done here).
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "02_train_test_split.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("Wrote", out)
