"""Generate notebooks/01_modeling_dataset_preparation.ipynb"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s))


md(
    """# 01 — ILD 4-class modeling dataset preparation

**Goal:** Build a reproducible modeling table for **IPF vs HP vs CTD vs 1_SAR** from Dataset B.

**Constraints for this stage:**
- Do **not** train a model
- Do **not** modify the raw CSV
- Do **not** use Dataset A
- Do **not** encode/scale features yet (except dtype casting needed for inspection)
- Do **not** drop suspicious rows automatically
- Do **not** apply class weights or resampling

**Raw source:** `database_final_anonymized_for_zenodo.csv` (semicolon-separated, European decimals)
"""
)

md("## 1. Imports")
code(
    """from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)
"""
)

md("## 2. Configuration / file paths")
code(
    '''PROJECT_ROOT = Path("..").resolve()
if not (PROJECT_ROOT / "database_final_anonymized_for_zenodo.csv").exists():
    # Fallback when the notebook kernel cwd is already the project root
    PROJECT_ROOT = Path(".").resolve()

RAW_PATH = PROJECT_ROOT / "database_final_anonymized_for_zenodo.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["IPF", "HP", "CTD", "1_SAR"]
FEATURE_COLS = [
    "fev1fvc_abs",
    "fev1_z",
    "fev1_pp",
    "fvc_z",
    "fvc_pp",
    "fev1fvc_z",
    "tlc_z",
    "tlc_pp",
    "tlco_z",
    "tlco_pp",
    "sex",
    "age at event",
    "Height",
    "Weight",
]
TARGET_COL = "diagnosis group"
HIGH_CORR_THRESHOLD = 0.90

EXCLUSION_RATIONALE = {
    "Case_number": "Identifier — not a clinical predictor; meaningless for new patients and risks identity leakage.",
    "First test": "Calendar date of baseline test — not a physiological predictor of ILD subtype; can leak cohort/time effects.",
    "event_date": "Follow-up/event date tied to outcomes — not available as a diagnosis-time feature.",
    "dead or alive": "Mortality outcome after follow-up — label leakage for diagnosis classification.",
    "survival time": "Time-to-event / follow-up duration derived from outcome timing — must not predict diagnosis.",
    "diagnosis group": "This is the classification target y, not a feature in X.",
}

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW_PATH exists:", RAW_PATH.exists())
'''
)

md("## 3. Load Dataset B")
code(
    '''# Semicolon separator + European decimal commas
raw = pd.read_csv(RAW_PATH, sep=";", decimal=",")
raw.columns = [c.strip() for c in raw.columns]
print("Raw shape:", raw.shape)
raw.head()
'''
)

md("## 4. Verify loading and data types")
code(
    '''print("Columns:")
print(list(raw.columns))
print("\\nDtypes:")
print(raw.dtypes)
print("\\nMissing values per column:")
print(raw.isna().sum())
print("\\nUnique diagnosis groups:")
print(raw[TARGET_COL].value_counts())
'''
)

md(
    """## 5. Filter the four diagnosis classes

Retain only: **IPF**, **HP**, **CTD**, **1_SAR**.
"""
)
code(
    '''filtered = raw[raw[TARGET_COL].isin(TARGET_CLASSES)].copy()
class_counts = filtered[TARGET_COL].value_counts().reindex(TARGET_CLASSES)
print("Records per retained class:")
print(class_counts)
print("\\nTotal retained:", len(filtered))
print("Excluded rows:", len(raw) - len(filtered))
'''
)

md(
    """## 6. Select features

### Predictors kept
PFT indices (absolute / z / % predicted), sex, age, height, weight.

### Fields excluded from X
"""
)
code(
    '''for col, reason in EXCLUSION_RATIONALE.items():
    print(f"- {col}: {reason}")

missing_features = [c for c in FEATURE_COLS + [TARGET_COL] if c not in filtered.columns]
assert not missing_features, f"Missing columns: {missing_features}"
'''
)

md("## 7. Define X and y")
code(
    '''modeling = filtered[FEATURE_COLS + [TARGET_COL]].copy()

# Cast numerics explicitly for inspection (no scaling / encoding yet)
num_cols = [c for c in FEATURE_COLS if c != "sex"]
for c in num_cols:
    modeling[c] = pd.to_numeric(modeling[c], errors="coerce")
modeling["sex"] = modeling["sex"].astype("category")
modeling[TARGET_COL] = modeling[TARGET_COL].astype("category")
modeling = modeling.reset_index(drop=True)

X = modeling[FEATURE_COLS].copy()
y = modeling[TARGET_COL].copy()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\\nX dtypes:\\n", X.dtypes)
print("\\ny dtype:", y.dtype)
X.head()
'''
)

md("## 8. Data-quality analysis")
code(
    '''rows = []
for c in X.columns:
    s = X[c]
    row = {
        "feature": c,
        "dtype": str(s.dtype),
        "n_missing": int(s.isna().sum()),
        "pct_missing": round(100 * s.isna().mean(), 3),
        "n_unique": int(s.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(s):
        row.update({
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
        })
    rows.append(row)

feature_summary = pd.DataFrame(rows)
feature_summary
'''
)
code(
    '''print("Categorical feature: sex")
print(X["sex"].value_counts(dropna=False))
print("\\nNumeric describe():")
X.select_dtypes(include=[np.number]).describe().T
'''
)

md(
    """## 9. Suspicious-value analysis

We **flag** suspicious values for a later decision. We do **not** delete them here.
"""
)
code(
    '''records = []

def flag(mask, feature, rule, severity, recommendation):
    for i in modeling.index[mask]:
        records.append({
            "row_index": int(i),
            "feature": feature,
            "value": modeling.at[i, feature],
            "sex": modeling.at[i, "sex"],
            "age at event": modeling.at[i, "age at event"],
            "diagnosis group": modeling.at[i, TARGET_COL],
            "rule": rule,
            "severity": severity,
            "recommendation": recommendation,
        })

flag(modeling["Weight"] == 8, "Weight", "Weight == 8 kg with adult age", "high",
     "Keep for now; likely data-entry error. Decide later: impute or sensitivity-exclude — do not silent-drop.")
flag((modeling["Weight"] < 30) & (modeling["Weight"] != 8), "Weight", "Weight < 30 kg", "high",
     "Review manually; defer drop/impute.")
flag(modeling["Weight"] > 200, "Weight", "Weight > 200 kg", "medium",
     "May be valid; keep unless confirmed error.")
flag((modeling["Height"] < 1.40) | (modeling["Height"] > 2.10), "Height", "Height outside 1.40-2.10 m", "medium",
     "Review extremes; values appear to be meters.")
flag((modeling["age at event"] < 18) | (modeling["age at event"] > 100), "age at event", "Age outside 18-100", "high",
     "Review; keep pending decision.")
flag((modeling["fev1fvc_abs"] < 0.20) | (modeling["fev1fvc_abs"] > 1.0), "fev1fvc_abs", "FEV1/FVC outside [0.20, 1.0]", "high",
     "Physiologically implausible as a ratio.")
for c in ["fev1_pp", "fvc_pp", "tlc_pp", "tlco_pp"]:
    flag((modeling[c] < 10) | (modeling[c] > 160), c, f"{c} outside [10, 160]", "medium",
         "Keep; possible severe disease or reference-equation edge case.")
for c, hi in [("fev1_z", 6), ("fvc_z", 6), ("fev1fvc_z", 6), ("tlc_z", 6), ("tlco_z", 5)]:
    sev = "high" if c == "tlco_z" else "medium"
    flag(modeling[c].abs() > hi, c, f"|{c}| > {hi}", sev,
         "Keep rows; extremes can reflect real impairment. Revisit winsorization only if needed later.")
flag(~modeling["sex"].isin(["M", "F"]), "sex", "sex not in {M, F}", "high",
     "Map/encode unexpected categories before training.")

suspicious = pd.DataFrame(records)
if len(suspicious):
    suspicious = suspicious.drop_duplicates(subset=["row_index", "feature", "rule"])

print("Flag records:", len(suspicious))
print("Unique rows flagged:", suspicious["row_index"].nunique() if len(suspicious) else 0)
if len(suspicious):
    display_cols = ["feature", "rule", "severity"]
    print(suspicious.groupby(display_cols).size().sort_values(ascending=False))
suspicious.head(20)
'''
)
code(
    '''# Spotlight: Weight = 8 and extreme TLCO z-scores
print("Weight == 8 kg rows:")
print(modeling.loc[modeling["Weight"] == 8, ["Weight", "Height", "age at event", "sex", TARGET_COL]])
print("\\n|tlco_z| > 5:", int((modeling["tlco_z"].abs() > 5).sum()))
print("|tlco_z| > 10:", int((modeling["tlco_z"].abs() > 10).sum()))
print("tlco_z min/max:", modeling["tlco_z"].min(), modeling["tlco_z"].max())
'''
)

md(
    """### Suspicious-value decision framework (no auto-deletion)

| Category | Finding | Recommended handling |
|---|---|---|
| Confirmed valid-looking | sex ∈ {M,F}; age 22–95; height in meters; FEV1/FVC in (0,1]; 0 missing | Keep as-is |
| Suspicious | Weight=8 kg (age 48, CTD) | Keep in table; later impute or sensitivity-exclude |
| Suspicious | Extreme `tlco_z` (|z|>5 common) | Keep — severe DLCO impairment is clinically plausible in ILD |
| Suspicious | Rare |z|>6 on FEV1/FVC/TLC; height <1.40 | Keep; document |
"""
)

md("## 10. Correlation analysis")
code(
    '''num_X = X.select_dtypes(include=[np.number])
corr = num_X.corr(method="pearson")

focus_pairs = [
    ("fev1_z", "fev1_pp"),
    ("fvc_z", "fvc_pp"),
    ("tlc_z", "tlc_pp"),
    ("tlco_z", "tlco_pp"),
    ("fev1fvc_abs", "fev1fvc_z"),
]
print("Focused pairwise correlations:")
for a, b in focus_pairs:
    print(f"  {a} vs {b}: r = {corr.loc[a, b]:.4f}")

high_pairs = []
cols = corr.columns.tolist()
for i, a in enumerate(cols):
    for b in cols[i + 1:]:
        r = corr.loc[a, b]
        if abs(r) >= HIGH_CORR_THRESHOLD:
            high_pairs.append({"feature_a": a, "feature_b": b, "pearson_r": float(r)})
high_pairs_df = pd.DataFrame(high_pairs).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
print(f"\\nPairs with |r| >= {HIGH_CORR_THRESHOLD}:")
high_pairs_df
'''
)
code(
    '''fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Pearson correlation — numerical modeling features")
fig.tight_layout()
fig_path = FIGURES_DIR / "ild_4class_correlation_heatmap.png"
fig.savefig(fig_path, dpi=150)
print("Saved", fig_path)
plt.show()
'''
)

md(
    """> **Note:** z-score and %-predicted pairs are expected to be highly collinear (same measurement, different reference scaling).  
> No correlated features are removed at this stage.
"""
)

md(
    """## 11. Class-distribution analysis

No class weights or resampling are applied yet.
"""
)
code(
    '''counts = y.value_counts().reindex(TARGET_CLASSES)
pct = (100 * counts / counts.sum()).round(2)
dist = pd.DataFrame({"count": counts, "percent": pct})
print(dist)
print("\\nTotal:", int(counts.sum()))
majority = counts.idxmax()
minority = counts.idxmin()
ratio = counts.max() / counts.min()
print(f"Majority: {majority} (n={int(counts.max())})")
print(f"Minority: {minority} (n={int(counts.min())})")
print(f"Majority/minority ratio: {ratio:.2f}:1")
'''
)
code(
    '''ax = counts.plot(kind="bar", color="#2F6F7E", figsize=(7, 4), rot=0)
ax.set_title("Class distribution — 4-class ILD modeling set")
ax.set_ylabel("Count")
ax.set_xlabel("diagnosis group")
plt.tight_layout()
plt.show()
'''
)

md(
    """### Why class imbalance matters later
`1_SAR` is the majority class. A model can achieve high **accuracy** by mostly predicting sarcoidosis while failing on IPF/HP/CTD.  
When we train (next stage), we should use a **stratified split** and report **per-class / macro** metrics. Weighting or resampling is deferred.
"""
)

md("## 12. Summary of the modeling dataset")
code(
    '''# Save derived artifacts only (raw CSV untouched)
out_csv = PROCESSED_DIR / "ild_4class_modeling_data.csv"
modeling.to_csv(out_csv, index=False)
feature_summary.to_csv(REPORTS_DIR / "ild_4class_feature_summary.csv", index=False)
corr.to_csv(REPORTS_DIR / "ild_4class_correlation_matrix.csv")
high_pairs_df.to_csv(REPORTS_DIR / "ild_4class_high_correlation_pairs.csv", index=False)
if len(suspicious):
    suspicious.to_csv(REPORTS_DIR / "ild_4class_suspicious_values.csv", index=False)

meta = {
    "n_rows": int(len(modeling)),
    "n_features": int(X.shape[1]),
    "features": FEATURE_COLS,
    "target": TARGET_COL,
    "class_counts": {k: int(v) for k, v in counts.items()},
    "missing_total": int(X.isna().sum().sum()),
    "majority_minority_ratio": float(ratio),
    "n_suspicious_flags": int(len(suspicious)),
    "high_corr_pairs": high_pairs_df.to_dict(orient="records"),
    "modeling_csv": str(out_csv.relative_to(PROJECT_ROOT)),
}
(REPORTS_DIR / "ild_4class_modeling_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("=== MODELING DATASET SUMMARY ===")
print(f"Rows: {len(modeling)}")
print(f"Features (X): {X.shape[1]}")
print(f"Feature names: {FEATURE_COLS}")
print(f"Target: {TARGET_COL}")
print("Class distribution:")
print(dist)
print(f"Missing values in X: {int(X.isna().sum().sum())}")
print(f"Suspicious flag records: {len(suspicious)}")
print(f"High-|r| pairs: {len(high_pairs_df)}")
print(f"Saved modeling CSV: {out_csv}")
print("\\nNo model was trained.")
'''
)

md(
    """## Stop point

Modeling dataset preparation and quality analysis are complete.  
**Next stage (not done here):** choose preprocessing (encoding/scaling), decide redundancy handling for z vs %predicted pairs, then train a simple baseline classifier.
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "01_modeling_dataset_preparation.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("Wrote", out)
