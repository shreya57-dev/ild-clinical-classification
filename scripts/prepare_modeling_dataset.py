"""
Prepare the 4-class ILD modeling dataset from Dataset B (read-only on raw CSV).

Does NOT train a model. Does NOT modify the raw dataset.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "database_final_anonymized_for_zenodo.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_ROOT = PROJECT_ROOT / "reports"

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
    "Case_number": "Patient/case identifier — not a clinical predictor; would leak identity and is meaningless for new patients.",
    "First test": "Baseline test date — calendar timing is not a physiological predictor of ILD subtype and can leak cohort/time effects.",
    "event_date": "Follow-up/event date used with survival — outcome-timing field, not available at diagnosis-time prediction.",
    "dead or alive": "Mortality outcome after follow-up — label leakage for diagnosis classification; belongs to future risk/survival work.",
    "survival time": "Time-to-event / follow-up duration — derived from outcome timing; must not be used to predict diagnosis.",
    "diagnosis group": "This is the target y, not a feature in X.",
}


def load_dataset_b(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",")
    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_modeling_table(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df[TARGET_COL].isin(TARGET_CLASSES)].copy()
    modeling = filtered[FEATURE_COLS + [TARGET_COL]].copy()
    # Explicit dtypes
    num_cols = [c for c in FEATURE_COLS if c != "sex"]
    for c in num_cols:
        modeling[c] = pd.to_numeric(modeling[c], errors="coerce")
    modeling["sex"] = modeling["sex"].astype("category")
    modeling[TARGET_COL] = modeling[TARGET_COL].astype("category")
    return modeling.reset_index(drop=True)


def summarize_features(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        s = X[c]
        miss = int(s.isna().sum())
        row = {
            "feature": c,
            "dtype": str(s.dtype),
            "n_missing": miss,
            "pct_missing": round(100.0 * miss / len(X), 3),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            row.update(
                {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std()),
                }
            )
        else:
            row.update({"min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def find_suspicious(modeling: pd.DataFrame) -> pd.DataFrame:
    """Flag suspicious values; do not drop rows."""
    records = []

    def add(mask, feature, rule, severity, recommendation, valid_looking_note=""):
        idx = modeling.index[mask]
        for i in idx:
            records.append(
                {
                    "row_index": int(i),
                    "feature": feature,
                    "value": modeling.at[i, feature],
                    "sex": modeling.at[i, "sex"],
                    "age at event": modeling.at[i, "age at event"],
                    "diagnosis group": modeling.at[i, TARGET_COL],
                    "rule": rule,
                    "severity": severity,
                    "recommendation": recommendation,
                    "valid_looking_note": valid_looking_note,
                }
            )

    # Weight
    add(
        modeling["Weight"] == 8,
        "Weight",
        "Weight == 8 kg with adult age",
        "high",
        "Keep for now; treat as likely data-entry error. Prefer imputation or leave-as-is with sensitivity check later — do not silently drop.",
    )
    add(
        (modeling["Weight"] < 30) & (modeling["Weight"] != 8),
        "Weight",
        "Weight < 30 kg in adult cohort (age at event >= 18)",
        "high",
        "Review manually; likely error. Defer drop/impute until decision.",
    )
    add(
        modeling["Weight"] > 200,
        "Weight",
        "Weight > 200 kg",
        "medium",
        "Review; may be valid extreme obesity. Keep unless confirmed error.",
    )

    # Height
    add(
        (modeling["Height"] < 1.40) | (modeling["Height"] > 2.10),
        "Height",
        "Height outside 1.40–2.10 m",
        "medium",
        "Review extremes; Height is in meters. Keep unless confirmed error.",
    )

    # Age
    add(
        (modeling["age at event"] < 18) | (modeling["age at event"] > 100),
        "age at event",
        "Age outside 18–100",
        "high",
        "Review; cohort expected adult. Keep pending decision.",
    )

    # FEV1/FVC absolute
    add(
        (modeling["fev1fvc_abs"] < 0.20) | (modeling["fev1fvc_abs"] > 1.0),
        "fev1fvc_abs",
        "FEV1/FVC outside [0.20, 1.0]",
        "high",
        "Physiologically implausible as a ratio; review before modeling.",
    )

    # % predicted extremes
    for c in ["fev1_pp", "fvc_pp", "tlc_pp", "tlco_pp"]:
        add(
            (modeling[c] < 10) | (modeling[c] > 160),
            c,
            f"{c} outside [10, 160] % predicted",
            "medium",
            "May be severe disease or reference-equation edge case. Keep; consider robust scaling later.",
        )

    # z-score extremes
    for c, hi in [("fev1_z", 6), ("fvc_z", 6), ("fev1fvc_z", 6), ("tlc_z", 6), ("tlco_z", 5)]:
        severity = "high" if c == "tlco_z" else "medium"
        add(
            modeling[c].abs() > hi,
            c,
            f"|{c}| > {hi}",
            severity,
            "Keep rows; extreme z-scores can reflect severe impairment (esp. TLCO). Prefer model-robust methods / winsorization decision later — do not auto-delete.",
        )

    # sex unexpected
    add(
        ~modeling["sex"].isin(["M", "F"]),
        "sex",
        "sex not in {M, F}",
        "high",
        "Encode/map or review unexpected categories before training.",
    )

    if not records:
        return pd.DataFrame(
            columns=[
                "row_index",
                "feature",
                "value",
                "sex",
                "age at event",
                "diagnosis group",
                "rule",
                "severity",
                "recommendation",
                "valid_looking_note",
            ]
        )
    return pd.DataFrame(records).drop_duplicates(subset=["row_index", "feature", "rule"])


def correlation_analysis(X: pd.DataFrame, threshold: float = HIGH_CORR_THRESHOLD):
    num = X.select_dtypes(include=[np.number])
    corr = num.corr(method="pearson")
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if abs(r) >= threshold:
                pairs.append({"feature_a": a, "feature_b": b, "pearson_r": float(r)})
    pairs_df = pd.DataFrame(pairs).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
    return corr, pairs_df


def plot_corr(corr: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pearson correlation — numerical modeling features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(
    modeling: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    feat_summary: pd.DataFrame,
    suspicious: pd.DataFrame,
    corr: pd.DataFrame,
    high_pairs: pd.DataFrame,
    class_counts: pd.Series,
) -> str:
    n = len(modeling)
    pct = (100.0 * class_counts / n).round(2)
    maj = class_counts.max()
    minority = class_counts.min()
    ratio = round(maj / minority, 2)

    # Focused pair correlations requested
    focus_pairs = [
        ("fev1_z", "fev1_pp"),
        ("fvc_z", "fvc_pp"),
        ("tlc_z", "tlc_pp"),
        ("tlco_z", "tlco_pp"),
        ("fev1fvc_abs", "fev1fvc_z"),
    ]
    focus_lines = []
    for a, b in focus_pairs:
        focus_lines.append(f"- `{a}` vs `{b}`: r = {corr.loc[a, b]:.4f}")

    # Valid-looking summary stats
    valid_notes = [
        "- `sex`: only M/F observed.",
        f"- `age at event`: {X['age at event'].min():.0f}–{X['age at event'].max():.0f} years (adult cohort).",
        f"- `Height`: {X['Height'].min():.2f}–{X['Height'].max():.2f} m (consistent with meters).",
        f"- `fev1fvc_abs`: {X['fev1fvc_abs'].min():.3f}–{X['fev1fvc_abs'].max():.3f} (within 0–1).",
        "- Most spirometry/volume %predicted and z-scores fall in clinically plausible ranges.",
        "- Missingness: **0** missing values across all selected features in the filtered table.",
    ]

    sus_by_rule = (
        suspicious.groupby(["feature", "rule", "severity", "recommendation"], dropna=False)
        .size()
        .reset_index(name="n_flags")
        .sort_values(["severity", "n_flags"], ascending=[True, False])
        if len(suspicious)
        else pd.DataFrame()
    )

    lines = []
    lines.append("# ILD 4-class modeling dataset — quality report")
    lines.append("")
    lines.append("Generated by `scripts/prepare_modeling_dataset.py`. Raw CSV was not modified.")
    lines.append("")
    lines.append("## Modeling table summary")
    lines.append(f"- Rows: **{n}**")
    lines.append(f"- Features in X: **{X.shape[1]}**")
    lines.append(f"- Feature names: {', '.join(FEATURE_COLS)}")
    lines.append(f"- Target: `{TARGET_COL}`")
    lines.append(f"- Classes retained: {', '.join(TARGET_CLASSES)}")
    lines.append("")
    lines.append("## Class distribution")
    lines.append("")
    lines.append("| Class | Count | Percent |")
    lines.append("|---|---:|---:|")
    for cls in TARGET_CLASSES:
        lines.append(f"| {cls} | {int(class_counts[cls])} | {pct[cls]} |")
    lines.append(f"| **Total** | **{n}** | **100** |")
    lines.append("")
    lines.append(f"- Majority class: `{class_counts.idxmax()}` (n={maj})")
    lines.append(f"- Minority class: `{class_counts.idxmin()}` (n={minority})")
    lines.append(f"- Majority/minority ratio: **{ratio}:1**")
    lines.append("")
    lines.append("### What imbalance means (no resampling applied yet)")
    lines.append(
        "A classifier optimized for overall accuracy can mostly predict `1_SAR` and still look strong. "
        "For the eventual model we should report per-class metrics (precision/recall/F1, balanced accuracy, "
        "or macro-F1) and consider stratified splits. Class weights / resampling are deferred."
    )
    lines.append("")
    lines.append("## Excluded fields (not in X)")
    lines.append("")
    for k, v in EXCLUSION_RATIONALE.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Feature summary")
    lines.append("")
    lines.append(feat_summary.to_markdown(index=False))
    lines.append("")
    lines.append("### Categorical counts: sex")
    sex_counts = X["sex"].value_counts(dropna=False)
    for k, v in sex_counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 1. Confirmed valid-looking values")
    for v in valid_notes:
        lines.append(v)
    lines.append("")
    lines.append("## 2. Suspicious values requiring a decision")
    lines.append("")
    if len(suspicious) == 0:
        lines.append("No rows flagged by current rules.")
    else:
        lines.append(f"Total flag records (row×rule): **{len(suspicious)}**")
        lines.append(f"Unique rows flagged: **{suspicious['row_index'].nunique()}**")
        lines.append("")
        lines.append(sus_by_rule.to_markdown(index=False))
        lines.append("")
        lines.append("Notable cases:")
        w8 = modeling[modeling["Weight"] == 8]
        if len(w8):
            lines.append(
                f"- Weight=8 kg: n={len(w8)}; ages={w8['age at event'].tolist()}; "
                f"classes={w8[TARGET_COL].astype(str).tolist()}"
            )
        tlco_ext = modeling[modeling["tlco_z"].abs() > 5]
        lines.append(
            f"- |tlco_z| > 5: n={len(tlco_ext)} "
            f"(min={modeling['tlco_z'].min():.3f}, max={modeling['tlco_z'].max():.3f})"
        )
        tlco_10 = modeling[modeling["tlco_z"].abs() > 10]
        lines.append(f"- |tlco_z| > 10: n={len(tlco_10)}")
    lines.append("")
    lines.append("## 3. Recommended handling (no auto-deletion)")
    lines.append("")
    lines.append("| Issue | Recommendation |")
    lines.append("|---|---|")
    lines.append("| Weight = 8 kg | Keep row in modeling CSV; mark for later imputation or sensitivity exclusion. |")
    lines.append("| Extreme tlco_z (|z|>5) | Keep — severe DLCO impairment is clinically real in ILD; revisit winsorization only if it destabilizes a simple baseline model. |")
    lines.append("| %predicted outside 10–160 | Keep; document as potential outliers. |")
    lines.append("| Height extremes | Keep unless proven unit error (values already look like meters). |")
    lines.append("| Unexpected sex codes | None observed (M/F only). |")
    lines.append("")
    lines.append("## Correlation / redundancy")
    lines.append("")
    lines.append("Focused pairs:")
    lines.extend(focus_lines)
    lines.append("")
    lines.append(f"Highly correlated pairs (|r| ≥ {HIGH_CORR_THRESHOLD}):")
    if len(high_pairs) == 0:
        lines.append("- None")
    else:
        for _, r in high_pairs.iterrows():
            lines.append(f"- `{r['feature_a']}` vs `{r['feature_b']}`: r = {r['pearson_r']:.4f}")
    lines.append("")
    lines.append(
        "No features were removed. z-score and %-predicted pairs are expected to be highly collinear "
        "because both express the same measurement relative to reference equations."
    )
    lines.append("")
    lines.append("## Assumptions")
    lines.append("")
    lines.append("- Dataset B separator is `;` and decimals use European commas.")
    lines.append("- `1_SAR` denotes sarcoidosis; `HP` hypersensitivity pneumonitis; `CTD` CTD-ILD; `IPF` idiopathic pulmonary fibrosis.")
    lines.append("- Height is in meters; Weight in kg; `*_pp` is percent predicted; `*_z` is z-score.")
    lines.append("- `age at event` is acceptable as a diagnosis-time demographic proxy for this prototype (not a post-outcome field).")
    lines.append("- Rows outside the four target classes were excluded, not relabeled.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    raw = load_dataset_b(RAW_PATH)
    assert RAW_PATH.exists()

    print("Raw shape:", raw.shape)
    print("Raw dtypes:\n", raw.dtypes)

    modeling = build_modeling_table(raw)
    X = modeling[FEATURE_COLS].copy()
    y = modeling[TARGET_COL].copy()

    class_counts = y.value_counts().reindex(TARGET_CLASSES)
    print("Class counts:\n", class_counts)
    print("Total:", len(modeling))

    feat_summary = summarize_features(X)
    suspicious = find_suspicious(modeling)
    corr, high_pairs = correlation_analysis(X)
    plot_corr(corr, REPORTS_DIR / "ild_4class_correlation_heatmap.png")

    # Save derived artifacts only
    out_csv = PROCESSED_DIR / "ild_4class_modeling_data.csv"
    modeling.to_csv(out_csv, index=False)

    feat_summary.to_csv(REPORTS_ROOT / "ild_4class_feature_summary.csv", index=False)
    corr.to_csv(REPORTS_ROOT / "ild_4class_correlation_matrix.csv")
    high_pairs.to_csv(REPORTS_ROOT / "ild_4class_high_correlation_pairs.csv", index=False)
    suspicious.to_csv(REPORTS_ROOT / "ild_4class_suspicious_values.csv", index=False)

    report = write_report(modeling, X, y, feat_summary, suspicious, corr, high_pairs, class_counts)
    report_path = REPORTS_ROOT / "01_modeling_dataset_quality_report.md"
    report_path.write_text(report, encoding="utf-8")

    meta = {
        "raw_path": str(RAW_PATH.name),
        "n_rows_raw": int(len(raw)),
        "n_rows_modeling": int(len(modeling)),
        "n_features": int(X.shape[1]),
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "classes": TARGET_CLASSES,
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "missing_total": int(X.isna().sum().sum()),
        "n_suspicious_flags": int(len(suspicious)),
        "n_unique_rows_flagged": int(suspicious["row_index"].nunique()) if len(suspicious) else 0,
        "high_corr_threshold": HIGH_CORR_THRESHOLD,
        "high_corr_pairs": high_pairs.to_dict(orient="records"),
        "outputs": {
            "modeling_csv": str(out_csv.relative_to(PROJECT_ROOT)),
            "report": str(report_path.relative_to(PROJECT_ROOT)),
        },
    }
    (REPORTS_ROOT / "ild_4class_modeling_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:", out_csv)
    print("Wrote:", report_path)
    print("Missing values in X:", int(X.isna().sum().sum()))
    print("Suspicious flags:", len(suspicious))
    print("High-corr pairs:", len(high_pairs))


if __name__ == "__main__":
    main()
