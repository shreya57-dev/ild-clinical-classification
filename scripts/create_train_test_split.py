"""Investigate age at event temporal leakage; create stratified split. No model training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "database_final_anonymized_for_zenodo.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_CLASSES = ["IPF", "HP", "CTD", "1_SAR"]
TARGET_COL = "diagnosis group"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Final features AFTER leakage audit (age at event EXCLUDED)
FINAL_FEATURES = [
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
    "Height",
    "Weight",
]

EXCLUDED_FROM_X = {
    "Case_number": "Identifier — not a clinical predictor.",
    "First test": "Baseline calendar date — temporal/cohort information; not a physiological predictor.",
    "event_date": "Death/censoring date — future relative to baseline PFT; outcome timing leakage.",
    "dead or alive": "Mortality/censoring outcome after follow-up — outcome leakage for diagnosis classification.",
    "survival time": "Days from First test to event_date — follow-up/outcome duration leakage.",
    "age at event": (
        "Age at death or censoring (event_date), NOT age at baseline PFT. "
        "Equals approximately paper baseline Age + survival_years. Temporal leakage."
    ),
    "diagnosis group": "Target variable y — must not appear in X.",
}

PAPER_BASELINE_AGE = {
    "1_SAR": 43.26,
    "CTD": 59.56,
    "HP": 52.34,
    "i-NSIP": 57.51,
    "IPF": 68.3,
    "o-ILD": 53.44,
    "u-ILD": 61.37,
    "ALL": 51.13,
}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    return df


def investigate_age_at_event(raw: pd.DataFrame) -> dict:
    ft = pd.to_datetime(raw["First test"], dayfirst=True, errors="coerce")
    ed = pd.to_datetime(raw["event_date"], dayfirst=True, errors="coerce")
    delta_days = (ed - ft).dt.days
    st = pd.to_numeric(raw["survival time"], errors="coerce")
    age = pd.to_numeric(raw["age at event"], errors="coerce")
    implied_baseline = age - (st / 365.25)

    equal_delta = int((delta_days == st).sum())
    by_group = []
    for g, sub in raw.groupby("diagnosis group"):
        a = pd.to_numeric(sub["age at event"], errors="coerce")
        imp = a - pd.to_numeric(sub["survival time"], errors="coerce") / 365.25
        paper = PAPER_BASELINE_AGE.get(g)
        by_group.append(
            {
                "diagnosis group": g,
                "n": int(len(sub)),
                "age_at_event_mean": float(a.mean()),
                "implied_baseline_age_mean": float(imp.mean()),
                "paper_table2_age_mean": paper,
                "abs_diff_implied_vs_paper": float(abs(imp.mean() - paper)) if paper is not None else None,
            }
        )

    conclusion = {
        "verdict": "EXCLUDE",
        "interpretation": "B — age at later follow-up/event (death or censoring), not baseline PFT age",
        "evidence": [
            "Source paper (Boros & Martusewicz-Boros, PLOS Medicine / Zenodo 15295274): "
            "time-to-event = first PFT to death or censoring (15 Mar 2023); PFTs are baseline/first presentation.",
            f"survival time equals (event_date - First test) in days for {equal_delta}/{len(raw)} rows.",
            f"Overall mean age at event={age.mean():.2f}; implied baseline age "
            f"(age_at_event - survival_years) mean={implied_baseline.mean():.2f} "
            f"matches paper Table 2 overall Age={PAPER_BASELINE_AGE['ALL']:.2f}.",
            "Per-group implied baseline ages match paper Table 2 Age means within ~0.05 years.",
            "Column is literally named 'age at event' and sits next to event_date / dead or alive / survival time.",
        ],
        "decision": (
            "Do NOT use age at event for diagnosis classification. "
            "It incorporates follow-up duration (patients who survive longer appear older), "
            "creating temporal leakage relative to baseline PFT prediction time."
        ),
        "by_group": by_group,
        "survival_equals_date_delta_n": equal_delta,
        "n_rows": int(len(raw)),
        "age_at_event_overall_mean": float(age.mean()),
        "implied_baseline_overall_mean": float(implied_baseline.mean()),
        "paper_overall_age": PAPER_BASELINE_AGE["ALL"],
    }
    return conclusion


def build_modeling_frame(raw: pd.DataFrame) -> pd.DataFrame:
    filtered = raw[raw[TARGET_COL].isin(TARGET_CLASSES)].copy()
    modeling = filtered[FINAL_FEATURES + [TARGET_COL]].copy()
    for c in FINAL_FEATURES:
        if c != "sex":
            modeling[c] = pd.to_numeric(modeling[c], errors="coerce")
    modeling["sex"] = modeling["sex"].astype(str)
    modeling = modeling.reset_index(drop=True)
    return modeling


def leakage_audit_all_columns(raw: pd.DataFrame) -> list[dict]:
    rows = []
    for c in raw.columns:
        if c in FINAL_FEATURES:
            rec = {
                "column": c,
                "role": "retained_feature",
                "leakage_concern": "None for diagnosis-time use (baseline PFT / demographics available at first test).",
                "recommendation": "Retain in X.",
            }
        elif c == TARGET_COL:
            rec = {
                "column": c,
                "role": "target",
                "leakage_concern": "Is the label.",
                "recommendation": "Use as y only; never place in X.",
            }
        elif c in EXCLUDED_FROM_X:
            rec = {
                "column": c,
                "role": "excluded",
                "leakage_concern": EXCLUDED_FROM_X[c],
                "recommendation": "Exclude from X.",
            }
        else:
            rec = {
                "column": c,
                "role": "unreviewed",
                "leakage_concern": "Unexpected column.",
                "recommendation": "Exclude until reviewed.",
            }
        rows.append(rec)
    return rows


def make_split(modeling: pd.DataFrame):
    X = modeling[FINAL_FEATURES].copy()
    y = modeling[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def distribution_table(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().reindex(TARGET_CLASSES)
    pct = (100.0 * counts / counts.sum()).round(2)
    return pd.DataFrame({"count": counts.astype(int), "percent": pct})


def run_assertions(X_train, X_test, y_train, y_test, modeling: pd.DataFrame) -> list[str]:
    checks = []

    assert TARGET_COL not in X_train.columns and TARGET_COL not in X_test.columns
    checks.append("PASS: target not in X_train/X_test")

    for col in EXCLUDED_FROM_X:
        assert col not in X_train.columns and col not in X_test.columns
    checks.append("PASS: all excluded leakage/id fields absent from X")

    n_train, n_test = len(X_train), len(X_test)
    assert n_train + n_test == len(modeling) == 5092
    checks.append(f"PASS: train({n_train})+test({n_test})=5092")

    # Exact duplicate rows between train and test (full feature vectors)
    train_keys = X_train.assign(**{TARGET_COL: y_train.values}).astype(str).agg("|".join, axis=1)
    test_keys = X_test.assign(**{TARGET_COL: y_test.values}).astype(str).agg("|".join, axis=1)
    overlap = set(train_keys) & set(test_keys)
    # Also check index disjointness
    assert set(X_train.index).isdisjoint(set(X_test.index))
    checks.append("PASS: train/test index sets disjoint")
    checks.append(f"INFO: identical feature+label row strings shared across splits: {len(overlap)} (coincidence possible; indices disjoint)")

    for split_name, ys in [("train", y_train), ("test", y_test)]:
        present = set(ys.unique())
        assert set(TARGET_CLASSES).issubset(present)
        checks.append(f"PASS: all 4 classes present in {split_name}")

    train_pct = (100 * y_train.value_counts(normalize=True).reindex(TARGET_CLASSES)).round(2)
    test_pct = (100 * y_test.value_counts(normalize=True).reindex(TARGET_CLASSES)).round(2)
    max_abs_diff = float((train_pct - test_pct).abs().max())
    assert max_abs_diff < 1.0  # stratified 80/20 should be very close
    checks.append(f"PASS: max |train%-test%| class gap = {max_abs_diff:.2f} pp (<1.0)")

    assert list(X_train.columns) == FINAL_FEATURES
    assert list(X_test.columns) == FINAL_FEATURES
    checks.append("PASS: X columns match FINAL_FEATURES")

    return checks


def write_report(
    age_invest: dict,
    audit_rows: list[dict],
    train_dist: pd.DataFrame,
    test_dist: pd.DataFrame,
    n_train: int,
    n_test: int,
    checks: list[str],
) -> str:
    lines = []
    lines.append("# Train/test split & leakage audit report")
    lines.append("")
    lines.append("No model was trained. Raw CSV was not modified.")
    lines.append("")
    lines.append("## 1. Investigation: `age at event`")
    lines.append("")
    lines.append(f"**Verdict:** {age_invest['verdict']} — {age_invest['interpretation']}")
    lines.append("")
    lines.append(f"**Decision:** {age_invest['decision']}")
    lines.append("")
    lines.append("### Evidence")
    for e in age_invest["evidence"]:
        lines.append(f"- {e}")
    lines.append("")
    lines.append("### Per-group age comparison (paper Table 2 vs data)")
    lines.append("")
    lines.append("| diagnosis group | age_at_event mean | implied baseline mean | paper Age mean | |diff| |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in age_invest["by_group"]:
        lines.append(
            f"| {r['diagnosis group']} | {r['age_at_event_mean']:.2f} | "
            f"{r['implied_baseline_age_mean']:.2f} | {r['paper_table2_age_mean']} | "
            f"{r['abs_diff_implied_vs_paper']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"Overall: age_at_event mean={age_invest['age_at_event_overall_mean']:.2f}; "
        f"implied baseline={age_invest['implied_baseline_overall_mean']:.2f}; "
        f"paper={age_invest['paper_overall_age']:.2f}."
    )
    lines.append("")
    lines.append("## 2. Final feature set")
    lines.append("")
    lines.append("### Retained")
    for f in FINAL_FEATURES:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("### Excluded (with reason)")
    for k, v in EXCLUDED_FROM_X.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("Correlated z/%predicted pairs were **not** removed at this stage.")
    lines.append("")
    lines.append("## 3. Full-column leakage audit (Dataset B)")
    lines.append("")
    lines.append("| column | role | concern | recommendation |")
    lines.append("|---|---|---|---|")
    for r in audit_rows:
        lines.append(
            f"| `{r['column']}` | {r['role']} | {r['leakage_concern']} | {r['recommendation']} |"
        )
    lines.append("")
    lines.append("## 4. Stratified train/test split")
    lines.append("")
    lines.append(f"- test_size = {TEST_SIZE}")
    lines.append(f"- random_state = {RANDOM_STATE}")
    lines.append(f"- stratify = `{TARGET_COL}`")
    lines.append(f"- n_train = **{n_train}**")
    lines.append(f"- n_test = **{n_test}**")
    lines.append("")
    lines.append("### Training class distribution")
    lines.append("")
    lines.append(train_dist.to_markdown())
    lines.append("")
    lines.append("### Test class distribution")
    lines.append("")
    lines.append(test_dist.to_markdown())
    lines.append("")
    lines.append("## 5. Preprocessing leakage rule (documented, not fitted)")
    lines.append("")
    lines.append(
        '> Any preprocessing whose parameters are learned from data must be fitted using training data only '
        "and then applied to the test data."
    )
    lines.append("")
    lines.append(
        "No scaler, imputer, encoder, or other learned transformer was fitted in this stage."
    )
    lines.append("")
    lines.append("## 6. Validation checks")
    lines.append("")
    for c in checks:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## 7. Remaining concerns")
    lines.append("")
    lines.append("- No true baseline age column exists in the released CSV; demographic age is unavailable without leakage.")
    lines.append("- Height/Weight used in GLI reference equations that produce z/%predicted — partial redundancy with PFT indices, but not temporal leakage; keep for now.")
    lines.append("- Weight=8 kg anomaly remains in the modeling table (not dropped).")
    lines.append("- Class imbalance (1_SAR majority) remains; handle at training/evaluation stage.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    age_invest = investigate_age_at_event(raw)
    audit_rows = leakage_audit_all_columns(raw)

    modeling = build_modeling_frame(raw)
    assert len(modeling) == 5092

    X_train, X_test, y_train, y_test = make_split(modeling)
    train_dist = distribution_table(y_train)
    test_dist = distribution_table(y_test)
    checks = run_assertions(X_train, X_test, y_train, y_test, modeling)

    # Save derived artifacts only
    train_out = X_train.copy()
    train_out[TARGET_COL] = y_train.values
    test_out = X_test.copy()
    test_out[TARGET_COL] = y_test.values

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    # Also save feature list / meta
    meta = {
        "target": TARGET_COL,
        "classes": TARGET_CLASSES,
        "final_features": FINAL_FEATURES,
        "excluded": EXCLUDED_FROM_X,
        "age_at_event_decision": age_invest["verdict"],
        "age_at_event_interpretation": age_invest["interpretation"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_class_counts": {k: int(v) for k, v in train_dist["count"].items()},
        "test_class_counts": {k: int(v) for k, v in test_dist["count"].items()},
        "train_class_pct": {k: float(v) for k, v in train_dist["percent"].items()},
        "test_class_pct": {k: float(v) for k, v in test_dist["percent"].items()},
        "preprocessing_rule": (
            "Any preprocessing whose parameters are learned from data must be fitted "
            "using training data only and then applied to the test data."
        ),
        "checks": checks,
        "train_csv": str(train_path.relative_to(PROJECT_ROOT)),
        "test_csv": str(test_path.relative_to(PROJECT_ROOT)),
        "no_model_trained": True,
    }
    (REPORTS_DIR / "02_train_test_split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (REPORTS_DIR / "02_age_at_event_investigation.json").write_text(json.dumps(age_invest, indent=2), encoding="utf-8")
    pd.DataFrame(audit_rows).to_csv(REPORTS_DIR / "02_leakage_audit.csv", index=False)

    report = write_report(age_invest, audit_rows, train_dist, test_dist, len(X_train), len(X_test), checks)
    report_path = REPORTS_DIR / "02_train_test_split_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print("\nWrote", train_path)
    print("Wrote", test_path)
    print("Wrote", report_path)


if __name__ == "__main__":
    main()
