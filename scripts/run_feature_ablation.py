"""
Feature ablation: full 13 features vs reduced 9 features (drop *_pp).

Selection uses ONLY training-set 5-fold Stratified CV.
Test set is evaluated once after selection.
Does NOT overwrite prior model artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COL = "diagnosis group"
CLASS_ORDER = ["IPF", "HP", "CTD", "1_SAR"]

FULL_FEATURES = [
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
REDUCED_FEATURES = [
    "fev1fvc_abs",
    "fev1_z",
    "fvc_z",
    "fev1fvc_z",
    "tlc_z",
    "tlco_z",
    "sex",
    "Height",
    "Weight",
]
REMOVED_FEATURES = ["fev1_pp", "fvc_pp", "tlc_pp", "tlco_pp"]

LR_PARAMS = dict(
    class_weight="balanced",
    solver="lbfgs",
    C=1.0,
    l1_ratio=0.0,
    max_iter=2000,
    random_state=42,
)

# Previous best (class-weighted LR, full features) on same held-out test set
PREV_BEST_TEST = {
    "accuracy": 0.713,
    "macro_f1": 0.582,
    "macro_recall": 0.6028,
    "IPF_recall": 0.717,
    "HP_recall": 0.408,
    "CTD_recall": 0.477,
    "1_SAR_recall": 0.810,
    "IPF_f1": 0.683,
    "HP_f1": 0.380,
    "CTD_f1": 0.401,
    "1_SAR_f1": 0.863,
}

PROTECTED_MODELS = [
    "baseline_logistic_regression.joblib",
    "class_weighted_logistic_regression.joblib",
    "random_forest_balanced.joblib",
]


def build_pipeline(feature_cols: list[str]) -> Pipeline:
    num = [c for c in feature_cols if c != "sex"]
    cat = ["sex"]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("clf", LogisticRegression(**LR_PARAMS)),
        ]
    )


def metrics_bundle(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)),
    }


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    return pd.DataFrame(
        {"class": CLASS_ORDER, "precision": p, "recall": r, "f1": f, "support": s.astype(int)}
    )


def run_cv(X: pd.DataFrame, y: pd.Series, feature_cols: list[str], cv) -> dict:
    pipe = build_pipeline(feature_cols)
    scoring = {
        "accuracy": "accuracy",
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro",
        "macro_f1": "f1_macro",
    }
    # Per-class metrics via custom loop for fold transparency on primary metric
    fold_macro_f1 = []
    fold_macro_recall = []
    fold_macro_precision = []
    fold_accuracy = []
    per_class_f1_folds = {c: [] for c in CLASS_ORDER}
    per_class_recall_folds = {c: [] for c in CLASS_ORDER}

    Xf = X[feature_cols]
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(Xf, y), start=1):
        X_tr, X_va = Xf.iloc[tr_idx], Xf.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = build_pipeline(feature_cols)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        fold_accuracy.append(accuracy_score(y_va, pred))
        fold_macro_precision.append(
            precision_score(y_va, pred, average="macro", labels=CLASS_ORDER, zero_division=0)
        )
        fold_macro_recall.append(
            recall_score(y_va, pred, average="macro", labels=CLASS_ORDER, zero_division=0)
        )
        fold_macro_f1.append(f1_score(y_va, pred, average="macro", labels=CLASS_ORDER, zero_division=0))
        _, r, f, _ = precision_recall_fscore_support(
            y_va, pred, labels=CLASS_ORDER, zero_division=0
        )
        for i, c in enumerate(CLASS_ORDER):
            per_class_f1_folds[c].append(float(f[i]))
            per_class_recall_folds[c].append(float(r[i]))

    # Also store sklearn cross_validate for sanity (same folds)
    cv_sk = cross_validate(
        pipe, Xf, y, cv=cv, scoring=scoring, n_jobs=None, return_train_score=False
    )

    return {
        "fold_macro_f1": fold_macro_f1,
        "fold_macro_recall": fold_macro_recall,
        "fold_macro_precision": fold_macro_precision,
        "fold_accuracy": fold_accuracy,
        "mean_macro_f1": float(np.mean(fold_macro_f1)),
        "std_macro_f1": float(np.std(fold_macro_f1, ddof=1)),
        "mean_macro_recall": float(np.mean(fold_macro_recall)),
        "mean_macro_precision": float(np.mean(fold_macro_precision)),
        "mean_accuracy": float(np.mean(fold_accuracy)),
        "std_accuracy": float(np.std(fold_accuracy, ddof=1)),
        "per_class_mean_f1": {c: float(np.mean(v)) for c, v in per_class_f1_folds.items()},
        "per_class_mean_recall": {c: float(np.mean(v)) for c, v in per_class_recall_folds.items()},
        "sklearn_cv_macro_f1_mean": float(np.mean(cv_sk["test_macro_f1"])),
    }


def write_report(
    full_cv: dict,
    red_cv: dict,
    selected: str,
    selected_features: list[str],
    test_metrics: dict,
    per_class: pd.DataFrame,
    cm: np.ndarray,
    protected_sizes: dict,
) -> str:
    lines = []
    lines.append("# Feature ablation report")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "Determine whether removing highly correlated percent-predicted (`*_pp`) PFT variables "
        "improves, worsens, or approximately preserves class-weighted Logistic Regression performance, "
        "using training-set CV only for selection."
    )
    lines.append("")
    lines.append("## 2. Why feature redundancy is being investigated")
    lines.append(
        "Prior correlation analysis showed |r| > 0.98 for several z-score vs %predicted pairs. "
        "Redundant representations can inflate coefficient instability and model complexity. "
        "Correlation alone does **not** prove a feature is useless; ablation measures predictive impact."
    )
    lines.append("")
    lines.append("## 3. Full feature set (13)")
    for f in FULL_FEATURES:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## 4. Reduced feature set (9)")
    lines.append("Removed: " + ", ".join(f"`{f}`" for f in REMOVED_FEATURES))
    lines.append("")
    lines.append("Retained:")
    for f in REDUCED_FEATURES:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## 5. CV methodology")
    lines.append("- Data: training set only (n=4073); test set held out until after selection")
    lines.append("- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`")
    lines.append("- Same folds for both feature sets")
    lines.append("- Model: class-weighted Logistic Regression (C=1.0, lbfgs, max_iter=2000)")
    lines.append("- Preprocessing inside Pipeline: StandardScaler + OneHotEncoder(sex)")
    lines.append("- Primary metric: macro F1")
    lines.append("")
    lines.append("## 6. Fold-by-fold macro F1")
    lines.append("")
    lines.append("| Fold | Full 13 | Reduced 9 | Difference (red - full) |")
    lines.append("|---:|---:|---:|---:|")
    for i in range(5):
        a, b = full_cv["fold_macro_f1"][i], red_cv["fold_macro_f1"][i]
        lines.append(f"| {i+1} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    lines.append("")
    lines.append("## 7. Mean / std CV results")
    lines.append("")
    lines.append("| Metric | Full 13 features | Reduced 9 features | Difference (red - full) |")
    lines.append("|---|---:|---:|---:|")
    rows = [
        ("mean CV macro F1", full_cv["mean_macro_f1"], red_cv["mean_macro_f1"]),
        ("std CV macro F1", full_cv["std_macro_f1"], red_cv["std_macro_f1"]),
        ("mean CV macro recall", full_cv["mean_macro_recall"], red_cv["mean_macro_recall"]),
        ("mean CV macro precision", full_cv["mean_macro_precision"], red_cv["mean_macro_precision"]),
        ("mean CV accuracy", full_cv["mean_accuracy"], red_cv["mean_accuracy"]),
    ]
    for name, a, b in rows:
        if "std" in name:
            lines.append(f"| {name} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
        else:
            lines.append(f"| {name} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    lines.append("")
    lines.append("### Per-class mean CV recall")
    lines.append("")
    lines.append("| Class | Full 13 | Reduced 9 | Diff |")
    lines.append("|---|---:|---:|---:|")
    for c in CLASS_ORDER:
        a = full_cv["per_class_mean_recall"][c]
        b = red_cv["per_class_mean_recall"][c]
        lines.append(f"| {c} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    lines.append("")
    lines.append("### Per-class mean CV F1")
    lines.append("")
    lines.append("| Class | Full 13 | Reduced 9 | Diff |")
    lines.append("|---|---:|---:|---:|")
    for c in CLASS_ORDER:
        a = full_cv["per_class_mean_f1"][c]
        b = red_cv["per_class_mean_f1"][c]
        lines.append(f"| {c} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    lines.append("")
    lines.append("## 8. Selected feature set")
    lines.append(f"**CV-selected representation: `{selected}`**")
    lines.append("")
    lines.append("Features used for final refit:")
    for f in selected_features:
        lines.append(f"- `{f}`")
    lines.append("")
    if selected.startswith("reduced"):
        lines.append(
            "The reduced representation was preferred by CV (higher mean macro F1)."
        )
    else:
        lines.append(
            "The ablation did **not** improve the representation; full features were preferred by CV."
        )
    lines.append("")
    lines.append("## 9. Final untouched test performance")
    lines.append("")
    lines.append(
        "Fitted on all 4073 training patients after selection; evaluated once on 1019-patient test set."
    )
    lines.append("")
    for k, v in test_metrics.items():
        lines.append(f"- {k}: **{v:.4f}**")
    lines.append("")
    lines.append("### Per-class test metrics")
    lines.append("")
    lines.append(per_class.to_markdown(index=False))
    lines.append("")
    lines.append("### Test confusion matrix")
    lines.append("")
    lines.append(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_markdown())
    lines.append("")
    lines.append("## 10. Comparison with previous best (class-weighted LR, full features)")
    lines.append("")
    lines.append("| Metric | Previous best | Ablation-selected | Diff |")
    lines.append("|---|---:|---:|---:|")
    pc = {r["class"]: r for r in per_class.to_dict(orient="records")}
    comps = [
        ("accuracy", PREV_BEST_TEST["accuracy"], test_metrics["accuracy"]),
        ("macro_f1", PREV_BEST_TEST["macro_f1"], test_metrics["macro_f1"]),
        ("macro_recall", PREV_BEST_TEST["macro_recall"], test_metrics["macro_recall"]),
        ("IPF_recall", PREV_BEST_TEST["IPF_recall"], float(pc["IPF"]["recall"])),
        ("HP_recall", PREV_BEST_TEST["HP_recall"], float(pc["HP"]["recall"])),
        ("CTD_recall", PREV_BEST_TEST["CTD_recall"], float(pc["CTD"]["recall"])),
        ("1_SAR_recall", PREV_BEST_TEST["1_SAR_recall"], float(pc["1_SAR"]["recall"])),
        ("IPF_f1", PREV_BEST_TEST["IPF_f1"], float(pc["IPF"]["f1"])),
        ("HP_f1", PREV_BEST_TEST["HP_f1"], float(pc["HP"]["f1"])),
        ("CTD_f1", PREV_BEST_TEST["CTD_f1"], float(pc["CTD"]["f1"])),
        ("1_SAR_f1", PREV_BEST_TEST["1_SAR_f1"], float(pc["1_SAR"]["f1"])),
    ]
    for name, a, b in comps:
        lines.append(f"| {name} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    lines.append("")
    lines.append("## 11. Interpretation")
    delta = red_cv["mean_macro_f1"] - full_cv["mean_macro_f1"]
    if abs(delta) < 0.005:
        perf = "approximately preserves"
    elif delta > 0:
        perf = "improves"
    else:
        perf = "worsens"
    lines.append(
        f"1. Removing `*_pp` variables **{perf}** mean CV macro F1 "
        f"(full={full_cv['mean_macro_f1']:.4f}, reduced={red_cv['mean_macro_f1']:.4f}, diff={delta:+.4f})."
    )
    # fold consistency: count wins
    red_wins = sum(
        1 for a, b in zip(full_cv["fold_macro_f1"], red_cv["fold_macro_f1"]) if b > a
    )
    full_wins = sum(
        1 for a, b in zip(full_cv["fold_macro_f1"], red_cv["fold_macro_f1"]) if a > b
    )
    lines.append(
        f"2. Fold stability: reduced wins {red_wins}/5 folds; full wins {full_wins}/5 folds; "
        f"std macro F1 full={full_cv['std_macro_f1']:.4f}, reduced={red_cv['std_macro_f1']:.4f}."
    )
    lines.append(
        "3. From a simplicity/interpretability perspective, the reduced set is preferable if CV performance "
        "is similar or better, because it removes near-collinear %predicted duplicates of z-scores."
    )
    if red_cv["mean_macro_f1"] < full_cv["mean_macro_f1"] - 0.005:
        lines.append(
            "4. Evidence suggests the removed `*_pp` variables may contain some useful predictive information "
            "(or complementary scaling relative to z-scores), because mean CV macro F1 dropped when they were removed."
        )
    elif red_cv["mean_macro_f1"] > full_cv["mean_macro_f1"] + 0.005:
        lines.append(
            "4. No evidence that removed `*_pp` variables were necessary; reduced set improved CV macro F1, "
            "consistent with redundancy/noise from near-collinear duplicates."
        )
    else:
        lines.append(
            "4. Little evidence that removed `*_pp` variables add substantial unique signal beyond z-scores "
            "under this linear model; CV macro F1 is essentially unchanged."
        )
    lines.append("")
    lines.append("## 12. Limitations")
    lines.append("- Only one ablation (drop all four `*_pp`); other subsets not tested.")
    lines.append("- Linear model only; nonlinear models might use redundant features differently.")
    lines.append("- No hyperparameter retuning after feature change.")
    lines.append("- Test set used once after selection; not for choosing features.")
    lines.append("")
    lines.append("## Artifact integrity")
    for name, size in protected_sizes.items():
        lines.append(f"- `{name}`: {size} bytes (unchanged)")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    protected_before = {
        name: (MODELS_DIR / name).stat().st_size for name in PROTECTED_MODELS
    }
    protected_mtime = {
        name: (MODELS_DIR / name).stat().st_mtime for name in PROTECTED_MODELS
    }

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    assert len(train) == 4073 and len(test) == 1019

    X_train = train.copy()
    y_train = train[TARGET_COL].copy()
    X_test = test.copy()
    y_test = test[TARGET_COL].copy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running CV on FULL features...")
    full_cv = run_cv(X_train, y_train, FULL_FEATURES, cv)
    print("Running CV on REDUCED features...")
    red_cv = run_cv(X_train, y_train, REDUCED_FEATURES, cv)

    # Selection by mean CV macro F1 only (training CV). Ties -> prefer reduced for simplicity.
    if red_cv["mean_macro_f1"] > full_cv["mean_macro_f1"]:
        selected = "reduced_9_features"
        selected_features = REDUCED_FEATURES
        model_name = "logistic_reduced_features.joblib"
    elif full_cv["mean_macro_f1"] > red_cv["mean_macro_f1"]:
        selected = "full_13_features"
        selected_features = FULL_FEATURES
        model_name = "logistic_full_features_cv_selected.joblib"
    else:
        selected = "reduced_9_features_tie_break_simplicity"
        selected_features = REDUCED_FEATURES
        model_name = "logistic_reduced_features.joblib"

    print(
        f"CV select: {selected} "
        f"(full macroF1={full_cv['mean_macro_f1']:.4f}, reduced={red_cv['mean_macro_f1']:.4f})"
    )

    # Final refit on all train, one test eval
    final_pipe = build_pipeline(selected_features)
    final_pipe.fit(X_train[selected_features], y_train)
    y_pred = final_pipe.predict(X_test[selected_features])
    test_metrics = metrics_bundle(y_test, y_pred)
    per_class = per_class_table(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    out_model = MODELS_DIR / model_name
    joblib.dump(final_pipe, out_model)

    # Protect prior models
    for name in PROTECTED_MODELS:
        assert (MODELS_DIR / name).stat().st_size == protected_before[name]
        assert (MODELS_DIR / name).stat().st_mtime == protected_mtime[name]

    # Save tables
    fold_df = pd.DataFrame(
        {
            "fold": list(range(1, 6)),
            "full_macro_f1": full_cv["fold_macro_f1"],
            "reduced_macro_f1": red_cv["fold_macro_f1"],
            "diff_red_minus_full": [
                b - a for a, b in zip(full_cv["fold_macro_f1"], red_cv["fold_macro_f1"])
            ],
        }
    )
    fold_df.to_csv(REPORTS_DIR / "06_cv_fold_macro_f1.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "Metric": "mean CV macro F1",
                "Full 13 features": full_cv["mean_macro_f1"],
                "Reduced 9 features": red_cv["mean_macro_f1"],
                "Difference": red_cv["mean_macro_f1"] - full_cv["mean_macro_f1"],
            },
            {
                "Metric": "std CV macro F1",
                "Full 13 features": full_cv["std_macro_f1"],
                "Reduced 9 features": red_cv["std_macro_f1"],
                "Difference": red_cv["std_macro_f1"] - full_cv["std_macro_f1"],
            },
            {
                "Metric": "mean CV macro recall",
                "Full 13 features": full_cv["mean_macro_recall"],
                "Reduced 9 features": red_cv["mean_macro_recall"],
                "Difference": red_cv["mean_macro_recall"] - full_cv["mean_macro_recall"],
            },
            {
                "Metric": "mean CV macro precision",
                "Full 13 features": full_cv["mean_macro_precision"],
                "Reduced 9 features": red_cv["mean_macro_precision"],
                "Difference": red_cv["mean_macro_precision"] - full_cv["mean_macro_precision"],
            },
            {
                "Metric": "mean CV accuracy",
                "Full 13 features": full_cv["mean_accuracy"],
                "Reduced 9 features": red_cv["mean_accuracy"],
                "Difference": red_cv["mean_accuracy"] - full_cv["mean_accuracy"],
            },
        ]
    )
    summary.to_csv(REPORTS_DIR / "06_cv_summary_comparison.csv", index=False)
    per_class.to_csv(REPORTS_DIR / "06_selected_model_test_per_class.csv", index=False)
    pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        REPORTS_DIR / "06_selected_model_test_confusion_matrix.csv"
    )

    report = write_report(
        full_cv,
        red_cv,
        selected,
        selected_features,
        test_metrics,
        per_class,
        cm,
        protected_before,
    )
    report_path = REPORTS_DIR / "06_feature_ablation_report.md"
    report_path.write_text(report, encoding="utf-8")

    meta = {
        "selected": selected,
        "selected_features": selected_features,
        "removed_features": REMOVED_FEATURES,
        "full_cv": {
            k: (v if not isinstance(v, dict) else v)
            for k, v in full_cv.items()
        },
        "reduced_cv": red_cv,
        "test_metrics": test_metrics,
        "model_path": str(out_model.relative_to(PROJECT_ROOT)),
        "prior_models_unchanged": True,
        "classification_report_test": classification_report(
            y_test, y_pred, labels=CLASS_ORDER, digits=4
        ),
    }
    # JSON-serialize folds
    def _ser(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, list):
                out[k] = [float(x) for x in v]
            elif isinstance(v, dict):
                out[k] = {kk: float(vv) for kk, vv in v.items()}
            else:
                out[k] = float(v) if isinstance(v, (float, np.floating)) else v
        return out

    meta["full_cv"] = _ser(full_cv)
    meta["reduced_cv"] = _ser(red_cv)
    (REPORTS_DIR / "06_feature_ablation_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(report.encode("ascii", errors="replace").decode("ascii"))
    print("\nSaved model:", out_model)
    print("Prior models unchanged.")


if __name__ == "__main__":
    main()
