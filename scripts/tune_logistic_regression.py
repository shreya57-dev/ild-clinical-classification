"""
Tune Logistic Regression C on training-set Stratified 5-fold CV only.

Final test evaluation happens once after C is selected.
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COL = "diagnosis group"
CLASS_ORDER = ["IPF", "HP", "CTD", "1_SAR"]
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
    "Height",
    "Weight",
]
NUM_FEATURES = [c for c in FEATURE_COLS if c != "sex"]
CAT_FEATURES = ["sex"]

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
RANDOM_STATE = 42

# Previous best: class-weighted LR with C=1.0 on same held-out test set
PREV_BEST = {
    "accuracy": 0.7134,
    "macro_f1": 0.5819,
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

PROTECTED = [
    "baseline_logistic_regression.joblib",
    "class_weighted_logistic_regression.joblib",
    "logistic_full_features_cv_selected.joblib",
    "random_forest_balanced.joblib",
]


def build_base_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_FEATURES,
            ),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        C=1.0,  # overridden by GridSearchCV
        l1_ratio=0.0,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


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
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=CLASS_ORDER, zero_division=0)
        ),
    }


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    return pd.DataFrame(
        {"class": CLASS_ORDER, "precision": p, "recall": r, "f1": f, "support": s.astype(int)}
    )


def write_report(
    cv_table: pd.DataFrame,
    best_C: float,
    train_metrics: dict,
    test_metrics: dict,
    per_class: pd.DataFrame,
    cm: np.ndarray,
    protected_sizes: dict,
) -> str:
    ranked = cv_table.sort_values("mean_macro_f1", ascending=False).reset_index(drop=True)
    top = ranked.iloc[0]
    second = ranked.iloc[1] if len(ranked) > 1 else None
    clear_gap = (
        second is not None and (top["mean_macro_f1"] - second["mean_macro_f1"]) >= 0.005
    )

    # Under/overfit pattern from CV curve
    small_c = cv_table.loc[cv_table["C"] == 0.001].iloc[0]
    large_c = cv_table.loc[cv_table["C"] == 100].iloc[0]
    mid_c = cv_table.loc[cv_table["C"] == 1].iloc[0]

    pc = {r["class"]: r for r in per_class.to_dict(orient="records")}
    gaps = {
        "accuracy": train_metrics["accuracy"] - test_metrics["accuracy"],
        "macro_f1": train_metrics["macro_f1"] - test_metrics["macro_f1"],
    }

    lines = []
    lines.append("# Logistic Regression hyperparameter tuning (C)")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "Select the Logistic Regression regularization strength `C` using training-set "
        "Stratified 5-fold CV only, then evaluate the chosen configuration once on the held-out test set."
    )
    lines.append("")
    lines.append("## 2. Why C is being tuned")
    lines.append(
        "`C` controls inverse regularization strength. With fixed `class_weight='balanced'` and the full "
        "13-feature representation, varying `C` can reduce underfitting (too strong regularization) or "
        "overfitting (too weak regularization) without changing the model family."
    )
    lines.append("")
    lines.append("## 3. Regularization and C")
    lines.append("- Smaller `C` => stronger L2 regularization => simpler decision boundaries.")
    lines.append("- Larger `C` => weaker regularization => model fits training data more closely.")
    lines.append("- sklearn LogisticRegression with `lbfgs` uses L2-equivalent regularization (`l1_ratio=0`).")
    lines.append("")
    lines.append("## 4. CV methodology")
    lines.append("- Training data only: n=4073")
    lines.append("- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`")
    lines.append("- Primary metric: `f1_macro`")
    lines.append("- Same folds for every `C` via `GridSearchCV`")
    lines.append("- Test set (n=1019) excluded from selection")
    lines.append("")
    lines.append("## 5. C values tested")
    lines.append(", ".join(str(c) for c in C_GRID))
    lines.append("")
    lines.append("## 6. CV results")
    lines.append("")
    lines.append(cv_table.to_markdown(index=False))
    lines.append("")
    lines.append("### Ranking by mean CV macro F1")
    lines.append("")
    lines.append(ranked[["C", "mean_macro_f1", "std_macro_f1"]].to_markdown(index=False))
    lines.append("")
    if clear_gap:
        lines.append(
            f"A relatively clear optimum exists at C={best_C} "
            f"(lead over next-best ~{top['mean_macro_f1'] - second['mean_macro_f1']:.4f} macro F1)."
        )
    else:
        lines.append(
            f"Several C values are close; selected C={best_C} by highest mean CV macro F1. "
            "Tiny differences should not be overinterpreted."
        )
    lines.append(
        f"Very small C (0.001) mean macro F1={small_c['mean_macro_f1']:.4f} vs C=1 ({mid_c['mean_macro_f1']:.4f}) "
        f"and C=100 ({large_c['mean_macro_f1']:.4f})."
    )
    lines.append("")
    lines.append("## 7. Selected C")
    lines.append(f"**Selected C = {best_C}** (highest mean CV macro F1 = {top['mean_macro_f1']:.4f}).")
    lines.append("")
    lines.append("## 8. Final test performance")
    lines.append("Refit on all 4073 training patients; evaluated once on untouched test set.")
    lines.append("")
    for k, v in test_metrics.items():
        lines.append(f"- {k}: **{v:.4f}**")
    lines.append("")
    lines.append("## 9. Comparison with previous best (C=1.0)")
    lines.append("")
    lines.append("| Metric | Previous best (C=1) | Tuned | Diff |")
    lines.append("|---|---:|---:|---:|")
    comps = [
        ("accuracy", PREV_BEST["accuracy"], test_metrics["accuracy"]),
        ("macro_f1", PREV_BEST["macro_f1"], test_metrics["macro_f1"]),
        ("macro_recall", PREV_BEST["macro_recall"], test_metrics["macro_recall"]),
        ("IPF_recall", PREV_BEST["IPF_recall"], float(pc["IPF"]["recall"])),
        ("HP_recall", PREV_BEST["HP_recall"], float(pc["HP"]["recall"])),
        ("CTD_recall", PREV_BEST["CTD_recall"], float(pc["CTD"]["recall"])),
        ("1_SAR_recall", PREV_BEST["1_SAR_recall"], float(pc["1_SAR"]["recall"])),
        ("IPF_f1", PREV_BEST["IPF_f1"], float(pc["IPF"]["f1"])),
        ("HP_f1", PREV_BEST["HP_f1"], float(pc["HP"]["f1"])),
        ("CTD_f1", PREV_BEST["CTD_f1"], float(pc["CTD"]["f1"])),
        ("1_SAR_f1", PREV_BEST["1_SAR_f1"], float(pc["1_SAR"]["f1"])),
    ]
    for name, a, b in comps:
        lines.append(f"| {name} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    lines.append("")
    lines.append("## 10. Per-class results (test)")
    lines.append("")
    lines.append(per_class.to_markdown(index=False))
    lines.append("")
    lines.append("## 11. Confusion matrix (test)")
    lines.append("")
    lines.append(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_markdown())
    lines.append("")
    lines.append("## 12. Overfitting analysis")
    lines.append(f"- Train accuracy: {train_metrics['accuracy']:.4f}")
    lines.append(f"- Test accuracy: {test_metrics['accuracy']:.4f}")
    lines.append(f"- Train macro F1: {train_metrics['macro_f1']:.4f}")
    lines.append(f"- Test macro F1: {test_metrics['macro_f1']:.4f}")
    lines.append(f"- Gaps (train - test): accuracy {gaps['accuracy']:+.4f}, macro F1 {gaps['macro_f1']:+.4f}")
    if gaps["macro_f1"] > 0.10:
        lines.append("- Large train-test macro-F1 gap suggests potential overfitting.")
    else:
        lines.append("- Train-test gap is modest for this linear model; no strong overfitting signal.")
    lines.append("")
    lines.append("## 13. Limitations")
    lines.append("- Only `C` was tuned; other LR settings held fixed.")
    lines.append("- Grid is coarse; intermediate C values not searched.")
    lines.append("- Selection used macro F1 only; clinical cost trade-offs not optimized.")
    lines.append("- Test set used once after selection.")
    lines.append("")
    lines.append("## Artifact integrity")
    for name, size in protected_sizes.items():
        lines.append(f"- `{name}`: {size} bytes (unchanged)")
    lines.append("")

    # Final conclusions
    improved_macro = test_metrics["macro_f1"] > PREV_BEST["macro_f1"] + 0.005
    improved_hp = float(pc["HP"]["recall"]) > PREV_BEST["HP_recall"] + 0.01
    improved_ctd = float(pc["CTD"]["recall"]) > PREV_BEST["CTD_recall"] + 0.01
    near_macro = abs(test_metrics["macro_f1"] - PREV_BEST["macro_f1"]) <= 0.005

    lines.append("## Final conclusions")
    lines.append("")
    lines.append(
        f"1. Did tuning C improve macro-F1? "
        f"{'YES' if improved_macro else ('NO meaningful change' if near_macro else 'NO')} "
        f"({PREV_BEST['macro_f1']:.4f} -> {test_metrics['macro_f1']:.4f})."
    )
    lines.append(
        f"2. Improve HP recall? {'YES' if improved_hp else 'NO'} "
        f"({PREV_BEST['HP_recall']:.3f} -> {float(pc['HP']['recall']):.3f})."
    )
    lines.append(
        f"3. Improve CTD recall? {'YES' if improved_ctd else 'NO'} "
        f"({PREV_BEST['CTD_recall']:.3f} -> {float(pc['CTD']['recall']):.3f})."
    )
    lines.append(
        f"4. Improve overall generalization? "
        f"{'YES' if improved_macro or (test_metrics['macro_f1'] >= PREV_BEST['macro_f1'] and gaps['macro_f1'] < 0.10) else 'NO clear gain'} "
        f"(judge primarily by held-out macro F1 and train-test gap)."
    )
    meaningfully_better = improved_macro or (
        test_metrics["macro_f1"] >= PREV_BEST["macro_f1"]
        and (improved_hp or improved_ctd)
        and not near_macro
    )
    if near_macro and not improved_hp and not improved_ctd:
        meaningfully = "NO - performance is essentially equivalent to C=1"
        best = "Class-weighted Logistic Regression with C=1.0 remains the simplest best candidate (tuned C not meaningfully better)"
        if best_C == 1 or best_C == 1.0:
            best = "Class-weighted Logistic Regression with C=1.0 (CV also selected C=1, confirming prior configuration)"
    elif meaningfully_better or improved_macro:
        meaningfully = "YES"
        best = f"CV-tuned Logistic Regression with C={best_C}"
    else:
        meaningfully = "NO / unclear"
        # Prefer higher test macro F1; tie -> C=1 for simplicity if selected C != 1
        if test_metrics["macro_f1"] > PREV_BEST["macro_f1"]:
            best = f"CV-tuned Logistic Regression with C={best_C} (marginal edge)"
        else:
            best = "Class-weighted Logistic Regression with C=1.0"
    lines.append(f"5. Is the tuned model meaningfully better than C=1? {meaningfully}.")
    lines.append(f"6. Current best candidate: **{best}**.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    protected_before = {n: (MODELS_DIR / n).stat().st_size for n in PROTECTED}
    protected_mtime = {n: (MODELS_DIR / n).stat().st_mtime for n in PROTECTED}

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    assert len(train) == 4073 and len(test) == 1019

    X_train = train[FEATURE_COLS].copy()
    y_train = train[TARGET_COL].copy()
    X_test = test[FEATURE_COLS].copy()
    y_test = test[TARGET_COL].copy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_base_pipeline()

    search = GridSearchCV(
        estimator=pipe,
        param_grid={"clf__C": C_GRID},
        scoring={
            "macro_f1": "f1_macro",
            "accuracy": "accuracy",
            "macro_precision": "precision_macro",
            "macro_recall": "recall_macro",
        },
        refit="macro_f1",
        cv=cv,
        n_jobs=None,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    # Build CV results table ranked later
    rows = []
    for i, c in enumerate(search.cv_results_["param_clf__C"]):
        rows.append(
            {
                "C": float(c),
                "mean_macro_f1": float(search.cv_results_["mean_test_macro_f1"][i]),
                "std_macro_f1": float(search.cv_results_["std_test_macro_f1"][i]),
                "mean_accuracy": float(search.cv_results_["mean_test_accuracy"][i]),
                "mean_macro_precision": float(search.cv_results_["mean_test_macro_precision"][i]),
                "mean_macro_recall": float(search.cv_results_["mean_test_macro_recall"][i]),
            }
        )
    cv_table = pd.DataFrame(rows).sort_values("C").reset_index(drop=True)
    cv_table.to_csv(REPORTS_DIR / "07_logistic_hyperparameter_cv_results.csv", index=False)

    best_C = float(search.best_params_["clf__C"])
    best_pipe = search.best_estimator_  # already refit on full training set by GridSearchCV

    y_train_pred = best_pipe.predict(X_train)
    y_test_pred = best_pipe.predict(X_test)
    train_metrics = metrics_bundle(y_train, y_train_pred)
    test_metrics = metrics_bundle(y_test, y_test_pred)
    per_class = per_class_table(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred, labels=CLASS_ORDER)

    out_model = MODELS_DIR / "logistic_regression_tuned.joblib"
    joblib.dump(best_pipe, out_model)

    for n in PROTECTED:
        assert (MODELS_DIR / n).stat().st_size == protected_before[n]
        assert (MODELS_DIR / n).stat().st_mtime == protected_mtime[n]

    per_class.to_csv(REPORTS_DIR / "07_tuned_test_per_class.csv", index=False)
    pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        REPORTS_DIR / "07_tuned_test_confusion_matrix.csv"
    )

    report = write_report(
        cv_table, best_C, train_metrics, test_metrics, per_class, cm, protected_before
    )
    report_path = REPORTS_DIR / "07_logistic_hyperparameter_tuning_report.md"
    report_path.write_text(report, encoding="utf-8")

    meta = {
        "best_C": best_C,
        "best_cv_macro_f1": float(search.best_score_),
        "cv_table": cv_table.to_dict(orient="records"),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_path": str(out_model.relative_to(PROJECT_ROOT)),
        "prior_models_unchanged": True,
        "classification_report_test": classification_report(
            y_test, y_test_pred, labels=CLASS_ORDER, digits=4
        ),
    }
    (REPORTS_DIR / "07_logistic_hyperparameter_tuning_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(report.encode("ascii", errors="replace").decode("ascii"))
    print("\nSaved:", out_model)
    print("Best C:", best_C)


if __name__ == "__main__":
    main()
