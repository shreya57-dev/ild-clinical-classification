"""
Controlled baseline: regularized HistGradientBoosting with balanced sample weights.

Compare to class-weighted Logistic Regression on the SAME Stratified 5-fold CV
(training set only). Evaluate the selected configuration once on the held-out test set.

Does NOT overwrite prior model artifacts. No hyperparameter tuning.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

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

HGB_PARAMS = dict(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=1.0,
    random_state=42,
)

LR_PARAMS = dict(
    class_weight="balanced",
    solver="lbfgs",
    C=1.0,
    l1_ratio=0.0,
    max_iter=2000,
    random_state=42,
)

# Prior confusion matrices for comparison (rows=true, cols=pred)
UNWEIGHTED_LR_CM = np.array(
    [[110, 7, 15, 20], [17, 24, 17, 45], [25, 10, 31, 62], [14, 5, 12, 605]], dtype=int
)
WEIGHTED_LR_CM = np.array(
    [[109, 17, 22, 4], [18, 42, 23, 20], [27, 21, 61, 19], [13, 38, 70, 515]], dtype=int
)
RF_CM = np.array(
    [[94, 14, 29, 15], [13, 36, 27, 27], [27, 24, 38, 39], [22, 11, 39, 564]], dtype=int
)

# Previous weighted LR held-out test (reference)
PREV_BEST_TEST = {
    "accuracy": 0.7134,
    "macro_f1": 0.5819,
    "macro_recall": 0.6028,
    "HP_recall": 0.408,
    "CTD_recall": 0.477,
    "1_SAR_recall": 0.810,
}

PROTECTED = [
    "baseline_logistic_regression.joblib",
    "class_weighted_logistic_regression.joblib",
    "logistic_full_features_cv_selected.joblib",
    "logistic_regression_tuned.joblib",
    "random_forest_balanced.joblib",
]


def build_hgb_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_FEATURES,
            ),
        ],
        remainder="drop",
    )
    clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def build_lr_pipeline() -> Pipeline:
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
    return Pipeline(
        [("preprocess", preprocess), ("clf", LogisticRegression(**LR_PARAMS))]
    )


def balanced_sample_weights(y: pd.Series | np.ndarray) -> np.ndarray:
    """weight_c = n / (n_classes * n_c); one weight per observation. TRAINING labels only."""
    return compute_sample_weight(class_weight="balanced", y=y)


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


def run_hgb_cv(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> dict:
    fold_rows = []
    val_pc_f1 = {c: [] for c in CLASS_ORDER}
    val_pc_rec = {c: [] for c in CLASS_ORDER}
    val_pc_prec = {c: [] for c in CLASS_ORDER}

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        sw = balanced_sample_weights(y_tr)

        # Verify formula using fold training counts
        counts = y_tr.value_counts()
        n, k = len(y_tr), len(CLASS_ORDER)
        for c in CLASS_ORDER:
            expected = n / (k * int(counts[c]))
            # sample weight for any row of class c equals expected
            assert abs(expected - float(sw[y_tr.to_numpy() == c][0])) < 1e-9

        pipe = build_hgb_pipeline()
        pipe.fit(X_tr, y_tr, clf__sample_weight=sw)

        pred_tr = pipe.predict(X_tr)
        pred_va = pipe.predict(X_va)
        tr_m = metrics_bundle(y_tr, pred_tr)
        va_m = metrics_bundle(y_va, pred_va)
        p, r, f, _ = precision_recall_fscore_support(
            y_va, pred_va, labels=CLASS_ORDER, zero_division=0
        )
        for i, c in enumerate(CLASS_ORDER):
            val_pc_prec[c].append(float(p[i]))
            val_pc_rec[c].append(float(r[i]))
            val_pc_f1[c].append(float(f[i]))

        fold_rows.append(
            {
                "fold": fold,
                "train_accuracy": tr_m["accuracy"],
                "train_macro_f1": tr_m["macro_f1"],
                "val_accuracy": va_m["accuracy"],
                "val_macro_f1": va_m["macro_f1"],
                "val_macro_precision": va_m["macro_precision"],
                "val_macro_recall": va_m["macro_recall"],
                "val_weighted_f1": va_m["weighted_f1"],
            }
        )

    folds = pd.DataFrame(fold_rows)
    return {
        "folds": folds,
        "mean_val_macro_f1": float(folds["val_macro_f1"].mean()),
        "std_val_macro_f1": float(folds["val_macro_f1"].std(ddof=1)),
        "mean_val_accuracy": float(folds["val_accuracy"].mean()),
        "mean_val_macro_precision": float(folds["val_macro_precision"].mean()),
        "mean_val_macro_recall": float(folds["val_macro_recall"].mean()),
        "mean_val_weighted_f1": float(folds["val_weighted_f1"].mean()),
        "mean_train_macro_f1": float(folds["train_macro_f1"].mean()),
        "mean_train_accuracy": float(folds["train_accuracy"].mean()),
        "mean_train_val_macro_f1_gap": float(
            (folds["train_macro_f1"] - folds["val_macro_f1"]).mean()
        ),
        "per_class_mean_val_f1": {c: float(np.mean(v)) for c, v in val_pc_f1.items()},
        "per_class_mean_val_recall": {c: float(np.mean(v)) for c, v in val_pc_rec.items()},
        "per_class_mean_val_precision": {c: float(np.mean(v)) for c, v in val_pc_prec.items()},
    }


def run_lr_cv(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> dict:
    fold_macro_f1 = []
    fold_acc = []
    fold_macro_rec = []
    for tr_idx, va_idx in cv.split(X, y):
        pipe = build_lr_pipeline()
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = pipe.predict(X.iloc[va_idx])
        yt = y.iloc[va_idx]
        fold_macro_f1.append(f1_score(yt, pred, average="macro", labels=CLASS_ORDER, zero_division=0))
        fold_acc.append(accuracy_score(yt, pred))
        fold_macro_rec.append(
            recall_score(yt, pred, average="macro", labels=CLASS_ORDER, zero_division=0)
        )
    return {
        "fold_macro_f1": fold_macro_f1,
        "mean_val_macro_f1": float(np.mean(fold_macro_f1)),
        "std_val_macro_f1": float(np.std(fold_macro_f1, ddof=1)),
        "mean_val_accuracy": float(np.mean(fold_acc)),
        "mean_val_macro_recall": float(np.mean(fold_macro_rec)),
    }


def write_report(
    hgb_cv: dict,
    lr_cv: dict,
    selected: str,
    train_full_metrics: dict | None,
    test_metrics: dict | None,
    per_class: pd.DataFrame | None,
    cm: np.ndarray | None,
    protected_sizes: dict,
    importance_note: str,
) -> str:
    lines = []
    lines.append("# HistGradientBoosting baseline report (balanced sample weights)")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "Determine whether a deliberately regularized nonlinear boosting model can capture "
        "useful interactions among PFT features without the severe overfitting seen with untuned Random Forest, "
        "and whether it improves on class-weighted Logistic Regression under training-only CV."
    )
    lines.append("")
    lines.append("## 2. Why Gradient Boosting was selected")
    lines.append("- Additive trees can model nonlinear interactions among correlated PFT indices.")
    lines.append("- Built-in regularization knobs (learning rate, leaf limits, L2) to limit RF-style memorization.")
    lines.append("- Still a classical ML baseline suitable for a prototype before any tuning.")
    lines.append("")
    lines.append("## 3. Difference between Random Forest and Gradient Boosting")
    lines.append(
        "- Random Forest averages many deep independent trees (high capacity; easy to overfit if unconstrained)."
    )
    lines.append(
        "- HistGradientBoosting builds shallow trees sequentially to correct residuals, with explicit "
        "learning-rate / leaf / L2 constraints that encourage smoother generalization."
    )
    lines.append("")
    lines.append("## 4. Feature set")
    for f in FEATURE_COLS:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## 5. Preprocessing")
    lines.append("- Numerical: passthrough (no StandardScaler)")
    lines.append("- Categorical `sex`: OneHotEncoder(handle_unknown='ignore', sparse_output=False)")
    lines.append("- No imputation (0 missing values)")
    lines.append("")
    lines.append("## 6. Class-weighting / sample-weight methodology")
    lines.append("- Formula: `weight_c = n_samples / (n_classes * n_c)`")
    lines.append("- Implemented via `compute_sample_weight(class_weight='balanced')`")
    lines.append("- Weights computed from **fold training labels only** (or full train labels for final fit)")
    lines.append("- Passed as `clf__sample_weight` to Pipeline.fit")
    lines.append("")
    lines.append("## 7. Model configuration")
    for k, v in HGB_PARAMS.items():
        lines.append(f"- `{k}` = `{v}`")
    lines.append("- No hyperparameter tuning")
    lines.append("")
    lines.append("## 8. 5-fold CV methodology")
    lines.append("- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)")
    lines.append("- Training set only (n=4073); test held out")
    lines.append("- Same folds for HGB and Logistic Regression comparison")
    lines.append("- Primary metric: macro F1")
    lines.append("")
    lines.append("## 9. Fold-level results (HGB)")
    lines.append("")
    lines.append(hgb_cv["folds"].to_markdown(index=False))
    lines.append("")
    lines.append("## 10. Mean / std CV results (HGB validation)")
    lines.append(f"- mean validation macro F1: **{hgb_cv['mean_val_macro_f1']:.4f}**")
    lines.append(f"- std validation macro F1: **{hgb_cv['std_val_macro_f1']:.4f}**")
    lines.append(f"- mean validation accuracy: **{hgb_cv['mean_val_accuracy']:.4f}**")
    lines.append(f"- mean validation macro precision: **{hgb_cv['mean_val_macro_precision']:.4f}**")
    lines.append(f"- mean validation macro recall: **{hgb_cv['mean_val_macro_recall']:.4f}**")
    lines.append(f"- mean validation weighted F1: **{hgb_cv['mean_val_weighted_f1']:.4f}**")
    lines.append("")
    lines.append("### Per-class mean validation recall / F1")
    lines.append("")
    lines.append("| Class | mean val recall | mean val F1 |")
    lines.append("|---|---:|---:|")
    for c in CLASS_ORDER:
        lines.append(
            f"| {c} | {hgb_cv['per_class_mean_val_recall'][c]:.4f} | {hgb_cv['per_class_mean_val_f1'][c]:.4f} |"
        )
    lines.append("")
    lines.append("## 11. Train-validation gap (HGB)")
    lines.append(f"- mean train macro F1: **{hgb_cv['mean_train_macro_f1']:.4f}**")
    lines.append(f"- mean validation macro F1: **{hgb_cv['mean_val_macro_f1']:.4f}**")
    lines.append(
        f"- mean train-validation macro F1 gap: **{hgb_cv['mean_train_val_macro_f1_gap']:+.4f}**"
    )
    lines.append("")
    lines.append("## 12. Comparison with weighted Logistic Regression (same CV folds)")
    lines.append("")
    lines.append("| Model | Mean CV Macro F1 | Std CV Macro F1 | Mean CV Accuracy | Mean CV Macro Recall |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Class-weighted Logistic Regression | {lr_cv['mean_val_macro_f1']:.4f} | "
        f"{lr_cv['std_val_macro_f1']:.4f} | {lr_cv['mean_val_accuracy']:.4f} | {lr_cv['mean_val_macro_recall']:.4f} |"
    )
    lines.append(
        f"| Regularized HistGradientBoosting | {hgb_cv['mean_val_macro_f1']:.4f} | "
        f"{hgb_cv['std_val_macro_f1']:.4f} | {hgb_cv['mean_val_accuracy']:.4f} | {hgb_cv['mean_val_macro_recall']:.4f} |"
    )
    lines.append("")
    delta = hgb_cv["mean_val_macro_f1"] - lr_cv["mean_val_macro_f1"]
    lines.append(f"CV macro F1 difference (HGB - LR): **{delta:+.4f}**")
    lines.append(f"**Configuration selected for final test evaluation: `{selected}`**")
    lines.append("")

    if test_metrics is not None and per_class is not None and cm is not None:
        lines.append("## 13. Final test performance")
        lines.append(
            "Fitted on all 4073 training patients with balanced sample weights from training labels only; "
            "evaluated once on the untouched 1019-patient test set."
        )
        lines.append("")
        for k, v in test_metrics.items():
            lines.append(f"- {k}: **{v:.4f}**")
        pc = {r["class"]: r for r in per_class.to_dict(orient="records")}
        lines.append(f"- HP recall: **{pc['HP']['recall']:.4f}**")
        lines.append(f"- CTD recall: **{pc['CTD']['recall']:.4f}**")
        lines.append(f"- 1_SAR recall: **{pc['1_SAR']['recall']:.4f}**")
        if train_full_metrics is not None:
            lines.append("")
            lines.append("### Full-train vs test macro F1")
            lines.append(f"- Full-train macro F1: {train_full_metrics['macro_f1']:.4f}")
            lines.append(f"- Test macro F1: {test_metrics['macro_f1']:.4f}")
            lines.append(
                f"- Gap: {train_full_metrics['macro_f1'] - test_metrics['macro_f1']:+.4f}"
            )
        lines.append("")
        lines.append("## 14. Per-class metrics (test)")
        lines.append("")
        lines.append(per_class.to_markdown(index=False))
        lines.append("")
        lines.append("## 15. Confusion matrix (test)")
        lines.append("")
        lines.append(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_markdown())
        lines.append("")
        hp_sar = int(cm[1, 3])
        ctd_sar = int(cm[2, 3])
        sar_min = int(cm[3, 0] + cm[3, 1] + cm[3, 2])
        lines.append("### Focus error comparison")
        lines.append("")
        lines.append("| Pattern | Unweighted LR | Weighted LR | Random Forest | HGB |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(
            f"| HP -> 1_SAR | {int(UNWEIGHTED_LR_CM[1,3])} | {int(WEIGHTED_LR_CM[1,3])} | {int(RF_CM[1,3])} | {hp_sar} |"
        )
        lines.append(
            f"| CTD -> 1_SAR | {int(UNWEIGHTED_LR_CM[2,3])} | {int(WEIGHTED_LR_CM[2,3])} | {int(RF_CM[2,3])} | {ctd_sar} |"
        )
        lines.append(
            f"| 1_SAR -> minority | {int(UNWEIGHTED_LR_CM[3,:3].sum())} | {int(WEIGHTED_LR_CM[3,:3].sum())} | {int(RF_CM[3,:3].sum())} | {sar_min} |"
        )
        lines.append("")
        lines.append("## 16. Overfitting analysis")
        gap_cv = hgb_cv["mean_train_val_macro_f1_gap"]
        lines.append(
            f"- CV mean train macro F1={hgb_cv['mean_train_macro_f1']:.4f}; "
            f"val={hgb_cv['mean_val_macro_f1']:.4f}; gap={gap_cv:+.4f}"
        )
        if train_full_metrics is not None:
            lines.append(
                f"- Full-train macro F1={train_full_metrics['macro_f1']:.4f}; "
                f"test={test_metrics['macro_f1']:.4f}; "
                f"gap={train_full_metrics['macro_f1'] - test_metrics['macro_f1']:+.4f}"
            )
        lines.append("- Random Forest reference: train macro F1=0.989, test macro F1=0.541 (gap ~0.45)")
        if gap_cv > 0.20:
            overfit_level = "severe overfitting"
        elif gap_cv > 0.08:
            overfit_level = "moderate overfitting"
        else:
            overfit_level = "little/no obvious overfitting"
        lines.append(f"- Assessment for HGB based on CV gap: **{overfit_level}**")
        lines.append("")
        lines.append("## Comparison vs previous best test metrics (Weighted LR)")
        lines.append("")
        lines.append("| Metric | Weighted LR (prior test) | HGB test | Diff |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            f"| accuracy | {PREV_BEST_TEST['accuracy']:.4f} | {test_metrics['accuracy']:.4f} | "
            f"{test_metrics['accuracy']-PREV_BEST_TEST['accuracy']:+.4f} |"
        )
        lines.append(
            f"| macro_f1 | {PREV_BEST_TEST['macro_f1']:.4f} | {test_metrics['macro_f1']:.4f} | "
            f"{test_metrics['macro_f1']-PREV_BEST_TEST['macro_f1']:+.4f} |"
        )
        lines.append(
            f"| HP_recall | {PREV_BEST_TEST['HP_recall']:.4f} | {pc['HP']['recall']:.4f} | "
            f"{pc['HP']['recall']-PREV_BEST_TEST['HP_recall']:+.4f} |"
        )
        lines.append(
            f"| CTD_recall | {PREV_BEST_TEST['CTD_recall']:.4f} | {pc['CTD']['recall']:.4f} | "
            f"{pc['CTD']['recall']-PREV_BEST_TEST['CTD_recall']:+.4f} |"
        )
        lines.append(
            f"| 1_SAR_recall | {PREV_BEST_TEST['1_SAR_recall']:.4f} | {pc['1_SAR']['recall']:.4f} | "
            f"{pc['1_SAR']['recall']-PREV_BEST_TEST['1_SAR_recall']:+.4f} |"
        )
    else:
        lines.append("## 13-16. Final test evaluation")
        lines.append(
            "HGB was **not** selected by CV for final test evaluation; "
            "no HGB test metrics are reported to avoid post-hoc selection."
        )
        # Still fit & evaluate for completeness? User said:
        # "If it does not clearly improve... state that... Do not change parameters after seeing test"
        # and "ONLY AFTER the model configuration has been fixed using CV" fit and evaluate selected model.
        # So if LR is selected, we should evaluate LR (already known) OR still evaluate HGB?
        # Task 10 says evaluate the selected model. If LR selected, evaluating LR again is redundant.
        # Task 14 says save hist_gradient_boosting model anyway.
        # I'll still fit HGB on full train and evaluate on test for the report artifact completeness
        # because Task 14 requires saving the HGB model and Task 10-12 ask for HGB test metrics / CM.
        # Re-read: "If HistGradientBoosting has clearly better... select it for final test evaluation.
        # If it does not clearly improve... state that the nonlinear model does not provide sufficient evidence"
        # Task 10: "ONLY AFTER the model configuration has been fixed... Fit the selected model"
        # So if LR is selected, we shouldn't necessarily run HGB test. But Task 14 still wants HGB model saved.
        # And Task 12-16 ask for HGB confusion matrix etc.
        # Practical approach: Always fit and save HGB on full train + one test eval for documentation,
        # but selection decision is based only on CV. Report both: CV decision + HGB test as
        # "final evaluation of the fixed HGB baseline configuration" since HGB config was fixed a priori
        # (not tuned). The "selected" for "best candidate" is separate.
        #
        # Actually re-read Task 9-10 more carefully. The HGB configuration is fixed (no tuning).
        # CV compares HGB vs LR. If HGB wins CV, it's the selected candidate for test.
        # If HGB loses, we state insufficient evidence - but Task 14 still saves HGB model.
        # Tasks 10-12 seem to assume we evaluate HGB on test. The fixed HGB baseline config was
        # chosen before seeing test - so evaluating HGB on test once is valid as evaluating this
        # experiment's model, while the "best candidate" decision uses CV comparison.
        #
        # I'll always do one HGB final test eval since the HGB hyperparams were fixed a priori
        # (not selected via test). CV comparison decides whether it's a stronger candidate than LR.

    lines.append("")
    lines.append("## 17. Feature importance")
    lines.append(importance_note)
    lines.append("")
    lines.append("## 18. Limitations")
    lines.append("- Single untuned HGB configuration; no grid search.")
    lines.append("- Sample weights approximate class balancing but differ from LR class_weight internals.")
    lines.append("- Correlated z/%predicted features retained.")
    lines.append("- Native feature importances unavailable in this sklearn build.")
    lines.append("")
    lines.append("## Artifact integrity")
    for name, size in protected_sizes.items():
        lines.append(f"- `{name}`: {size} bytes (unchanged)")
    lines.append("")

    # Final conclusions need CV + optional test
    clear_cv_win = hgb_cv["mean_val_macro_f1"] > lr_cv["mean_val_macro_f1"] + 0.005
    lines.append("## 19. Final conclusion")
    lines.append("")
    lines.append(
        f"1. Outperform weighted LR in training-only CV? "
        f"{'YES' if clear_cv_win else 'NO'} "
        f"(HGB={hgb_cv['mean_val_macro_f1']:.4f}, LR={lr_cv['mean_val_macro_f1']:.4f})."
    )
    if test_metrics is not None and per_class is not None and cm is not None:
        pc = {r["class"]: r for r in per_class.to_dict(orient="records")}
        lines.append(
            f"2. Improve HP recall vs prior weighted LR test? "
            f"{'YES' if pc['HP']['recall'] > PREV_BEST_TEST['HP_recall'] + 0.01 else 'NO'} "
            f"({PREV_BEST_TEST['HP_recall']:.3f} -> {pc['HP']['recall']:.3f})."
        )
        lines.append(
            f"3. Improve CTD recall vs prior weighted LR test? "
            f"{'YES' if pc['CTD']['recall'] > PREV_BEST_TEST['CTD_recall'] + 0.01 else 'NO'} "
            f"({PREV_BEST_TEST['CTD_recall']:.3f} -> {pc['CTD']['recall']:.3f})."
        )
        hp_sar = int(cm[1, 3])
        ctd_sar = int(cm[2, 3])
        lines.append(
            f"4. Reduce HP/CTD -> 1_SAR errors vs weighted LR? "
            f"{'YES' if (hp_sar + ctd_sar) < (20 + 19) else 'NO'} "
            f"({20+19} -> {hp_sar + ctd_sar})."
        )
        gap_cv = hgb_cv["mean_train_val_macro_f1_gap"]
        if gap_cv > 0.20:
            overfit_q = "YES (substantial)"
        elif gap_cv > 0.08:
            overfit_q = "MODERATE"
        else:
            overfit_q = "NO / little"
        lines.append(f"5. Overfit substantially? {overfit_q} (CV train-val macro F1 gap={gap_cv:+.4f}).")
        stronger = clear_cv_win and test_metrics["macro_f1"] >= PREV_BEST_TEST["macro_f1"]
        lines.append(
            f"6. Stronger candidate than class-weighted LR? "
            f"{'YES' if stronger else 'NO'} "
            f"(CV decision + held-out macro F1 {test_metrics['macro_f1']:.4f} vs LR {PREV_BEST_TEST['macro_f1']:.4f})."
        )
        if clear_cv_win and stronger:
            next_step = "Proceed to careful tuning of HistGradientBoosting"
        else:
            next_step = "Retain class-weighted Logistic Regression as the primary candidate; do not prioritize HGB tuning yet"
        lines.append(f"7. Next step: **{next_step}**.")
    else:
        lines.append("2-4. Test metrics not applicable under non-selection path.")
        lines.append(
            f"5. Overfit substantially? "
            f"{'YES' if hgb_cv['mean_train_val_macro_f1_gap'] > 0.20 else 'MODERATE / limited'}."
        )
        lines.append("6. Stronger candidate than LR? NO (insufficient CV evidence).")
        lines.append("7. Next step: **Retain class-weighted Logistic Regression**.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Only assert protected files that exist
    existing_protected = [n for n in PROTECTED if (MODELS_DIR / n).exists()]
    protected_before = {n: (MODELS_DIR / n).stat().st_size for n in existing_protected}
    protected_mtime = {n: (MODELS_DIR / n).stat().st_mtime for n in existing_protected}

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    assert len(train) == 4073 and len(test) == 1019

    X_train = train[FEATURE_COLS].copy()
    y_train = train[TARGET_COL].copy()
    X_test = test[FEATURE_COLS].copy()
    y_test = test[TARGET_COL].copy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running HGB CV...")
    hgb_cv = run_hgb_cv(X_train, y_train, cv)
    print("Running LR CV on same folds...")
    lr_cv = run_lr_cv(X_train, y_train, cv)

    # Selection based on CV only (clear improvement threshold 0.005)
    if hgb_cv["mean_val_macro_f1"] > lr_cv["mean_val_macro_f1"] + 0.005:
        selected = "hist_gradient_boosting_balanced"
        print("CV selects HGB")
    else:
        selected = "class_weighted_logistic_regression"
        print(
            "CV does not clearly prefer HGB; "
            f"HGB={hgb_cv['mean_val_macro_f1']:.4f} vs LR={lr_cv['mean_val_macro_f1']:.4f}"
        )

    # Fixed HGB configuration was chosen a priori; always fit once for this experiment's artifact
    # and one held-out test evaluation (no retuning after seeing test).
    sw_full = balanced_sample_weights(y_train)
    # Report effective weights from full training labels
    counts = y_train.value_counts()
    weight_map = {
        c: len(y_train) / (len(CLASS_ORDER) * int(counts[c])) for c in CLASS_ORDER
    }
    print("Full-train balanced class weights:", weight_map)

    hgb_pipe = build_hgb_pipeline()
    hgb_pipe.fit(X_train, y_train, clf__sample_weight=sw_full)
    y_tr_pred = hgb_pipe.predict(X_train)
    y_te_pred = hgb_pipe.predict(X_test)
    train_full_metrics = metrics_bundle(y_train, y_tr_pred)
    test_metrics = metrics_bundle(y_test, y_te_pred)
    per_class = per_class_table(y_test, y_te_pred)
    cm = confusion_matrix(y_test, y_te_pred, labels=CLASS_ORDER)

    out_model = MODELS_DIR / "hist_gradient_boosting_balanced.joblib"
    joblib.dump(hgb_pipe, out_model)

    for n in existing_protected:
        assert (MODELS_DIR / n).stat().st_size == protected_before[n]
        assert (MODELS_DIR / n).stat().st_mtime == protected_mtime[n]

    # Feature importance
    clf = hgb_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        names = hgb_pipe.named_steps["preprocess"].get_feature_names_out()
        imp = pd.DataFrame(
            {"feature": names, "importance": clf.feature_importances_}
        ).sort_values("importance", ascending=False)
        imp.to_csv(REPORTS_DIR / "08_hgb_feature_importance.csv", index=False)
        importance_note = (
            "Native `feature_importances_` available. Values reflect model association/usage, not causality.\n\n"
            + imp.to_markdown(index=False)
        )
    else:
        importance_note = (
            "Native `feature_importances_` is **unavailable** for `HistGradientBoostingClassifier` "
            f"in sklearn {__import__('sklearn').__version__}. No alternative importance method was added in this experiment."
        )

    # Plots
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("HGB balanced - test confusion matrix")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_hgb_confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    # Save tables
    hgb_cv["folds"].to_csv(REPORTS_DIR / "08_hgb_cv_folds.csv", index=False)
    per_class.to_csv(REPORTS_DIR / "08_hgb_test_per_class.csv", index=False)
    pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        REPORTS_DIR / "08_hgb_test_confusion_matrix.csv"
    )
    comparison = pd.DataFrame(
        [
            {
                "Model": "Class-weighted Logistic Regression",
                "Mean CV Macro F1": lr_cv["mean_val_macro_f1"],
                "Std CV Macro F1": lr_cv["std_val_macro_f1"],
                "Mean CV Accuracy": lr_cv["mean_val_accuracy"],
                "Mean CV Macro Recall": lr_cv["mean_val_macro_recall"],
            },
            {
                "Model": "Regularized HistGradientBoosting",
                "Mean CV Macro F1": hgb_cv["mean_val_macro_f1"],
                "Std CV Macro F1": hgb_cv["std_val_macro_f1"],
                "Mean CV Accuracy": hgb_cv["mean_val_accuracy"],
                "Mean CV Macro Recall": hgb_cv["mean_val_macro_recall"],
            },
        ]
    )
    comparison.to_csv(REPORTS_DIR / "08_cv_model_comparison.csv", index=False)

    report = write_report(
        hgb_cv,
        lr_cv,
        selected,
        train_full_metrics,
        test_metrics,
        per_class,
        cm,
        protected_before,
        importance_note,
    )
    report_path = REPORTS_DIR / "08_hist_gradient_boosting_baseline_report.md"
    report_path.write_text(report, encoding="utf-8")

    meta = {
        "hgb_params": HGB_PARAMS,
        "selected_by_cv": selected,
        "train_class_weights": weight_map,
        "hgb_cv_summary": {
            k: v
            for k, v in hgb_cv.items()
            if k != "folds"
        },
        "lr_cv": lr_cv,
        "train_full_metrics": train_full_metrics,
        "test_metrics": test_metrics,
        "model_path": str(out_model.relative_to(PROJECT_ROOT)),
        "feature_importances_available": hasattr(clf, "feature_importances_"),
        "classification_report_test": classification_report(
            y_test, y_te_pred, labels=CLASS_ORDER, digits=4
        ),
    }
    # serialize nested
    meta["hgb_cv_summary"]["per_class_mean_val_f1"] = hgb_cv["per_class_mean_val_f1"]
    meta["hgb_cv_summary"]["per_class_mean_val_recall"] = hgb_cv["per_class_mean_val_recall"]
    meta["hgb_cv_summary"]["per_class_mean_val_precision"] = hgb_cv[
        "per_class_mean_val_precision"
    ]
    (REPORTS_DIR / "08_hist_gradient_boosting_baseline_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(report.encode("ascii", errors="replace").decode("ascii"))
    print("\nSaved:", out_model)


if __name__ == "__main__":
    main()
