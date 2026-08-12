"""
Controlled experiment: class-weighted Random Forest baseline.

Does NOT modify train/test CSVs, raw data, or prior Logistic Regression models.
No hyperparameter tuning.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RF_MODEL_PATH = MODELS_DIR / "random_forest_balanced.joblib"
BASELINE_LR_PATH = MODELS_DIR / "baseline_logistic_regression.joblib"
WEIGHTED_LR_PATH = MODELS_DIR / "class_weighted_logistic_regression.joblib"

TARGET_COL = "diagnosis group"
CLASS_ORDER = ["IPF", "HP", "CTD", "1_SAR"]

NUM_FEATURES = [
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
    "Height",
    "Weight",
]
CAT_FEATURES = ["sex"]
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

RANDOM_STATE = 42
RF_PARAMS = dict(
    n_estimators=300,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Previously recorded Logistic Regression results on THIS same held-out test set
UNWEIGHTED_LR = {
    "accuracy": 0.7556,
    "macro_precision": 0.6061,
    "macro_recall": 0.5375,
    "macro_f1": 0.5510,
    "weighted_f1": 0.7262,  # from experiment 2 comparison table
    "IPF_recall": 0.724,
    "HP_recall": 0.233,
    "CTD_recall": 0.242,
    "1_SAR_recall": 0.951,
    "IPF_f1": 0.692,
    "HP_f1": 0.322,
    "CTD_f1": 0.305,
    "1_SAR_f1": 0.885,
}
WEIGHTED_LR = {
    "accuracy": 0.713,
    "macro_precision": 0.5695,
    "macro_recall": 0.6028,
    "macro_f1": 0.582,
    "weighted_f1": 0.7292,
    "IPF_recall": 0.717,
    "HP_recall": 0.408,
    "CTD_recall": 0.477,
    "1_SAR_recall": 0.810,
    "IPF_f1": 0.683,
    "HP_f1": 0.380,
    "CTD_f1": 0.401,
    "1_SAR_f1": 0.863,
}

# Confusion matrices from prior experiments (rows=true, cols=pred, order IPF/HP/CTD/1_SAR)
UNWEIGHTED_LR_CM = np.array(
    [[110, 7, 15, 20], [17, 24, 17, 45], [25, 10, 31, 62], [14, 5, 12, 605]],
    dtype=int,
)
WEIGHTED_LR_CM = np.array(
    [[109, 17, 22, 4], [18, 42, 23, 20], [27, 21, 61, 19], [13, 38, 70, 515]],
    dtype=int,
)


def metrics_bundle(y_true, y_pred, labels=CLASS_ORDER) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
    }


def per_class_table(y_true, y_pred, labels=CLASS_ORDER) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {"class": labels, "precision": p, "recall": r, "f1": f, "support": s.astype(int)}
    )


def build_pipeline() -> Pipeline:
    # Trees do not need StandardScaler; encode sex only.
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
    clf = RandomForestClassifier(**RF_PARAMS)
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def feature_importance_table(pipe: Pipeline) -> pd.DataFrame:
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pipe.named_steps["clf"].feature_importances_
    df = pd.DataFrame({"feature": names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_cm(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    train_metrics: dict,
    test_metrics: dict,
    per_class: pd.DataFrame,
    cm: np.ndarray,
    comparison: pd.DataFrame,
    importances: pd.DataFrame,
    gaps: dict,
    lr_sizes: dict,
) -> str:
    pc = {r["class"]: r for r in per_class.to_dict(orient="records")}

    hp_to_sar = int(cm[1, 3])
    ctd_to_sar = int(cm[2, 3])
    sar_to_min = int(cm[3, 0] + cm[3, 1] + cm[3, 2])

    lines = []
    lines.append("# Random Forest baseline report (class_weight=balanced)")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "Establish whether a nonlinear tree ensemble can extract additional predictive signal "
        "from the same 13 features versus the two Logistic Regression baselines, using the same "
        "held-out test set and the same class-imbalance strategy (`class_weight='balanced'`)."
    )
    lines.append("")
    lines.append("## 2. Why Random Forest was selected")
    lines.append("- Captures nonlinear interactions among PFT indices without manual feature engineering.")
    lines.append("- Robust to feature scale (no StandardScaler required).")
    lines.append("- Provides native multiclass support and feature-importance diagnostics.")
    lines.append("- Remains interpretable enough for a prototype, unlike boosting/NN stacks.")
    lines.append("")
    lines.append("## 3. Feature set (unchanged)")
    for f in FEATURE_COLS:
        lines.append(f"- `{f}`")
    lines.append(f"- Target: `{TARGET_COL}`")
    lines.append("")
    lines.append("## 4. Preprocessing")
    lines.append("- Numerical: passthrough (no scaling)")
    lines.append("- Categorical `sex`: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`")
    lines.append("- ColumnTransformer + Pipeline; fitted on training data only")
    lines.append("")
    lines.append("## 5. Model configuration")
    for k, v in RF_PARAMS.items():
        lines.append(f"- `{k}` = `{v}`")
    lines.append("- No hyperparameter tuning")
    lines.append("")
    lines.append("## 6. Training metrics")
    for k, v in train_metrics.items():
        lines.append(f"- {k}: **{v:.4f}**")
    lines.append("")
    lines.append("## 7. Test metrics")
    for k, v in test_metrics.items():
        lines.append(f"- {k}: **{v:.4f}**")
    lines.append("")
    lines.append("## 8. Per-class metrics (test)")
    lines.append("")
    lines.append(per_class.to_markdown(index=False))
    lines.append("")
    lines.append("## 9. Confusion matrix (test; rows=true, cols=pred)")
    lines.append("")
    lines.append(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_markdown())
    lines.append("")
    lines.append("### Focus error comparison")
    lines.append("")
    lines.append("| Pattern | Unweighted LR | Weighted LR | Random Forest |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| HP -> 1_SAR | {int(UNWEIGHTED_LR_CM[1,3])} | {int(WEIGHTED_LR_CM[1,3])} | {hp_to_sar} |"
    )
    lines.append(
        f"| CTD -> 1_SAR | {int(UNWEIGHTED_LR_CM[2,3])} | {int(WEIGHTED_LR_CM[2,3])} | {ctd_to_sar} |"
    )
    lines.append(
        f"| 1_SAR -> minority | {int(UNWEIGHTED_LR_CM[3,:3].sum())} | {int(WEIGHTED_LR_CM[3,:3].sum())} | {sar_to_min} |"
    )
    lines.append("")
    lines.append("## 10. Feature importance (model usage; not causal)")
    lines.append("")
    lines.append(importances.to_markdown(index=False))
    lines.append("")
    lines.append("## 11. Comparison against Logistic Regression baselines")
    lines.append("")
    lines.append(comparison.to_markdown(index=False))
    lines.append("")
    lines.append("## 12. Overfitting analysis")
    lines.append(f"- Train-test accuracy gap: {gaps['accuracy']:+.4f}")
    lines.append(f"- Train-test macro-F1 gap: {gaps['macro_f1']:+.4f}")
    if gaps["accuracy"] > 0.10 or gaps["macro_f1"] > 0.10:
        lines.append(
            "- Large train-test gap indicates **potential overfitting** under the untuned RF configuration."
        )
    else:
        lines.append("- Train-test gap is modest; no strong overfitting signal relative to a large gap threshold (0.10).")
    lines.append("")
    lines.append("## 13. Limitations")
    lines.append("- Untuned RF (default depth/split constraints can memorize training data).")
    lines.append("- Correlated z/%predicted features retained.")
    lines.append("- Feature importances are impurity-based associations, not causal effects.")
    lines.append("- No calibration or thresholding explored.")
    lines.append("")
    lines.append("## Artifact integrity")
    lines.append(f"- Baseline LR unchanged: {lr_sizes['baseline']} bytes")
    lines.append(f"- Class-weighted LR unchanged: {lr_sizes['weighted']} bytes")
    lines.append(f"- New RF model: `{RF_MODEL_PATH.relative_to(PROJECT_ROOT)}`")
    lines.append("")

    # Final conclusions
    rf_macro = test_metrics["macro_f1"]
    rf_hp = float(pc["HP"]["recall"])
    rf_ctd = float(pc["CTD"]["recall"])
    better_macro_both = rf_macro > UNWEIGHTED_LR["macro_f1"] and rf_macro > WEIGHTED_LR["macro_f1"]
    better_macro_either = rf_macro > UNWEIGHTED_LR["macro_f1"] or rf_macro > WEIGHTED_LR["macro_f1"]
    hp_vs_wlr = rf_hp > WEIGHTED_LR["HP_recall"]
    ctd_vs_wlr = rf_ctd > WEIGHTED_LR["CTD_recall"]
    fewer_min_to_sar_vs_wlr = (hp_to_sar + ctd_to_sar) < (
        int(WEIGHTED_LR_CM[1, 3]) + int(WEIGHTED_LR_CM[2, 3])
    )
    overfit = gaps["accuracy"] > 0.10 or gaps["macro_f1"] > 0.10

    # Choose strongest candidate by macro-F1 primarily, then minority recall trade-offs
    candidates = {
        "Unweighted Logistic Regression": UNWEIGHTED_LR["macro_f1"],
        "Class-weighted Logistic Regression": WEIGHTED_LR["macro_f1"],
        "Class-weighted Random Forest": rf_macro,
    }
    strongest = max(candidates, key=candidates.get)

    lines.append("## Final conclusions")
    lines.append("")
    lines.append(
        f"1. Macro-F1 over both LR baselines? "
        f"{'YES' if better_macro_both else ('PARTIALLY (beats at least one)' if better_macro_either else 'NO')} "
        f"(RF={rf_macro:.4f}; unweighted LR={UNWEIGHTED_LR['macro_f1']:.4f}; weighted LR={WEIGHTED_LR['macro_f1']:.4f})."
    )
    lines.append(
        f"2. Improve HP recall vs class-weighted LR? "
        f"{'YES' if hp_vs_wlr else 'NO'} ({WEIGHTED_LR['HP_recall']:.3f} -> {rf_hp:.3f})."
    )
    lines.append(
        f"3. Improve CTD recall vs class-weighted LR? "
        f"{'YES' if ctd_vs_wlr else 'NO'} ({WEIGHTED_LR['CTD_recall']:.3f} -> {rf_ctd:.3f})."
    )
    lines.append(
        f"4. Reduce HP/CTD -> 1_SAR errors vs class-weighted LR? "
        f"{'YES' if fewer_min_to_sar_vs_wlr else 'NO'} "
        f"({int(WEIGHTED_LR_CM[1,3])+int(WEIGHTED_LR_CM[2,3])} -> {hp_to_sar + ctd_to_sar})."
    )
    lines.append(
        f"5. Significant overfitting? {'YES' if overfit else 'NO / modest'} "
        f"(acc gap={gaps['accuracy']:+.4f}, macro-F1 gap={gaps['macro_f1']:+.4f})."
    )
    lines.append(
        f"6. Strongest current candidate: **{strongest}** "
        f"(primary criterion: held-out macro-F1 = {candidates[strongest]:.4f}), "
        "considering also minority-class recall and HP/CTD->SAR error patterns."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    assert BASELINE_LR_PATH.exists() and WEIGHTED_LR_PATH.exists()
    lr_sizes_before = {
        "baseline": BASELINE_LR_PATH.stat().st_size,
        "weighted": WEIGHTED_LR_PATH.stat().st_size,
        "baseline_mtime": BASELINE_LR_PATH.stat().st_mtime,
        "weighted_mtime": WEIGHTED_LR_PATH.stat().st_mtime,
    }

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    assert len(train) == 4073 and len(test) == 1019
    assert list(train.columns) == FEATURE_COLS + [TARGET_COL]
    assert set(train[TARGET_COL].unique()) == set(CLASS_ORDER)
    assert set(test[TARGET_COL].unique()) == set(CLASS_ORDER)

    X_train = train[FEATURE_COLS].copy()
    y_train = train[TARGET_COL].copy()
    X_test = test[FEATURE_COLS].copy()
    y_test = test[TARGET_COL].copy()

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    train_metrics = metrics_bundle(y_train, y_train_pred)
    test_metrics = metrics_bundle(y_test, y_test_pred)
    gaps = {
        "accuracy": train_metrics["accuracy"] - test_metrics["accuracy"],
        "macro_f1": train_metrics["macro_f1"] - test_metrics["macro_f1"],
    }
    per_class = per_class_table(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred, labels=CLASS_ORDER)
    importances = feature_importance_table(pipe)

    pc = {r["class"]: r for r in per_class.to_dict(orient="records")}
    rf_row = {
        "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "IPF_recall": float(pc["IPF"]["recall"]),
        "HP_recall": float(pc["HP"]["recall"]),
        "CTD_recall": float(pc["CTD"]["recall"]),
        "1_SAR_recall": float(pc["1_SAR"]["recall"]),
        "IPF_f1": float(pc["IPF"]["f1"]),
        "HP_f1": float(pc["HP"]["f1"]),
        "CTD_f1": float(pc["CTD"]["f1"]),
        "1_SAR_f1": float(pc["1_SAR"]["f1"]),
    }

    metric_names = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "IPF_recall",
        "HP_recall",
        "CTD_recall",
        "1_SAR_recall",
        "IPF_f1",
        "HP_f1",
        "CTD_f1",
        "1_SAR_f1",
    ]
    comparison = pd.DataFrame(
        {
            "Metric": metric_names,
            "Unweighted LR": [UNWEIGHTED_LR[m] for m in metric_names],
            "Class-weighted LR": [WEIGHTED_LR[m] for m in metric_names],
            "Class-weighted RF": [rf_row[m] for m in metric_names],
        }
    )

    # Save RF only
    joblib.dump(pipe, RF_MODEL_PATH)

    # Ensure LR models untouched
    assert BASELINE_LR_PATH.stat().st_size == lr_sizes_before["baseline"]
    assert WEIGHTED_LR_PATH.stat().st_size == lr_sizes_before["weighted"]
    assert BASELINE_LR_PATH.stat().st_mtime == lr_sizes_before["baseline_mtime"]
    assert WEIGHTED_LR_PATH.stat().st_mtime == lr_sizes_before["weighted_mtime"]

    plot_cm(cm, FIGURES_DIR / "05_random_forest_confusion_matrix_test.png", "RF balanced - test CM")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, mat, title in [
        (axes[0], UNWEIGHTED_LR_CM, "Unweighted LR"),
        (axes[1], WEIGHTED_LR_CM, "Weighted LR"),
        (axes[2], cm, "Random Forest"),
    ]:
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_ORDER, fontsize=8)
        ax.set_yticklabels(CLASS_ORDER, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_confusion_matrix_three_way.png", dpi=150)
    plt.close(fig)

    # Importance bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importances["feature"][::-1], importances["importance"][::-1], color="#2F6F7E")
    ax.set_xlabel("Impurity-based importance")
    ax.set_title("Random Forest feature importances")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_random_forest_feature_importance.png", dpi=150)
    plt.close(fig)

    per_class.to_csv(REPORTS_DIR / "05_random_forest_per_class_metrics.csv", index=False)
    comparison.to_csv(REPORTS_DIR / "05_model_comparison.csv", index=False)
    importances.to_csv(REPORTS_DIR / "05_random_forest_feature_importance.csv", index=False)
    pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        REPORTS_DIR / "05_random_forest_confusion_matrix_test.csv"
    )

    report = write_report(
        train_metrics,
        test_metrics,
        per_class,
        cm,
        comparison,
        importances,
        gaps,
        {"baseline": lr_sizes_before["baseline"], "weighted": lr_sizes_before["weighted"]},
    )
    report_path = REPORTS_DIR / "05_random_forest_baseline_report.md"
    report_path.write_text(report, encoding="utf-8")

    meta = {
        "model": "RandomForestClassifier",
        "rf_params": RF_PARAMS,
        "preprocessing": {
            "numeric": "passthrough (no scaling)",
            "categorical": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
        },
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "gaps_train_minus_test": gaps,
        "model_path": str(RF_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "prior_lr_models_unchanged": True,
        "classification_report_test": classification_report(
            y_test, y_test_pred, labels=CLASS_ORDER, digits=4
        ),
    }
    (REPORTS_DIR / "05_random_forest_baseline_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(report.encode("ascii", errors="replace").decode("ascii"))
    print("\nSaved RF:", RF_MODEL_PATH)
    print("LR models untouched.")


if __name__ == "__main__":
    main()
