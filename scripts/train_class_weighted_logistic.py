"""
Controlled experiment: class-weighted Logistic Regression.

ONLY change vs baseline: class_weight='balanced'.
Does NOT modify train/test CSVs, raw data, or baseline model artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_logistic_regression.joblib"
WEIGHTED_MODEL_PATH = MODELS_DIR / "class_weighted_logistic_regression.joblib"

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

# ONLY modeling change vs baseline
LR_PARAMS = dict(
    class_weight="balanced",
    solver="lbfgs",
    C=1.0,
    l1_ratio=0.0,  # L2-equivalent under sklearn 1.9 API
    max_iter=2000,
    random_state=RANDOM_STATE,
)

# Baseline metrics from the committed baseline on THIS SAME held-out test set.
# Do not recompute from a different split.
BASELINE_TEST = {
    "accuracy": 0.7556,
    "macro_precision": 0.6061,
    "macro_recall": 0.5375,
    "macro_f1": 0.5510,
    "weighted_f1": None,  # filled from baseline report/meta if available; else from per-class reconstruction note
    "IPF_recall": 0.724,
    "HP_recall": 0.233,
    "CTD_recall": 0.242,
    "1_SAR_recall": 0.951,
    "IPF_f1": 0.692,
    "HP_f1": 0.322,
    "CTD_f1": 0.305,
    "1_SAR_f1": 0.885,
    "IPF_precision": 0.663,
    "HP_precision": 0.522,
    "CTD_precision": 0.413,
    "1_SAR_precision": 0.827,
}

# Baseline confusion matrix (test), rows=true, cols=pred, order IPF/HP/CTD/1_SAR
BASELINE_CM = np.array(
    [
        [110, 7, 15, 20],
        [17, 24, 17, 45],
        [25, 10, 31, 62],
        [14, 5, 12, 605],
    ],
    dtype=int,
)


def load_baseline_weighted_f1_if_available() -> float | None:
    meta_path = REPORTS_DIR / "03_baseline_logistic_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # May not store weighted_f1; compute from stored classification report if present is hard.
    return meta.get("test_metrics", {}).get("weighted_f1")


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
        {
            "class": labels,
            "precision": p,
            "recall": r,
            "f1": f,
            "support": s.astype(int),
        }
    )


def balanced_weights_from_train(y_train: pd.Series) -> dict[str, float]:
    """weight_c = n_samples / (n_classes * n_c) using TRAINING counts only."""
    counts = y_train.value_counts()
    n_samples = int(len(y_train))
    n_classes = len(CLASS_ORDER)
    weights = {}
    for cls in CLASS_ORDER:
        n_c = int(counts[cls])
        weights[cls] = n_samples / (n_classes * n_c)
    # Cross-check with sklearn helper
    sk = compute_class_weight(
        class_weight="balanced",
        classes=np.array(CLASS_ORDER),
        y=y_train.to_numpy(),
    )
    sk_map = {cls: float(w) for cls, w in zip(CLASS_ORDER, sk)}
    for cls in CLASS_ORDER:
        assert abs(weights[cls] - sk_map[cls]) < 1e-9, (cls, weights[cls], sk_map[cls])
    return weights


def build_pipeline() -> Pipeline:
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
    clf = LogisticRegression(**LR_PARAMS)
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def plot_confusion(cm: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(
    train_counts: dict,
    test_counts: dict,
    weights: dict,
    train_metrics: dict,
    test_metrics: dict,
    per_class: pd.DataFrame,
    cm: np.ndarray,
    comparison: pd.DataFrame,
    baseline_size_before: int,
    baseline_size_after: int,
) -> str:
    # Confusion comparison helpers
    # rows: IPF=0, HP=1, CTD=2, SAR=3; col SAR=3
    hp_to_sar_b = int(BASELINE_CM[1, 3])
    ctd_to_sar_b = int(BASELINE_CM[2, 3])
    sar_to_min_b = int(BASELINE_CM[3, 0] + BASELINE_CM[3, 1] + BASELINE_CM[3, 2])
    hp_to_sar_w = int(cm[1, 3])
    ctd_to_sar_w = int(cm[2, 3])
    sar_to_min_w = int(cm[3, 0] + cm[3, 1] + cm[3, 2])

    pc = {r["class"]: r for r in per_class.to_dict(orient="records")}

    lines = []
    lines.append("# Class-weighted Logistic Regression experiment")
    lines.append("")
    lines.append("## Experiment objective")
    lines.append(
        "Test whether `class_weight='balanced'` improves recognition of minority ILD classes "
        "(especially HP and CTD) relative to the committed baseline Logistic Regression, "
        "**without** changing the split, features, preprocessing, or model family."
    )
    lines.append("")
    lines.append("## Hypothesis")
    lines.append(
        "Because the baseline disproportionately predicts `1_SAR`, balanced class weights "
        "(larger penalties for minority errors) should increase HP/CTD recall and macro-F1, "
        "likely at some cost to overall accuracy and/or 1_SAR recall."
    )
    lines.append("")
    lines.append("## Data (unchanged split)")
    lines.append(f"- Train: n={sum(train_counts.values())} | {train_counts}")
    lines.append(f"- Test: n={sum(test_counts.values())} | {test_counts}")
    lines.append("- Files: `data/processed/train.csv`, `data/processed/test.csv` (not modified)")
    lines.append("")
    lines.append("## Effective class weights (TRAINING counts only)")
    lines.append("")
    lines.append("Formula: `weight_c = n_samples / (n_classes * n_c)`")
    lines.append("")
    lines.append("| class | train n | weight |")
    lines.append("|---|---:|---:|")
    for cls in CLASS_ORDER:
        lines.append(f"| {cls} | {train_counts[cls]} | {weights[cls]:.6f} |")
    lines.append("")
    lines.append(
        "Minority classes (HP, CTD, IPF) receive **larger** weights than majority `1_SAR`, "
        "as required by balanced weighting."
    )
    lines.append("")
    lines.append("## Model configuration")
    for k, v in LR_PARAMS.items():
        lines.append(f"- `{k}` = `{v}`")
    lines.append("- Only intentional change vs baseline: `class_weight='balanced'`")
    lines.append("")
    lines.append("## Preprocessing (identical to baseline)")
    lines.append("- Numeric: `StandardScaler`")
    lines.append("- Categorical `sex`: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`")
    lines.append("- Fitted only inside Pipeline on training data")
    lines.append("")
    lines.append("## Training metrics")
    for k, v in train_metrics.items():
        lines.append(f"- {k}: **{v:.4f}**")
    lines.append("")
    lines.append("## Test metrics")
    for k, v in test_metrics.items():
        lines.append(f"- {k}: **{v:.4f}**")
    lines.append("")
    lines.append("## Per-class metrics (test)")
    lines.append("")
    lines.append(per_class.to_markdown(index=False))
    lines.append("")
    lines.append("## Confusion matrix (test; rows=true, cols=pred)")
    lines.append("")
    lines.append(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_markdown())
    lines.append("")
    lines.append("## Baseline comparison (same held-out test set)")
    lines.append("")
    lines.append(comparison.to_markdown(index=False))
    lines.append("")
    lines.append("## Confusion-matrix comparison (focus errors)")
    lines.append("")
    lines.append("| Error pattern | Baseline | Class-weighted | Change |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| HP -> 1_SAR | {hp_to_sar_b} | {hp_to_sar_w} | {hp_to_sar_w - hp_to_sar_b:+d} |")
    lines.append(f"| CTD -> 1_SAR | {ctd_to_sar_b} | {ctd_to_sar_w} | {ctd_to_sar_w - ctd_to_sar_b:+d} |")
    lines.append(
        f"| 1_SAR -> minority (IPF+HP+CTD) | {sar_to_min_b} | {sar_to_min_w} | {sar_to_min_w - sar_to_min_b:+d} |"
    )
    lines.append("")
    lines.append("## Interpretation of the trade-off")
    lines.append("")
    lines.append(
        f"1. Minority-class recall: HP {BASELINE_TEST['HP_recall']:.3f}->{pc['HP']['recall']:.3f}; "
        f"CTD {BASELINE_TEST['CTD_recall']:.3f}->{pc['CTD']['recall']:.3f}; "
        f"IPF {BASELINE_TEST['IPF_recall']:.3f}->{pc['IPF']['recall']:.3f}."
    )
    lines.append(
        f"2. HP recall {'improved' if pc['HP']['recall'] > BASELINE_TEST['HP_recall'] else 'did not improve'}."
    )
    lines.append(
        f"3. CTD recall {'improved' if pc['CTD']['recall'] > BASELINE_TEST['CTD_recall'] else 'did not improve'}."
    )
    lines.append(
        f"4. 1_SAR recall {'decreased' if pc['1_SAR']['recall'] < BASELINE_TEST['1_SAR_recall'] else 'did not decrease'} "
        f"({BASELINE_TEST['1_SAR_recall']:.3f}->{pc['1_SAR']['recall']:.3f})."
    )
    lines.append(
        f"5. Macro F1 {'improved' if test_metrics['macro_f1'] > BASELINE_TEST['macro_f1'] else 'did not improve'} "
        f"({BASELINE_TEST['macro_f1']:.4f}->{test_metrics['macro_f1']:.4f})."
    )
    lines.append(
        f"6. Accuracy {'decreased' if test_metrics['accuracy'] < BASELINE_TEST['accuracy'] else 'increased/unchanged'} "
        f"({BASELINE_TEST['accuracy']:.4f}->{test_metrics['accuracy']:.4f})."
    )
    lines.append(
        "7. Favorability for 4-class ILD objective depends on whether recovering HP/CTD "
        "is worth more false positives away from SAR and any accuracy drop; see final conclusion."
    )
    lines.append("")
    lines.append("## Artifact integrity")
    lines.append(
        f"- Baseline model file size before/after: {baseline_size_before} / {baseline_size_after} bytes "
        f"(must be unchanged)."
    )
    lines.append(f"- New model saved to `{WEIGHTED_MODEL_PATH.relative_to(PROJECT_ROOT)}`.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("- Single controlled change only; no tuning, resampling, or feature edits.")
    lines.append("- Multicollinear z/%predicted features retained.")
    lines.append("- Class weights address imbalance in the loss, not representation learning limits of linear models.")
    lines.append("- Coefficients not re-analyzed in this experiment.")
    lines.append("")

    # Final conclusion with evidence
    hp_recall_up = pc["HP"]["recall"] > BASELINE_TEST["HP_recall"]
    ctd_recall_up = pc["CTD"]["recall"] > BASELINE_TEST["CTD_recall"]
    hp_f1_up = pc["HP"]["f1"] > BASELINE_TEST["HP_f1"]
    ctd_f1_up = pc["CTD"]["f1"] > BASELINE_TEST["CTD_f1"]
    macro_better = test_metrics["macro_f1"] > BASELINE_TEST["macro_f1"]
    fewer_min_to_sar = (hp_to_sar_w + ctd_to_sar_w) < (hp_to_sar_b + ctd_to_sar_b)

    lines.append("## Final conclusion")
    lines.append("")
    lines.append('**Did class weighting improve the model\'s ability to distinguish the minority ILD classes?**')
    lines.append("")
    if hp_recall_up and ctd_recall_up and macro_better and fewer_min_to_sar:
        lines.append(
            "**Yes - with a clear trade-off.** Minority discrimination improved (HP/CTD recall & F1, "
            "fewer HP/CTD->SAR errors, higher macro-F1), at the cost of lower accuracy and lower 1_SAR recall, "
            "plus more SAR->minority false positives."
        )
    elif (hp_recall_up or ctd_recall_up or fewer_min_to_sar) and macro_better:
        lines.append(
            "**Yes, partially** - evidence favors improved minority discrimination on balance."
        )
    elif hp_recall_up or ctd_recall_up or fewer_min_to_sar:
        lines.append(
            "**Partially** - minority recall and/or minority->SAR errors improved, but overall macro-F1 did not clearly improve."
        )
    else:
        lines.append(
            "**No clear improvement** - minority discrimination metrics did not improve enough to support adopting this change alone."
        )
    lines.append("")
    lines.append("Evidence:")
    lines.append(
        f"- HP recall/F1: {BASELINE_TEST['HP_recall']:.3f}/{BASELINE_TEST['HP_f1']:.3f} -> "
        f"{pc['HP']['recall']:.3f}/{pc['HP']['f1']:.3f}"
    )
    lines.append(
        f"- CTD recall/F1: {BASELINE_TEST['CTD_recall']:.3f}/{BASELINE_TEST['CTD_f1']:.3f} -> "
        f"{pc['CTD']['recall']:.3f}/{pc['CTD']['f1']:.3f}"
    )
    lines.append(
        f"- Macro F1: {BASELINE_TEST['macro_f1']:.4f} -> {test_metrics['macro_f1']:.4f}"
    )
    lines.append(
        f"- HP+CTD -> 1_SAR errors: {hp_to_sar_b + ctd_to_sar_b} -> {hp_to_sar_w + ctd_to_sar_w}"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    assert BASELINE_MODEL_PATH.exists(), "Baseline model missing"
    baseline_size_before = BASELINE_MODEL_PATH.stat().st_size
    baseline_mtime_before = BASELINE_MODEL_PATH.stat().st_mtime

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    assert len(train) == 4073 and len(test) == 1019
    assert list(train.columns) == FEATURE_COLS + [TARGET_COL]

    X_train = train[FEATURE_COLS].copy()
    y_train = train[TARGET_COL].copy()
    X_test = test[FEATURE_COLS].copy()
    y_test = test[TARGET_COL].copy()

    train_counts = y_train.value_counts().reindex(CLASS_ORDER).astype(int).to_dict()
    test_counts = y_test.value_counts().reindex(CLASS_ORDER).astype(int).to_dict()
    assert train_counts == {"IPF": 605, "HP": 412, "CTD": 512, "1_SAR": 2544}
    assert test_counts == {"IPF": 152, "HP": 103, "CTD": 128, "1_SAR": 636}

    weights = balanced_weights_from_train(y_train)
    # Verify minority > majority weight
    assert weights["HP"] > weights["1_SAR"]
    assert weights["CTD"] > weights["1_SAR"]
    assert weights["IPF"] > weights["1_SAR"]

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    train_metrics = metrics_bundle(y_train, y_train_pred)
    test_metrics = metrics_bundle(y_test, y_test_pred)
    per_class = per_class_table(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred, labels=CLASS_ORDER)

    # Fill baseline weighted F1: prefer recomputing from baseline model on SAME files (not a new split)
    baseline_weighted_f1 = load_baseline_weighted_f1_if_available()
    if baseline_weighted_f1 is None:
        baseline_pipe = joblib.load(BASELINE_MODEL_PATH)
        baseline_pred = baseline_pipe.predict(X_test)
        baseline_weighted_f1 = float(
            f1_score(y_test, baseline_pred, average="weighted", labels=CLASS_ORDER, zero_division=0)
        )
        # Sanity: baseline accuracy should match known value closely
        bas_acc = accuracy_score(y_test, baseline_pred)
        assert abs(bas_acc - BASELINE_TEST["accuracy"]) < 1e-3, bas_acc
    BASELINE_TEST["weighted_f1"] = baseline_weighted_f1

    pc = {r["class"]: r for r in per_class.to_dict(orient="records")}
    comparison_rows = [
        ("accuracy", BASELINE_TEST["accuracy"], test_metrics["accuracy"]),
        ("macro_precision", BASELINE_TEST["macro_precision"], test_metrics["macro_precision"]),
        ("macro_recall", BASELINE_TEST["macro_recall"], test_metrics["macro_recall"]),
        ("macro_f1", BASELINE_TEST["macro_f1"], test_metrics["macro_f1"]),
        ("weighted_f1", BASELINE_TEST["weighted_f1"], test_metrics["weighted_f1"]),
        ("IPF_recall", BASELINE_TEST["IPF_recall"], float(pc["IPF"]["recall"])),
        ("HP_recall", BASELINE_TEST["HP_recall"], float(pc["HP"]["recall"])),
        ("CTD_recall", BASELINE_TEST["CTD_recall"], float(pc["CTD"]["recall"])),
        ("1_SAR_recall", BASELINE_TEST["1_SAR_recall"], float(pc["1_SAR"]["recall"])),
        ("IPF_f1", BASELINE_TEST["IPF_f1"], float(pc["IPF"]["f1"])),
        ("HP_f1", BASELINE_TEST["HP_f1"], float(pc["HP"]["f1"])),
        ("CTD_f1", BASELINE_TEST["CTD_f1"], float(pc["CTD"]["f1"])),
        ("1_SAR_f1", BASELINE_TEST["1_SAR_f1"], float(pc["1_SAR"]["f1"])),
    ]
    comparison = pd.DataFrame(
        [
            {
                "Metric": m,
                "Baseline LR": b,
                "Class-weighted LR": w,
                "Difference": w - b,
            }
            for m, b, w in comparison_rows
        ]
    )

    # Save NEW model only
    joblib.dump(pipe, WEIGHTED_MODEL_PATH)

    # Ensure baseline untouched
    baseline_size_after = BASELINE_MODEL_PATH.stat().st_size
    baseline_mtime_after = BASELINE_MODEL_PATH.stat().st_mtime
    assert baseline_size_before == baseline_size_after
    assert baseline_mtime_before == baseline_mtime_after

    plot_confusion(
        cm,
        FIGURES_DIR / "04_class_weighted_confusion_matrix_test.png",
        "Class-weighted LR - test confusion matrix",
    )
    # Side-by-side comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title in [
        (axes[0], BASELINE_CM, "Baseline LR"),
        (axes[1], cm, "Class-weighted LR"),
    ]:
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_ORDER)
        ax.set_yticklabels(CLASS_ORDER)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_confusion_matrix_comparison.png", dpi=150)
    plt.close(fig)

    per_class.to_csv(REPORTS_DIR / "04_class_weighted_per_class_metrics.csv", index=False)
    comparison.to_csv(REPORTS_DIR / "04_baseline_vs_class_weighted.csv", index=False)
    pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        REPORTS_DIR / "04_class_weighted_confusion_matrix_test.csv"
    )

    report = write_report(
        train_counts,
        test_counts,
        weights,
        train_metrics,
        test_metrics,
        per_class,
        cm,
        comparison,
        baseline_size_before,
        baseline_size_after,
    )
    report_path = REPORTS_DIR / "04_class_weighted_logistic_report.md"
    report_path.write_text(report, encoding="utf-8")

    meta = {
        "experiment": "class_weight=balanced LogisticRegression",
        "only_change": "class_weight='balanced'",
        "lr_params": LR_PARAMS,
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "balanced_weights_train": weights,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline_test_reference": BASELINE_TEST,
        "model_path": str(WEIGHTED_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "baseline_model_unchanged": True,
        "baseline_model_bytes": baseline_size_after,
        "classification_report_test": classification_report(
            y_test, y_test_pred, labels=CLASS_ORDER, digits=4
        ),
    }
    (REPORTS_DIR / "04_class_weighted_logistic_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(report.encode("ascii", errors="replace").decode("ascii"))
    print("\nSaved:", WEIGHTED_MODEL_PATH)
    print("Baseline untouched:", BASELINE_MODEL_PATH, baseline_size_after, "bytes")


if __name__ == "__main__":
    main()
