"""
Train and evaluate a baseline multiclass Logistic Regression model.

Does NOT modify train/test CSVs or raw data.
Does NOT perform hyperparameter tuning or try other models.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET_COL = "diagnosis group"
CLASS_ORDER = ["IPF", "HP", "CTD", "1_SAR"]
LEAKAGE_COLS = [
    "Case_number",
    "First test",
    "event_date",
    "dead or alive",
    "survival time",
    "age at event",
]

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
# Column order matches data/processed/train.csv and test.csv
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

# Baseline Logistic Regression configuration (no tuning)
# sklearn 1.9: L2 regularization via default (l1_ratio=0); avoid deprecated penalty/multi_class args.
LR_PARAMS = dict(
    solver="lbfgs",
    C=1.0,
    l1_ratio=0.0,  # L2-equivalent under the new API
    max_iter=2000,
    random_state=RANDOM_STATE,
)


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def verify_split(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    checks = []
    assert len(train) == 4073, f"train rows={len(train)}"
    assert len(test) == 1019, f"test rows={len(test)}"
    checks.append("PASS: row counts 4073 / 1019")

    expected_cols = FEATURE_COLS + [TARGET_COL]
    assert list(train.columns) == expected_cols, train.columns.tolist()
    assert list(test.columns) == expected_cols, test.columns.tolist()
    checks.append("PASS: expected columns present")

    for name, df in [("train", train), ("test", test)]:
        assert set(df[TARGET_COL].unique()) == set(CLASS_ORDER)
        checks.append(f"PASS: {name} has all 4 classes")
        for leak in LEAKAGE_COLS:
            assert leak not in df.columns
        checks.append(f"PASS: {name} has no leakage columns")
        assert df[FEATURE_COLS].isna().sum().sum() == 0
        checks.append(f"PASS: {name} has 0 missing values in features")
    return checks


def build_pipeline() -> Pipeline:
    # No imputation: audit found 0 missing values in the modeling features.
    # StandardScaler + OneHotEncoder are fitted only via pipeline.fit(X_train, y_train).
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
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", clf),
        ]
    )


def metrics_bundle(y_true, y_pred, labels=CLASS_ORDER) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
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


def majority_baseline_accuracy(y_train, y_test) -> tuple[str, float]:
    majority = y_train.value_counts().idxmax()
    preds = np.full(shape=len(y_test), fill_value=majority)
    return str(majority), float(accuracy_score(y_test, preds))


def get_feature_names(pipe: Pipeline) -> np.ndarray:
    return pipe.named_steps["preprocess"].get_feature_names_out()


def coefficient_table(pipe: Pipeline) -> pd.DataFrame:
    clf = pipe.named_steps["clf"]
    names = get_feature_names(pipe)
    # classes_ order from the fitted classifier
    classes = list(clf.classes_)
    rows = []
    for i, cls in enumerate(classes):
        coefs = clf.coef_[i]
        for name, coef in zip(names, coefs):
            rows.append(
                {
                    "class": cls,
                    "feature": name,
                    "coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                }
            )
    return pd.DataFrame(rows)


def top_coefficients(coef_df: pd.DataFrame, k: int = 5) -> dict:
    out = {}
    for cls in CLASS_ORDER:
        sub = coef_df[coef_df["class"] == cls].sort_values("coefficient", ascending=False)
        out[cls] = {
            "top_positive": sub.head(k)[["feature", "coefficient"]].to_dict(orient="records"),
            "top_negative": sub.tail(k).sort_values("coefficient")[["feature", "coefficient"]].to_dict(
                orient="records"
            ),
        }
    return out


def plot_confusion(cm: np.ndarray, labels, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(
    checks: list[str],
    train_metrics: dict,
    test_metrics: dict,
    majority_class: str,
    majority_acc: float,
    cm: np.ndarray,
    per_class: pd.DataFrame,
    top_coef: dict,
    gaps: dict,
) -> str:
    lines = []
    lines.append("# Baseline Logistic Regression report")
    lines.append("")
    lines.append("First baseline only. No hyperparameter tuning. No other models.")
    lines.append("")
    lines.append("## Preprocessing")
    lines.append("- Numerical: `StandardScaler` (zero mean / unit variance), fitted on training fold only via Pipeline")
    lines.append("- Categorical (`sex`): `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`")
    lines.append("- No imputer (0 missing values in modeling features)")
    lines.append("- ColumnTransformer + Pipeline so transform parameters never see the test set during fit")
    lines.append("")
    lines.append("## Logistic Regression configuration")
    for k, v in LR_PARAMS.items():
        lines.append(f"- `{k}` = `{v}`")
    lines.append("")
    lines.append("## Data checks")
    for c in checks:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## Training metrics")
    for k, v in train_metrics.items():
        lines.append(f"- {k}: **{v:.4f}**")
    lines.append("")
    lines.append("## Test metrics")
    for k, v in test_metrics.items():
        lines.append(f"- {k}: **{v:.4f}**")
    lines.append("")
    lines.append("## Majority-class baseline")
    lines.append(f"- Always predict `{majority_class}`")
    lines.append(f"- Test accuracy: **{majority_acc:.4f}**")
    lines.append(
        "- Accuracy alone is insufficient: predicting always 1_SAR yields ~62% test accuracy "
        "while completely failing IPF/HP/CTD. Macro-F1 / per-class metrics are required."
    )
    lines.append("")
    lines.append("## Train-test gap")
    for k, v in gaps.items():
        lines.append(f"- {k}: {v:+.4f} (train - test)")
    lines.append("")
    lines.append("## Confusion matrix (test, rows=true, cols=predicted, order IPF/HP/CTD/1_SAR)")
    lines.append("")
    cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
    lines.append(cm_df.to_markdown())
    lines.append("")
    lines.append("## Per-class metrics (test)")
    lines.append("")
    lines.append(per_class.to_markdown(index=False))
    lines.append("")
    lines.append("## Coefficient highlights (association with class log-odds; not causal)")
    lines.append("")
    for cls, d in top_coef.items():
        lines.append(f"### {cls}")
        lines.append("Top positive:")
        for r in d["top_positive"]:
            lines.append(f"- `{r['feature']}`: {r['coefficient']:.4f}")
        lines.append("Top negative:")
        for r in d["top_negative"]:
            lines.append(f"- `{r['feature']}`: {r['coefficient']:.4f}")
        lines.append("")
    lines.append("## Limitations / concerns")
    lines.append("- Highly correlated z/%predicted pairs retained (multicollinearity can inflate coefficient variance).")
    lines.append("- Class imbalance (1_SAR majority) remains; no class_weight applied yet.")
    lines.append("- Weight=8 anomaly retained.")
    lines.append("- No baseline age available after leakage exclusion.")
    lines.append("- Coefficients are associations under the model, not clinical causal effects.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split(TRAIN_PATH)
    test = load_split(TEST_PATH)
    checks = verify_split(train, test)

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
    gaps = {k: train_metrics[k] - test_metrics[k] for k in train_metrics}

    majority_class, majority_acc = majority_baseline_accuracy(y_train, y_test)
    cm = confusion_matrix(y_test, y_test_pred, labels=CLASS_ORDER)
    per_class = per_class_table(y_test, y_test_pred)
    coef_df = coefficient_table(pipe)
    top_coef = top_coefficients(coef_df, k=5)

    # Save artifacts
    model_path = MODELS_DIR / "baseline_logistic_regression.joblib"
    joblib.dump(pipe, model_path)

    plot_confusion(
        cm,
        CLASS_ORDER,
        FIGURES_DIR / "03_baseline_confusion_matrix_test.png",
        "Baseline LR - test confusion matrix",
    )

    per_class.to_csv(REPORTS_DIR / "03_baseline_per_class_metrics.csv", index=False)
    coef_df.to_csv(REPORTS_DIR / "03_baseline_coefficients.csv", index=False)
    pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        REPORTS_DIR / "03_baseline_confusion_matrix_test.csv"
    )

    report = write_report(
        checks,
        train_metrics,
        test_metrics,
        majority_class,
        majority_acc,
        cm,
        per_class,
        top_coef,
        gaps,
    )
    report_path = REPORTS_DIR / "03_baseline_logistic_report.md"
    report_path.write_text(report, encoding="utf-8")

    meta = {
        "model": "LogisticRegression",
        "lr_params": LR_PARAMS,
        "preprocessing": {
            "numeric": "StandardScaler",
            "categorical": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "imputer": None,
        },
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "gaps_train_minus_test": gaps,
        "majority_class": majority_class,
        "majority_test_accuracy": majority_acc,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "checks": checks,
        "classification_report_test": classification_report(
            y_test, y_test_pred, labels=CLASS_ORDER, digits=4
        ),
    }
    (REPORTS_DIR / "03_baseline_logistic_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(report.encode("ascii", errors="replace").decode("ascii"))
    print("\nSaved model:", model_path)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
