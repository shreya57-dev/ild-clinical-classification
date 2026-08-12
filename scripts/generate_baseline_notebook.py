"""Generate notebooks/03_baseline_logistic_regression.ipynb (does not overwrite other notebooks)."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s))


md(
    """# 03 — Baseline Logistic Regression

First interpretable baseline for 4-class ILD classification (`IPF`, `HP`, `CTD`, `1_SAR`).

**Rules for this stage:**
- Train only Logistic Regression (no RF/XGBoost/NN/LLM)
- No hyperparameter tuning
- No feature dropping / outlier removal
- Fit preprocessing **only** on training data via a Pipeline
- Do not modify `train.csv` / `test.csv`
"""
)

md("## 1. Imports")
code(
    """from pathlib import Path
import json

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

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)
"""
)

md("## 2. Load train/test data")
code(
    '''PROJECT_ROOT = Path("..").resolve()
if not (PROJECT_ROOT / "data" / "processed" / "train.csv").exists():
    PROJECT_ROOT = Path(".").resolve()

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "diagnosis group"
CLASS_ORDER = ["IPF", "HP", "CTD", "1_SAR"]
LEAKAGE_COLS = [
    "Case_number", "First test", "event_date",
    "dead or alive", "survival time", "age at event",
]

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print("train", train.shape)
print("test", test.shape)
print(train.head(3))
'''
)

md("## 3. Define X / y and verify integrity")
code(
    '''NUM_FEATURES = [
    "fev1fvc_abs", "fev1_z", "fev1_pp",
    "fvc_z", "fvc_pp", "fev1fvc_z",
    "tlc_z", "tlc_pp", "tlco_z", "tlco_pp",
    "Height", "Weight",
]
CAT_FEATURES = ["sex"]
# Order matches train.csv / test.csv
FEATURE_COLS = [
    "fev1fvc_abs", "fev1_z", "fev1_pp",
    "fvc_z", "fvc_pp", "fev1fvc_z",
    "tlc_z", "tlc_pp", "tlco_z", "tlco_pp",
    "sex", "Height", "Weight",
]

assert len(train) == 4073 and len(test) == 1019
assert list(train.columns) == FEATURE_COLS + [TARGET_COL]
assert list(test.columns) == FEATURE_COLS + [TARGET_COL]
for df_name, df in [("train", train), ("test", test)]:
    assert set(df[TARGET_COL].unique()) == set(CLASS_ORDER)
    for leak in LEAKAGE_COLS:
        assert leak not in df.columns
    assert df[FEATURE_COLS].isna().sum().sum() == 0
    print(df_name, "OK")

X_train = train[FEATURE_COLS].copy()
y_train = train[TARGET_COL].copy()
X_test = test[FEATURE_COLS].copy()
y_test = test[TARGET_COL].copy()
print("X_train", X_train.shape, "X_test", X_test.shape)
'''
)

md(
    """## 4–6. Preprocessing + Logistic Regression pipeline

**Numerical:** `StandardScaler`  
**Categorical:** `OneHotEncoder(handle_unknown="ignore")`  
**No imputer** (0 missing values).

All transformers are fitted only inside `pipeline.fit(X_train, y_train)`.
"""
)
code(
    '''RANDOM_STATE = 42
# sklearn 1.9: L2 via l1_ratio=0; multinomial multiclass is default for lbfgs.
LR_PARAMS = dict(
    solver="lbfgs",
    C=1.0,
    l1_ratio=0.0,
    max_iter=2000,
    random_state=RANDOM_STATE,
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ],
    remainder="drop",
)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(**LR_PARAMS)),
])
pipe
'''
)

md("## 7. Train")
code(
    '''pipe.fit(X_train, y_train)
print("Fitted classes:", list(pipe.named_steps["clf"].classes_))
'''
)

md("## 8. Evaluate (train + test)")
code(
    '''def metrics_bundle(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0),
    }

y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

train_metrics = metrics_bundle(y_train, y_train_pred)
test_metrics = metrics_bundle(y_test, y_test_pred)
gaps = {k: train_metrics[k] - test_metrics[k] for k in train_metrics}

print("TRAIN:", {k: round(v, 4) for k, v in train_metrics.items()})
print("TEST :", {k: round(v, 4) for k, v in test_metrics.items()})
print("GAP  :", {k: round(v, 4) for k, v in gaps.items()})
'''
)

md(
    """## 9. Majority-class baseline

Always predict `1_SAR` (majority in training). Accuracy can look decent under imbalance while per-class performance is zero for minority classes.
"""
)
code(
    '''majority = y_train.value_counts().idxmax()
majority_pred = np.full(len(y_test), majority)
majority_acc = accuracy_score(y_test, majority_pred)
print(f"Majority class: {majority}")
print(f"Majority baseline test accuracy: {majority_acc:.4f}")
print(f"LR test accuracy: {test_metrics['accuracy']:.4f}")
print(f"LR test macro-F1: {test_metrics['macro_f1']:.4f}")
'''
)

md("## 10. Confusion matrix (test)")
code(
    '''cm = confusion_matrix(y_test, y_test_pred, labels=CLASS_ORDER)
cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
print(cm_df)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CLASS_ORDER)))
ax.set_yticks(range(len(CLASS_ORDER)))
ax.set_xticklabels(CLASS_ORDER)
ax.set_yticklabels(CLASS_ORDER)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Baseline LR — test confusion matrix")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "03_baseline_confusion_matrix_test.png", dpi=150)
plt.show()
'''
)

md("## 11. Per-class metrics (test)")
code(
    '''p, r, f, s = precision_recall_fscore_support(
    y_test, y_test_pred, labels=CLASS_ORDER, zero_division=0
)
per_class = pd.DataFrame({
    "class": CLASS_ORDER,
    "precision": p,
    "recall": r,
    "f1": f,
    "support": s.astype(int),
})
print(per_class)
print("\\nClassification report:\\n")
print(classification_report(y_test, y_test_pred, labels=CLASS_ORDER, digits=4))
'''
)

md(
    """## 12–13. Coefficient analysis & interpretation

Coefficients are **associations** with the model's predicted class log-odds after scaling/encoding.  
They are **not** causal clinical effects.
"""
)
code(
    '''feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
clf = pipe.named_steps["clf"]
rows = []
for i, cls in enumerate(clf.classes_):
    for name, coef in zip(feature_names, clf.coef_[i]):
        rows.append({"class": cls, "feature": name, "coefficient": coef, "abs_coefficient": abs(coef)})
coef_df = pd.DataFrame(rows)

for cls in CLASS_ORDER:
    sub = coef_df[coef_df["class"] == cls].sort_values("coefficient", ascending=False)
    print("=" * 60)
    print(cls)
    print("Top positive:")
    print(sub.head(5)[["feature", "coefficient"]].to_string(index=False))
    print("Top negative:")
    print(sub.tail(5).sort_values("coefficient")[["feature", "coefficient"]].to_string(index=False))
'''
)

md(
    """## 14. Limitations

- Correlated z-score / %predicted pairs retained → multicollinearity can make individual coefficients unstable.
- Severe class imbalance (1_SAR majority); no `class_weight` yet.
- No usable baseline age after excluding `age at event`.
- Weight=8 anomaly retained.
- This is an untuned first baseline only.
"""
)

md("## Save fitted pipeline")
code(
    '''model_path = MODELS_DIR / "baseline_logistic_regression.joblib"
joblib.dump(pipe, model_path)
print("Saved full pipeline (preprocess + model) to", model_path)

meta = {
    "lr_params": LR_PARAMS,
    "train_metrics": {k: float(v) for k, v in train_metrics.items()},
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    "majority_class": str(majority),
    "majority_test_accuracy": float(majority_acc),
    "model_path": str(model_path.relative_to(PROJECT_ROOT)),
}
(REPORTS_DIR / "03_baseline_logistic_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
per_class.to_csv(REPORTS_DIR / "03_baseline_per_class_metrics.csv", index=False)
coef_df.to_csv(REPORTS_DIR / "03_baseline_coefficients.csv", index=False)
print("Done. No hyperparameter tuning performed.")
'''
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "03_baseline_logistic_regression.ipynb"
assert not out.exists() or True  # may regenerate this notebook only
nbf.write(nb, out)
print("Wrote", out)
