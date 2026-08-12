"""Generate notebooks/04_class_weighted_logistic_regression.ipynb"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s))


md(
    """# 04 — Class-weighted Logistic Regression (controlled experiment)

**Only change vs baseline:** `LogisticRegression(class_weight="balanced")`

Unchanged:
- train/test split files
- feature set
- preprocessing
- solver / C / max_iter / random_state
- no tuning, no resampling, no other models

Baseline model artifact must remain untouched.
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
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
"""
)

md("## 2. Load existing train/test split (do not recreate)")
code(
    '''PROJECT_ROOT = Path("..").resolve()
if not (PROJECT_ROOT / "data" / "processed" / "train.csv").exists():
    PROJECT_ROOT = Path(".").resolve()

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
BASELINE_MODEL = MODELS_DIR / "baseline_logistic_regression.joblib"

TARGET_COL = "diagnosis group"
CLASS_ORDER = ["IPF", "HP", "CTD", "1_SAR"]
FEATURE_COLS = [
    "fev1fvc_abs", "fev1_z", "fev1_pp", "fvc_z", "fvc_pp", "fev1fvc_z",
    "tlc_z", "tlc_pp", "tlco_z", "tlco_pp", "sex", "Height", "Weight",
]
NUM_FEATURES = [c for c in FEATURE_COLS if c != "sex"]
CAT_FEATURES = ["sex"]

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
assert len(train) == 4073 and len(test) == 1019
print(train[TARGET_COL].value_counts().reindex(CLASS_ORDER))
print(test[TARGET_COL].value_counts().reindex(CLASS_ORDER))
print("Baseline model exists:", BASELINE_MODEL.exists(), "bytes:", BASELINE_MODEL.stat().st_size)
baseline_bytes_before = BASELINE_MODEL.stat().st_size
'''
)

md("## 3. Define X / y")
code(
    '''X_train = train[FEATURE_COLS].copy()
y_train = train[TARGET_COL].copy()
X_test = test[FEATURE_COLS].copy()
y_test = test[TARGET_COL].copy()
'''
)

md(
    """## 4. Effective class weights (training counts only)

`weight_c = n_samples / (n_classes * n_c)`
"""
)
code(
    '''n_samples = len(y_train)
n_classes = 4
counts = y_train.value_counts()
weights = {c: n_samples / (n_classes * int(counts[c])) for c in CLASS_ORDER}
sk = compute_class_weight("balanced", classes=np.array(CLASS_ORDER), y=y_train.to_numpy())
print(pd.DataFrame({
    "class": CLASS_ORDER,
    "train_n": [int(counts[c]) for c in CLASS_ORDER],
    "weight_formula": [weights[c] for c in CLASS_ORDER],
    "sklearn_compute_class_weight": list(sk),
}))
print("Minority weights > SAR:", weights["HP"] > weights["1_SAR"], weights["CTD"] > weights["1_SAR"])
'''
)

md("## 5–6. Same preprocessing + class-weighted LR pipeline")
code(
    '''preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ],
    remainder="drop",
)
pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        C=1.0,
        l1_ratio=0.0,
        max_iter=2000,
        random_state=42,
    )),
])
pipe.fit(X_train, y_train)
print("Fitted.")
'''
)

md("## 7. Evaluate")
code(
    '''def bundle(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", labels=CLASS_ORDER, zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", labels=CLASS_ORDER, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=CLASS_ORDER, zero_division=0),
    }

y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)
train_metrics = bundle(y_train, y_train_pred)
test_metrics = bundle(y_test, y_test_pred)
print("TRAIN", {k: round(v, 4) for k, v in train_metrics.items()})
print("TEST ", {k: round(v, 4) for k, v in test_metrics.items()})

p, r, f, s = precision_recall_fscore_support(y_test, y_test_pred, labels=CLASS_ORDER, zero_division=0)
per_class = pd.DataFrame({"class": CLASS_ORDER, "precision": p, "recall": r, "f1": f, "support": s.astype(int)})
print(per_class)
cm = confusion_matrix(y_test, y_test_pred, labels=CLASS_ORDER)
print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))
'''
)

md("## 8. Direct comparison with committed baseline metrics")
code(
    '''BASELINE = {
    "accuracy": 0.7556, "macro_precision": 0.6061, "macro_recall": 0.5375, "macro_f1": 0.5510,
    "IPF_recall": 0.724, "HP_recall": 0.233, "CTD_recall": 0.242, "1_SAR_recall": 0.951,
    "IPF_f1": 0.692, "HP_f1": 0.322, "CTD_f1": 0.305, "1_SAR_f1": 0.885,
}
# weighted F1 from baseline model on the SAME test file
base_pipe = joblib.load(BASELINE_MODEL)
base_pred = base_pipe.predict(X_test)
BASELINE["weighted_f1"] = f1_score(y_test, base_pred, average="weighted", labels=CLASS_ORDER, zero_division=0)

pc = per_class.set_index("class")
rows = [
    ("accuracy", BASELINE["accuracy"], test_metrics["accuracy"]),
    ("macro_precision", BASELINE["macro_precision"], test_metrics["macro_precision"]),
    ("macro_recall", BASELINE["macro_recall"], test_metrics["macro_recall"]),
    ("macro_f1", BASELINE["macro_f1"], test_metrics["macro_f1"]),
    ("weighted_f1", BASELINE["weighted_f1"], test_metrics["weighted_f1"]),
    ("IPF_recall", BASELINE["IPF_recall"], float(pc.loc["IPF", "recall"])),
    ("HP_recall", BASELINE["HP_recall"], float(pc.loc["HP", "recall"])),
    ("CTD_recall", BASELINE["CTD_recall"], float(pc.loc["CTD", "recall"])),
    ("1_SAR_recall", BASELINE["1_SAR_recall"], float(pc.loc["1_SAR", "recall"])),
    ("IPF_f1", BASELINE["IPF_f1"], float(pc.loc["IPF", "f1"])),
    ("HP_f1", BASELINE["HP_f1"], float(pc.loc["HP", "f1"])),
    ("CTD_f1", BASELINE["CTD_f1"], float(pc.loc["CTD", "f1"])),
    ("1_SAR_f1", BASELINE["1_SAR_f1"], float(pc.loc["1_SAR", "f1"])),
]
comparison = pd.DataFrame([
    {"Metric": m, "Baseline LR": b, "Class-weighted LR": w, "Difference": w - b}
    for m, b, w in rows
])
comparison
'''
)

md("## 9. Confusion matrix comparison")
code(
    '''BASELINE_CM = np.array([[110,7,15,20],[17,24,17,45],[25,10,31,62],[14,5,12,605]])
print("HP->SAR:", BASELINE_CM[1,3], "->", cm[1,3])
print("CTD->SAR:", BASELINE_CM[2,3], "->", cm[2,3])
print("SAR->minority:", BASELINE_CM[3,:3].sum(), "->", cm[3,:3].sum())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, mat, title in [(axes[0], BASELINE_CM, "Baseline"), (axes[1], cm, "Class-weighted")]:
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_ORDER); ax.set_yticklabels(CLASS_ORDER)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center")
plt.tight_layout(); plt.show()
'''
)

md("## 10. Save weighted model (do not overwrite baseline)")
code(
    '''out = MODELS_DIR / "class_weighted_logistic_regression.joblib"
joblib.dump(pipe, out)
assert BASELINE_MODEL.stat().st_size == baseline_bytes_before
print("Saved", out)
print("Baseline unchanged:", BASELINE_MODEL, baseline_bytes_before, "bytes")
print(classification_report(y_test, y_test_pred, labels=CLASS_ORDER, digits=4))
'''
)

md(
    """## Stop

Single controlled experiment complete. No further optimization in this notebook.
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "04_class_weighted_logistic_regression.ipynb"
nbf.write(nb, out)
print("Wrote", out)
