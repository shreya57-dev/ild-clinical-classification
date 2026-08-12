"""Generate notebooks/05_random_forest_baseline.ipynb"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s))


md(
    """# 05 — Class-weighted Random Forest baseline

Controlled comparison vs:
1. Unweighted Logistic Regression
2. Class-weighted Logistic Regression

**Unchanged:** train/test split, 13 features, target, `class_weight='balanced'` strategy.  
**Changed:** model family to Random Forest; **no** StandardScaler (trees do not need it).  
**No tuning.** Do not overwrite LR model artifacts.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
"""
)

md("## 2. Load existing split")
code(
    '''PROJECT_ROOT = Path("..").resolve()
if not (PROJECT_ROOT / "data" / "processed" / "train.csv").exists():
    PROJECT_ROOT = Path(".").resolve()

train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
TARGET_COL = "diagnosis group"
CLASS_ORDER = ["IPF", "HP", "CTD", "1_SAR"]
FEATURE_COLS = [
    "fev1fvc_abs", "fev1_z", "fev1_pp", "fvc_z", "fvc_pp", "fev1fvc_z",
    "tlc_z", "tlc_pp", "tlco_z", "tlco_pp", "sex", "Height", "Weight",
]
NUM_FEATURES = [c for c in FEATURE_COLS if c != "sex"]
CAT_FEATURES = ["sex"]

assert len(train) == 4073 and len(test) == 1019
assert list(train.columns) == FEATURE_COLS + [TARGET_COL]
print(train[TARGET_COL].value_counts().reindex(CLASS_ORDER))
print(test[TARGET_COL].value_counts().reindex(CLASS_ORDER))

baseline_lr = PROJECT_ROOT / "models" / "baseline_logistic_regression.joblib"
weighted_lr = PROJECT_ROOT / "models" / "class_weighted_logistic_regression.joblib"
b_before, w_before = baseline_lr.stat().st_size, weighted_lr.stat().st_size
'''
)

md("## 3–4. Preprocessing + Random Forest pipeline")
code(
    '''X_train = train[FEATURE_COLS].copy()
y_train = train[TARGET_COL].copy()
X_test = test[FEATURE_COLS].copy()
y_test = test[TARGET_COL].copy()

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ],
    remainder="drop",
)
pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])
pipe.fit(X_train, y_train)
print("Fitted on train only.")
'''
)

md("## 5. Evaluate")
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

y_tr = pipe.predict(X_train)
y_te = pipe.predict(X_test)
train_metrics, test_metrics = bundle(y_train, y_tr), bundle(y_test, y_te)
print("TRAIN", {k: round(v, 4) for k, v in train_metrics.items()})
print("TEST ", {k: round(v, 4) for k, v in test_metrics.items()})
print("GAP accuracy", train_metrics["accuracy"] - test_metrics["accuracy"])
print("GAP macro_f1", train_metrics["macro_f1"] - test_metrics["macro_f1"])

p, r, f, s = precision_recall_fscore_support(y_test, y_te, labels=CLASS_ORDER, zero_division=0)
per_class = pd.DataFrame({"class": CLASS_ORDER, "precision": p, "recall": r, "f1": f, "support": s.astype(int)})
print(per_class)
cm = confusion_matrix(y_test, y_te, labels=CLASS_ORDER)
print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))
'''
)

md("## 6. Compare to recorded LR baselines")
code(
    '''UNWEIGHTED_LR = dict(accuracy=0.7556, macro_precision=0.6061, macro_recall=0.5375, macro_f1=0.5510,
    weighted_f1=0.7262, IPF_recall=0.724, HP_recall=0.233, CTD_recall=0.242, **{"1_SAR_recall": 0.951},
    IPF_f1=0.692, HP_f1=0.322, CTD_f1=0.305, **{"1_SAR_f1": 0.885})
WEIGHTED_LR = dict(accuracy=0.713, macro_precision=0.5695, macro_recall=0.6028, macro_f1=0.582,
    weighted_f1=0.7292, IPF_recall=0.717, HP_recall=0.408, CTD_recall=0.477, **{"1_SAR_recall": 0.810},
    IPF_f1=0.683, HP_f1=0.380, CTD_f1=0.401, **{"1_SAR_f1": 0.863})
pc = per_class.set_index("class")
rf = dict(
    accuracy=test_metrics["accuracy"], macro_precision=test_metrics["macro_precision"],
    macro_recall=test_metrics["macro_recall"], macro_f1=test_metrics["macro_f1"],
    weighted_f1=test_metrics["weighted_f1"],
    IPF_recall=float(pc.loc["IPF","recall"]), HP_recall=float(pc.loc["HP","recall"]),
    CTD_recall=float(pc.loc["CTD","recall"]), **{"1_SAR_recall": float(pc.loc["1_SAR","recall"])},
    IPF_f1=float(pc.loc["IPF","f1"]), HP_f1=float(pc.loc["HP","f1"]),
    CTD_f1=float(pc.loc["CTD","f1"]), **{"1_SAR_f1": float(pc.loc["1_SAR","f1"])},
)
metrics = ["accuracy","macro_precision","macro_recall","macro_f1","weighted_f1",
           "IPF_recall","HP_recall","CTD_recall","1_SAR_recall","IPF_f1","HP_f1","CTD_f1","1_SAR_f1"]
comparison = pd.DataFrame({
    "Metric": metrics,
    "Unweighted LR": [UNWEIGHTED_LR[m] for m in metrics],
    "Class-weighted LR": [WEIGHTED_LR[m] for m in metrics],
    "Class-weighted RF": [rf[m] for m in metrics],
})
comparison
'''
)

md("## 7. Confusion focus + feature importance")
code(
    '''print("HP->SAR", cm[1,3], "CTD->SAR", cm[2,3], "SAR->minority", cm[3,:3].sum())
names = pipe.named_steps["preprocess"].get_feature_names_out()
imp = pd.DataFrame({"feature": names, "importance": pipe.named_steps["clf"].feature_importances_})
imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
imp
'''
)

md("## 8. Save RF model only")
code(
    '''out = PROJECT_ROOT / "models" / "random_forest_balanced.joblib"
joblib.dump(pipe, out)
assert baseline_lr.stat().st_size == b_before
assert weighted_lr.stat().st_size == w_before
print("Saved", out)
print("LR artifacts unchanged.")
print(classification_report(y_test, y_te, labels=CLASS_ORDER, digits=4))
'''
)

md("## Stop\nSingle controlled RF baseline experiment complete. No tuning.")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "05_random_forest_baseline.ipynb"
nbf.write(nb, out)
print("Wrote", out)
