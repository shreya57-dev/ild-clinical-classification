"""Generate notebooks/06_feature_ablation.ipynb"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md(
    """# 06 — Feature ablation (z-score vs %predicted redundancy)

Compare **full 13 features** vs **reduced 9 features** (drop `fev1_pp`, `fvc_pp`, `tlc_pp`, `tlco_pp`).

**Selection:** 5-fold Stratified CV on **training set only**.  
**Test set:** evaluated once after selection.  
**Model:** class-weighted Logistic Regression (no tuning).  
Do not overwrite previous models.
"""
)

md("## 1. Imports & setup")
code(
    """from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path("..").resolve()
if not (PROJECT_ROOT / "data" / "processed" / "train.csv").exists():
    PROJECT_ROOT = Path(".").resolve()

TARGET_COL = "diagnosis group"
CLASS_ORDER = ["IPF", "HP", "CTD", "1_SAR"]
FULL = [
    "fev1fvc_abs","fev1_z","fev1_pp","fvc_z","fvc_pp","fev1fvc_z",
    "tlc_z","tlc_pp","tlco_z","tlco_pp","sex","Height","Weight",
]
REDUCED = [
    "fev1fvc_abs","fev1_z","fvc_z","fev1fvc_z","tlc_z","tlco_z","sex","Height","Weight",
]
LR = dict(class_weight="balanced", solver="lbfgs", C=1.0, l1_ratio=0.0, max_iter=2000, random_state=42)
"""
)

md("## 2. Load train/test (do not recreate split)")
code(
    '''train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
assert len(train) == 4073 and len(test) == 1019
y_train = train[TARGET_COL]
y_test = test[TARGET_COL]
print(y_train.value_counts().reindex(CLASS_ORDER))
'''
)

md("## 3. Pipeline builder")
code(
    '''def build_pipeline(feats):
    num = [c for c in feats if c != "sex"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex"]),
    ], remainder="drop")
    return Pipeline([("preprocess", pre), ("clf", LogisticRegression(**LR))])
'''
)

md("## 4. Training-only 5-fold Stratified CV")
code(
    '''cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_cv(feats):
    fold_f1, fold_rec, fold_prec, fold_acc = [], [], [], []
    pc_f1 = {c: [] for c in CLASS_ORDER}
    pc_rec = {c: [] for c in CLASS_ORDER}
    X = train[feats]
    for tr, va in cv.split(X, y_train):
        model = build_pipeline(feats)
        model.fit(X.iloc[tr], y_train.iloc[tr])
        pred = model.predict(X.iloc[va])
        yt = y_train.iloc[va]
        fold_acc.append(accuracy_score(yt, pred))
        fold_prec.append(precision_score(yt, pred, average="macro", labels=CLASS_ORDER, zero_division=0))
        fold_rec.append(recall_score(yt, pred, average="macro", labels=CLASS_ORDER, zero_division=0))
        fold_f1.append(f1_score(yt, pred, average="macro", labels=CLASS_ORDER, zero_division=0))
        _, r, f, _ = precision_recall_fscore_support(yt, pred, labels=CLASS_ORDER, zero_division=0)
        for i, c in enumerate(CLASS_ORDER):
            pc_f1[c].append(f[i]); pc_rec[c].append(r[i])
    return {
        "fold_macro_f1": fold_f1,
        "mean_macro_f1": float(np.mean(fold_f1)),
        "std_macro_f1": float(np.std(fold_f1, ddof=1)),
        "mean_macro_recall": float(np.mean(fold_rec)),
        "mean_macro_precision": float(np.mean(fold_prec)),
        "mean_accuracy": float(np.mean(fold_acc)),
        "per_class_mean_f1": {c: float(np.mean(v)) for c, v in pc_f1.items()},
        "per_class_mean_recall": {c: float(np.mean(v)) for c, v in pc_rec.items()},
    }

full_cv = evaluate_cv(FULL)
red_cv = evaluate_cv(REDUCED)
print("Full mean macro F1", full_cv["mean_macro_f1"], "+/-", full_cv["std_macro_f1"])
print("Reduced mean macro F1", red_cv["mean_macro_f1"], "+/-", red_cv["std_macro_f1"])
print("Fold F1 full   ", [round(x,4) for x in full_cv["fold_macro_f1"]])
print("Fold F1 reduced", [round(x,4) for x in red_cv["fold_macro_f1"]])
'''
)

md("## 5. Select by CV macro F1 (no test peeking)")
code(
    '''if red_cv["mean_macro_f1"] >= full_cv["mean_macro_f1"]:
    selected, feats, model_name = "reduced_9", REDUCED, "logistic_reduced_features.joblib"
else:
    selected, feats, model_name = "full_13", FULL, "logistic_full_features_cv_selected.joblib"
print("CV-selected:", selected, feats)
'''
)

md("## 6. Final train refit + one test evaluation")
code(
    '''pipe = build_pipeline(feats)
pipe.fit(train[feats], y_train)
pred = pipe.predict(test[feats])
print("TEST accuracy", accuracy_score(y_test, pred))
print("TEST macro F1", f1_score(y_test, pred, average="macro", labels=CLASS_ORDER))
print("TEST macro recall", recall_score(y_test, pred, average="macro", labels=CLASS_ORDER))
p,r,f,s = precision_recall_fscore_support(y_test, pred, labels=CLASS_ORDER, zero_division=0)
print(pd.DataFrame({"class": CLASS_ORDER, "precision": p, "recall": r, "f1": f, "support": s}))
print(pd.DataFrame(confusion_matrix(y_test, pred, labels=CLASS_ORDER), index=CLASS_ORDER, columns=CLASS_ORDER))
out = PROJECT_ROOT / "models" / model_name
joblib.dump(pipe, out)
print("Saved", out)
'''
)

md("## Stop\nFeature ablation complete. No hyperparameter tuning.")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "06_feature_ablation.ipynb"
nbf.write(nb, out)
print("Wrote", out)
