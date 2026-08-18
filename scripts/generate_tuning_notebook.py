"""Generate notebooks/07_logistic_hyperparameter_tuning.ipynb"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md(
    """# 07 — Logistic Regression hyperparameter tuning (`C`)

Tune `C` with **training-set Stratified 5-fold CV only**.

- Features: full 13
- `class_weight='balanced'`
- Test set used **once** after selection
- Do not overwrite prior models
"""
)

md("## 1. Imports")
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
"""
)

md("## 2. Load existing split")
code(
    '''PROJECT_ROOT = Path("..").resolve()
if not (PROJECT_ROOT / "data" / "processed" / "train.csv").exists():
    PROJECT_ROOT = Path(".").resolve()

train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
assert len(train) == 4073 and len(test) == 1019

TARGET_COL = "diagnosis group"
CLASS_ORDER = ["IPF", "HP", "CTD", "1_SAR"]
FEATURE_COLS = [
    "fev1fvc_abs","fev1_z","fev1_pp","fvc_z","fvc_pp","fev1fvc_z",
    "tlc_z","tlc_pp","tlco_z","tlco_pp","sex","Height","Weight",
]
NUM = [c for c in FEATURE_COLS if c != "sex"]
X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]
print(y_train.value_counts().reindex(CLASS_ORDER))
'''
)

md("## 3–5. Pipeline + GridSearchCV on C")
code(
    '''pipe = Pipeline([
    ("preprocess", ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex"]),
    ], remainder="drop")),
    ("clf", LogisticRegression(
        class_weight="balanced", solver="lbfgs", C=1.0, l1_ratio=0.0,
        max_iter=2000, random_state=42,
    )),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipe,
    param_grid={"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    scoring={
        "macro_f1": "f1_macro",
        "accuracy": "accuracy",
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro",
    },
    refit="macro_f1",
    cv=cv,
)
search.fit(X_train, y_train)

rows = []
for i, c in enumerate(search.cv_results_["param_clf__C"]):
    rows.append({
        "C": float(c),
        "mean_macro_f1": search.cv_results_["mean_test_macro_f1"][i],
        "std_macro_f1": search.cv_results_["std_test_macro_f1"][i],
        "mean_accuracy": search.cv_results_["mean_test_accuracy"][i],
        "mean_macro_precision": search.cv_results_["mean_test_macro_precision"][i],
        "mean_macro_recall": search.cv_results_["mean_test_macro_recall"][i],
    })
cv_table = pd.DataFrame(rows).sort_values("mean_macro_f1", ascending=False)
print(cv_table)
print("Selected C:", search.best_params_["clf__C"], "CV macro F1:", search.best_score_)
'''
)

md("## 6. Final test evaluation (once)")
code(
    '''best = search.best_estimator_
y_tr = best.predict(X_train)
y_te = best.predict(X_test)
print("TRAIN acc/macroF1", accuracy_score(y_train, y_tr), f1_score(y_train, y_tr, average="macro", labels=CLASS_ORDER))
print("TEST  acc/macroF1", accuracy_score(y_test, y_te), f1_score(y_test, y_te, average="macro", labels=CLASS_ORDER))
p,r,f,s = precision_recall_fscore_support(y_test, y_te, labels=CLASS_ORDER, zero_division=0)
print(pd.DataFrame({"class": CLASS_ORDER, "precision": p, "recall": r, "f1": f, "support": s}))
print(pd.DataFrame(confusion_matrix(y_test, y_te, labels=CLASS_ORDER), index=CLASS_ORDER, columns=CLASS_ORDER))
'''
)

md("## 7. Save tuned model")
code(
    '''out = PROJECT_ROOT / "models" / "logistic_regression_tuned.joblib"
joblib.dump(best, out)
print("Saved", out)
'''
)

md("## Stop\nHyperparameter selection complete. No further tuning in this notebook.")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "07_logistic_hyperparameter_tuning.ipynb"
nbf.write(nb, out)
print("Wrote", out)
