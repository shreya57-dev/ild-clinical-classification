"""Generate notebooks/08_hist_gradient_boosting_baseline.ipynb"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md(
    """# 08 — Regularized HistGradientBoosting baseline

Controlled comparison vs class-weighted Logistic Regression using **training-only** Stratified 5-fold CV.

- Full 13 features
- Balanced **sample weights** computed within each fold's training split
- Fixed HGB hyperparameters (no tuning)
- Test set evaluated once after CV comparison
"""
)

md("## 1. Imports")
code(
    """from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
"""
)

md("## 2. Load data")
code(
    '''PROJECT_ROOT = Path("..").resolve()
if not (PROJECT_ROOT / "data" / "processed" / "train.csv").exists():
    PROJECT_ROOT = Path(".").resolve()
train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
assert len(train) == 4073 and len(test) == 1019
TARGET = "diagnosis group"
CLASSES = ["IPF", "HP", "CTD", "1_SAR"]
FEATS = [
    "fev1fvc_abs","fev1_z","fev1_pp","fvc_z","fvc_pp","fev1fvc_z",
    "tlc_z","tlc_pp","tlco_z","tlco_pp","sex","Height","Weight",
]
NUM = [c for c in FEATS if c != "sex"]
X_train, y_train = train[FEATS], train[TARGET]
X_test, y_test = test[FEATS], test[TARGET]
'''
)

md("## 3. Pipelines")
code(
    '''def hgb_pipe():
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", "passthrough", NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex"]),
        ], remainder="drop")),
        ("clf", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0, random_state=42,
        )),
    ])

def lr_pipe():
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex"]),
        ], remainder="drop")),
        ("clf", LogisticRegression(
            class_weight="balanced", solver="lbfgs", C=1.0, l1_ratio=0.0,
            max_iter=2000, random_state=42,
        )),
    ])
'''
)

md("## 4–7. Same-fold CV comparison")
code(
    '''cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
hgb_rows, lr_f1 = [], []
for fold, (tr, va) in enumerate(cv.split(X_train, y_train), 1):
    sw = compute_sample_weight("balanced", y_train.iloc[tr])
    h = hgb_pipe()
    h.fit(X_train.iloc[tr], y_train.iloc[tr], clf__sample_weight=sw)
    pred_tr = h.predict(X_train.iloc[tr])
    pred_va = h.predict(X_train.iloc[va])
    hgb_rows.append({
        "fold": fold,
        "train_macro_f1": f1_score(y_train.iloc[tr], pred_tr, average="macro", labels=CLASSES),
        "val_macro_f1": f1_score(y_train.iloc[va], pred_va, average="macro", labels=CLASSES),
        "val_accuracy": accuracy_score(y_train.iloc[va], pred_va),
        "val_macro_recall": recall_score(y_train.iloc[va], pred_va, average="macro", labels=CLASSES),
    })
    lr = lr_pipe()
    lr.fit(X_train.iloc[tr], y_train.iloc[tr])
    lr_pred = lr.predict(X_train.iloc[va])
    lr_f1.append(f1_score(y_train.iloc[va], lr_pred, average="macro", labels=CLASSES))

hgb_cv = pd.DataFrame(hgb_rows)
print(hgb_cv)
print("HGB mean val macro F1", hgb_cv.val_macro_f1.mean(), "gap", (hgb_cv.train_macro_f1 - hgb_cv.val_macro_f1).mean())
print("LR  mean val macro F1", np.mean(lr_f1))
'''
)

md("## 8. Final HGB fit + one test evaluation")
code(
    '''sw = compute_sample_weight("balanced", y_train)
model = hgb_pipe()
model.fit(X_train, y_train, clf__sample_weight=sw)
pred = model.predict(X_test)
print("test macro F1", f1_score(y_test, pred, average="macro", labels=CLASSES))
print("test accuracy", accuracy_score(y_test, pred))
p,r,f,s = precision_recall_fscore_support(y_test, pred, labels=CLASSES, zero_division=0)
print(pd.DataFrame({"class": CLASSES, "precision": p, "recall": r, "f1": f, "support": s}))
print(pd.DataFrame(confusion_matrix(y_test, pred, labels=CLASSES), index=CLASSES, columns=CLASSES))
print("feature_importances_ available:", hasattr(model.named_steps["clf"], "feature_importances_"))
out = PROJECT_ROOT / "models" / "hist_gradient_boosting_balanced.joblib"
joblib.dump(model, out)
print("Saved", out)
'''
)

md("## Stop\nControlled HGB baseline complete. No tuning.")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "08_hist_gradient_boosting_baseline.ipynb"
nbf.write(nb, out)
print("Wrote", out)
