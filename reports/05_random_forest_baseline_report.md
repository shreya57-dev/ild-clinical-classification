# Random Forest baseline report (class_weight=balanced)

## 1. Objective
Establish whether a nonlinear tree ensemble can extract additional predictive signal from the same 13 features versus the two Logistic Regression baselines, using the same held-out test set and the same class-imbalance strategy (`class_weight='balanced'`).

## 2. Why Random Forest was selected
- Captures nonlinear interactions among PFT indices without manual feature engineering.
- Robust to feature scale (no StandardScaler required).
- Provides native multiclass support and feature-importance diagnostics.
- Remains interpretable enough for a prototype, unlike boosting/NN stacks.

## 3. Feature set (unchanged)
- `fev1fvc_abs`
- `fev1_z`
- `fev1_pp`
- `fvc_z`
- `fvc_pp`
- `fev1fvc_z`
- `tlc_z`
- `tlc_pp`
- `tlco_z`
- `tlco_pp`
- `sex`
- `Height`
- `Weight`
- Target: `diagnosis group`

## 4. Preprocessing
- Numerical: passthrough (no scaling)
- Categorical `sex`: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- ColumnTransformer + Pipeline; fitted on training data only

## 5. Model configuration
- `n_estimators` = `300`
- `class_weight` = `balanced`
- `random_state` = `42`
- `n_jobs` = `-1`
- No hyperparameter tuning

## 6. Training metrics
- accuracy: **0.9904**
- macro_precision: **0.9815**
- macro_recall: **0.9962**
- macro_f1: **0.9887**
- weighted_precision: **0.9907**
- weighted_recall: **0.9904**
- weighted_f1: **0.9904**

## 7. Test metrics
- accuracy: **0.7184**
- macro_precision: **0.5466**
- macro_recall: **0.5379**
- macro_f1: **0.5413**
- weighted_precision: **0.7143**
- weighted_recall: **0.7184**
- weighted_f1: **0.7159**

## 8. Per-class metrics (test)

| class   |   precision |   recall |       f1 |   support |
|:--------|------------:|---------:|---------:|----------:|
| IPF     |    0.602564 | 0.618421 | 0.61039  |       152 |
| HP      |    0.423529 | 0.349515 | 0.382979 |       103 |
| CTD     |    0.285714 | 0.296875 | 0.291188 |       128 |
| 1_SAR   |    0.874419 | 0.886792 | 0.880562 |       636 |

## 9. Confusion matrix (test; rows=true, cols=pred)

|       |   IPF |   HP |   CTD |   1_SAR |
|:------|------:|-----:|------:|--------:|
| IPF   |    94 |   14 |    29 |      15 |
| HP    |    13 |   36 |    27 |      27 |
| CTD   |    27 |   24 |    38 |      39 |
| 1_SAR |    22 |   11 |    39 |     564 |

### Focus error comparison

| Pattern | Unweighted LR | Weighted LR | Random Forest |
|---|---:|---:|---:|
| HP -> 1_SAR | 45 | 20 | 27 |
| CTD -> 1_SAR | 62 | 19 | 39 |
| 1_SAR -> minority | 31 | 121 | 72 |

## 10. Feature importance (model usage; not causal)

| feature          |   importance |
|:-----------------|-------------:|
| num__tlco_pp     |    0.133701  |
| num__tlco_z      |    0.107623  |
| num__fev1_z      |    0.0805668 |
| num__tlc_pp      |    0.0791497 |
| num__fev1fvc_z   |    0.074237  |
| num__fvc_z       |    0.0742341 |
| num__tlc_z       |    0.0727077 |
| num__fev1fvc_abs |    0.0722725 |
| num__fvc_pp      |    0.0718562 |
| num__Height      |    0.0708274 |
| num__fev1_pp     |    0.0694176 |
| num__Weight      |    0.0628623 |
| cat__sex_M       |    0.015961  |
| cat__sex_F       |    0.0145831 |

## 11. Comparison against Logistic Regression baselines

| Metric          |   Unweighted LR |   Class-weighted LR |   Class-weighted RF |
|:----------------|----------------:|--------------------:|--------------------:|
| accuracy        |          0.7556 |              0.713  |            0.718351 |
| macro_precision |          0.6061 |              0.5695 |            0.546557 |
| macro_recall    |          0.5375 |              0.6028 |            0.537901 |
| macro_f1        |          0.551  |              0.582  |            0.54128  |
| weighted_f1     |          0.7262 |              0.7292 |            0.715933 |
| IPF_recall      |          0.724  |              0.717  |            0.618421 |
| HP_recall       |          0.233  |              0.408  |            0.349515 |
| CTD_recall      |          0.242  |              0.477  |            0.296875 |
| 1_SAR_recall    |          0.951  |              0.81   |            0.886792 |
| IPF_f1          |          0.692  |              0.683  |            0.61039  |
| HP_f1           |          0.322  |              0.38   |            0.382979 |
| CTD_f1          |          0.305  |              0.401  |            0.291188 |
| 1_SAR_f1        |          0.885  |              0.863  |            0.880562 |

## 12. Overfitting analysis
- Train-test accuracy gap: +0.2721
- Train-test macro-F1 gap: +0.4474
- Large train-test gap indicates **potential overfitting** under the untuned RF configuration.

## 13. Limitations
- Untuned RF (default depth/split constraints can memorize training data).
- Correlated z/%predicted features retained.
- Feature importances are impurity-based associations, not causal effects.
- No calibration or thresholding explored.

## Artifact integrity
- Baseline LR unchanged: 4281 bytes
- Class-weighted LR unchanged: 4281 bytes
- New RF model: `models\random_forest_balanced.joblib`

## Final conclusions

1. Macro-F1 over both LR baselines? NO (RF=0.5413; unweighted LR=0.5510; weighted LR=0.5820).
2. Improve HP recall vs class-weighted LR? NO (0.408 -> 0.350).
3. Improve CTD recall vs class-weighted LR? NO (0.477 -> 0.297).
4. Reduce HP/CTD -> 1_SAR errors vs class-weighted LR? NO (39 -> 66).
5. Significant overfitting? YES (acc gap=+0.2721, macro-F1 gap=+0.4474).
6. Strongest current candidate: **Class-weighted Logistic Regression** (primary criterion: held-out macro-F1 = 0.5820), considering also minority-class recall and HP/CTD->SAR error patterns.
