# HistGradientBoosting baseline report (balanced sample weights)

## 1. Objective
Determine whether a deliberately regularized nonlinear boosting model can capture useful interactions among PFT features without the severe overfitting seen with untuned Random Forest, and whether it improves on class-weighted Logistic Regression under training-only CV.

## 2. Why Gradient Boosting was selected
- Additive trees can model nonlinear interactions among correlated PFT indices.
- Built-in regularization knobs (learning rate, leaf limits, L2) to limit RF-style memorization.
- Still a classical ML baseline suitable for a prototype before any tuning.

## 3. Difference between Random Forest and Gradient Boosting
- Random Forest averages many deep independent trees (high capacity; easy to overfit if unconstrained).
- HistGradientBoosting builds shallow trees sequentially to correct residuals, with explicit learning-rate / leaf / L2 constraints that encourage smoother generalization.

## 4. Feature set
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

## 5. Preprocessing
- Numerical: passthrough (no StandardScaler)
- Categorical `sex`: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
- No imputation (0 missing values)

## 6. Class-weighting / sample-weight methodology
- Formula: `weight_c = n_samples / (n_classes * n_c)`
- Implemented via `compute_sample_weight(class_weight='balanced')`
- Weights computed from **fold training labels only** (or full train labels for final fit)
- Passed as `clf__sample_weight` to Pipeline.fit

## 7. Model configuration
- `max_iter` = `200`
- `learning_rate` = `0.05`
- `max_leaf_nodes` = `15`
- `min_samples_leaf` = `30`
- `l2_regularization` = `1.0`
- `random_state` = `42`
- No hyperparameter tuning

## 8. 5-fold CV methodology
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Training set only (n=4073); test held out
- Same folds for HGB and Logistic Regression comparison
- Primary metric: macro F1

## 9. Fold-level results (HGB)

|   fold |   train_accuracy |   train_macro_f1 |   val_accuracy |   val_macro_f1 |   val_macro_precision |   val_macro_recall |   val_weighted_f1 |
|-------:|-----------------:|-----------------:|---------------:|---------------:|----------------------:|-------------------:|------------------:|
|      1 |         0.93892  |         0.920992 |       0.710429 |       0.540746 |              0.536055 |           0.546892 |          0.71892  |
|      2 |         0.93892  |         0.92386  |       0.728834 |       0.544997 |              0.547221 |           0.543213 |          0.728106 |
|      3 |         0.943217 |         0.928655 |       0.707975 |       0.550405 |              0.542969 |           0.559676 |          0.715995 |
|      4 |         0.940166 |         0.924107 |       0.733415 |       0.569052 |              0.559661 |           0.582002 |          0.741548 |
|      5 |         0.939552 |         0.924152 |       0.727273 |       0.572541 |              0.56295  |           0.58527  |          0.73618  |

## 10. Mean / std CV results (HGB validation)
- mean validation macro F1: **0.5555**
- std validation macro F1: **0.0144**
- mean validation accuracy: **0.7216**
- mean validation macro precision: **0.5498**
- mean validation macro recall: **0.5634**
- mean validation weighted F1: **0.7281**

### Per-class mean validation recall / F1

| Class | mean val recall | mean val F1 |
|---|---:|---:|
| IPF | 0.6413 | 0.6190 |
| HP | 0.3884 | 0.3755 |
| CTD | 0.3556 | 0.3379 |
| 1_SAR | 0.8683 | 0.8898 |

## 11. Train-validation gap (HGB)
- mean train macro F1: **0.9244**
- mean validation macro F1: **0.5555**
- mean train-validation macro F1 gap: **+0.3688**

## 12. Comparison with weighted Logistic Regression (same CV folds)

| Model | Mean CV Macro F1 | Std CV Macro F1 | Mean CV Accuracy | Mean CV Macro Recall |
|---|---:|---:|---:|---:|
| Class-weighted Logistic Regression | 0.5669 | 0.0072 | 0.7024 | 0.5878 |
| Regularized HistGradientBoosting | 0.5555 | 0.0144 | 0.7216 | 0.5634 |

CV macro F1 difference (HGB - LR): **-0.0114**
**Configuration selected for final test evaluation: `class_weighted_logistic_regression`**

## 13. Final test performance
Fitted on all 4073 training patients with balanced sample weights from training labels only; evaluated once on the untouched 1019-patient test set.

- accuracy: **0.7223**
- macro_precision: **0.5422**
- macro_recall: **0.5492**
- macro_f1: **0.5452**
- weighted_f1: **0.7245**
- HP recall: **0.3689**
- CTD recall: **0.3125**
- 1_SAR recall: **0.8836**

### Full-train vs test macro F1
- Full-train macro F1: 0.8940
- Test macro F1: 0.5452
- Gap: +0.3488

## 14. Per-class metrics (test)

| class   |   precision |   recall |       f1 |   support |
|:--------|------------:|---------:|---------:|----------:|
| IPF     |    0.611465 | 0.631579 | 0.621359 |       152 |
| HP      |    0.324786 | 0.368932 | 0.345455 |       103 |
| CTD     |    0.333333 | 0.3125   | 0.322581 |       128 |
| 1_SAR   |    0.8992   | 0.883648 | 0.891356 |       636 |

## 15. Confusion matrix (test)

|       |   IPF |   HP |   CTD |   1_SAR |
|:------|------:|-----:|------:|--------:|
| IPF   |    96 |   25 |    24 |       7 |
| HP    |    16 |   38 |    20 |      29 |
| CTD   |    30 |   31 |    40 |      27 |
| 1_SAR |    15 |   23 |    36 |     562 |

### Focus error comparison

| Pattern | Unweighted LR | Weighted LR | Random Forest | HGB |
|---|---:|---:|---:|---:|
| HP -> 1_SAR | 45 | 20 | 27 | 29 |
| CTD -> 1_SAR | 62 | 19 | 39 | 27 |
| 1_SAR -> minority | 31 | 121 | 72 | 74 |

## 16. Overfitting analysis
- CV mean train macro F1=0.9244; val=0.5555; gap=+0.3688
- Full-train macro F1=0.8940; test=0.5452; gap=+0.3488
- Random Forest reference: train macro F1=0.989, test macro F1=0.541 (gap ~0.45)
- Assessment for HGB based on CV gap: **severe overfitting**

## Comparison vs previous best test metrics (Weighted LR)

| Metric | Weighted LR (prior test) | HGB test | Diff |
|---|---:|---:|---:|
| accuracy | 0.7134 | 0.7223 | +0.0089 |
| macro_f1 | 0.5819 | 0.5452 | -0.0367 |
| HP_recall | 0.4080 | 0.3689 | -0.0391 |
| CTD_recall | 0.4770 | 0.3125 | -0.1645 |
| 1_SAR_recall | 0.8100 | 0.8836 | +0.0736 |

## 17. Feature importance
Native `feature_importances_` is **unavailable** for `HistGradientBoostingClassifier` in sklearn 1.9.0. No alternative importance method was added in this experiment.

## 18. Limitations
- Single untuned HGB configuration; no grid search.
- Sample weights approximate class balancing but differ from LR class_weight internals.
- Correlated z/%predicted features retained.
- Native feature importances unavailable in this sklearn build.

## Artifact integrity
- `baseline_logistic_regression.joblib`: 4281 bytes (unchanged)
- `class_weighted_logistic_regression.joblib`: 4281 bytes (unchanged)
- `logistic_full_features_cv_selected.joblib`: 4281 bytes (unchanged)
- `logistic_regression_tuned.joblib`: 4281 bytes (unchanged)
- `random_forest_balanced.joblib`: 46865874 bytes (unchanged)

## 19. Final conclusion

1. Outperform weighted LR in training-only CV? NO (HGB=0.5555, LR=0.5669).
2. Improve HP recall vs prior weighted LR test? NO (0.408 -> 0.369).
3. Improve CTD recall vs prior weighted LR test? NO (0.477 -> 0.312).
4. Reduce HP/CTD -> 1_SAR errors vs weighted LR? NO (39 -> 56).
5. Overfit substantially? YES (substantial) (CV train-val macro F1 gap=+0.3688).
6. Stronger candidate than class-weighted LR? NO (CV decision + held-out macro F1 0.5452 vs LR 0.5819).
7. Next step: **Retain class-weighted Logistic Regression as the primary candidate; do not prioritize HGB tuning yet**.
