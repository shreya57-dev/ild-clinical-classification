# Logistic Regression hyperparameter tuning (C)

## 1. Objective
Select the Logistic Regression regularization strength `C` using training-set Stratified 5-fold CV only, then evaluate the chosen configuration once on the held-out test set.

## 2. Why C is being tuned
`C` controls inverse regularization strength. With fixed `class_weight='balanced'` and the full 13-feature representation, varying `C` can reduce underfitting (too strong regularization) or overfitting (too weak regularization) without changing the model family.

## 3. Regularization and C
- Smaller `C` => stronger L2 regularization => simpler decision boundaries.
- Larger `C` => weaker regularization => model fits training data more closely.
- sklearn LogisticRegression with `lbfgs` uses L2-equivalent regularization (`l1_ratio=0`).

## 4. CV methodology
- Training data only: n=4073
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Primary metric: `f1_macro`
- Same folds for every `C` via `GridSearchCV`
- Test set (n=1019) excluded from selection

## 5. C values tested
0.001, 0.01, 0.1, 1, 10, 100

## 6. CV results

|       C |   mean_macro_f1 |   std_macro_f1 |   mean_accuracy |   mean_macro_precision |   mean_macro_recall |
|--------:|----------------:|---------------:|----------------:|-----------------------:|--------------------:|
|   0.001 |        0.533889 |     0.0107622  |        0.686225 |               0.529131 |            0.543434 |
|   0.01  |        0.548454 |     0.00581161 |        0.692856 |               0.540296 |            0.563575 |
|   0.1   |        0.570276 |     0.00715383 |        0.703413 |               0.560144 |            0.59046  |
|   1     |        0.56692  |     0.00646914 |        0.702432 |               0.556694 |            0.587803 |
|  10     |        0.570848 |     0.00907524 |        0.706605 |               0.55999  |            0.592363 |
| 100     |        0.573445 |     0.00605347 |        0.708571 |               0.562148 |            0.595324 |

### Ranking by mean CV macro F1

|       C |   mean_macro_f1 |   std_macro_f1 |
|--------:|----------------:|---------------:|
| 100     |        0.573445 |     0.00605347 |
|  10     |        0.570848 |     0.00907524 |
|   0.1   |        0.570276 |     0.00715383 |
|   1     |        0.56692  |     0.00646914 |
|   0.01  |        0.548454 |     0.00581161 |
|   0.001 |        0.533889 |     0.0107622  |

Several C values are close; selected C=100.0 by highest mean CV macro F1. Tiny differences should not be overinterpreted.
Very small C (0.001) mean macro F1=0.5339 vs C=1 (0.5669) and C=100 (0.5734).

## 7. Selected C
**Selected C = 100.0** (highest mean CV macro F1 = 0.5734).

## 8. Final test performance
Refit on all 4073 training patients; evaluated once on untouched test set.

- accuracy: **0.7115**
- macro_precision: **0.5660**
- macro_recall: **0.5978**
- macro_f1: **0.5780**
- weighted_f1: **0.7263**

## 9. Comparison with previous best (C=1.0)

| Metric | Previous best (C=1) | Tuned | Diff |
|---|---:|---:|---:|
| accuracy | 0.7134 | 0.7115 | -0.0019 |
| macro_f1 | 0.5819 | 0.5780 | -0.0039 |
| macro_recall | 0.6028 | 0.5978 | -0.0050 |
| IPF_recall | 0.7170 | 0.7303 | +0.0133 |
| HP_recall | 0.4080 | 0.3981 | -0.0099 |
| CTD_recall | 0.4770 | 0.4531 | -0.0239 |
| 1_SAR_recall | 0.8100 | 0.8097 | -0.0003 |
| IPF_f1 | 0.6830 | 0.6894 | +0.0064 |
| HP_f1 | 0.3800 | 0.3779 | -0.0021 |
| CTD_f1 | 0.4010 | 0.3841 | -0.0169 |
| 1_SAR_f1 | 0.8630 | 0.8605 | -0.0025 |

## 10. Per-class results (test)

| class   |   precision |   recall |       f1 |   support |
|:--------|------------:|---------:|---------:|----------:|
| IPF     |    0.652941 | 0.730263 | 0.689441 |       152 |
| HP      |    0.359649 | 0.398058 | 0.37788  |       103 |
| CTD     |    0.333333 | 0.453125 | 0.384106 |       128 |
| 1_SAR   |    0.918004 | 0.809748 | 0.860485 |       636 |

## 11. Confusion matrix (test)

|       |   IPF |   HP |   CTD |   1_SAR |
|:------|------:|-----:|------:|--------:|
| IPF   |   111 |   15 |    22 |       4 |
| HP    |    18 |   41 |    24 |      20 |
| CTD   |    28 |   20 |    58 |      22 |
| 1_SAR |    13 |   38 |    70 |     515 |

## 12. Overfitting analysis
- Train accuracy: 0.7103
- Test accuracy: 0.7115
- Train macro F1: 0.5760
- Test macro F1: 0.5780
- Gaps (train - test): accuracy -0.0012, macro F1 -0.0019
- Train-test gap is modest for this linear model; no strong overfitting signal.

## 13. Limitations
- Only `C` was tuned; other LR settings held fixed.
- Grid is coarse; intermediate C values not searched.
- Selection used macro F1 only; clinical cost trade-offs not optimized.
- Test set used once after selection.

## Artifact integrity
- `baseline_logistic_regression.joblib`: 4281 bytes (unchanged)
- `class_weighted_logistic_regression.joblib`: 4281 bytes (unchanged)
- `logistic_full_features_cv_selected.joblib`: 4281 bytes (unchanged)
- `random_forest_balanced.joblib`: 46865874 bytes (unchanged)

## Final conclusions

1. Did tuning C improve macro-F1? NO meaningful change (0.5819 -> 0.5780).
2. Improve HP recall? NO (0.408 -> 0.398).
3. Improve CTD recall? NO (0.477 -> 0.453).
4. Improve overall generalization? NO clear gain (judge primarily by held-out macro F1 and train-test gap).
5. Is the tuned model meaningfully better than C=1? NO - performance is essentially equivalent to C=1.
6. Current best candidate: **Class-weighted Logistic Regression with C=1.0 remains the simplest best candidate (tuned C not meaningfully better)**.
