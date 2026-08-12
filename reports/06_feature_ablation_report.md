# Feature ablation report

## 1. Objective
Determine whether removing highly correlated percent-predicted (`*_pp`) PFT variables improves, worsens, or approximately preserves class-weighted Logistic Regression performance, using training-set CV only for selection.

## 2. Why feature redundancy is being investigated
Prior correlation analysis showed |r| > 0.98 for several z-score vs %predicted pairs. Redundant representations can inflate coefficient instability and model complexity. Correlation alone does **not** prove a feature is useless; ablation measures predictive impact.

## 3. Full feature set (13)
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

## 4. Reduced feature set (9)
Removed: `fev1_pp`, `fvc_pp`, `tlc_pp`, `tlco_pp`

Retained:
- `fev1fvc_abs`
- `fev1_z`
- `fvc_z`
- `fev1fvc_z`
- `tlc_z`
- `tlco_z`
- `sex`
- `Height`
- `Weight`

## 5. CV methodology
- Data: training set only (n=4073); test set held out until after selection
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Same folds for both feature sets
- Model: class-weighted Logistic Regression (C=1.0, lbfgs, max_iter=2000)
- Preprocessing inside Pipeline: StandardScaler + OneHotEncoder(sex)
- Primary metric: macro F1

## 6. Fold-by-fold macro F1

| Fold | Full 13 | Reduced 9 | Difference (red - full) |
|---:|---:|---:|---:|
| 1 | 0.5696 | 0.5729 | +0.0033 |
| 2 | 0.5740 | 0.5685 | -0.0055 |
| 3 | 0.5553 | 0.5543 | -0.0010 |
| 4 | 0.5652 | 0.5555 | -0.0097 |
| 5 | 0.5706 | 0.5698 | -0.0008 |

## 7. Mean / std CV results

| Metric | Full 13 features | Reduced 9 features | Difference (red - full) |
|---|---:|---:|---:|
| mean CV macro F1 | 0.5669 | 0.5642 | -0.0027 |
| std CV macro F1 | 0.0072 | 0.0087 | +0.0014 |
| mean CV macro recall | 0.5878 | 0.5860 | -0.0018 |
| mean CV macro precision | 0.5567 | 0.5532 | -0.0035 |
| mean CV accuracy | 0.7024 | 0.6997 | -0.0027 |

### Per-class mean CV recall

| Class | Full 13 | Reduced 9 | Diff |
|---|---:|---:|---:|
| IPF | 0.6992 | 0.6926 | -0.0066 |
| HP | 0.4201 | 0.4225 | +0.0024 |
| CTD | 0.4277 | 0.4277 | +0.0000 |
| 1_SAR | 0.8042 | 0.8011 | -0.0031 |

### Per-class mean CV F1

| Class | Full 13 | Reduced 9 | Diff |
|---|---:|---:|---:|
| IPF | 0.6573 | 0.6429 | -0.0143 |
| HP | 0.3920 | 0.3968 | +0.0048 |
| CTD | 0.3569 | 0.3574 | +0.0005 |
| 1_SAR | 0.8615 | 0.8595 | -0.0019 |

## 8. Selected feature set
**CV-selected representation: `full_13_features`**

Features used for final refit:
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

The ablation did **not** improve the representation; full features were preferred by CV.

## 9. Final untouched test performance

Fitted on all 4073 training patients after selection; evaluated once on 1019-patient test set.

- accuracy: **0.7134**
- macro_precision: **0.5695**
- macro_recall: **0.6028**
- macro_f1: **0.5819**

### Per-class test metrics

| class   |   precision |   recall |       f1 |   support |
|:--------|------------:|---------:|---------:|----------:|
| IPF     |    0.652695 | 0.717105 | 0.683386 |       152 |
| HP      |    0.355932 | 0.407767 | 0.38009  |       103 |
| CTD     |    0.346591 | 0.476562 | 0.401316 |       128 |
| 1_SAR   |    0.922939 | 0.809748 | 0.862647 |       636 |

### Test confusion matrix

|       |   IPF |   HP |   CTD |   1_SAR |
|:------|------:|-----:|------:|--------:|
| IPF   |   109 |   17 |    22 |       4 |
| HP    |    18 |   42 |    23 |      20 |
| CTD   |    27 |   21 |    61 |      19 |
| 1_SAR |    13 |   38 |    70 |     515 |

## 10. Comparison with previous best (class-weighted LR, full features)

| Metric | Previous best | Ablation-selected | Diff |
|---|---:|---:|---:|
| accuracy | 0.7130 | 0.7134 | +0.0004 |
| macro_f1 | 0.5820 | 0.5819 | -0.0001 |
| macro_recall | 0.6028 | 0.6028 | -0.0000 |
| IPF_recall | 0.7170 | 0.7171 | +0.0001 |
| HP_recall | 0.4080 | 0.4078 | -0.0002 |
| CTD_recall | 0.4770 | 0.4766 | -0.0004 |
| 1_SAR_recall | 0.8100 | 0.8097 | -0.0003 |
| IPF_f1 | 0.6830 | 0.6834 | +0.0004 |
| HP_f1 | 0.3800 | 0.3801 | +0.0001 |
| CTD_f1 | 0.4010 | 0.4013 | +0.0003 |
| 1_SAR_f1 | 0.8630 | 0.8626 | -0.0004 |

## 11. Interpretation
1. Removing `*_pp` variables **approximately preserves** mean CV macro F1 (full=0.5669, reduced=0.5642, diff=-0.0027).
2. Fold stability: reduced wins 1/5 folds; full wins 4/5 folds; std macro F1 full=0.0072, reduced=0.0087.
3. From a simplicity/interpretability perspective, the reduced set is preferable if CV performance is similar or better, because it removes near-collinear %predicted duplicates of z-scores.
4. Little evidence that removed `*_pp` variables add substantial unique signal beyond z-scores under this linear model; CV macro F1 is essentially unchanged.

## 12. Limitations
- Only one ablation (drop all four `*_pp`); other subsets not tested.
- Linear model only; nonlinear models might use redundant features differently.
- No hyperparameter retuning after feature change.
- Test set used once after selection; not for choosing features.

## Artifact integrity
- `baseline_logistic_regression.joblib`: 4281 bytes (unchanged)
- `class_weighted_logistic_regression.joblib`: 4281 bytes (unchanged)
- `random_forest_balanced.joblib`: 46865874 bytes (unchanged)
