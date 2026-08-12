# Baseline Logistic Regression report

First baseline only. No hyperparameter tuning. No other models.

## Preprocessing
- Numerical: `StandardScaler` (zero mean / unit variance), fitted on training fold only via Pipeline
- Categorical (`sex`): `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- No imputer (0 missing values in modeling features)
- ColumnTransformer + Pipeline so transform parameters never see the test set during fit

## Logistic Regression configuration
- `solver` = `lbfgs`
- `C` = `1.0`
- `l1_ratio` = `0.0`
- `max_iter` = `2000`
- `random_state` = `42`

## Data checks
- PASS: row counts 4073 / 1019
- PASS: expected columns present
- PASS: train has all 4 classes
- PASS: train has no leakage columns
- PASS: train has 0 missing values in features
- PASS: test has all 4 classes
- PASS: test has no leakage columns
- PASS: test has 0 missing values in features

## Training metrics
- accuracy: **0.7609**
- macro_precision: **0.6039**
- macro_recall: **0.5478**
- macro_f1: **0.5591**

## Test metrics
- accuracy: **0.7556**
- macro_precision: **0.6061**
- macro_recall: **0.5375**
- macro_f1: **0.5510**

## Majority-class baseline
- Always predict `1_SAR`
- Test accuracy: **0.6241**
- Accuracy alone is insufficient: predicting always 1_SAR yields ~62% test accuracy while completely failing IPF/HP/CTD. Macro-F1 / per-class metrics are required.

## Train-test gap
- accuracy: +0.0052 (train - test)
- macro_precision: -0.0022 (train - test)
- macro_recall: +0.0103 (train - test)
- macro_f1: +0.0082 (train - test)

## Confusion matrix (test, rows=true, cols=predicted, order IPF/HP/CTD/1_SAR)

|       |   IPF |   HP |   CTD |   1_SAR |
|:------|------:|-----:|------:|--------:|
| IPF   |   110 |    7 |    15 |      20 |
| HP    |    17 |   24 |    17 |      45 |
| CTD   |    25 |   10 |    31 |      62 |
| 1_SAR |    14 |    5 |    12 |     605 |

## Per-class metrics (test)

| class   |   precision |   recall |       f1 |   support |
|:--------|------------:|---------:|---------:|----------:|
| IPF     |    0.662651 | 0.723684 | 0.691824 |       152 |
| HP      |    0.521739 | 0.23301  | 0.322148 |       103 |
| CTD     |    0.413333 | 0.242188 | 0.305419 |       128 |
| 1_SAR   |    0.826503 | 0.951258 | 0.884503 |       636 |

## Coefficient highlights (association with class log-odds; not causal)

### IPF
Top positive:
- `num__fev1fvc_z`: 2.4779
- `num__fev1_pp`: 1.0342
- `num__fvc_z`: 0.8622
- `num__fev1_z`: 0.6215
- `num__Weight`: 0.2627
Top negative:
- `num__fev1fvc_abs`: -2.9222
- `num__fvc_pp`: -1.5626
- `num__tlco_pp`: -1.0353
- `num__tlc_pp`: -0.5340
- `num__Height`: -0.4349

### HP
Top positive:
- `num__fev1fvc_abs`: 1.1618
- `num__fvc_pp`: 0.6650
- `num__tlc_pp`: 0.5860
- `num__fev1_pp`: 0.1326
- `num__Weight`: 0.1249
Top negative:
- `num__fev1_z`: -1.3547
- `num__fev1fvc_z`: -0.3725
- `num__tlco_pp`: -0.3402
- `num__fvc_z`: -0.2197
- `num__Height`: -0.2155

### CTD
Top positive:
- `num__fev1_pp`: 0.6363
- `num__fev1fvc_z`: 0.4718
- `cat__sex_F`: 0.3547
- `num__fvc_z`: 0.2424
- `num__tlc_pp`: 0.2003
Top negative:
- `num__fev1_z`: -0.5733
- `num__fvc_pp`: -0.5023
- `num__fev1fvc_abs`: -0.4568
- `cat__sex_M`: -0.3687
- `num__Weight`: -0.1633

### 1_SAR
Top positive:
- `num__fev1fvc_abs`: 2.2171
- `num__tlco_pp`: 1.4348
- `num__fvc_pp`: 1.3999
- `num__fev1_z`: 1.3065
- `num__Height`: 0.6562
Top negative:
- `num__fev1fvc_z`: -2.5773
- `num__fev1_pp`: -1.8031
- `num__fvc_z`: -0.8849
- `num__tlc_pp`: -0.2523
- `num__Weight`: -0.2242

## Limitations / concerns
- Highly correlated z/%predicted pairs retained (multicollinearity can inflate coefficient variance).
- Class imbalance (1_SAR majority) remains; no class_weight applied yet.
- Weight=8 anomaly retained.
- No baseline age available after leakage exclusion.
- Coefficients are associations under the model, not clinical causal effects.
