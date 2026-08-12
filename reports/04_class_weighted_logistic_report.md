# Class-weighted Logistic Regression experiment

## Experiment objective
Test whether `class_weight='balanced'` improves recognition of minority ILD classes (especially HP and CTD) relative to the committed baseline Logistic Regression, **without** changing the split, features, preprocessing, or model family.

## Hypothesis
Because the baseline disproportionately predicts `1_SAR`, balanced class weights (larger penalties for minority errors) should increase HP/CTD recall and macro-F1, likely at some cost to overall accuracy and/or 1_SAR recall.

## Data (unchanged split)
- Train: n=4073 | {'IPF': 605, 'HP': 412, 'CTD': 512, '1_SAR': 2544}
- Test: n=1019 | {'IPF': 152, 'HP': 103, 'CTD': 128, '1_SAR': 636}
- Files: `data/processed/train.csv`, `data/processed/test.csv` (not modified)

## Effective class weights (TRAINING counts only)

Formula: `weight_c = n_samples / (n_classes * n_c)`

| class | train n | weight |
|---|---:|---:|
| IPF | 605 | 1.683058 |
| HP | 412 | 2.471481 |
| CTD | 512 | 1.988770 |
| 1_SAR | 2544 | 0.400256 |

Minority classes (HP, CTD, IPF) receive **larger** weights than majority `1_SAR`, as required by balanced weighting.

## Model configuration
- `class_weight` = `balanced`
- `solver` = `lbfgs`
- `C` = `1.0`
- `l1_ratio` = `0.0`
- `max_iter` = `2000`
- `random_state` = `42`
- Only intentional change vs baseline: `class_weight='balanced'`

## Preprocessing (identical to baseline)
- Numeric: `StandardScaler`
- Categorical `sex`: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- Fitted only inside Pipeline on training data

## Training metrics
- accuracy: **0.7073**
- macro_precision: **0.5614**
- macro_recall: **0.5958**
- macro_f1: **0.5740**
- weighted_precision: **0.7506**
- weighted_recall: **0.7073**
- weighted_f1: **0.7245**

## Test metrics
- accuracy: **0.7134**
- macro_precision: **0.5695**
- macro_recall: **0.6028**
- macro_f1: **0.5819**
- weighted_precision: **0.7529**
- weighted_recall: **0.7134**
- weighted_f1: **0.7292**

## Per-class metrics (test)

| class   |   precision |   recall |       f1 |   support |
|:--------|------------:|---------:|---------:|----------:|
| IPF     |    0.652695 | 0.717105 | 0.683386 |       152 |
| HP      |    0.355932 | 0.407767 | 0.38009  |       103 |
| CTD     |    0.346591 | 0.476562 | 0.401316 |       128 |
| 1_SAR   |    0.922939 | 0.809748 | 0.862647 |       636 |

## Confusion matrix (test; rows=true, cols=pred)

|       |   IPF |   HP |   CTD |   1_SAR |
|:------|------:|-----:|------:|--------:|
| IPF   |   109 |   17 |    22 |       4 |
| HP    |    18 |   42 |    23 |      20 |
| CTD   |    27 |   21 |    61 |      19 |
| 1_SAR |    13 |   38 |    70 |     515 |

## Baseline comparison (same held-out test set)

| Metric          |   Baseline LR |   Class-weighted LR |   Difference |
|:----------------|--------------:|--------------------:|-------------:|
| accuracy        |      0.7556   |            0.713445 |  -0.0421554  |
| macro_precision |      0.6061   |            0.569539 |  -0.0365608  |
| macro_recall    |      0.5375   |            0.602796 |   0.0652958  |
| macro_f1        |      0.551    |            0.58186  |   0.0308596  |
| weighted_f1     |      0.726179 |            0.729181 |   0.00300262 |
| IPF_recall      |      0.724    |            0.717105 |  -0.00689474 |
| HP_recall       |      0.233    |            0.407767 |   0.174767   |
| CTD_recall      |      0.242    |            0.476562 |   0.234563   |
| 1_SAR_recall    |      0.951    |            0.809748 |  -0.141252   |
| IPF_f1          |      0.692    |            0.683386 |  -0.00861442 |
| HP_f1           |      0.322    |            0.38009  |   0.0580905  |
| CTD_f1          |      0.305    |            0.401316 |   0.0963158  |
| 1_SAR_f1        |      0.885    |            0.862647 |  -0.0223534  |

## Confusion-matrix comparison (focus errors)

| Error pattern | Baseline | Class-weighted | Change |
|---|---:|---:|---:|
| HP -> 1_SAR | 45 | 20 | -25 |
| CTD -> 1_SAR | 62 | 19 | -43 |
| 1_SAR -> minority (IPF+HP+CTD) | 31 | 121 | +90 |

## Interpretation of the trade-off

1. Minority-class recall: HP 0.233->0.408; CTD 0.242->0.477; IPF 0.724->0.717.
2. HP recall improved.
3. CTD recall improved.
4. 1_SAR recall decreased (0.951->0.810).
5. Macro F1 improved (0.5510->0.5819).
6. Accuracy decreased (0.7556->0.7134).
7. Favorability for 4-class ILD objective depends on whether recovering HP/CTD is worth more false positives away from SAR and any accuracy drop; see final conclusion.

## Artifact integrity
- Baseline model file size before/after: 4281 / 4281 bytes (must be unchanged).
- New model saved to `models\class_weighted_logistic_regression.joblib`.

## Limitations
- Single controlled change only; no tuning, resampling, or feature edits.
- Multicollinear z/%predicted features retained.
- Class weights address imbalance in the loss, not representation learning limits of linear models.
- Coefficients not re-analyzed in this experiment.

## Final conclusion

**Did class weighting improve the model's ability to distinguish the minority ILD classes?**

**Yes - with a clear trade-off.** Minority discrimination improved (HP/CTD recall & F1, fewer HP/CTD->SAR errors, higher macro-F1), at the cost of lower accuracy and lower 1_SAR recall, plus more SAR->minority false positives.

Evidence:
- HP recall/F1: 0.233/0.322 -> 0.408/0.380
- CTD recall/F1: 0.242/0.305 -> 0.477/0.401
- Macro F1: 0.5510 -> 0.5819
- HP+CTD -> 1_SAR errors: 107 -> 39
