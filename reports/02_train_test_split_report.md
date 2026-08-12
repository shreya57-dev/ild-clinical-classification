# Train/test split & leakage audit report

No model was trained. Raw CSV was not modified.

## 1. Investigation: `age at event`

**Verdict:** EXCLUDE — B — age at later follow-up/event (death or censoring), not baseline PFT age

**Decision:** Do NOT use age at event for diagnosis classification. It incorporates follow-up duration (patients who survive longer appear older), creating temporal leakage relative to baseline PFT prediction time.

### Evidence
- Source paper (Boros & Martusewicz-Boros, PLOS Medicine / Zenodo 15295274): time-to-event = first PFT to death or censoring (15 Mar 2023); PFTs are baseline/first presentation.
- survival time equals (event_date - First test) in days for 6785/6808 rows.
- Overall mean age at event=58.49; implied baseline age (age_at_event - survival_years) mean=51.12 matches paper Table 2 overall Age=51.13.
- Per-group implied baseline ages match paper Table 2 Age means within ~0.05 years.
- Column is literally named 'age at event' and sits next to event_date / dead or alive / survival time.

### Per-group age comparison (paper Table 2 vs data)

| diagnosis group | age_at_event mean | implied baseline mean | paper Age mean | |diff| |
|---|---:|---:|---:|---:|
| 1_SAR | 52.18 | 43.24 | 43.26 | 0.016 |
| CTD | 65.96 | 59.53 | 59.56 | 0.029 |
| HP | 59.16 | 52.32 | 52.34 | 0.022 |
| IPF | 72.82 | 68.26 | 68.3 | 0.038 |
| i-NSIP | 63.35 | 57.50 | 57.51 | 0.012 |
| o-ILD | 59.75 | 53.43 | 53.44 | 0.011 |
| u-ILD | 66.96 | 61.35 | 61.37 | 0.019 |

Overall: age_at_event mean=58.49; implied baseline=51.12; paper=51.13.

## 2. Final feature set

### Retained
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

### Excluded (with reason)
- `Case_number`: Identifier — not a clinical predictor.
- `First test`: Baseline calendar date — temporal/cohort information; not a physiological predictor.
- `event_date`: Death/censoring date — future relative to baseline PFT; outcome timing leakage.
- `dead or alive`: Mortality/censoring outcome after follow-up — outcome leakage for diagnosis classification.
- `survival time`: Days from First test to event_date — follow-up/outcome duration leakage.
- `age at event`: Age at death or censoring (event_date), NOT age at baseline PFT. Equals approximately paper baseline Age + survival_years. Temporal leakage.
- `diagnosis group`: Target variable y — must not appear in X.

Correlated z/%predicted pairs were **not** removed at this stage.

## 3. Full-column leakage audit (Dataset B)

| column | role | concern | recommendation |
|---|---|---|---|
| `Case_number` | excluded | Identifier — not a clinical predictor. | Exclude from X. |
| `First test` | excluded | Baseline calendar date — temporal/cohort information; not a physiological predictor. | Exclude from X. |
| `sex` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `fev1fvc_abs` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `fev1_z` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `fev1_pp` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `fvc_z` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `fvc_pp` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `fev1fvc_z` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `tlc_z` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `tlc_pp` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `tlco_z` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `tlco_pp` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `Height` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `Weight` | retained_feature | None for diagnosis-time use (baseline PFT / demographics available at first test). | Retain in X. |
| `event_date` | excluded | Death/censoring date — future relative to baseline PFT; outcome timing leakage. | Exclude from X. |
| `dead or alive` | excluded | Mortality/censoring outcome after follow-up — outcome leakage for diagnosis classification. | Exclude from X. |
| `age at event` | excluded | Age at death or censoring (event_date), NOT age at baseline PFT. Equals approximately paper baseline Age + survival_years. Temporal leakage. | Exclude from X. |
| `diagnosis group` | target | Is the label. | Use as y only; never place in X. |
| `survival time` | excluded | Days from First test to event_date — follow-up/outcome duration leakage. | Exclude from X. |

## 4. Stratified train/test split

- test_size = 0.2
- random_state = 42
- stratify = `diagnosis group`
- n_train = **4073**
- n_test = **1019**

### Training class distribution

| diagnosis group   |   count |   percent |
|:------------------|--------:|----------:|
| IPF               |     605 |     14.85 |
| HP                |     412 |     10.12 |
| CTD               |     512 |     12.57 |
| 1_SAR             |    2544 |     62.46 |

### Test class distribution

| diagnosis group   |   count |   percent |
|:------------------|--------:|----------:|
| IPF               |     152 |     14.92 |
| HP                |     103 |     10.11 |
| CTD               |     128 |     12.56 |
| 1_SAR             |     636 |     62.41 |

## 5. Preprocessing leakage rule (documented, not fitted)

> Any preprocessing whose parameters are learned from data must be fitted using training data only and then applied to the test data.

No scaler, imputer, encoder, or other learned transformer was fitted in this stage.

## 6. Validation checks

- PASS: target not in X_train/X_test
- PASS: all excluded leakage/id fields absent from X
- PASS: train(4073)+test(1019)=5092
- PASS: train/test index sets disjoint
- INFO: identical feature+label row strings shared across splits: 0 (coincidence possible; indices disjoint)
- PASS: all 4 classes present in train
- PASS: all 4 classes present in test
- PASS: max |train%-test%| class gap = 0.07 pp (<1.0)
- PASS: X columns match FINAL_FEATURES

## 7. Remaining concerns

- No true baseline age column exists in the released CSV; demographic age is unavailable without leakage.
- Height/Weight used in GLI reference equations that produce z/%predicted — partial redundancy with PFT indices, but not temporal leakage; keep for now.
- Weight=8 kg anomaly remains in the modeling table (not dropped).
- Class imbalance (1_SAR majority) remains; handle at training/evaluation stage.
