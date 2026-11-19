# Modern predictive maintenance of power transformers: machine-learning fault diagnosis from dissolved gas analysis

Final-year project at KNUST (2024), supervised by Dr. E. K. Anto, extended in 2026 into an audit of how far such a classifier can be trusted. All code is MATLAB.

Author: Adu-Gyamfi Kwadwo, Department of Electrical and Electronic Engineering, KNUST, Ghana.

## Why this project exists

A transformer usually develops a fault slowly, for example overheating or electrical discharge in the windings, core or insulation. If nobody notices, it fails: an outage, an expensive repair and a risk to the staff. Predictive maintenance tries to find the fault early. The standard way is to test the oil. Faults make the oil release gases, and a laboratory measures five of them (H2, CH4, C2H6, C2H4, C2H2). The classical rules for reading these numbers (Duval Triangle, IEC ratio table) use fixed limits and sometimes give no answer. Machine learning can learn the pattern from past records.

This project compares four classifiers (random forest, KNN, SVM, neural network) for naming one of six faults from the five gases: partial discharge (PD), low-energy discharge (D1), high-energy discharge (D2), and thermal faults in three temperature bands (T1, T2, T3). It then asks whether the accuracy can be trusted when the data come from somewhere else. The paper is in docs/DGA_Audit_Paper_v2.pdf. It covers the diagnosis step only, not forecasting how long a transformer will last.

## Main results

| Question | Answer | Source |
|---|---|---|
| Best model on one 80/20 split (117 test records) | Random forest 83.76%, KNN 76.07%, SVM 76.07%, ANN 70.94% | code/DGA_Model_Comparison.m |
| Same forest, clean cross-validation | 83.2% | Rigorous_Eval, part C |
| Same forest, labels made by the Duval rule | 99.2% | Rigorous_Eval, part C |
| Same forest, SMOTE before splitting | 89.3% | Rigorous_Eval, part C |
| One data source against the others | 94.4% against 80.7% | Extended, experiment 2 |
| Records of one transformer in train and test | 93.2% against 72.0% when transformers are kept apart | Extended, experiment 3 |
| 465 independent records | 55.5% | Extended, experiment 4 |
| Duval Triangle on the 589 rows | 62.8% | Rigorous_Eval |
| Answering only the 50% most confident records | 96.0% inside the data, 62.2% outside | Extended, experiment 4 |
| 25 local records only, on a new utility | 65.1%, against 54.3% for the 589 rows plus the same 25 | Extended, experiment 5 |
| Prediction sets aimed at 90% coverage | 62.2% when calibrated on the 589 rows | Extended, experiment 5 |

The full MATLAB output is in results/matlab_extended_run_output.txt. The two scripts use fixed seeds, but another MATLAB version can shift a value by about one point.

## Folders

```
code/      MATLAB scripts
data/      datasets and data notes (see data/README.md)
results/   output printed by MATLAB
figures/   figures of the original scripts (figures/*.png) and audit figures (figures/audit/)
docs/      DGA_Audit_Paper_v2 (docx and pdf), REFERENCES.md; the 2024 report is in docs/v1/
```

## Scripts

| Script | What it does | Time |
|---|---|---|
| DGA_Model_Comparison.m | The original four-model comparison and its figures. | few minutes |
| DGA_RandomForest_v2.m | The original random forest with SMOTE and repeated cross-validation. | few minutes |
| DGA_Rigorous_Eval.m | Duval and IEC baselines, TC 10 test, leakage and label-origin tests, independent records. Set RUN_NESTED = true for the slow nested comparison of four models. | minutes |
| DGA_Extended_Experiments.m | Experiments 1 to 5 of the paper: per-class results, one source against others, same-transformer leakage, abstention, local records and prediction sets. Writes CSV files to a results folder. | 15 to 40 min |

Needs the Statistics and Machine Learning Toolbox. The neural network in the original scripts needs the Deep Learning Toolbox.

## Run it

1. Copy the files of data/ and code/ into one folder and make it the MATLAB current folder.
2. Run DGA_Rigorous_Eval.m, then DGA_Extended_Experiments.m.
3. Warnings about a missing class in some cross-validation folds are expected. A few classes have very few transformers.

## What changed since the 2024 report

- The cross-dataset test of 66.48% and 67.12% on sets of 1,748 and 1,679 samples is withdrawn. The data behind them cannot be reproduced from this repository, so the claim is withdrawn and the tests above replace it.
- The claim to be the first to measure the drop is removed.
- The data source is corrected (see data/README.md).
- The 2024 report is kept in docs/v1/ for the record. The original two scripts are unchanged.

## Limits

Public records only, five gases, one six-class label scheme, no transformer in service. The 83.76% figure comes from a single split. Read the limits in Section 7 of the paper before quoting any number.
