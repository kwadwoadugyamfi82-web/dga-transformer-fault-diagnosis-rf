# Dissolved Gas Analysis-Based Power Transformer Fault Diagnosis Using Random Forest

**Final Year Project Report** submitted in partial fulfilment of the requirements for the degree of Bachelor of Science in Electrical and Electronic Engineering, Kwame Nkrumah University of Science and Technology (KNUST), Ghana.

A comparative machine-learning study evaluating **Random Forest**, **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **Artificial Neural Network (ANN)** classifiers for Dissolved Gas Analysis (DGA)-based power transformer fault diagnosis — including an explicit **cross-dataset generalization test**, a gap left open by prior published work in this area.

**Author:** Adu-Gyamfi Kwadwo — Department of Electrical and Electronic Engineering, KNUST, Ghana
**Supervisor:** Dr. E. K. Anto

📄 **Full project report:** [`docs/DGA_RandomForest_ResearchPaper.docx`](docs/DGA_RandomForest_ResearchPaper.docx) · [PDF version](docs/DGA_RandomForest_ResearchPaper.pdf)

---

## Summary of Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | **83.76%** | 84.12% | 83.32% | 83.44% |
| K-Nearest Neighbors (K=1) | 76.07% | 76.17% | 75.28% | 75.53% |
| Support Vector Machine (RBF) | 76.07% | 75.58% | 74.19% | 74.38% |
| Artificial Neural Network | 70.94% | 71.71% | 69.70% | 69.70% |

**Key finding:** the Random Forest model, when evaluated on independent DGA datasets from different sources, dropped from 83.76% to 66–67% accuracy — a substantial domain-shift effect not quantified in the reviewed literature. See Section 5.6 of the paper for full discussion.

---

## Repository Structure

```
├── docs/
│   ├── DGA_RandomForest_ResearchPaper.docx   Full project report (with clickable references)
│   └── DGA_RandomForest_ResearchPaper.pdf    Same report, PDF version
├── code/
│   ├── DGA_Model_Comparison.m                MATLAB: RF vs KNN vs SVM vs ANN, same split, all figures
│   └── DGA_RandomForest_v2.m                 MATLAB: Random Forest with SMOTE + repeated CV tuning
├── data/
│   └── DGA_transformer_dataset.csv           589-sample cleaned DGA dataset (5 gases + fault label)
└── figures/                                  All result figures (PNG) referenced in the report
```

---

## Dataset

589 field-verified DGA records (H₂, CH₄, C₂H₆, C₂H₄, C₂H₂ in ppm) across six IEC 60599 fault classes (PD, D1, D2, T1, T2, T3), sourced from:

> E. Li, "Dissolved gas data in transformer oil — Fault diagnosis of power transformers with membership degree," IEEE Dataport, Jan. 2019. doi: [10.21227/h8g0-8z59](https://dx.doi.org/10.21227/h8g0-8z59)

## How to Reproduce

1. Open MATLAB (Statistics and Machine Learning Toolbox + Deep Learning Toolbox required).
2. Place `data/DGA_transformer_dataset.csv` in the same folder as the scripts in `code/`.
3. Run `DGA_Model_Comparison.m` to reproduce all four models and every figure in the `figures/` folder.

## Citation

If you reference this work, please cite the accompanying project report in `docs/`.

## License

This repository is shared for academic and educational purposes. The dataset retains the license terms of its original source (see citation above).
