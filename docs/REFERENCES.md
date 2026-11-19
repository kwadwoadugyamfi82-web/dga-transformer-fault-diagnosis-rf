# References and data sources

Cited in docs/DGA_Audit_Paper_v2.pdf.

1. IEC 60599: Mineral oil-filled electrical equipment in service. Guidance on the interpretation of dissolved and free gases analysis. International Electrotechnical Commission.
2. M. Duval and A. dePablo, "Interpretation of gas-in-oil analysis using new IEC publication 60599 and IEC TC 10 databases," IEEE Electrical Insulation Magazine, vol. 17, no. 2, pp. 31-41, 2001.
3. A. Nanfak, S. Eke, C. H. Kom, R. Mouangue and I. Fofana, "Interpreting dissolved gases in transformer oil: A new method based on the analysis of labelled fault data," IET Generation, Transmission & Distribution, 2021, doi: 10.1049/gtd2.12239.
4. A. Ibrahim, M. Ghoneim and I. Taha, "DGALab: an extensible software implementation for DGA," IET Generation, Transmission & Distribution, 2018, doi: 10.1049/iet-gtd.2018.5564.
5. L. Breiman, "Random forests," Machine Learning, vol. 45, pp. 5-32, 2001.
6. N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," Journal of Artificial Intelligence Research, vol. 16, pp. 321-357, 2002.
7. C. Nadeau and Y. Bengio, "Inference for the generalization error," Machine Learning, vol. 52, pp. 239-281, 2003.
8. A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," arXiv:2107.07511, 2021.
9. E. Li, "Dissolved gas data in transformer oil: Fault diagnosis of power transformers with membership degree," IEEE DataPort, 2019, doi: 10.21227/h8g0-8z59 (cited as a related data set; not the source of the 589 rows).

## Data sources used in this repository

- 589 training records: DGALab package, file dataset_(589).xlsx (github.com/Saleh860/DGA). Cleaned copy: data/DGA_transformer_dataset.csv.
- IEC TC 10 cases: Duval and dePablo (2001), typed in as data/tc10_cases.csv.
- 1,622 labelled records from a public compilation (PICL data file): data/picl_records.csv and data/dga_provenance.csv.
- Optional single-source file dataset1_(240).xlsx, from the DGALab package. Download it from that repository if you want to rerun experiment 2.

See data/README.md for how the independent test set was chosen and for the licence note.
