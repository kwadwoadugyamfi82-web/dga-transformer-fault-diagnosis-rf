# Data notes

| File | Rows | What it is |
|---|---|---|
| DGA_transformer_dataset.csv | 589 | Training set: five gases in ppm and a fault label. |
| tc10_cases.csv | 116 | Inspection-confirmed cases of the IEC TC 10 database, typed in from Duval and dePablo (2001). Labels PD, D1, D2, T1T2, T3. |
| picl_records.csv | 1,622 | Labelled records taken from a public data file (PICL). Columns: source, case_id, fault_type, label_confirmed, is_duplicate, above_attention, H2, CH4, C2H6, C2H4, C2H2. |
| dga_provenance.csv | 1,622 | Same records with source, equipment, label basis and flags, used to pick the independent set. |

Where the 589 rows come from: the DGALab package (github.com/Saleh860/DGA), file dataset_(589).xlsx. The first version of this project named the IEEE DataPort set of E. Li (2019) as the source. That was wrong for these rows.

The independent test set is the part of picl_records.csv with label_confirmed = true, is_duplicate = false, above_attention = true and not already among the 589 rows. That gives 465 records from 189 transformers or cases (NCEPR 351, IEEE DataPort 43, NE Grid 38, Published cases 16, Fujian 13, IEC TC 10 4). Records from the Fujian source that were labelled by a rule are not used as ground truth.

Licence: I did not check the licence of every source file. The files are shared here for study and for checking the paper. If you reuse them, go back to the original sources.

Optional file: dataset1_(240).xlsx from the DGALab package is needed only for experiment 2 of DGA_Extended_Experiments.m. It is not copied here.
