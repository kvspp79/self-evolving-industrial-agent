# Step 10J — Final Research Evidence Synthesis

Audit generated (UTC): 2026-10-25T10:00:37.418580+00:00

## Purpose

This step inventories Step 10F–10I artifacts, checks CSV/JSON readability, records schemas and basic integrity indicators, and exports copies of evidence tables. It does not retrain models or modify source experiment outputs.

## Audit coverage

- Stage directories: 4
- CSV discovered/readable/errors: 17/17/0
- JSON discovered/readable/errors: 14/14/0
- Other files: 6
- Evidence table copies: 16

## Integrity observations

- Errors: 0
- Warnings: 0
- Informational observations: 0

See `step10J_issues.csv` for file-specific details. Contextually non-applicable fields are excluded only for the two documented Step 10I schemas; original source CSVs are not changed.

## Interpretation limits

- This audit checks files and schemas; it does not independently reproduce experiments.
- Readable files are not proof that experiments are methodologically valid.
- Repeated seeds are not independent industrial datasets.
- Synthetic sensor shifts do not establish real-world concept drift.
- Feature importance/SHAP does not establish causation.
- Simulated maintenance costs are not real operational costs.
- NASA PHM score is asymmetric and should be reported separately from RMSE and MAE.
- Do not use held-out results to revise acceptance criteria or select a model after evaluation.
- Statistical inference must respect the experimental unit and dependence structure.

This audit does not independently reproduce the underlying experiments. Verify statistical calculations from seed-level source data before publication.
