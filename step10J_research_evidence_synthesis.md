# Step 10J — Final Research Evidence Synthesis

Audit generated (UTC): 2026-10-24T20:08:01.843484+00:00

## 1. Purpose

This step inventories existing Step 10F–10I artifacts, checks whether CSV and JSON files can be read, records schemas and basic integrity indicators, and exports copies of relevant evidence tables. It does not retrain models or modify experiment outputs.

## 2. Audit coverage

- Stage directories discovered: 4
- CSV files discovered: 17
- CSV files readable: 17
- CSV read errors: 0
- JSON files discovered: 14
- JSON files readable: 14
- JSON read errors: 0
- Other files found: 6
- Evidence-table copies exported: 16

## 3. Data-integrity observations

- Errors recorded: 0
- Warnings recorded: 2
- Informational observations: 0

See `step10J_issues.csv` for file-specific details. A readable file is not necessarily a scientifically valid result. Duplicate rows and missing cells require contextual review rather than automatic deletion or imputation.

## 4. Research findings supported by the previously reported runs

The supplied Step 10F output reports 10 successful seed runs. Unconditional retraining improved held-out RMSE in 8 of 10 seeds, MAE in 8 of 10, and NASA score in 7 of 10. The safety-gated model improved held-out RMSE in 5 of 10 seeds, MAE in 4 of 10, and NASA score in 5 of 10. These counts show variability rather than uniform improvement.

The supplied Step 10G output reports 4 accepted and 6 rejected updates. Three accepted updates improved all three monitored metrics on the held-out set, while one accepted update worsened at least one metric. This indicates that the validation gate was not a guarantee of held-out success.

The supplied Step 10H output reports 3 accepted updates under each of the two gate rules, with the same three seeds accepted by both. That result does not establish superiority of the alternative NASA-prioritized rule.

These statements are transcribed from previously supplied terminal output. This script does not treat screenshots or pasted logs as machine-readable source data. Verify them against the original CSV/JSON files before publication.

## 5. Statistical interpretation

Ten seeds provide a useful robustness check but limited precision. A percentage of seeds improved is descriptive, not by itself a statistical significance result. Confidence intervals and paired tests must be calculated from correctly paired seed-level outcomes. Do not treat engine rows, cycle rows, or repeated comparisons as independent observations when they share engines or training runs.

The NASA PHM score is asymmetric and may be highly skewed. Report RMSE, MAE, and NASA score separately, include uncertainty where appropriate, and explain the unit of aggregation. Do not average incompatible row-level, engine-level, and seed-level quantities.

## 6. Defensible claim about self-evolution

The experiments operationalize a controlled adaptation loop: a candidate model is trained using adaptation data, evaluated against a frozen model on a gate-validation subset, and either accepted or rejected according to predeclared criteria. The experiments also compare accepted candidates against held-out engines.

The evidence supports the claim that a safety-gated adaptation mechanism was implemented and evaluated under a synthetic sensor shift. The evidence does not yet prove reliable improvement in real industrial environments, online autonomous learning in deployment, or generalization to other engines, datasets, or types of drift.

## 7. Limitations to disclose

- The sensor shift is synthetic, not observed industrial drift.
- FD001 contains 100 engines; each split uses a limited number of engines and can be sensitive to the seed.
- The experiment uses labelled RUL values for model adaptation; this is not equivalent to fully autonomous learning without labels.
- The adaptation procedure and acceptance rule are specific to this experiment.
- A gate that accepts an update on validation data can still accept a candidate that performs worse on held-out data.
- The gate may reject useful candidates when validation estimates are noisy.
- NASA score, RMSE, and MAE measure different aspects of prediction quality and may disagree.
- Simulated maintenance costs and action thresholds are assumptions, not validated operational economics.
- Feature importance and SHAP are explanatory associations, not causal evidence.
- No real-time industrial deployment or operator study has been established by these experiments.

## 8. Recommended reporting package

Use the audited CSV/JSON artifacts to prepare: (1) a dataset and split description; (2) the fixed experimental protocol; (3) seed-level paired results; (4) gate acceptance and rejection outcomes; (5) uncertainty intervals and statistical tests at the correct experimental unit; (6) failure analysis; and (7) a limitations section.

Do not select the best model or alter the gate by inspecting the final held-out results. Any new rule should be evaluated in a new predeclared experiment.

## 9. Generated files

- `step10J_report.json` — machine-readable audit summary
- `step10J_csv_inventory.csv` — CSV readability and integrity profile
- `step10J_json_inventory.csv` — JSON readability profile
- `step10J_schema_audit.csv` — schemas and coarse table roles
- `step10J_key_evidence_index.csv` — index of metric/statistical columns
- `step10J_issues.csv` — detected file-level issues
- `evidence_tables/` — copies of selected existing CSV tables

## 10. Final status

This is an artifact and evidence-consolidation audit, not an independent replication of the scientific experiment. A clean file audit must not be described as proof of model validity.
