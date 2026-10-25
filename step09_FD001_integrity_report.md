# Step 9 — Final Experimental Integrity Audit

**Dataset:** NASA C-MAPSS FD001
**Generated (UTC):** 2026-10-25T09:57:28.547093+00:00

## Scope

This is a read-only audit of stored artifacts and report structures. It does not reproduce every model fit or independently validate every metric.

## Summary

- Required artifacts: 25
- Missing required artifacts: 0
- Files inventoried: 230
- JSON reports parsed: 38
- CSV files read: 136
- ERROR findings: 0
- WARNING findings: 3
- INFO findings: 6

## Evaluation boundaries

- Keep all-cycle validation metrics separate from final-cycle engine metrics.
- Treat calibration as post-processing, not model retraining.
- Treat simulated maintenance costs as illustrative assumptions.
- A validation sample of 20 engines does not establish real-world safety.
- Train/test sensor shift alone does not prove temporal concept drift.
- Do not tune using test labels or test-set results.
- Existing steps do not yet demonstrate continuous online self-evolution.

## Findings

- **WARNING — EMPTY_CSV** (`C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS\outputs\step10A\step10A_scan_errors.csv`): CSV has headers or content but contains zero data rows.
- **WARNING — EMPTY_CSV** (`C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS\outputs\step10J_final_evidence_synthesis\step10J_issues.csv`): CSV has headers or content but contains zero data rows.
- **INFO — EVALUATION_UNITS** (`Step 4 / Step 8.5 / Step 8.7`): All-cycle validation metrics and final-cycle engine metrics are different evaluation units. Do not compare them as though they measured the same set of observations.
- **INFO — CALIBRATION** (`Step 8.5`): Calibration is post-processing fitted using training-partition out-of-fold predictions. It is not model retraining. Review the saved report and metrics before making improvement claims.
- **INFO — POLICY_COSTS** (`Steps 6–8`): Maintenance costs are illustrative assumptions, not validated industrial cost estimates.
- **WARNING — SMALL_VALIDATION_SAMPLE** (`Step 8 / Step 8.7`): The final-cycle validation policy evidence involves only 20 engines. Zero observed misses does not establish zero real-world failure risk.
- **INFO — DISTRIBUTION_SHIFT** (`Step 8.2`): Training-to-test sensor distribution differences do not alone prove temporal concept drift, causal degradation, or policy failure.
- **INFO — SELF_EVOLUTION_CLAIM** (`Steps 8.1–8.8`): Policy selection, calibration, drift detection, and auditing do not by themselves demonstrate continuous online learning. A controlled feedback-loop experiment is still needed to support a strong self-evolution claim.
- **INFO — TEST_SET_BOUNDARY** (`All steps`): Test-set metrics are for final reporting only. Do not use them to tune models, calibration, thresholds, or claims.

## Metric table index

| Artifact | Status | Rows | Evaluation unit |
|---|---|---:|---|
| `step04/validation_predictions_FD001.csv` | readable | 4070 | All validation cycle rows; predictions and actual RUL. |
| `step04/test_predictions_FD001.csv` | readable | 100 | Final-cycle test engine predictions, as reported in Step 4. |
| `step08_5/validation_calibration_metrics_FD001.csv` | readable | 4 | Calibration evaluation across validation rows. |
| `step08_6/step08_6_validation_metrics_FD001.csv` | readable | 4 | Step 8.6 policy evaluation; inspect the report for exact unit. |
| `step08_7/step08_7_final_cycle_regression_metrics_FD001.csv` | readable | 4 | Final-cycle validation engine regression metrics. |
| `step08_7/step08_7_engine_bootstrap_FD001.csv` | readable | 4 | Engine-level bootstrap comparisons; inspect difference direction. |

## Interpretation

A successful script run confirms that this audit completed. It does not prove that every model, metric, policy, or scientific claim is correct. Review warnings and compare key metrics against their original reports before writing final conclusions.
