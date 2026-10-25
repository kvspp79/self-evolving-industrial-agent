# Step 8.8 — Evidence Synthesis and Research-Integrity Audit

**Dataset:** NASA C-MAPSS FD001
**Generated (UTC):** 2026-10-25T09:57:27.734433+00:00

## Purpose

This audit consolidates existing outputs and checks interpretation and evaluation boundaries. It does not retrain models, modify a maintenance policy, or use test labels for selection.

## Main findings

- Step 8.5 fitted linear calibration using training-partition out-of-fold predictions only. Validation labels were evaluation-only; no test labels were used.
- baseline_calibrator: intercept=3.2400420699467816, slope=0.9680938848346163.
- temporal_calibrator: intercept=7.437800121419201, slope=0.9545655135187104.
- Step 8.7 final-cycle regression metrics were read from step08_7_final_cycle_regression_metrics_FD001.csv; they refer to 20 validation engines, not all 4,070 validation cycle rows.

## Bootstrap interpretation audit

- Status: **corrected_copy_created**
- Input: `C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS\outputs\step08_7\step08_7_engine_bootstrap_FD001.csv`
- Corrected copy: `C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS\outputs\step08_8\step08_8_corrected_bootstrap_interpretations_FD001.csv`

For a comparison A_minus_B defined as RMSE(A) - RMSE(B):
- A 95% interval entirely below zero favors A.
- A 95% interval entirely above zero favors B.
- An interval containing zero is inconclusive at this interval level.
- These labels describe the interval; they do not prove practical importance or broad generalization.

## Interpretation safeguards

1. All-cycle metrics and final-cycle engine metrics are different evaluation units and must not be compared as if they were the same.
2. Calibration is post-processing, not model retraining.
3. Maintenance costs and cost-optimal thresholds depend on illustrative assumptions; they are not validated industrial cost estimates.
4. A validation set of 20 engines gives uncertain safety estimates. Zero observed misses does not establish a zero real-world failure rate.
5. Distribution shift between train and test sensor rows does not, by itself, establish temporal concept drift or causal degradation.
6. The current experiments do not establish continuous online self-evolution or safe autonomous deployment.
7. Test-set results are for final reporting only and must not be used to tune models, calibration, thresholds, or claims.

## Artifact inventory

- Files inventoried: 135
- See `step08_8_artifact_inventory_FD001.csv` for paths and presence.

## Limitations

- Only C-MAPSS FD001 is covered by this audit.
- Earlier step reports are summarized as recorded; this script does not independently reproduce every model fit or metric.
- A successful audit script run does not itself validate model correctness, safety, or deployment readiness.
