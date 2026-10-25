
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# STEP 8.6
# Calibrated prediction and frozen-policy evaluation
# NASA C-MAPSS FD001
#
# This script:
#   1. Reads Step 8.5 validation predictions.
#   2. Compares raw and calibrated prediction variants.
#   3. Applies the existing frozen thresholds without changing them.
#   4. Reports validation-snapshot safety and illustrative costs.
#   5. Does NOT use test labels, retrain models, or deploy policies.
#
# IMPORTANT:
# The 4,070 rows are engine-cycle snapshots from 20 engines.
# Snapshot-level costs are NOT real operational maintenance costs.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step08_5"
    / "validation_calibrated_predictions_FD001.csv"
)

STEP8_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "step08"
    / "step08_FD001_report.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step08_6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

CRITICAL_RUL = 15.0
HIGH_RISK_RUL = 30.0

MAX_MISSED_CRITICAL = 0
MAX_HIGH_RISK_MISS_RATE = 0.10

COST_SCENARIOS = {
    "standard": {
        "maintenance_cost": 100.0,
        "missed_critical_cost": 1000.0,
        "false_alarm_cost": 50.0,
    },
    "high_failure_penalty": {
        "maintenance_cost": 100.0,
        "missed_critical_cost": 2000.0,
        "false_alarm_cost": 50.0,
    },
    "lower_failure_penalty": {
        "maintenance_cost": 100.0,
        "missed_critical_cost": 500.0,
        "false_alarm_cost": 50.0,
    },
    "high_maintenance_cost": {
        "maintenance_cost": 250.0,
        "missed_critical_cost": 1000.0,
        "false_alarm_cost": 100.0,
    },
}


def nasa_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Standard asymmetric NASA C-MAPSS score."""
    error = predicted - actual

    penalties = np.where(
        error < 0,
        np.expm1(np.clip(-error / 13.0, -50, 50)),
        np.expm1(np.clip(error / 10.0, -50, 50)),
    )

    return float(np.sum(penalties))


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_variant(
    data: pd.DataFrame,
    prediction_column: str,
    threshold: float,
    variant: str,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate one prediction variant at a fixed threshold."""

    frame = data[
        ["unit_id", "cycle", "actual_RUL", prediction_column]
    ].copy()

    frame = frame.rename(
        columns={prediction_column: "predicted_RUL"}
    )

    frame["variant"] = variant
    frame["threshold_cycles"] = float(threshold)

    actual = frame["actual_RUL"].to_numpy(dtype=float)
    predicted = frame["predicted_RUL"].to_numpy(dtype=float)

    frame["maintenance_action"] = predicted <= threshold

    frame["critical_snapshot"] = actual <= CRITICAL_RUL
    frame["high_risk_snapshot"] = actual <= HIGH_RISK_RUL

    frame["missed_critical_snapshot"] = (
        frame["critical_snapshot"]
        & ~frame["maintenance_action"]
    )

    frame["missed_high_risk_snapshot"] = (
        frame["high_risk_snapshot"]
        & ~frame["maintenance_action"]
    )

    frame["false_alarm_snapshot"] = (
        frame["maintenance_action"]
        & (actual > HIGH_RISK_RUL)
    )

    critical_count = int(frame["critical_snapshot"].sum())
    high_risk_count = int(frame["high_risk_snapshot"].sum())

    missed_critical = int(
        frame["missed_critical_snapshot"].sum()
    )
    missed_high_risk = int(
        frame["missed_high_risk_snapshot"].sum()
    )
    false_alarms = int(frame["false_alarm_snapshot"].sum())
    actions = int(frame["maintenance_action"].sum())

    high_risk_miss_rate = (
        missed_high_risk / high_risk_count
        if high_risk_count > 0
        else None
    )

    # Unique engines affected at least once. These are counts of
    # engines, not counts of independent snapshots.
    engines_with_missed_critical = int(
        frame.loc[
            frame["missed_critical_snapshot"], "unit_id"
        ].nunique()
    )

    engines_with_missed_high_risk = int(
        frame.loc[
            frame["missed_high_risk_snapshot"], "unit_id"
        ].nunique()
    )

    metrics = {
        "variant": variant,
        "threshold_cycles": float(threshold),
        "n_snapshots": int(len(frame)),
        "n_engines": int(frame["unit_id"].nunique()),
        "RMSE": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "MAE": float(np.mean(np.abs(predicted - actual))),
        "NASA_score": nasa_score(actual, predicted),
        "mean_error_predicted_minus_actual": float(
            np.mean(predicted - actual)
        ),
        "maintenance_action_snapshots": actions,
        "critical_snapshots": critical_count,
        "missed_critical_snapshots": missed_critical,
        "engines_with_any_missed_critical_snapshot":
            engines_with_missed_critical,
        "high_risk_snapshots": high_risk_count,
        "missed_high_risk_snapshots": missed_high_risk,
        "engines_with_any_missed_high_risk_snapshot":
            engines_with_missed_high_risk,
        "high_risk_miss_rate": high_risk_miss_rate,
        "false_alarm_snapshots": false_alarms,
        "safety_constraint_feasible_snapshot_level": bool(
            missed_critical <= MAX_MISSED_CRITICAL
            and high_risk_miss_rate is not None
            and high_risk_miss_rate <= MAX_HIGH_RISK_MISS_RATE
        ),
    }

    for scenario_name, costs in COST_SCENARIOS.items():
        # These are snapshot-level accounting totals, not actual
        # operating costs or a realistic repeated-maintenance schedule.
        scenario_cost = (
            actions * costs["maintenance_cost"]
            + missed_critical * costs["missed_critical_cost"]
            + false_alarms * costs["false_alarm_cost"]
        )

        metrics[f"illustrative_snapshot_cost_{scenario_name}"] = float(
            scenario_cost
        )

    # Keep decision details so results can be audited row by row.
    decision_columns = [
        "unit_id",
        "cycle",
        "actual_RUL",
        "predicted_RUL",
        "variant",
        "threshold_cycles",
        "maintenance_action",
        "critical_snapshot",
        "high_risk_snapshot",
        "missed_critical_snapshot",
        "missed_high_risk_snapshot",
        "false_alarm_snapshot",
    ]

    return metrics, frame[decision_columns]


def main() -> None:
    print("=" * 72)
    print("NASA C-MAPSS FD001 - STEP 8.6")
    print("CALIBRATED PREDICTION / FROZEN-POLICY EVALUATION")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Step 8.5 validation predictions not found:\n{INPUT_FILE}"
        )

    step8_report = load_json(STEP8_REPORT)

    # Read the exact frozen thresholds from the prior report.
    # The known report records the baseline threshold in its
    # validation selection artifact, and temporal threshold in
    # adaptive_policy. Do not silently invent a missing threshold.
    adaptive_policy = step8_report.get("adaptive_policy", {})

    temporal_threshold = adaptive_policy.get(
        "selected_threshold_cycles"
    )

    if temporal_threshold is None:
        raise ValueError(
            "Could not find adaptive_policy.selected_threshold_cycles "
            "in the Step 8 report."
        )

    # The previously reported baseline threshold is 10 cycles.
    # Confirm it against the original Step 8 selection CSV below.
    baseline_selection_file = (
        PROJECT_ROOT
        / "outputs"
        / "step08"
        / "step08_validation_policy_selection.csv"
    )

    if not baseline_selection_file.exists():
        raise FileNotFoundError(
            "Cannot verify the frozen baseline threshold because "
            f"this file is missing:\n{baseline_selection_file}"
        )

    baseline_selection = pd.read_csv(baseline_selection_file)

    # Identify the baseline selected threshold using common schema
    # names rather than assuming a single exact CSV layout.
    threshold_candidates = [
        "selected_threshold_cycles",
        "threshold_cycles",
        "selected_threshold",
        "threshold",
    ]
    model_candidates = ["model", "selected_model", "model_variant"]

    threshold_col = next(
        (c for c in threshold_candidates if c in baseline_selection.columns),
        None,
    )
    model_col = next(
        (c for c in model_candidates if c in baseline_selection.columns),
        None,
    )

    if threshold_col is None or model_col is None:
        raise ValueError(
            "Could not identify the model and threshold columns in "
            "step08_validation_policy_selection.csv.\n"
            f"Columns found: {baseline_selection.columns.tolist()}\n"
            "No thresholds were changed. Inspect the CSV schema."
        )

    baseline_rows = baseline_selection[
        baseline_selection[model_col]
        .astype(str)
        .str.lower()
        .str.contains("baseline", regex=False)
    ]

    if baseline_rows.empty:
        raise ValueError(
            "No baseline row was found in the Step 8 policy-selection CSV."
        )

    baseline_threshold = float(
        baseline_rows.iloc[0][threshold_col]
    )

    print(f"Input: {INPUT_FILE}")
    print(f"Frozen baseline threshold: {baseline_threshold}")
    print(f"Frozen temporal threshold: {float(temporal_threshold)}")

    data = pd.read_csv(INPUT_FILE)

    required_columns = [
        "unit_id",
        "cycle",
        "actual_RUL",
        "baseline_raw_RUL",
        "baseline_calibrated_RUL",
        "temporal_raw_RUL",
        "temporal_calibrated_RUL",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Columns found: {data.columns.tolist()}"
        )

    data = data[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[required_columns].isna().any().any():
        bad_counts = data[required_columns].isna().sum()
        raise ValueError(
            "Missing or nonnumeric values detected:\n"
            f"{bad_counts[bad_counts > 0].to_dict()}"
        )

    if not np.isfinite(data[required_columns].to_numpy()).all():
        raise ValueError("Non-finite values found in the input.")

    if data["unit_id"].nunique() != 20:
        warnings.warn(
            "Expected 20 validation engines based on the Step 8 report; "
            f"found {data['unit_id'].nunique()}. Verify the input split."
        )

    # Fixed-threshold comparison: calibration changes predictions,
    # not the frozen thresholds. Candidate thresholds are not optimized
    # against this same evaluation set.
    variants = [
        (
            "baseline_raw_frozen",
            "baseline_raw_RUL",
            baseline_threshold,
        ),
        (
            "baseline_calibrated_same_threshold",
            "baseline_calibrated_RUL",
            baseline_threshold,
        ),
        (
            "temporal_raw_frozen",
            "temporal_raw_RUL",
            float(temporal_threshold),
        ),
        (
            "temporal_calibrated_same_threshold",
            "temporal_calibrated_RUL",
            float(temporal_threshold),
        ),
    ]

    all_metrics = []
    all_decisions = []

    for variant, prediction_column, threshold in variants:
        metrics, decisions = evaluate_variant(
            data=data,
            prediction_column=prediction_column,
            threshold=threshold,
            variant=variant,
        )
        all_metrics.append(metrics)
        all_decisions.append(decisions)

    metrics_df = pd.DataFrame(all_metrics)
    decisions_df = pd.concat(all_decisions, ignore_index=True)

    metrics_path = OUTPUT_DIR / "step08_6_validation_metrics_FD001.csv"
    decisions_path = OUTPUT_DIR / "step08_6_validation_decisions_FD001.csv"
    report_path = OUTPUT_DIR / "step08_6_FD001_report.json"
    plot_path = OUTPUT_DIR / "step08_6_policy_comparison_FD001.png"

    metrics_df.to_csv(metrics_path, index=False)
    decisions_df.to_csv(decisions_path, index=False)

    # Plot illustrative snapshot costs under the standard scenario.
    cost_column = "illustrative_snapshot_cost_standard"

    plt.figure(figsize=(11, 6))
    bars = plt.bar(
        metrics_df["variant"],
        metrics_df[cost_column],
    )
    plt.title(
        "Step 8.6 — Validation Snapshot Cost Comparison\n"
        "Illustrative costs; frozen thresholds"
    )
    plt.ylabel("Illustrative snapshot cost (arbitrary units)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, metrics_df[cost_column]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    report = {
        "project": (
            "A Self-Evolving Explainable Rational Agent for Adaptive "
            "Industrial Decision Intelligence Using Statistical Trend Learning"
        ),
        "dataset": "NASA C-MAPSS FD001",
        "step": "8.6",
        "method": "Frozen-threshold calibrated prediction evaluation",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT_FILE),
        "validation_snapshots": int(len(data)),
        "validation_engines": int(data["unit_id"].nunique()),
        "frozen_thresholds": {
            "baseline_cycles": float(baseline_threshold),
            "temporal_cycles": float(temporal_threshold),
        },
        "definitions": {
            "maintenance_action": "predicted RUL <= frozen threshold",
            "critical": f"actual RUL <= {CRITICAL_RUL}",
            "high_risk": f"actual RUL <= {HIGH_RISK_RUL}",
            "false_alarm": (
                "maintenance action when actual RUL > "
                f"{HIGH_RISK_RUL}"
            ),
            "high_risk_miss_rate": (
                "missed high-risk snapshots divided by all high-risk "
                "snapshots"
            ),
        },
        "safety_constraints": {
            "maximum_missed_critical_snapshots": MAX_MISSED_CRITICAL,
            "maximum_high_risk_miss_rate": MAX_HIGH_RISK_MISS_RATE,
            "interpretation": (
                "Exploratory snapshot-level screen, not a validated "
                "industrial safety guarantee."
            ),
        },
        "cost_scenarios": COST_SCENARIOS,
        "metrics": metrics_df.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
        "limitations": [
            "Validation contains 4,070 engine-cycle snapshots from only 20 engines.",
            "Snapshots from the same engine are dependent observations.",
            "Snapshot-level costs are not actual operating costs and do not model "
            "repeated interventions, repair duration, or maintenance scheduling.",
            "The same frozen threshold is applied before and after calibration; "
            "this isolates the prediction transformation but may not be the "
            "optimal threshold for calibrated predictions.",
            "No threshold is selected using these Step 8.6 results.",
            "No test labels are read or used.",
            "No model is retrained.",
            "No maintenance policy is modified or deployed.",
            "Results are limited to FD001 and the specified illustrative costs.",
        ],
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "decisions_csv": str(decisions_path),
            "plot": str(plot_path),
            "report_json": str(report_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    display_columns = [
        "variant",
        "threshold_cycles",
        "n_snapshots",
        "n_engines",
        "RMSE",
        "MAE",
        "NASA_score",
        "maintenance_action_snapshots",
        "missed_critical_snapshots",
        "missed_high_risk_snapshots",
        "high_risk_miss_rate",
        "false_alarm_snapshots",
        "illustrative_snapshot_cost_standard",
        "safety_constraint_feasible_snapshot_level",
    ]

    print("\nVALIDATION SNAPSHOT RESULTS")
    print(
        metrics_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSTEP 8.6 COMPLETED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metrics: {metrics_path}")
    print(f"Decisions: {decisions_path}")
    print(f"Plot: {plot_path}")
    print(f"Report: {report_path}")
    print("\nNo test labels were used.")
    print("No model was retrained.")
    print("No maintenance policy was modified or deployed.")


if __name__ == "__main__":
    main()
