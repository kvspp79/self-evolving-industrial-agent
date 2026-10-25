
"""
NASA C-MAPSS FD001 — STEP 6
UNCERTAINTY-AWARE RUL AND MAINTENANCE DECISIONS

Inputs:
    outputs/step04/validation_predictions_FD001.csv
    outputs/step04/test_predictions_FD001.csv

Outputs:
    outputs/step06/

This script does not retrain models or modify Steps 1–5.

Research notes:
- Calibration uses one final observed row per validation engine.
- Test data are not used to calibrate intervals.
- Costs and risk thresholds are illustrative assumptions.
- Small-sample calibration does not guarantee future coverage.
"""

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

STEP04_DIR = ROOT / "outputs" / "step04"
OUTPUT_DIR = ROOT / "outputs" / "step06"

SUBSET = "FD001"

# Nominal split-conformal interval level.
ALPHA = 0.10

# Illustrative RUL risk thresholds, in cycles.
CRITICAL_RUL = 15.0
HIGH_RUL = 30.0
MEDIUM_RUL = 60.0

# Illustrative costs, NOT empirical plant costs.
MAINTENANCE_COST = 100.0
MISSED_CRITICAL_COST = 1000.0

# Recommend maintenance if the lower RUL bound is at or
# below this threshold.
ACTION_RUL_THRESHOLD = HIGH_RUL


# ============================================================
# 1. INPUT VALIDATION
# ============================================================

REQUIRED_COLUMNS = [
    "unit_id",
    "actual_RUL",
    "baseline_predicted_RUL",
    "temporal_predicted_RUL",
]


def load_predictions(path, description):
    """Load and validate an existing Step 4 prediction file."""

    if not path.exists():
        raise FileNotFoundError(
            f"\n{description} file was not found:\n{path}\n\n"
            "Check that Step 4 completed successfully and that "
            "the filename matches the expected name."
        )

    df = pd.read_csv(path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\nMissing columns in {path.name}: {missing}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    if df.empty:
        raise ValueError(f"{description} contains no rows.")

    df = df.copy()

    numeric_columns = [
        "actual_RUL",
        "baseline_predicted_RUL",
        "temporal_predicted_RUL",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            f"{description} contains missing values in required columns."
        )

    numeric_values = df[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            f"{description} contains infinite or non-finite values."
        )

    if "cycle" in df.columns:
        df["cycle"] = pd.to_numeric(df["cycle"], errors="raise")

    return df


def final_row_per_engine(df):
    """
    Return the last observed row for each engine.

    Sorting by cycle ensures that the final observed cycle is selected
    when a cycle column is present.
    """

    data = df.copy()

    if "cycle" in data.columns:
        data = data.sort_values(
            ["unit_id", "cycle"],
            kind="stable",
        )

    result = (
        data.groupby("unit_id", sort=False, group_keys=False)
        .tail(1)
        .copy()
        .reset_index(drop=True)
    )

    if result["unit_id"].duplicated().any():
        raise RuntimeError(
            "Internal error: duplicate engine IDs after final-row selection."
        )

    return result


# ============================================================
# 2. CONFORMAL INTERVAL CALIBRATION
# ============================================================

def conformal_residual_quantile(residuals, alpha):
    """
    Calculate a finite-sample split-conformal absolute-residual quantile.

    For n calibration observations, use the order statistic with
    one-based rank ceil((n + 1) * (1 - alpha)), capped at n.

    With a small calibration sample, the resulting radius can be
    unstable. The exchangeability assumption may also fail when
    engine observations are dependent or the deployment distribution
    differs from the calibration distribution.
    """

    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError(
            "Cannot calibrate intervals: no finite residuals."
        )

    if not 0.0 < alpha < 1.0:
        raise ValueError("ALPHA must be strictly between 0 and 1.")

    values = np.sort(values)
    n = len(values)

    rank = math.ceil((n + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), n)

    return float(values[rank - 1])


def calibrate_model(validation, prediction_column):
    """Calibrate an interval radius using validation engines only."""

    actual = validation["actual_RUL"].to_numpy(dtype=float)
    predicted = validation[prediction_column].to_numpy(dtype=float)

    residuals = np.abs(actual - predicted)

    radius = conformal_residual_quantile(residuals, ALPHA)

    return {
        "radius": radius,
        "n_calibration_engines": int(len(residuals)),
        "alpha": float(ALPHA),
        "nominal_coverage": float(1.0 - ALPHA),
        "mean_absolute_calibration_error": float(np.mean(residuals)),
        "maximum_absolute_calibration_error": float(np.max(residuals)),
    }


def add_interval(df, prediction_column, radius, prefix):
    """Add nonnegative lower and upper RUL bounds."""

    result = df.copy()

    prediction = result[prediction_column].to_numpy(dtype=float)

    lower = np.maximum(0.0, prediction - radius)
    upper = np.maximum(0.0, prediction + radius)

    result[f"{prefix}_lower"] = lower
    result[f"{prefix}_upper"] = upper
    result[f"{prefix}_width"] = upper - lower

    return result


# ============================================================
# 3. RISK CLASSIFICATION
# ============================================================

def classify_rul(values):
    """
    Point-estimate risk categories.

    Critical: RUL <= 15
    High:     15 < RUL <= 30
    Medium:   30 < RUL <= 60
    Low:      RUL > 60
    """

    values = np.asarray(values, dtype=float)

    return np.select(
        [
            values <= CRITICAL_RUL,
            values <= HIGH_RUL,
            values <= MEDIUM_RUL,
        ],
        [
            "Critical",
            "High",
            "Medium",
        ],
        default="Low",
    )


def add_model_decisions(df, prefix):
    """Add risk category, uncertainty warning, and action recommendation."""

    result = df.copy()

    prediction_column = f"{prefix}_predicted_RUL"
    lower_column = f"{prefix}_lower"

    result[f"{prefix}_point_risk"] = classify_rul(
        result[prediction_column].to_numpy(dtype=float)
    )

    # Conservative decision rule: act if the lower RUL bound
    # falls within the maintenance action horizon.
    result[f"{prefix}_maintenance_action"] = (
        result[lower_column] <= ACTION_RUL_THRESHOLD
    )

    # Flag predictions whose point estimate appears safe but whose
    # lower interval bound enters the high-risk horizon.
    result[f"{prefix}_uncertainty_warning"] = (
        (result[lower_column] <= HIGH_RUL)
        & (result[prediction_column] > HIGH_RUL)
    )

    return result


# ============================================================
# 4. REGRESSION METRICS
# ============================================================

def regression_metrics(actual, predicted):
    """Calculate RMSE, MAE, and the asymmetric NASA scoring function."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - actual

    exponent = np.where(
        error < 0,
        -error / 13.0,
        error / 10.0,
    )

    nasa_score = float(
        np.sum(np.exp(np.clip(exponent, 0, 50)) - 1.0)
    )

    return {
        "n": int(len(actual)),
        "RMSE": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "NASA_score": nasa_score,
    }


# ============================================================
# 5. RISK CLASSIFICATION METRICS
# ============================================================

def risk_classification_metrics(actual_risk, predicted_risk):
    """Evaluate four-category risk classification."""

    labels = ["Critical", "High", "Medium", "Low"]

    matrix = confusion_matrix(
        actual_risk,
        predicted_risk,
        labels=labels,
    )

    n = int(matrix.sum())

    accuracy = (
        float(np.trace(matrix) / n)
        if n > 0
        else None
    )

    actual_critical = np.asarray(actual_risk) == "Critical"
    predicted_critical = np.asarray(predicted_risk) == "Critical"

    tp = int(np.sum(actual_critical & predicted_critical))
    fp = int(np.sum(~actual_critical & predicted_critical))
    fn = int(np.sum(actual_critical & ~predicted_critical))
    tn = int(np.sum(~actual_critical & ~predicted_critical))

    precision = (
        float(tp / (tp + fp))
        if (tp + fp) > 0
        else None
    )

    recall = (
        float(tp / (tp + fn))
        if (tp + fn) > 0
        else None
    )

    return {
        "class_order": labels,
        "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
        "accuracy": accuracy,
        "critical_screening": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
        },
    }


# ============================================================
# 6. SIMULATED MAINTENANCE DECISION EVALUATION
# ============================================================

def evaluate_decisions(df, prefix):
    """
    Evaluate a fixed maintenance rule.

    Simulated total cost =
        maintenance actions * assumed maintenance cost
        + missed critical engines * assumed missed-critical cost.

    A false alarm here means an action was recommended even though
    actual RUL was greater than the configured HIGH_RUL threshold.

    These assumptions are for methodological demonstration only.
    """

    actions = df[f"{prefix}_maintenance_action"].to_numpy(dtype=bool)
    actual = df["actual_RUL"].to_numpy(dtype=float)

    actual_critical = actual <= CRITICAL_RUL
    actual_high_or_critical = actual <= HIGH_RUL

    missed_critical = actual_critical & ~actions
    missed_high_or_critical = actual_high_or_critical & ~actions

    false_alarm_actions = actions & (actual > HIGH_RUL)

    n_actions = int(np.sum(actions))
    n_missed_critical = int(np.sum(missed_critical))

    maintenance_cost = n_actions * MAINTENANCE_COST
    missed_critical_cost = n_missed_critical * MISSED_CRITICAL_COST

    return {
        "n_engines": int(len(df)),
        "maintenance_actions": n_actions,
        "actual_critical_engines": int(np.sum(actual_critical)),
        "actual_high_or_critical_engines": int(
            np.sum(actual_high_or_critical)
        ),
        "missed_critical_engines": n_missed_critical,
        "missed_high_or_critical_engines": int(
            np.sum(missed_high_or_critical)
        ),
        "false_alarm_actions": int(np.sum(false_alarm_actions)),
        "assumed_maintenance_cost_per_action": float(MAINTENANCE_COST),
        "assumed_missed_critical_cost_per_engine": float(
            MISSED_CRITICAL_COST
        ),
        "simulated_maintenance_cost": float(maintenance_cost),
        "simulated_missed_critical_cost": float(missed_critical_cost),
        "simulated_total_cost": float(
            maintenance_cost + missed_critical_cost
        ),
        "costs_are_empirical": False,
    }


# ============================================================
# 7. ROBUST VISUALIZATIONS
# ============================================================

def plot_interval_width(df, output_path):
    """
    Compare interval widths using boxplots instead of histograms.

    Boxplots do not require histogram bin edges, so this remains
    valid when widths are constant or nearly constant.
    """

    columns = ["baseline_width", "temporal_width"]

    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Cannot plot interval widths; missing columns: {missing}"
        )

    baseline = pd.to_numeric(
        df["baseline_width"], errors="raise"
    ).to_numpy(dtype=float)

    temporal = pd.to_numeric(
        df["temporal_width"], errors="raise"
    ).to_numpy(dtype=float)

    if len(baseline) == 0 or len(temporal) == 0:
        raise ValueError("Cannot plot empty interval-width data.")

    if (
        not np.isfinite(baseline).all()
        or not np.isfinite(temporal).all()
    ):
        raise ValueError("Interval widths contain non-finite values.")

    fig, ax = plt.subplots(figsize=(9, 5))

    # Use set_xticks/set_xticklabels for broader Matplotlib compatibility.
    ax.boxplot(
        [baseline, temporal],
        showmeans=True,
        showfliers=True,
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "Temporal"])

    summary = (
        f"Baseline: mean={np.mean(baseline):.2f}, "
        f"median={np.median(baseline):.2f}\n"
        f"Temporal: mean={np.mean(temporal):.2f}, "
        f"median={np.median(temporal):.2f}"
    )

    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    ax.set_ylabel("Prediction interval width (RUL cycles)")
    ax.set_title("Test Prediction Interval Width")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_risk_counts(df, output_path):
    """Compare predicted point-estimate risk categories."""

    labels = ["Critical", "High", "Medium", "Low"]

    baseline_counts = (
        df["baseline_point_risk"]
        .value_counts()
        .reindex(labels, fill_value=0)
    )

    temporal_counts = (
        df["temporal_point_risk"]
        .value_counts()
        .reindex(labels, fill_value=0)
    )

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        baseline_counts.to_numpy(),
        width,
        label="Baseline",
    )

    ax.bar(
        x + width / 2,
        temporal_counts.to_numpy(),
        width,
        label="Temporal",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of engines")
    ax.set_title("Test Engine Risk Classification")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(df, output_path):
    """Plot actual RUL against each model's prediction."""

    actual = df["actual_RUL"].to_numpy(dtype=float)
    baseline = df["baseline_predicted_RUL"].to_numpy(dtype=float)
    temporal = df["temporal_predicted_RUL"].to_numpy(dtype=float)

    low = float(min(np.min(actual), np.min(baseline), np.min(temporal)))
    high = float(max(np.max(actual), np.max(baseline), np.max(temporal)))

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(actual, baseline, alpha=0.7, label="Baseline")
    ax.scatter(actual, temporal, alpha=0.7, label="Temporal")
    ax.plot([low, high], [low, high], linestyle="--", label="Ideal")

    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Final-Cycle Test Predictions")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("NASA C-MAPSS — STEP 6")
    print("UNCERTAINTY-AWARE RUL AND MAINTENANCE DECISIONS")
    print(f"Subset: {SUBSET}")
    print("=" * 68)

    validation_path = (
        STEP04_DIR / f"validation_predictions_{SUBSET}.csv"
    )

    test_path = STEP04_DIR / f"test_predictions_{SUBSET}.csv"

    validation_all = load_predictions(
        validation_path,
        "Validation predictions",
    )

    test = load_predictions(
        test_path,
        "Test predictions",
    )

    validation = final_row_per_engine(validation_all)
    test = final_row_per_engine(test)

    if len(validation) < 10:
        warnings.warn(
            "Fewer than 10 validation engines are available. "
            "Interval calibration is likely to be unstable."
        )

    print("\nCALIBRATION DATA")
    print("-" * 48)
    print("Validation rows available:", len(validation_all))
    print("Final validation engines:", len(validation))
    print("Final test engines:", len(test))
    print(f"Nominal interval level: {(1 - ALPHA) * 100:.0f}%")

    # --------------------------------------------------------
    # A. Calibrate using validation engines only.
    # --------------------------------------------------------

    calibration = {
        "baseline": calibrate_model(
            validation,
            "baseline_predicted_RUL",
        ),
        "temporal": calibrate_model(
            validation,
            "temporal_predicted_RUL",
        ),
    }

    print(
        "Baseline interval radius:",
        f"{calibration['baseline']['radius']:.3f} cycles",
    )

    print(
        "Temporal interval radius:",
        f"{calibration['temporal']['radius']:.3f} cycles",
    )

    # Save calibration residuals for reproducibility.
    calibration_rows = []

    for _, row in validation.iterrows():
        record = {
            "unit_id": row["unit_id"],
            "actual_RUL": float(row["actual_RUL"]),
            "baseline_predicted_RUL": float(
                row["baseline_predicted_RUL"]
            ),
            "temporal_predicted_RUL": float(
                row["temporal_predicted_RUL"]
            ),
        }

        record["baseline_absolute_residual"] = abs(
            record["actual_RUL"] - record["baseline_predicted_RUL"]
        )

        record["temporal_absolute_residual"] = abs(
            record["actual_RUL"] - record["temporal_predicted_RUL"]
        )

        calibration_rows.append(record)

    pd.DataFrame(calibration_rows).to_csv(
        OUTPUT_DIR / f"validation_calibration_residuals_{SUBSET}.csv",
        index=False,
    )

    # --------------------------------------------------------
    # B. Add intervals to validation and test data.
    # --------------------------------------------------------

    validation_result = validation.copy()
    test_result = test.copy()

    for prefix, prediction_column in [
        ("baseline", "baseline_predicted_RUL"),
        ("temporal", "temporal_predicted_RUL"),
    ]:
        radius = calibration[prefix]["radius"]

        validation_result = add_interval(
            validation_result,
            prediction_column,
            radius,
            prefix,
        )

        test_result = add_interval(
            test_result,
            prediction_column,
            radius,
            prefix,
        )

    # --------------------------------------------------------
    # C. Risk categories and maintenance recommendations.
    # --------------------------------------------------------

    for prefix in ["baseline", "temporal"]:
        validation_result = add_model_decisions(
            validation_result,
            prefix,
        )

        test_result = add_model_decisions(
            test_result,
            prefix,
        )

    validation_result["actual_risk"] = classify_rul(
        validation_result["actual_RUL"].to_numpy(dtype=float)
    )

    test_result["actual_risk"] = classify_rul(
        test_result["actual_RUL"].to_numpy(dtype=float)
    )

    validation_result.to_csv(
        OUTPUT_DIR / f"validation_uncertainty_risk_{SUBSET}.csv",
        index=False,
    )

    test_result.to_csv(
        OUTPUT_DIR / f"test_uncertainty_risk_{SUBSET}.csv",
        index=False,
    )

    # --------------------------------------------------------
    # D. Evaluate intervals.
    # Test coverage is descriptive only; it does not calibrate
    # the interval or determine the decision policy.
    # --------------------------------------------------------

    interval_metrics = {}

    for prefix in ["baseline", "temporal"]:
        actual = test_result["actual_RUL"].to_numpy(dtype=float)
        lower = test_result[f"{prefix}_lower"].to_numpy(dtype=float)
        upper = test_result[f"{prefix}_upper"].to_numpy(dtype=float)

        covered = (actual >= lower) & (actual <= upper)
        widths = upper - lower

        interval_metrics[prefix] = {
            "nominal_coverage": float(1.0 - ALPHA),
            "observed_test_coverage": float(np.mean(covered)),
            "covered_test_engines": int(np.sum(covered)),
            "test_engines": int(len(covered)),
            "mean_interval_width": float(np.mean(widths)),
            "median_interval_width": float(np.median(widths)),
            "evaluation_note": (
                "Descriptive test-set result; test labels were not "
                "used to calibrate intervals."
            ),
        }

    # --------------------------------------------------------
    # E. Point-estimate risk classification metrics.
    # --------------------------------------------------------

    risk_metrics = {}

    for prefix in ["baseline", "temporal"]:
        predicted_risk = classify_rul(
            test_result[f"{prefix}_predicted_RUL"].to_numpy(dtype=float)
        )

        risk_metrics[prefix] = risk_classification_metrics(
            test_result["actual_risk"].to_numpy(),
            predicted_risk,
        )

    # --------------------------------------------------------
    # F. Simulated maintenance decisions and costs.
    # --------------------------------------------------------

    decision_metrics = {}

    for prefix in ["baseline", "temporal"]:
        decision_metrics[prefix] = evaluate_decisions(
            test_result,
            prefix,
        )

    # --------------------------------------------------------
    # G. Regression metrics on the same final-cycle test engines.
    # --------------------------------------------------------

    actual_test = test_result["actual_RUL"].to_numpy(dtype=float)

    test_regression_metrics = {
        "baseline": regression_metrics(
            actual_test,
            test_result["baseline_predicted_RUL"].to_numpy(dtype=float),
        ),
        "temporal": regression_metrics(
            actual_test,
            test_result["temporal_predicted_RUL"].to_numpy(dtype=float),
        ),
    }

    # --------------------------------------------------------
    # H. Generate plots.
    # --------------------------------------------------------

    plot_interval_width(
        test_result,
        OUTPUT_DIR / f"test_interval_width_{SUBSET}.png",
    )

    plot_risk_counts(
        test_result,
        OUTPUT_DIR / f"test_risk_counts_{SUBSET}.png",
    )

    plot_actual_vs_predicted(
        test_result,
        OUTPUT_DIR / f"test_actual_vs_predicted_{SUBSET}.png",
    )

    # --------------------------------------------------------
    # I. Build the report.
    # --------------------------------------------------------

    report = {
        "dataset": "NASA C-MAPSS",
        "subset": SUBSET,
        "step": 6,
        "purpose": (
            "Uncertainty-aware remaining useful life prediction, "
            "risk classification, and illustrative maintenance decisions."
        ),
        "calibration_method": (
            "Split conformal absolute-residual quantiles calculated "
            "from one final observed validation row per engine."
        ),
        "calibration": calibration,
        "interval_evaluation_on_test": interval_metrics,
        "test_regression_metrics": test_regression_metrics,
        "risk_thresholds_cycles": {
            "critical": f"RUL <= {CRITICAL_RUL}",
            "high": f"{CRITICAL_RUL} < RUL <= {HIGH_RUL}",
            "medium": f"{HIGH_RUL} < RUL <= {MEDIUM_RUL}",
            "low": f"RUL > {MEDIUM_RUL}",
        },
        "maintenance_policy": {
            "action_rule": (
                "Recommend maintenance if the lower prediction "
                "interval bound is at or below the action threshold."
            ),
            "action_threshold_cycles": ACTION_RUL_THRESHOLD,
        },
        "risk_classification_on_test": risk_metrics,
        "simulated_decision_evaluation_on_test": decision_metrics,
        "cost_assumptions": {
            "maintenance_cost_per_action": MAINTENANCE_COST,
            "missed_critical_cost_per_engine": MISSED_CRITICAL_COST,
            "currency": "unspecified illustrative units",
            "empirical_costs": False,
        },
        "limitations": [
            (
                "Only 20 held-out validation engines were available "
                "in the previously reported experiment; confirm the "
                "actual count in this run."
            ),
            (
                "Finite-sample conformal coverage relies on exchangeability "
                "between calibration and future cases."
            ),
            (
                "Temporal dependence, distribution shift, and a small "
                "calibration sample can weaken coverage guarantees."
            ),
            (
                "Clipping the lower RUL endpoint at zero changes the "
                "unclipped interval and may affect coverage."
            ),
            (
                "The test set is used only for final descriptive evaluation, "
                "not for calibration or threshold selection."
            ),
            (
                "Risk thresholds and costs are illustrative and require "
                "plant-specific validation before operational use."
            ),
            (
                "These model outputs are not a substitute for industrial "
                "safety systems or maintenance engineering review."
            ),
        ],
    }

    report_path = OUTPUT_DIR / f"step06_{SUBSET}_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, allow_nan=False)

    # --------------------------------------------------------
    # J. Console summary.
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("TEST REGRESSION METRICS")
    print("=" * 68)

    for model_name, metrics in test_regression_metrics.items():
        print(f"\n{model_name.upper()}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float)
                  else f"  {key}: {value}")

    print("\n" + "=" * 68)
    print("TEST INTERVAL METRICS")
    print("=" * 68)

    for model_name, metrics in interval_metrics.items():
        print(f"\n{model_name.upper()}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 68)
    print("SIMULATED MAINTENANCE DECISIONS")
    print("=" * 68)

    for model_name, metrics in decision_metrics.items():
        print(f"\n{model_name.upper()}")
        print(
            "  Maintenance actions:",
            metrics["maintenance_actions"],
        )
        print(
            "  Missed critical engines:",
            metrics["missed_critical_engines"],
        )
        print(
            "  Missed high-or-critical engines:",
            metrics["missed_high_or_critical_engines"],
        )
        print(
            "  False-alarm actions:",
            metrics["false_alarm_actions"],
        )
        print(
            "  Simulated total cost:",
            metrics["simulated_total_cost"],
        )

    print("\n" + "=" * 68)
    print("STEP 6 COMPLETED SUCCESSFULLY")
    print("=" * 68)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Research report: {report_path}")
    print("Steps 1–5 and their model artifacts were not modified.")


if __name__ == "__main__":
    main()
