
"""
NASA C-MAPSS FD001 — STEP 8.3
Explainable Adaptation Audit

Combines:
- Step 8.2 sensor-distribution-shift results
- Step 5 SHAP feature importance
- Step 4 baseline-versus-temporal prediction comparison
- Step 8 frozen-policy selection report
- Step 7.2 robustness report, when available

This is an audit and recommendation layer.
It does not retrain models or change/deploy maintenance policies.
Test RUL labels are not used to select policies.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "step08_3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_PATH = (
    ROOT / "outputs" / "step08_2"
    / "sensor_distribution_shift_FD001.csv"
)

STEP05_DIR = ROOT / "outputs" / "step05"
STEP04_DIR = ROOT / "outputs" / "step04"
STEP08_DIR = ROOT / "outputs" / "step08"
STEP072_DIR = ROOT / "outputs" / "step07_2"

REPORT_PATH = OUTPUT_DIR / "step08_3_FD001_report.json"
AUDIT_PATH = OUTPUT_DIR / "explainable_adaptation_audit_FD001.csv"
PLOT_PATH = OUTPUT_DIR / "adaptation_audit_FD001.png"

# These are review-screening settings, not learned parameters.
MIN_SENSORS_FOR_REVIEW = 3
MIN_SHIFTED_FRACTION_FOR_REVIEW = 0.20
PSI_REVIEW_LEVEL = 0.10
SMD_REVIEW_LEVEL = 0.50


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_csv_if_available(path, description):
    if not path.exists():
        print(f"[OPTIONAL INPUT NOT FOUND] {description}: {path}")
        return None

    try:
        df = pd.read_csv(path)
        print(f"[LOADED] {description}: {path}")
        print(f"         Rows: {len(df):,}")
        return df
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def load_json_if_available(path, description):
    if not path.exists():
        print(f"[OPTIONAL INPUT NOT FOUND] {description}: {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        print(f"[LOADED] {description}: {path}")
        return data
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def find_existing_csv(directory, keywords):
    if not directory.exists():
        return None

    candidates = sorted(directory.glob("*.csv"))

    for path in candidates:
        name = path.name.lower()
        if all(keyword.lower() in name for keyword in keywords):
            return path

    return None


def find_existing_json(directory, preferred_name):
    preferred = directory / preferred_name
    if preferred.exists():
        return preferred

    if directory.exists():
        candidates = sorted(directory.glob("*FD001*report*.json"))
        if candidates:
            return candidates[0]

        candidates = sorted(directory.glob("*report*.json"))
        if candidates:
            return candidates[0]

    return None


def safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def safe_read_shap():
    """
    Locate the Step 5 SHAP importance table using common filenames.
    """
    candidates = [
        STEP05_DIR / "shap_feature_importance_FD001.csv",
        STEP05_DIR / "shap_importance_FD001.csv",
        STEP05_DIR / "temporal_shap_feature_importance_FD001.csv",
        STEP05_DIR / "shap_global_importance_FD001.csv",
    ]

    for path in candidates:
        if path.exists():
            result = load_csv_if_available(
                path, "Step 5 SHAP importance"
            )
            if result is not None:
                return result, path

    # Fallback: find a CSV containing both "shap" and "importance".
    path = find_existing_csv(
        STEP05_DIR, ["shap", "importance"]
    )

    if path is not None:
        return (
            load_csv_if_available(path, "Step 5 SHAP importance"),
            path,
        )

    print(
        "[OPTIONAL INPUT NOT FOUND] Step 5 SHAP importance CSV. "
        "The audit will continue without feature-level SHAP matching."
    )
    return None, None


def find_feature_column(df):
    for candidate in [
        "feature", "Feature", "feature_name", "sensor"
    ]:
        if candidate in df.columns:
            return candidate
    return None


def find_shap_value_column(df):
    for candidate in [
        "mean_absolute_SHAP",
        "mean_abs_shap",
        "mean_abs_SHAP",
        "mean_absolute_shap",
        "mean_abs_shap_value",
        "shap_importance",
        "importance",
    ]:
        if candidate in df.columns:
            return candidate

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if "rank" not in c.lower()
        and "index" not in c.lower()
    ]

    return numeric_cols[0] if numeric_cols else None


def read_prediction_metrics():
    """
    Recalculate regression metrics from Step 4 prediction tables.

    Validation metrics use all validation prediction rows.
    Test metrics use final-cycle test predictions supplied by Step 4.
    Test metrics are descriptive only; they do not tune the policy.
    """
    results = []

    files = [
        (
            STEP04_DIR / "validation_predictions_FD001.csv",
            "validation",
        ),
        (
            STEP04_DIR / "test_predictions_FD001.csv",
            "test_descriptive",
        ),
    ]

    for path, split in files:
        df = load_csv_if_available(path, f"{split} predictions")

        if df is None:
            continue

        required = [
            "actual_RUL",
            "baseline_predicted_RUL",
            "temporal_predicted_RUL",
        ]

        if not all(c in df.columns for c in required):
            print(
                f"[SKIPPED] {path.name}: expected prediction columns "
                f"were not all found."
            )
            continue

        actual = pd.to_numeric(
            df["actual_RUL"], errors="coerce"
        ).to_numpy(dtype=float)

        for model, col in [
            ("baseline", "baseline_predicted_RUL"),
            ("temporal", "temporal_predicted_RUL"),
        ]:
            predicted = pd.to_numeric(
                df[col], errors="coerce"
            ).to_numpy(dtype=float)

            valid = np.isfinite(actual) & np.isfinite(predicted)

            if valid.sum() == 0:
                continue

            error = predicted[valid] - actual[valid]
            absolute_error = np.abs(error)

            # Standard NASA C-MAPSS asymmetric score:
            # overestimation is penalized more slowly than
            # underestimation in the commonly used formulation.
            # The convention is stated explicitly in the report.
            d = predicted[valid] - actual[valid]
            score = np.where(
                d < 0,
                np.expm1(-d / 13.0),
                np.expm1(d / 10.0),
            )
            score = np.clip(score, 0, 1e100)

            results.append({
                "split": split,
                "model": model,
                "n_rows": int(valid.sum()),
                "RMSE": float(np.sqrt(np.mean(error ** 2))),
                "MAE": float(np.mean(absolute_error)),
                "NASA_score": float(np.sum(score)),
                "mean_error_predicted_minus_actual": float(
                    np.mean(error)
                ),
            })

    return pd.DataFrame(results)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():
    section(
        "NASA C-MAPSS — STEP 8.3\n"
        "EXPLAINABLE ADAPTATION AUDIT"
    )
    print("Subset: FD001")
    print(
        "UTC run time:",
        datetime.now(timezone.utc).isoformat(),
    )

    # --------------------------------------------------------
    # 1. Load the drift evidence
    # --------------------------------------------------------

    section("1. SENSOR-DRIFT EVIDENCE")

    drift = load_csv_if_available(
        DRIFT_PATH, "Step 8.2 sensor distribution-shift results"
    )

    if drift is None:
        raise FileNotFoundError(
            "Required Step 8.2 CSV is missing:\n"
            f"{DRIFT_PATH}\n"
            "Run Step 8.2 successfully before Step 8.3."
        )

    if "sensor" not in drift.columns:
        raise ValueError(
            "Step 8.2 CSV does not contain the expected 'sensor' column."
        )

    if "drift_status" not in drift.columns:
        raise ValueError(
            "Step 8.2 CSV does not contain the expected 'drift_status' column."
        )

    if "psi" not in drift.columns:
        drift["psi"] = np.nan

    if "absolute_standardized_mean_difference" not in drift.columns:
        drift["absolute_standardized_mean_difference"] = np.nan

    drift["psi"] = pd.to_numeric(
        drift["psi"], errors="coerce"
    )
    drift["absolute_standardized_mean_difference"] = pd.to_numeric(
        drift["absolute_standardized_mean_difference"],
        errors="coerce",
    )

    moderate_major = drift["drift_status"].isin(
        ["moderate_shift", "major_shift"]
    )

    elevated_effect = (
        (drift["psi"] >= PSI_REVIEW_LEVEL)
        | (
            drift["absolute_standardized_mean_difference"]
            >= SMD_REVIEW_LEVEL
        )
    )

    # Reassess from recorded sensor evidence, using the predefined
    # screening rule rather than assuming Step 8.2's conclusion.
    review_sensor_mask = moderate_major | elevated_effect
    review_sensors = drift.loc[review_sensor_mask].copy()

    total_sensors = int(len(drift))
    n_review_sensors = int(len(review_sensors))
    review_fraction = n_review_sensors / max(total_sensors, 1)

    drift_review_trigger = (
        n_review_sensors >= MIN_SENSORS_FOR_REVIEW
        or review_fraction >= MIN_SHIFTED_FRACTION_FOR_REVIEW
    )

    print(f"Sensors assessed: {total_sensors}")
    print(f"Sensors flagged for review: {n_review_sensors}")
    print(f"Flagged fraction: {review_fraction:.3f}")
    print(f"Review trigger: {drift_review_trigger}")

    # --------------------------------------------------------
    # 2. Match drifted sensors to SHAP feature importance
    # --------------------------------------------------------

    section("2. EXPLAINABILITY ALIGNMENT")

    shap_df, shap_path = safe_read_shap()
    feature_column = None
    shap_value_column = None

    if shap_df is not None:
        feature_column = find_feature_column(shap_df)
        shap_value_column = find_shap_value_column(shap_df)

    if (
        shap_df is not None
        and feature_column is not None
        and shap_value_column is not None
    ):
        shap_df = shap_df.copy()
        shap_df[feature_column] = (
            shap_df[feature_column].astype(str)
        )
        shap_df[shap_value_column] = pd.to_numeric(
            shap_df[shap_value_column], errors="coerce"
        )

        # Aggregate SHAP values to sensor-level by matching either
        # exact sensor names or feature names beginning "sensor_NN_".
        shap_df["_sensor_key"] = shap_df[feature_column].str.extract(
            r"(sensor_\d+)", expand=False
        )

        shap_sensor = (
            shap_df.dropna(subset=["_sensor_key"])
            .groupby("_sensor_key", as_index=False)[shap_value_column]
            .max()
            .rename(columns={
                "_sensor_key": "sensor",
                shap_value_column: "max_mean_absolute_SHAP",
            })
        )

        audit = drift.merge(
            shap_sensor, on="sensor", how="left"
        )
    else:
        print(
            "SHAP table could not be matched using known columns. "
            "Continuing with sensor statistics only."
        )
        audit = drift.copy()
        audit["max_mean_absolute_SHAP"] = np.nan

    audit["review_flag"] = (
        audit["drift_status"].isin(
            ["moderate_shift", "major_shift"]
        )
        | (audit["psi"] >= PSI_REVIEW_LEVEL)
        | (
            audit["absolute_standardized_mean_difference"]
            >= SMD_REVIEW_LEVEL
        )
    )

    # Rank only within this audit. A high SHAP value means the feature
    # contributed to model predictions; it does not imply causation.
    audit["explainability_note"] = np.where(
        audit["max_mean_absolute_SHAP"].notna(),
        "Matched to Step 5 SHAP sensor feature(s); association is not causation.",
        "No matched sensor-level SHAP value; investigate feature-name mapping.",
    )

    audit = audit.sort_values(
        by=[
            "review_flag",
            "psi",
            "max_mean_absolute_SHAP",
        ],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 3. Recalculate baseline/temporal prediction metrics
    # --------------------------------------------------------

    section("3. PREDICTION-PERFORMANCE CONTEXT")

    metrics = read_prediction_metrics()

    if metrics.empty:
        print(
            "No compatible Step 4 prediction files were found. "
            "The audit will report sensor/SHAP evidence only."
        )
    else:
        print(metrics.to_string(index=False))

    validation_metrics = (
        metrics[metrics["split"] == "validation"]
        if not metrics.empty else pd.DataFrame()
    )

    performance_comparison = {}

    if not validation_metrics.empty:
        indexed = validation_metrics.set_index("model")

        if {"baseline", "temporal"}.issubset(indexed.index):
            for metric in ["RMSE", "MAE", "NASA_score"]:
                baseline_value = safe_float(
                    indexed.loc["baseline", metric]
                )
                temporal_value = safe_float(
                    indexed.loc["temporal", metric]
                )

                if (
                    baseline_value is not None
                    and temporal_value is not None
                    and baseline_value != 0
                ):
                    performance_comparison[
                        f"{metric}_temporal_improvement_percent"
                    ] = float(
                        100.0
                        * (baseline_value - temporal_value)
                        / abs(baseline_value)
                    )

    if performance_comparison:
        print("\nValidation temporal-versus-baseline changes:")
        for key, value in performance_comparison.items():
            print(f"{key}: {value:.3f}%")

    # --------------------------------------------------------
    # 4. Read prior policy reports without changing them
    # --------------------------------------------------------

    section("4. FROZEN-POLICY CONTEXT")

    step08_report_path = find_existing_json(
        STEP08_DIR, "step08_FD001_report.json"
    )
    step072_report_path = find_existing_json(
        STEP072_DIR, "step07_2_FD001_report.json"
    )

    step08_report = (
        load_json_if_available(
            step08_report_path, "Step 8 policy report"
        )
        if step08_report_path else None
    )

    step072_report = (
        load_json_if_available(
            step072_report_path, "Step 7.2 robustness report"
        )
        if step072_report_path else None
    )

    print(
        "Policy reports are read-only evidence. "
        "This script does not change their thresholds."
    )

    # --------------------------------------------------------
    # 5. Transparent audit recommendations
    # --------------------------------------------------------

    section("5. AUDIT DECISION")

    recommendations = []

    if drift_review_trigger:
        recommendations.append(
            "Review sensor distributions and feature stability before "
            "relying on the current policy under changed conditions."
        )
    else:
        recommendations.append(
            "Continue monitoring; the configured sensor-shift "
            "screening rule was not met."
        )

    shap_matched = (
        "max_mean_absolute_SHAP" in audit.columns
        and audit["max_mean_absolute_SHAP"].notna().any()
    )

    if shap_matched:
        recommendations.append(
            "Prioritize review of sensors that are both shifted and "
            "associated with high SHAP importance, while treating SHAP "
            "as predictive attribution rather than causal evidence."
        )
    else:
        recommendations.append(
            "Verify Step 5 SHAP output column names and sensor-feature "
            "mapping before claiming that explanations align with drift."
        )

    if performance_comparison:
        rmse_change = performance_comparison.get(
            "RMSE_temporal_improvement_percent"
        )

        if rmse_change is not None and rmse_change < 0:
            recommendations.append(
                "Temporal model validation RMSE is worse than baseline "
                "in this comparison; investigate before preferring the "
                "temporal model."
            )
        elif rmse_change is not None:
            recommendations.append(
                "Temporal model validation RMSE is better in this "
                "comparison, but confirm stability across additional "
                "engine-level splits before making a broad claim."
            )

    recommendations.append(
        "Any recalibration or policy update must be evaluated using "
        "labeled validation data and compared with the frozen policy."
    )

    recommendations.append(
        "Do not interpret training-to-test sensor shift alone as "
        "confirmed temporal drift, causal degradation, or proof of "
        "maintenance-policy failure."
    )

    print("Recommendations:")
    for i, recommendation in enumerate(recommendations, start=1):
        print(f"{i}. {recommendation}")

    # --------------------------------------------------------
    # 6. Save audit table and plots
    # --------------------------------------------------------

    audit.to_csv(AUDIT_PATH, index=False)

    plot_df = audit.copy()
    plot_df = plot_df.sort_values("psi", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot_df["sensor"], plot_df["psi"])
    ax.axvline(
        PSI_REVIEW_LEVEL,
        linestyle="--",
        label=f"PSI review level ({PSI_REVIEW_LEVEL})",
    )
    ax.set_xlabel("Population Stability Index")
    ax.set_ylabel("Sensor")
    ax.set_title(
        "Step 8.3 — Sensor Shift Evidence for Adaptation Review"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)

    report = {
        "step": "8.3",
        "project": "NASA C-MAPSS",
        "subset": "FD001",
        "run_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "purpose": (
            "Auditable synthesis of sensor-shift evidence, SHAP "
            "feature attribution, model metrics, and frozen-policy context."
        ),
        "drift_screening": {
            "sensors_assessed": total_sensors,
            "sensors_flagged_for_review": n_review_sensors,
            "flagged_sensor_fraction": float(review_fraction),
            "review_triggered": bool(drift_review_trigger),
            "minimum_sensor_count": MIN_SENSORS_FOR_REVIEW,
            "minimum_fraction": MIN_SHIFTED_FRACTION_FOR_REVIEW,
        },
        "explainability": {
            "shap_file_found": shap_path is not None,
            "shap_file": str(shap_path) if shap_path else None,
            "feature_column": feature_column,
            "shap_value_column": shap_value_column,
            "sensor_level_shap_match_available": bool(shap_matched),
            "interpretation": (
                "SHAP values describe predictive attribution, not causation."
            ),
        },
        "prediction_metrics": (
            metrics.replace({np.nan: None}).to_dict(orient="records")
            if not metrics.empty else []
        ),
        "validation_temporal_vs_baseline": performance_comparison,
        "policy_context": {
            "step08_report_found": step08_report is not None,
            "step072_report_found": step072_report is not None,
            "policy_modified": False,
            "model_retrained": False,
            "test_labels_used_for_policy_selection": False,
        },
        "recommendations": recommendations,
        "limitations": [
            "This is a review and audit layer, not an adaptation algorithm.",
            "Distribution shift does not prove concept drift or causal degradation.",
            "SHAP feature importance is not causal evidence.",
            "Prediction metrics from test labels, if present in the Step 4 "
            "artifact, are descriptive and are not used for policy selection.",
            "A deployment claim requires repeated engine-level validation, "
            "a frozen-policy comparison, and explicit safety constraints.",
        ],
        "files": {
            "audit_csv": str(AUDIT_PATH),
            "audit_plot": str(PLOT_PATH),
            "report_json": str(REPORT_PATH),
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    section("STEP 8.3 COMPLETED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Audit table: {AUDIT_PATH}")
    print(f"Audit plot: {PLOT_PATH}")
    print(f"Research report: {REPORT_PATH}")
    print("\nNo model was retrained.")
    print("No maintenance policy was modified or deployed.")
    print("Steps 1–8.2 were not modified.")


if __name__ == "__main__":
    main()
