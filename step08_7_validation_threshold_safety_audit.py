
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 8.7
# Validation-only threshold reassessment and safety audit
# NASA C-MAPSS FD001
#
# Uses validation data only.
# Does not load test predictions or test labels.
# Does not retrain models or modify/deploy the frozen policy.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "step08_7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATED_FILE = (
    PROJECT_DIR / "outputs" / "step08_5"
    / "validation_calibrated_predictions_FD001.csv"
)

RAW_VALIDATION_FILE = (
    PROJECT_DIR / "outputs" / "step04"
    / "validation_predictions_FD001.csv"
)

# Illustrative assumptions inherited from Step 8.
# These are NOT real industrial cost estimates.
CRITICAL_RUL = 15.0
HIGH_RISK_RUL = 30.0

MAINTENANCE_COST = 100.0
MISSED_CRITICAL_COST = 1000.0
FALSE_ALARM_COST = 50.0

MAX_MISSED_CRITICAL_ENGINES = 0
MAX_HIGH_RISK_MISS_RATE = 0.10

THRESHOLDS = np.arange(0.0, 200.0 + 5.0, 5.0)

FROZEN_THRESHOLDS = {
    "baseline": 10.0,
    "temporal": 5.0,
}

RANDOM_SEED = 42
BOOTSTRAP_DRAWS = 2000


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """Find a column by case-insensitive exact name."""
    mapping = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in mapping:
            return mapping[key]

    raise ValueError(
        f"Cannot find {description} column.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def nasa_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    NASA C-MAPSS asymmetric score.

    Error = predicted RUL - actual RUL.
    Positive error (RUL overestimated): denominator 10.
    Negative error (RUL underestimated): denominator 13.
    """
    error = predicted - actual
    penalties = np.where(
        error < 0,
        np.expm1(np.clip(-error / 13.0, 0, 700)),
        np.expm1(np.clip(error / 10.0, 0, 700)),
    )
    return float(np.sum(penalties))


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    error = predicted - actual

    return {
        "n_engines": int(len(actual)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "NASA_score": nasa_score(actual, predicted),
        "mean_error_predicted_minus_actual": float(np.mean(error)),
    }


def policy_metrics(
    df: pd.DataFrame,
    prediction_col: str,
    threshold: float,
) -> dict:
    actual = df["actual_RUL"].to_numpy(dtype=float)
    predicted = df[prediction_col].to_numpy(dtype=float)

    # One decision per engine, using its final observed validation cycle.
    action = predicted <= threshold

    critical = actual <= CRITICAL_RUL
    high_risk = actual <= HIGH_RISK_RUL

    missed_critical = int(np.sum(critical & ~action))
    missed_high_risk = int(np.sum(high_risk & ~action))
    n_high_risk = int(np.sum(high_risk))
    n_critical = int(np.sum(critical))

    false_alarms = int(np.sum(action & ~high_risk))
    actions = int(np.sum(action))

    high_risk_miss_rate = (
        float(missed_high_risk / n_high_risk)
        if n_high_risk > 0
        else 0.0
    )

    cost = (
        actions * MAINTENANCE_COST
        + missed_critical * MISSED_CRITICAL_COST
        + false_alarms * FALSE_ALARM_COST
    )

    feasible = (
        missed_critical <= MAX_MISSED_CRITICAL_ENGINES
        and high_risk_miss_rate <= MAX_HIGH_RISK_MISS_RATE
    )

    return {
        "threshold_cycles": float(threshold),
        "n_engines": int(len(df)),
        "maintenance_actions": actions,
        "critical_engines": n_critical,
        "missed_critical_engines": missed_critical,
        "high_risk_engines": n_high_risk,
        "missed_high_risk_engines": missed_high_risk,
        "high_risk_miss_rate": high_risk_miss_rate,
        "false_alarm_engines": false_alarms,
        "illustrative_cost": float(cost),
        "safety_feasible": bool(feasible),
    }


def final_snapshot_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Keep exactly one final observed row for each validation engine."""
    required = {"unit_id", "cycle", "actual_RUL"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()

    for col in ["unit_id", "cycle", "actual_RUL"]:
        out[col] = pd.to_numeric(out[col], errors="raise")

    out = out.sort_values(["unit_id", "cycle"])
    out = out.groupby("unit_id", as_index=False, sort=True).tail(1)
    out = out.sort_values("unit_id").reset_index(drop=True)

    if out["unit_id"].duplicated().any():
        raise RuntimeError("More than one final row remains for an engine.")

    return out


def bootstrap_engine_difference(
    df: pd.DataFrame,
    prediction_a: str,
    prediction_b: str,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict:
    """
    Paired engine-level bootstrap for RMSE difference:
    RMSE(A) - RMSE(B).

    Each resample samples engines, not individual time-series rows.
    This is exploratory with only 20 validation engines.
    """
    actual = df["actual_RUL"].to_numpy(dtype=float)
    a = df[prediction_a].to_numpy(dtype=float)
    b = df[prediction_b].to_numpy(dtype=float)

    observed = (
        np.sqrt(np.mean((a - actual) ** 2))
        - np.sqrt(np.mean((b - actual) ** 2))
    )

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df)
    samples = np.empty(draws, dtype=float)

    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        rmse_a = np.sqrt(np.mean((a[idx] - actual[idx]) ** 2))
        rmse_b = np.sqrt(np.mean((b[idx] - actual[idx]) ** 2))
        samples[i] = rmse_a - rmse_b

    lower, upper = np.quantile(samples, [0.025, 0.975])

    return {
        "comparison": f"{prediction_a}_minus_{prediction_b}",
        "observed_RMSE_difference": float(observed),
        "bootstrap_CI_lower_95": float(lower),
        "bootstrap_CI_upper_95": float(upper),
        "interpretation": (
            "CI favors B"
            if upper < 0
            else "CI favors A"
            if lower > 0
            else "inconclusive: CI includes zero"
        ),
        "bootstrap_draws": draws,
        "resampling_unit": "validation engine",
    }


def main() -> None:
    print("=" * 72)
    print("NASA C-MAPSS FD001 - STEP 8.7")
    print("VALIDATION-ONLY THRESHOLD REASSESSMENT AND SAFETY AUDIT")
    print("=" * 72)

    if not CALIBRATED_FILE.exists():
        raise FileNotFoundError(
            f"Required Step 8.5 file not found:\n{CALIBRATED_FILE}"
        )

    calibrated = pd.read_csv(CALIBRATED_FILE)

    unit_col = find_column(
        calibrated, ["unit_id", "engine_id", "unit"], "engine ID"
    )
    cycle_col = find_column(
        calibrated, ["cycle", "last_observed_cycle"], "cycle"
    )
    actual_col = find_column(
        calibrated, ["actual_RUL", "actual_rul", "RUL"], "actual RUL"
    )

    baseline_raw_col = find_column(
        calibrated,
        ["baseline_raw_RUL", "baseline_predicted_RUL",
         "baseline_raw", "baseline_prediction"],
        "baseline raw prediction",
    )
    baseline_cal_col = find_column(
        calibrated,
        ["baseline_calibrated_RUL", "baseline_calibrated",
         "baseline_calibrated_prediction"],
        "baseline calibrated prediction",
    )
    temporal_raw_col = find_column(
        calibrated,
        ["temporal_raw_RUL", "temporal_predicted_RUL",
         "temporal_raw", "temporal_prediction"],
        "temporal raw prediction",
    )
    temporal_cal_col = find_column(
        calibrated,
        ["temporal_calibrated_RUL", "temporal_calibrated",
         "temporal_calibrated_prediction"],
        "temporal calibrated prediction",
    )

    # Normalize column names into a stable internal schema.
    data = calibrated[
        [
            unit_col,
            cycle_col,
            actual_col,
            baseline_raw_col,
            baseline_cal_col,
            temporal_raw_col,
            temporal_cal_col,
        ]
    ].copy()

    data.columns = [
        "unit_id",
        "cycle",
        "actual_RUL",
        "baseline_raw_RUL",
        "baseline_calibrated_RUL",
        "temporal_raw_RUL",
        "temporal_calibrated_RUL",
    ]

    numeric_cols = [
        "unit_id", "cycle", "actual_RUL",
        "baseline_raw_RUL", "baseline_calibrated_RUL",
        "temporal_raw_RUL", "temporal_calibrated_RUL",
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="raise")

    if data[numeric_cols].isna().any().any():
        raise ValueError("Missing numeric values found in validation input.")

    # The Step 8.5 file should contain the validation rows only.
    # This script never opens a test prediction file.
    final_df = final_snapshot_per_engine(data)

    if len(final_df) < 2:
        raise ValueError("Too few validation engines for this audit.")

    variants = {
        "baseline_raw": "baseline_raw_RUL",
        "baseline_calibrated": "baseline_calibrated_RUL",
        "temporal_raw": "temporal_raw_RUL",
        "temporal_calibrated": "temporal_calibrated_RUL",
    }

    print(f"\nInput: {CALIBRATED_FILE}")
    print(f"Validation rows in input: {len(data):,}")
    print(f"Final-cycle validation engines: {len(final_df)}")
    print("Test predictions/labels: NOT USED")
    print("Model retraining: NOT PERFORMED")
    print("Frozen policy modification/deployment: NOT PERFORMED")

    # --------------------------------------------------------
    # 1. Final-cycle regression metrics
    # --------------------------------------------------------
    regression_rows = []

    for variant, pred_col in variants.items():
        row = {"variant": variant, "prediction_column": pred_col}
        row.update(
            regression_metrics(
                final_df["actual_RUL"].to_numpy(dtype=float),
                final_df[pred_col].to_numpy(dtype=float),
            )
        )
        regression_rows.append(row)

    regression_df = pd.DataFrame(regression_rows)
    regression_path = (
        OUTPUT_DIR / "step08_7_final_cycle_regression_metrics_FD001.csv"
    )
    regression_df.to_csv(regression_path, index=False)

    print("\nFINAL-CYCLE REGRESSION METRICS")
    print(regression_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --------------------------------------------------------
    # 2. Evaluate every candidate threshold on validation
    # --------------------------------------------------------
    candidate_rows = []

    for variant, pred_col in variants.items():
        for threshold in THRESHOLDS:
            metrics = policy_metrics(final_df, pred_col, float(threshold))
            metrics["variant"] = variant
            metrics["prediction_column"] = pred_col
            candidate_rows.append(metrics)

    candidates = pd.DataFrame(candidate_rows)

    # Add the frozen-threshold results for direct comparison.
    frozen_rows = []

    for model_name, threshold in FROZEN_THRESHOLDS.items():
        for variant in [f"{model_name}_raw", f"{model_name}_calibrated"]:
            pred_col = variants[variant]
            metrics = policy_metrics(final_df, pred_col, threshold)
            metrics["variant"] = variant
            metrics["prediction_column"] = pred_col
            metrics["evaluation_type"] = "frozen_threshold_reference"
            frozen_rows.append(metrics)

    frozen_df = pd.DataFrame(frozen_rows)

    candidates["evaluation_type"] = "candidate_threshold"
    all_policy_rows = pd.concat(
        [candidates, frozen_df], ignore_index=True
    )

    candidates_path = (
        OUTPUT_DIR / "step08_7_all_validation_threshold_candidates_FD001.csv"
    )
    frozen_path = (
        OUTPUT_DIR / "step08_7_frozen_threshold_reference_FD001.csv"
    )

    candidates.to_csv(candidates_path, index=False)
    frozen_df.to_csv(frozen_path, index=False)

    # --------------------------------------------------------
    # 3. Select the lowest-cost feasible candidate per variant.
    # If no candidate is feasible, report that explicitly.
    # --------------------------------------------------------
    selected_rows = []

    for variant in variants:
        subset = candidates[candidates["variant"] == variant].copy()
        feasible = subset[subset["safety_feasible"]].copy()

        if feasible.empty:
            selected_rows.append({
                "variant": variant,
                "selection_status": "NO_FEASIBLE_THRESHOLD",
                "selected_threshold_cycles": None,
                "n_engines": int(len(final_df)),
                "maintenance_actions": None,
                "missed_critical_engines": None,
                "missed_high_risk_engines": None,
                "high_risk_miss_rate": None,
                "false_alarm_engines": None,
                "illustrative_cost": None,
                "safety_feasible": False,
            })
            continue

        # Deterministic tie-break: lower cost, then fewer actions,
        # then lower threshold.
        feasible = feasible.sort_values(
            ["illustrative_cost", "maintenance_actions", "threshold_cycles"],
            ascending=[True, True, True],
        )
        best = feasible.iloc[0].to_dict()
        best["selection_status"] = "LOWEST_COST_FEASIBLE_VALIDATION_CANDIDATE"
        best["selected_threshold_cycles"] = float(best["threshold_cycles"])
        selected_rows.append(best)

    selected_df = pd.DataFrame(selected_rows)
    selected_path = (
        OUTPUT_DIR / "step08_7_selected_validation_candidates_FD001.csv"
    )
    selected_df.to_csv(selected_path, index=False)

    print("\nFROZEN-THRESHOLD REFERENCES")
    print(
        frozen_df[
            [
                "variant", "threshold_cycles", "maintenance_actions",
                "missed_critical_engines", "missed_high_risk_engines",
                "high_risk_miss_rate", "false_alarm_engines",
                "illustrative_cost", "safety_feasible",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nLOWEST-COST FEASIBLE VALIDATION CANDIDATES")
    display_cols = [
        "variant", "selection_status", "selected_threshold_cycles",
        "maintenance_actions", "missed_critical_engines",
        "missed_high_risk_engines", "high_risk_miss_rate",
        "false_alarm_engines", "illustrative_cost", "safety_feasible",
    ]
    print(
        selected_df[display_cols].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # 4. Paired bootstrap on final-cycle engine predictions
    # --------------------------------------------------------
    bootstrap_rows = []

    for a, b in [
        ("baseline_raw_RUL", "baseline_calibrated_RUL"),
        ("temporal_raw_RUL", "temporal_calibrated_RUL"),
        ("baseline_raw_RUL", "temporal_raw_RUL"),
        ("baseline_calibrated_RUL", "temporal_calibrated_RUL"),
    ]:
        bootstrap_rows.append(
            bootstrap_engine_difference(final_df, a, b)
        )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_path = (
        OUTPUT_DIR / "step08_7_engine_bootstrap_FD001.csv"
    )
    bootstrap_df.to_csv(bootstrap_path, index=False)

    print("\nPAIRED ENGINE-LEVEL BOOTSTRAP: RMSE DIFFERENCES")
    print(bootstrap_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --------------------------------------------------------
    # 5. Plot candidate cost and safety feasibility
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for variant in variants:
        subset = candidates[candidates["variant"] == variant]
        axes[0].plot(
            subset["threshold_cycles"],
            subset["illustrative_cost"],
            marker="o",
            markersize=3,
            label=variant,
        )

        axes[1].plot(
            subset["threshold_cycles"],
            subset["high_risk_miss_rate"],
            marker="o",
            markersize=3,
            label=variant,
        )

    axes[0].set_title("Validation Cost by Candidate Threshold")
    axes[0].set_xlabel("Maintenance threshold (RUL cycles)")
    axes[0].set_ylabel("Illustrative cost")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axhline(
        MAX_HIGH_RISK_MISS_RATE,
        linestyle="--",
        label="Maximum allowed high-risk miss rate",
    )
    axes[1].set_title("High-Risk Miss Rate by Threshold")
    axes[1].set_xlabel("Maintenance threshold (RUL cycles)")
    axes[1].set_ylabel("High-risk miss rate")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Step 8.7 — Final-Cycle Validation Threshold and Safety Audit"
    )
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "step08_7_threshold_safety_audit_FD001.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 6. Engine-level decision audit at selected thresholds
    # --------------------------------------------------------
    decision_df = final_df[
        ["unit_id", "cycle", "actual_RUL"]
    ].copy()

    decision_df["actual_critical"] = (
        decision_df["actual_RUL"] <= CRITICAL_RUL
    )
    decision_df["actual_high_risk"] = (
        decision_df["actual_RUL"] <= HIGH_RISK_RUL
    )

    for variant, pred_col in variants.items():
        decision_df[f"{variant}_prediction"] = final_df[pred_col].to_numpy()

        selected = selected_df[selected_df["variant"] == variant]
        if (
            len(selected) == 1
            and pd.notna(selected.iloc[0]["selected_threshold_cycles"])
        ):
            threshold = float(selected.iloc[0]["selected_threshold_cycles"])
            decision_df[f"{variant}_candidate_threshold"] = threshold
            decision_df[f"{variant}_candidate_action"] = (
                decision_df[f"{variant}_prediction"] <= threshold
            )
        else:
            decision_df[f"{variant}_candidate_threshold"] = np.nan
            decision_df[f"{variant}_candidate_action"] = pd.NA

    decisions_path = (
        OUTPUT_DIR / "step08_7_final_cycle_engine_decisions_FD001.csv"
    )
    decision_df.to_csv(decisions_path, index=False)

    # --------------------------------------------------------
    # 7. Machine-readable report
    # --------------------------------------------------------
    report = {
        "project": (
            "Self-Evolving Explainable Rational Agent for Adaptive "
            "Industrial Decision Intelligence Using Statistical Trend Learning"
        ),
        "dataset": "NASA C-MAPSS",
        "subset": "FD001",
        "step": "8.7",
        "method": "Validation-only threshold reassessment and safety audit",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(CALIBRATED_FILE),
        "validation_input_rows": int(len(data)),
        "final_cycle_validation_engines": int(len(final_df)),
        "test_data_used": False,
        "test_labels_used": False,
        "model_retrained": False,
        "frozen_policy_modified": False,
        "policy_deployed": False,
        "decision_unit": "one final observed validation cycle per engine",
        "prediction_variants": list(variants.keys()),
        "safety_constraints": {
            "critical_RUL_cycles": CRITICAL_RUL,
            "high_risk_RUL_cycles": HIGH_RISK_RUL,
            "maximum_missed_critical_engines": MAX_MISSED_CRITICAL_ENGINES,
            "maximum_high_risk_miss_rate": MAX_HIGH_RISK_MISS_RATE,
        },
        "illustrative_cost_assumptions": {
            "maintenance_cost": MAINTENANCE_COST,
            "missed_critical_cost": MISSED_CRITICAL_COST,
            "false_alarm_cost": FALSE_ALARM_COST,
            "real_industrial_cost_estimates": False,
        },
        "frozen_thresholds": FROZEN_THRESHOLDS,
        "selected_validation_candidates": selected_df.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
        "final_cycle_regression_metrics": regression_df.to_dict(
            orient="records"
        ),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": RANDOM_SEED,
            "resampling_unit": "validation engine",
            "results": bootstrap_df.to_dict(orient="records"),
        },
        "limitations": [
            "Only 20 validation engines are available; safety estimates are unstable.",
            "Thresholds are selected and evaluated on the same small validation set, so reported candidate costs are optimistic selection estimates.",
            "Bootstrap intervals are exploratory and cannot compensate for the small number of engines.",
            "Costs are illustrative assumptions, not validated industrial costs.",
            "A feasible threshold on this validation set is not proof of operational safety.",
            "No independent test evaluation is performed in this step.",
            "This step does not demonstrate continuous online learning or deployment.",
        ],
        "artifacts": {
            "regression_metrics_csv": str(regression_path),
            "all_threshold_candidates_csv": str(candidates_path),
            "frozen_threshold_reference_csv": str(frozen_path),
            "selected_validation_candidates_csv": str(selected_path),
            "bootstrap_results_csv": str(bootstrap_path),
            "engine_decisions_csv": str(decisions_path),
            "plot": str(plot_path),
        },
    }

    report_path = OUTPUT_DIR / "step08_7_FD001_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print("\nSTEP 8.7 COMPLETED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Regression metrics: {regression_path}")
    print(f"Threshold candidates: {candidates_path}")
    print(f"Frozen-threshold reference: {frozen_path}")
    print(f"Selected candidates: {selected_path}")
    print(f"Bootstrap results: {bootstrap_path}")
    print(f"Engine decisions: {decisions_path}")
    print(f"Plot: {plot_path}")
    print(f"Research report: {report_path}")
    print("\nNo test data or labels were used.")
    print("No model was retrained.")
    print("No frozen maintenance policy was modified or deployed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nSTEP 8.7 FAILED")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
