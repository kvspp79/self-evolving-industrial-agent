
"""
NASA C-MAPSS FD001 — Step 8.1
Pseudo-test validation and robust maintenance policy selection.

Uses:
    outputs/step04/validation_predictions_FD001.csv
    outputs/step04/test_predictions_FD001.csv

Writes new outputs only to:
    outputs/step08_1/

Protocol:
1. Select representative pseudo-test snapshots from validation trajectories.
2. Select policies and thresholds using validation data only.
3. Freeze the selected policies.
4. Evaluate the frozen policies on final-cycle test engines.
5. Never use test labels to select a policy or threshold.

Maintenance costs and safety limits are illustrative assumptions,
not measured industrial costs.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

VALIDATION_FILE = (
    ROOT / "outputs" / "step04" / "validation_predictions_FD001.csv"
)
TEST_FILE = (
    ROOT / "outputs" / "step04" / "test_predictions_FD001.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "step08_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSET = "FD001"

CRITICAL_RUL = 15.0
HIGH_RISK_RUL = 30.0

# Illustrative policy-selection constraints.
MAX_MISSED_CRITICAL = 0
MAX_HIGH_RISK_MISS_RATE = 0.10

# Candidate decision thresholds in cycles.
THRESHOLDS = np.arange(0.0, 201.0, 5.0)

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

MODEL_COLUMNS = {
    "baseline": "baseline_predicted_RUL",
    "temporal": "temporal_predicted_RUL",
}

RUL_COLUMN = "actual_RUL"
ENGINE_COLUMN = "unit_id"


# ============================================================
# INPUT VALIDATION
# ============================================================

def load_predictions(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} file not found:\n{path}\n"
            "Run Step 4 successfully before running Step 8.1."
        )

    df = pd.read_csv(path)

    required = {
        ENGINE_COLUMN,
        RUL_COLUMN,
        *MODEL_COLUMNS.values(),
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{description} file is missing required columns: "
            f"{sorted(missing)}\nFound columns: {df.columns.tolist()}"
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(required)).copy()

    if df.empty:
        raise ValueError(f"{description} data is empty after cleaning.")

    if (df[RUL_COLUMN] < 0).any():
        raise ValueError(
            f"{description} contains negative actual RUL values."
        )

    for col in MODEL_COLUMNS.values():
        if not np.isfinite(df[col]).all():
            raise ValueError(
                f"{description} contains non-finite predictions in {col}."
            )

    df[ENGINE_COLUMN] = df[ENGINE_COLUMN].astype(int)
    df = df.sort_values(ENGINE_COLUMN).reset_index(drop=True)

    return df


# ============================================================
# BUILD PSEUDO-TEST SNAPSHOTS
# ============================================================

def build_pseudo_test_snapshots(validation):
    """
    Select up to one validation row per engine in each RUL band.

    This approximates decisions at different points in an engine's
    life. actual_RUL is available here because these are held-out
    training engines, not the final test engines.

    RUL bands are:
        0-15, 16-30, 31-60, 61-100,
        101-150, 151-200, 201-300, >300.

    For each engine and band, select the row closest to the band's
    representative target RUL. This reduces over-representation of
    long trajectories compared with using every cycle.
    """

    bands = [
        ("critical_0_15", 0.0, 15.0, 7.5),
        ("high_16_30", 15.0, 30.0, 22.5),
        ("early_31_60", 30.0, 60.0, 45.0),
        ("early_61_100", 60.0, 100.0, 80.0),
        ("mid_101_150", 100.0, 150.0, 125.0),
        ("mid_151_200", 150.0, 200.0, 175.0),
        ("late_201_300", 200.0, 300.0, 250.0),
        ("late_over_300", 300.0, np.inf, 320.0),
    ]

    snapshots = []

    for engine_id, engine_df in validation.groupby(ENGINE_COLUMN):
        engine_df = engine_df.copy()

        for band_name, low, high, target in bands:
            if np.isinf(high):
                candidates = engine_df[
                    engine_df[RUL_COLUMN] > low
                ]
            elif low == 0:
                candidates = engine_df[
                    (engine_df[RUL_COLUMN] >= low)
                    & (engine_df[RUL_COLUMN] <= high)
                ]
            else:
                candidates = engine_df[
                    (engine_df[RUL_COLUMN] > low)
                    & (engine_df[RUL_COLUMN] <= high)
                ]

            if candidates.empty:
                continue

            # Prefer a row closest to the representative RUL.
            # Deterministic tie-breaking by cycle, when available.
            candidates = candidates.assign(
                _target_distance=(
                    candidates[RUL_COLUMN] - target
                ).abs()
            )

            sort_cols = ["_target_distance"]
            if "cycle" in candidates.columns:
                sort_cols.append("cycle")

            chosen = candidates.sort_values(sort_cols).iloc[0].copy()
            chosen["snapshot_band"] = band_name
            snapshots.append(chosen)

    if not snapshots:
        raise ValueError(
            "Could not construct pseudo-test snapshots from validation data."
        )

    result = pd.DataFrame(snapshots).reset_index(drop=True)

    return result


# ============================================================
# POLICY METRICS AND COST
# ============================================================

def evaluate_policy(df, model_name, threshold, cost_config):
    """
    Evaluate a single threshold.

    Action rule:
        Maintain if predicted RUL <= threshold.

    A missed critical engine is an engine/snapshot with actual RUL
    <= CRITICAL_RUL for which maintenance was not triggered.

    A high-risk miss is actual RUL <= HIGH_RISK_RUL with no action.

    A false alarm is an action when actual RUL > HIGH_RISK_RUL.
    These definitions are illustrative and should be stated in the
    research paper.
    """

    pred_col = MODEL_COLUMNS[model_name]

    actual = df[RUL_COLUMN].to_numpy(dtype=float)
    predicted = df[pred_col].to_numpy(dtype=float)

    maintenance = predicted <= float(threshold)

    critical = actual <= CRITICAL_RUL
    high_risk = actual <= HIGH_RISK_RUL
    false_alarm = maintenance & (actual > HIGH_RISK_RUL)

    missed_critical = critical & (~maintenance)
    missed_high_risk = high_risk & (~maintenance)

    maintenance_count = int(maintenance.sum())
    missed_critical_count = int(missed_critical.sum())
    missed_high_risk_count = int(missed_high_risk.sum())
    false_alarm_count = int(false_alarm.sum())

    high_risk_count = int(high_risk.sum())
    high_risk_miss_rate = (
        float(missed_high_risk_count / high_risk_count)
        if high_risk_count > 0
        else 0.0
    )

    maintenance_cost = (
        maintenance_count * cost_config["maintenance_cost"]
    )
    missed_critical_cost = (
        missed_critical_count * cost_config["missed_critical_cost"]
    )
    false_alarm_cost = (
        false_alarm_count * cost_config["false_alarm_cost"]
    )

    total_cost = (
        maintenance_cost
        + missed_critical_cost
        + false_alarm_cost
    )

    return {
        "model": model_name,
        "threshold": float(threshold),
        "n_snapshots": int(len(df)),
        "n_engines": int(df[ENGINE_COLUMN].nunique()),
        "maintenance_actions": maintenance_count,
        "missed_critical_snapshots": missed_critical_count,
        "missed_high_risk_snapshots": missed_high_risk_count,
        "high_risk_snapshots": high_risk_count,
        "high_risk_miss_rate": high_risk_miss_rate,
        "false_alarm_actions": false_alarm_count,
        "maintenance_cost": float(maintenance_cost),
        "missed_critical_cost": float(missed_critical_cost),
        "false_alarm_cost": float(false_alarm_cost),
        "total_simulated_cost": float(total_cost),
        "cost_per_snapshot": float(total_cost / len(df)),
        "safety_constraints_satisfied": bool(
            missed_critical_count <= MAX_MISSED_CRITICAL
            and high_risk_miss_rate <= MAX_HIGH_RISK_MISS_RATE
        ),
    }


def select_policy(validation_snapshots, model_name, scenario_name, costs):
    results = []

    for threshold in THRESHOLDS:
        result = evaluate_policy(
            validation_snapshots,
            model_name,
            threshold,
            costs,
        )
        result["cost_scenario"] = scenario_name
        results.append(result)

    table = pd.DataFrame(results)

    feasible = table[table["safety_constraints_satisfied"]].copy()

    if not feasible.empty:
        # Lowest validation cost among policies meeting constraints.
        selected = feasible.sort_values(
            ["total_simulated_cost", "threshold"],
            ascending=[True, True],
        ).iloc[0].to_dict()

        status = "feasible_lowest_validation_cost"
    else:
        # Do not falsely claim the safety requirements were met.
        # Fallback: minimize validation cost, then missed critical
        # snapshots, then high-risk miss rate.
        selected = table.sort_values(
            [
                "total_simulated_cost",
                "missed_critical_snapshots",
                "high_risk_miss_rate",
                "threshold",
            ],
            ascending=[True, True, True, True],
        ).iloc[0].to_dict()

        status = "no_feasible_threshold"

    selected["selection_status"] = status
    selected["safety_constraints_satisfied"] = bool(
        selected["safety_constraints_satisfied"]
    )

    return selected, table


# ============================================================
# PLOTS
# ============================================================

def plot_snapshot_distribution(snapshots):
    counts = (
        snapshots["snapshot_band"]
        .value_counts()
        .reindex(
            [
                "critical_0_15",
                "high_16_30",
                "early_31_60",
                "early_61_100",
                "mid_101_150",
                "mid_151_200",
                "late_201_300",
                "late_over_300",
            ],
            fill_value=0,
        )
    )

    plt.figure(figsize=(11, 6))
    counts.plot(kind="bar")
    plt.title("Step 8.1 — Pseudo-Test Validation Snapshots by RUL Band")
    plt.xlabel("Actual RUL band (validation engines)")
    plt.ylabel("Number of selected snapshots")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "pseudo_test_snapshot_distribution.png",
        dpi=180,
    )
    plt.close()


def plot_validation_cost_curves(all_threshold_results):
    for scenario_name in COST_SCENARIOS:
        subset = all_threshold_results[
            all_threshold_results["cost_scenario"] == scenario_name
        ]

        plt.figure(figsize=(9, 6))

        for model_name in MODEL_COLUMNS:
            model_data = subset[subset["model"] == model_name]
            if not model_data.empty:
                model_data = model_data.sort_values("threshold")
                plt.plot(
                    model_data["threshold"],
                    model_data["total_simulated_cost"],
                    marker="o",
                    label=model_name,
                )

        plt.title(
            f"Validation Policy Cost by Threshold — {scenario_name}"
        )
        plt.xlabel("Maintenance threshold (predicted RUL cycles)")
        plt.ylabel("Illustrative validation cost")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = scenario_name.replace(" ", "_")
        plt.savefig(
            OUTPUT_DIR / f"validation_cost_curve_{safe_name}.png",
            dpi=180,
        )
        plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 76)
    print("NASA C-MAPSS — STEP 8.1")
    print("PSEUDO-TEST VALIDATION AND ROBUST POLICY SELECTION")
    print(f"Subset: {SUBSET}")
    print("=" * 76)

    validation = load_predictions(
        VALIDATION_FILE, "Validation predictions"
    )
    test = load_predictions(TEST_FILE, "Test predictions")

    # The validation file is row-level. Test predictions should be
    # one final observed row per engine.
    if "cycle" not in validation.columns:
        raise ValueError(
            "Validation predictions must contain a 'cycle' column."
        )

    if "last_observed_cycle" in test.columns:
        test = test.sort_values(
            [ENGINE_COLUMN, "last_observed_cycle"]
        )

    # Ensure one final prediction per test engine.
    test = (
        test.sort_values(
            [ENGINE_COLUMN]
            + (
                ["last_observed_cycle"]
                if "last_observed_cycle" in test.columns
                else []
            )
        )
        .groupby(ENGINE_COLUMN, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    print("\nDATA SUMMARY")
    print("-" * 60)
    print(f"Validation rows: {len(validation):,}")
    print(f"Validation engines: {validation[ENGINE_COLUMN].nunique():,}")
    print(f"Test final-cycle rows: {len(test):,}")
    print(f"Test engines: {test[ENGINE_COLUMN].nunique():,}")

    if validation[ENGINE_COLUMN].nunique() < 30:
        warnings.warn(
            "Validation has fewer than 30 engines. Policy estimates "
            "may be unstable; treat this analysis as exploratory."
        )

    print("\nConstructing pseudo-test snapshots from validation engines...")
    snapshots = build_pseudo_test_snapshots(validation)

    print(f"Pseudo-test snapshots: {len(snapshots):,}")
    print(
        "Validation engines represented: "
        f"{snapshots[ENGINE_COLUMN].nunique():,}"
    )

    band_summary = (
        snapshots.groupby("snapshot_band")
        .agg(
            snapshots=(ENGINE_COLUMN, "size"),
            engines=(ENGINE_COLUMN, "nunique"),
            mean_actual_RUL=(RUL_COLUMN, "mean"),
        )
        .reset_index()
    )

    print("\nSNAPSHOT DISTRIBUTION")
    print(band_summary.to_string(index=False))

    snapshots.to_csv(
        OUTPUT_DIR / "pseudo_test_validation_snapshots.csv",
        index=False,
    )
    band_summary.to_csv(
        OUTPUT_DIR / "pseudo_test_snapshot_band_summary.csv",
        index=False,
    )

    # Save baseline/temporal RUL regression metrics on the same
    # pseudo-test snapshots.
    regression_rows = []

    for model_name, pred_col in MODEL_COLUMNS.items():
        error = (
            snapshots[pred_col].to_numpy(dtype=float)
            - snapshots[RUL_COLUMN].to_numpy(dtype=float)
        )
        squared = error ** 2

        # Standard NASA C-MAPSS asymmetric score:
        # overestimation of RUL (positive error) receives a slower
        # penalty than underestimation (negative error).
        score_terms = np.where(
            error < 0,
            np.exp(np.clip(-error / 13.0, -50, 50)) - 1.0,
            np.exp(np.clip(error / 10.0, -50, 50)) - 1.0,
        )

        regression_rows.append(
            {
                "model": model_name,
                "n_snapshots": len(snapshots),
                "n_engines": snapshots[ENGINE_COLUMN].nunique(),
                "RMSE": float(np.sqrt(np.mean(squared))),
                "MAE": float(np.mean(np.abs(error))),
                "NASA_score": float(np.sum(score_terms)),
            }
        )

    regression_metrics = pd.DataFrame(regression_rows)
    regression_metrics.to_csv(
        OUTPUT_DIR / "pseudo_test_regression_metrics.csv",
        index=False,
    )

    print("\nPSEUDO-TEST RUL METRICS")
    print(regression_metrics.to_string(index=False))

    # Select a policy independently for each model and cost scenario.
    selected_policies = []
    all_threshold_tables = []

    print("\nVALIDATION-ONLY POLICY SELECTION")
    print("-" * 76)
    print(
        f"Safety constraints: missed critical snapshots <= "
        f"{MAX_MISSED_CRITICAL}; high-risk miss rate <= "
        f"{MAX_HIGH_RISK_MISS_RATE:.0%}"
    )

    for scenario_name, costs in COST_SCENARIOS.items():
        for model_name in MODEL_COLUMNS:
            selected, threshold_table = select_policy(
                snapshots,
                model_name,
                scenario_name,
                costs,
            )

            selected_policies.append(selected)
            all_threshold_tables.append(threshold_table)

            print(
                f"{model_name:8s} | {scenario_name:22s} | "
                f"threshold={selected['threshold']:6.1f} | "
                f"validation cost={selected['total_simulated_cost']:9.1f} | "
                f"missed critical={selected['missed_critical_snapshots']} | "
                f"high-risk miss rate="
                f"{selected['high_risk_miss_rate']:.3f} | "
                f"{selected['selection_status']}"
            )

    selected_df = pd.DataFrame(selected_policies)
    threshold_df = pd.concat(
        all_threshold_tables, ignore_index=True
    )

    selected_df.to_csv(
        OUTPUT_DIR / "selected_validation_policies.csv",
        index=False,
    )
    threshold_df.to_csv(
        OUTPUT_DIR / "all_validation_threshold_results.csv",
        index=False,
    )

    # The selection is now frozen. Test labels are used below only
    # to measure outcomes, never to choose thresholds or models.
    print("\nPOLICIES FROZEN.")
    print("Evaluating frozen policies on held-out test engines...")

    test_results = []

    for _, policy in selected_df.iterrows():
        model_name = policy["model"]
        scenario_name = policy["cost_scenario"]
        threshold = float(policy["threshold"])
        costs = COST_SCENARIOS[scenario_name]

        result = evaluate_policy(
            test,
            model_name,
            threshold,
            costs,
        )

        result["cost_scenario"] = scenario_name
        result["selection_status"] = policy["selection_status"]
        result["validation_safety_constraints_satisfied"] = bool(
            policy["safety_constraints_satisfied"]
        )

        # Test safety constraints are reported descriptively only.
        # They do not alter the frozen threshold.
        test_results.append(result)

    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(
        OUTPUT_DIR / "frozen_policy_test_results.csv",
        index=False,
    )

    print("\nHELD-OUT TEST EVALUATION OF FROZEN POLICIES")
    print("-" * 76)
    display_cols = [
        "model",
        "cost_scenario",
        "threshold",
        "n_snapshots",
        "maintenance_actions",
        "missed_critical_snapshots",
        "missed_high_risk_snapshots",
        "high_risk_miss_rate",
        "false_alarm_actions",
        "total_simulated_cost",
        "cost_per_snapshot",
        "selection_status",
    ]
    print(test_results_df[display_cols].to_string(index=False))

    # Visualizations.
    plot_snapshot_distribution(snapshots)
    plot_validation_cost_curves(threshold_df)

    # Report explicitly documents the protocol and limitations.
    report = {
        "step": "8.1",
        "subset": SUBSET,
        "protocol": {
            "validation_source": str(VALIDATION_FILE),
            "test_source": str(TEST_FILE),
            "validation_rows": int(len(validation)),
            "validation_engines": int(
                validation[ENGINE_COLUMN].nunique()
            ),
            "pseudo_test_snapshots": int(len(snapshots)),
            "pseudo_test_engines": int(
                snapshots[ENGINE_COLUMN].nunique()
            ),
            "test_final_cycle_engines": int(
                test[ENGINE_COLUMN].nunique()
            ),
            "threshold_selection_data": "validation pseudo-test snapshots only",
            "test_labels_used_for_selection": False,
            "test_labels_used_for_descriptive_evaluation": True,
            "threshold_grid": [float(x) for x in THRESHOLDS],
            "critical_rul_cycles": CRITICAL_RUL,
            "high_risk_rul_cycles": HIGH_RISK_RUL,
            "max_missed_critical_snapshots": MAX_MISSED_CRITICAL,
            "max_high_risk_miss_rate": MAX_HIGH_RISK_MISS_RATE,
        },
        "cost_scenarios": COST_SCENARIOS,
        "pseudo_test_regression_metrics": regression_metrics.to_dict(
            orient="records"
        ),
        "selected_validation_policies": selected_df.to_dict(
            orient="records"
        ),
        "frozen_test_policy_results": test_results_df.to_dict(
            orient="records"
        ),
        "limitations": [
            "Validation uses only 20 held-out training engines, so "
            "policy estimates may be unstable.",
            "Snapshots from the same engine are correlated and must not "
            "be treated as independent engines.",
            "Pseudo-test snapshots are sampled from observed training "
            "trajectories and do not perfectly reproduce the censoring "
            "mechanism of the official test set.",
            "Maintenance costs and safety constraints are illustrative "
            "assumptions, not measured industrial costs.",
            "Test labels are used only for final descriptive evaluation, "
            "not for selecting thresholds or policies.",
        ],
    }

    report_path = OUTPUT_DIR / "step08_1_FD001_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 76)
    print("STEP 8.1 COMPLETED SUCCESSFULLY")
    print("=" * 76)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Research report: {report_path}")
    print("Validation data selected the thresholds.")
    print("Test labels did not select or modify the frozen policies.")
    print("Steps 1–8 were not modified.")


if __name__ == "__main__":
    main()
