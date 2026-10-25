
"""
NASA C-MAPSS FD001 — STEP 7.1
SAFETY-CONSTRAINED ADAPTIVE MAINTENANCE POLICY
AND COST-SENSITIVITY ANALYSIS

Inputs:
    outputs/step04/validation_predictions_FD001.csv
    outputs/step04/test_predictions_FD001.csv

Outputs:
    outputs/step07_1/

Important research safeguards:
- Thresholds are selected using validation data only.
- Test labels are used only for final evaluation.
- Engine IDs are not compared across train/test namespaces.
- Validation and test are evaluated at one final observed cycle per engine.
- Safety constraints and cost assumptions are reported explicitly.
- This is threshold optimization, not reinforcement learning.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "outputs" / "step04"
OUTPUT_DIR = ROOT / "outputs" / "step07_1"

SUBSET = "FD001"

CRITICAL_RUL = 15.0
HIGH_RUL = 30.0

# Safety constraints applied during validation selection.
MAX_MISSED_CRITICAL_VALIDATION = 0

# At most 10% of actual high-or-critical validation engines
# may be missed. This is a design constraint, not a standard.
MAX_MISSED_HIGH_RISK_RATE = 0.10

# Explore thresholds from 0 to 200 cycles in 5-cycle increments.
THRESHOLDS = np.arange(0.0, 201.0, 5.0)

# Multiple illustrative cost scenarios.
# Threshold selection is repeated on validation data for each scenario.
COST_SCENARIOS = {
    "standard": {
        "maintenance": 100.0,
        "missed_critical": 1000.0,
        "false_alarm": 50.0,
    },
    "high_failure_penalty": {
        "maintenance": 100.0,
        "missed_critical": 2000.0,
        "false_alarm": 50.0,
    },
    "lower_failure_penalty": {
        "maintenance": 100.0,
        "missed_critical": 500.0,
        "false_alarm": 50.0,
    },
    "high_maintenance_cost": {
        "maintenance": 250.0,
        "missed_critical": 1000.0,
        "false_alarm": 100.0,
    },
}

PREDICTION_COLUMNS = {
    "baseline": "baseline_predicted_RUL",
    "temporal": "temporal_predicted_RUL",
}

REQUIRED_COLUMNS = [
    "unit_id",
    "actual_RUL",
    "baseline_predicted_RUL",
    "temporal_predicted_RUL",
]


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_predictions(path, split_name):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {split_name} predictions:\n{path}\n\n"
            "Run Step 4 first and check the prediction filenames."
        )

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{split_name} file is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if df.empty:
        raise ValueError(f"{split_name} file is empty.")

    df = df.copy()

    for col in [
        "actual_RUL",
        "baseline_predicted_RUL",
        "temporal_predicted_RUL",
    ]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    numeric = df[
        [
            "actual_RUL",
            "baseline_predicted_RUL",
            "temporal_predicted_RUL",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError(
            f"{split_name} contains NaN or infinite prediction values."
        )

    if "cycle" in df.columns:
        df["cycle"] = pd.to_numeric(df["cycle"], errors="raise")

    return df


def final_row_per_engine(df):
    """
    Keep the last observed row for each engine.

    For training/validation trajectories, this means the final
    observed cycle for each validation engine.
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
        .reset_index(drop=True)
    )

    if result["unit_id"].duplicated().any():
        raise ValueError(
            "Failed to reduce data to one row per engine."
        )

    return result


# ============================================================
# SAFETY AND COST METRICS
# ============================================================

def evaluate_threshold(
    df,
    model_name,
    threshold,
    cost_scenario,
):
    prediction_column = PREDICTION_COLUMNS[model_name]

    actual = df["actual_RUL"].to_numpy(dtype=float)
    predicted = df[prediction_column].to_numpy(dtype=float)

    action = predicted <= float(threshold)

    critical = actual <= CRITICAL_RUL
    high_or_critical = actual <= HIGH_RUL

    missed_critical = critical & ~action
    missed_high = high_or_critical & ~action

    # An action is considered a false alarm if the engine's actual
    # RUL is greater than the defined high-risk boundary.
    false_alarm = action & (actual > HIGH_RUL)

    n = len(df)
    actual_critical_n = int(critical.sum())
    actual_high_n = int(high_or_critical.sum())

    action_n = int(action.sum())
    missed_critical_n = int(missed_critical.sum())
    missed_high_n = int(missed_high.sum())
    false_alarm_n = int(false_alarm.sum())

    high_risk_miss_rate = (
        missed_high_n / actual_high_n
        if actual_high_n > 0
        else 0.0
    )

    costs = COST_SCENARIOS[cost_scenario]

    maintenance_cost = action_n * costs["maintenance"]
    missed_critical_cost = (
        missed_critical_n * costs["missed_critical"]
    )
    false_alarm_cost = false_alarm_n * costs["false_alarm"]

    total_cost = (
        maintenance_cost
        + missed_critical_cost
        + false_alarm_cost
    )

    # A candidate is feasible only if it satisfies both safety rules.
    feasible = (
        missed_critical_n <= MAX_MISSED_CRITICAL_VALIDATION
        and high_risk_miss_rate <= MAX_MISSED_HIGH_RISK_RATE
    )

    return {
        "model": model_name,
        "cost_scenario": cost_scenario,
        "threshold_cycles": float(threshold),
        "n_engines": int(n),
        "actual_critical_engines": actual_critical_n,
        "actual_high_or_critical_engines": actual_high_n,
        "maintenance_actions": action_n,
        "maintenance_action_rate": (
            float(action_n / n) if n else 0.0
        ),
        "missed_critical_engines": missed_critical_n,
        "missed_high_or_critical_engines": missed_high_n,
        "high_risk_miss_rate": float(high_risk_miss_rate),
        "false_alarm_actions": false_alarm_n,
        "maintenance_cost": float(maintenance_cost),
        "missed_critical_cost": float(missed_critical_cost),
        "false_alarm_cost": float(false_alarm_cost),
        "total_simulated_cost": float(total_cost),
        "simulated_cost_per_engine": (
            float(total_cost / n) if n else 0.0
        ),
        "safety_constraints_satisfied": bool(feasible),
    }


# ============================================================
# VALIDATION-ONLY POLICY SELECTION
# ============================================================

def select_safety_constrained_policy(validation, model_name, scenario):
    """
    Search candidate thresholds on validation data only.

    Selection:
    1. If feasible thresholds exist, choose the lowest-cost feasible one.
    2. Otherwise, select lexicographically by:
       missed critical engines,
       missed high-risk engines,
       simulated cost,
       threshold.

    The second case is explicitly reported as safety-infeasible.
    """

    rows = [
        evaluate_threshold(
            validation,
            model_name,
            threshold,
            scenario,
        )
        for threshold in THRESHOLDS
    ]

    results = pd.DataFrame(rows)

    feasible = results[
        results["safety_constraints_satisfied"]
    ].copy()

    if not feasible.empty:
        selected = feasible.sort_values(
            [
                "total_simulated_cost",
                "missed_critical_engines",
                "missed_high_or_critical_engines",
                "maintenance_actions",
                "threshold_cycles",
            ],
            ascending=True,
            kind="stable",
        ).iloc[0].to_dict()

        selection_status = "feasible_lowest_cost"
    else:
        # Safety-first fallback: do not pretend that a feasible
        # policy exists if the validation data cannot support one.
        selected = results.sort_values(
            [
                "missed_critical_engines",
                "missed_high_or_critical_engines",
                "total_simulated_cost",
                "threshold_cycles",
            ],
            ascending=True,
            kind="stable",
        ).iloc[0].to_dict()

        selection_status = "no_feasible_threshold"

    selected["selection_status"] = selection_status

    results["selection_status"] = selection_status

    return selected, results


# ============================================================
# COST-SENSITIVITY EVALUATION
# ============================================================

def evaluate_frozen_test_policy(test, model_name, selected):
    """
    Apply a threshold selected on validation to the test set.
    Test outcomes do not change the selected threshold.
    """

    return evaluate_threshold(
        test,
        model_name,
        selected["threshold_cycles"],
        selected["cost_scenario"],
    )


def build_engine_decisions(df, selected_policies, split_name):
    result = df.copy()
    result["split"] = split_name

    for key, selected in selected_policies.items():
        model_name, scenario = key
        prediction_column = PREDICTION_COLUMNS[model_name]

        suffix = f"{model_name}_{scenario}"
        threshold = float(selected["threshold_cycles"])

        result[f"{suffix}_threshold"] = threshold
        result[f"{suffix}_maintenance_action"] = (
            result[prediction_column] <= threshold
        )

        result[f"{suffix}_missed_critical"] = (
            (result["actual_RUL"] <= CRITICAL_RUL)
            & ~result[f"{suffix}_maintenance_action"]
        )

        result[f"{suffix}_missed_high_or_critical"] = (
            (result["actual_RUL"] <= HIGH_RUL)
            & ~result[f"{suffix}_maintenance_action"]
        )

    return result


# ============================================================
# FIGURES
# ============================================================

def plot_threshold_costs(all_search_results, output_path):
    models = list(PREDICTION_COLUMNS.keys())
    scenarios = list(COST_SCENARIOS.keys())

    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(11, 5 * len(models)),
        squeeze=False,
    )

    for row, model_name in enumerate(models):
        ax = axes[row, 0]

        for scenario in scenarios:
            subset = all_search_results[
                (all_search_results["model"] == model_name)
                & (
                    all_search_results["cost_scenario"]
                    == scenario
                )
            ].sort_values("threshold_cycles")

            ax.plot(
                subset["threshold_cycles"],
                subset["total_simulated_cost"],
                marker="o",
                markersize=3,
                label=scenario,
            )

        ax.set_title(
            f"{model_name.capitalize()}: Validation Threshold Search"
        )
        ax.set_xlabel("Maintenance threshold (RUL cycles)")
        ax.set_ylabel("Validation simulated cost")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_test_policy_comparison(test_results, output_path):
    data = pd.DataFrame(test_results)

    if data.empty:
        return

    data["label"] = (
        data["model"] + "\n" + data["cost_scenario"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(
        data["label"],
        data["missed_critical_engines"],
    )
    axes[0].set_title("Test Missed Critical Engines")
    axes[0].set_ylabel("Number of engines")
    axes[0].tick_params(axis="x", rotation=65)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        data["label"],
        data["total_simulated_cost"],
    )
    axes[1].set_title("Test Simulated Cost")
    axes[1].set_ylabel("Illustrative cost units")
    axes[1].tick_params(axis="x", rotation=65)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("NASA C-MAPSS — STEP 7.1")
    print("SAFETY-CONSTRAINED POLICY AND COST-SENSITIVITY ANALYSIS")
    print(f"Subset: {SUBSET}")
    print("=" * 72)

    validation_path = (
        INPUT_DIR / f"validation_predictions_{SUBSET}.csv"
    )
    test_path = INPUT_DIR / f"test_predictions_{SUBSET}.csv"

    validation_all = load_predictions(
        validation_path,
        "validation",
    )
    test_all = load_predictions(test_path, "test")

    validation = final_row_per_engine(validation_all)
    test = final_row_per_engine(test_all)

    print("\nDATA SUMMARY")
    print("-" * 50)
    print("Validation source rows:", len(validation_all))
    print("Final validation engines:", len(validation))
    print("Test source rows:", len(test_all))
    print("Final test engines:", len(test))

    if len(validation) < 30:
        warnings.warn(
            f"Only {len(validation)} validation engines are available. "
            "Safety and threshold estimates may be unstable."
        )

    print("\nSAFETY CONSTRAINTS")
    print("-" * 50)
    print(
        "Maximum missed critical validation engines:",
        MAX_MISSED_CRITICAL_VALIDATION,
    )
    print(
        "Maximum missed high-risk validation rate:",
        f"{MAX_MISSED_HIGH_RISK_RATE:.1%}",
    )
    print("Threshold candidates:", THRESHOLDS.tolist())

    print("\nCOST SCENARIOS")
    print("-" * 50)
    for name, costs in COST_SCENARIOS.items():
        print(
            f"{name}: maintenance={costs['maintenance']}, "
            f"missed_critical={costs['missed_critical']}, "
            f"false_alarm={costs['false_alarm']}"
        )

    # --------------------------------------------------------
    # 1. Select thresholds using validation data only.
    # --------------------------------------------------------

    selected_policies = {}
    all_search_tables = []
    validation_selected_rows = []

    for model_name in PREDICTION_COLUMNS:
        for scenario in COST_SCENARIOS:
            selected, search_table = (
                select_safety_constrained_policy(
                    validation,
                    model_name,
                    scenario,
                )
            )

            key = (model_name, scenario)
            selected_policies[key] = selected

            all_search_tables.append(search_table)
            validation_selected_rows.append(selected)

            print(
                f"\n{model_name.upper()} | {scenario}"
            )
            print("-" * 50)
            print(
                "Selected threshold:",
                selected["threshold_cycles"],
            )
            print(
                "Selection status:",
                selected["selection_status"],
            )
            print(
                "Safety constraints satisfied:",
                selected["safety_constraints_satisfied"],
            )
            print(
                "Validation missed critical:",
                selected["missed_critical_engines"],
            )
            print(
                "Validation missed high-or-critical:",
                selected["missed_high_or_critical_engines"],
            )
            print(
                "Validation maintenance actions:",
                selected["maintenance_actions"],
            )
            print(
                "Validation simulated cost:",
                selected["total_simulated_cost"],
            )

    all_search = pd.concat(
        all_search_tables,
        ignore_index=True,
    )

    all_search.to_csv(
        OUTPUT_DIR / f"validation_threshold_search_{SUBSET}.csv",
        index=False,
    )

    pd.DataFrame(validation_selected_rows).to_csv(
        OUTPUT_DIR / f"selected_validation_policies_{SUBSET}.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Evaluate frozen thresholds on test data.
    # --------------------------------------------------------

    test_results = []

    for (model_name, scenario), selected in selected_policies.items():
        result = evaluate_frozen_test_policy(
            test,
            model_name,
            selected,
        )

        result["selection_status"] = selected["selection_status"]
        result["validation_safety_feasible"] = selected[
            "safety_constraints_satisfied"
        ]

        test_results.append(result)

    test_results_df = pd.DataFrame(test_results)

    test_results_df.to_csv(
        OUTPUT_DIR / f"held_out_test_policy_results_{SUBSET}.csv",
        index=False,
    )

    # Per-engine decisions for reproducibility.
    validation_decisions = build_engine_decisions(
        validation,
        selected_policies,
        "validation",
    )

    test_decisions = build_engine_decisions(
        test,
        selected_policies,
        "test",
    )

    validation_decisions.to_csv(
        OUTPUT_DIR / f"validation_engine_decisions_{SUBSET}.csv",
        index=False,
    )

    test_decisions.to_csv(
        OUTPUT_DIR / f"test_engine_decisions_{SUBSET}.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Generate figures.
    # --------------------------------------------------------

    plot_threshold_costs(
        all_search,
        OUTPUT_DIR / f"validation_threshold_cost_sensitivity_{SUBSET}.png",
    )

    plot_test_policy_comparison(
        test_results,
        OUTPUT_DIR / f"test_policy_sensitivity_comparison_{SUBSET}.png",
    )

    # --------------------------------------------------------
    # 4. Research report.
    # --------------------------------------------------------

    selected_report = {}

    for (model_name, scenario), selected in selected_policies.items():
        key = f"{model_name}__{scenario}"

        selected_report[key] = {
            k: (
                bool(v)
                if isinstance(v, (np.bool_,))
                else (
                    int(v)
                    if isinstance(v, (np.integer,))
                    else (
                        float(v)
                        if isinstance(v, (np.floating,))
                        else v
                    )
                )
            )
            for k, v in selected.items()
        }

    test_report = test_results_df.replace(
        [np.inf, -np.inf], np.nan
    ).to_dict(orient="records")

    report = {
        "dataset": "NASA C-MAPSS",
        "subset": SUBSET,
        "step": "7.1",
        "method": (
            "Validation-only safety-constrained threshold selection "
            "with cost-sensitivity analysis and frozen test evaluation."
        ),
        "data": {
            "validation_source_rows": int(len(validation_all)),
            "validation_final_engines": int(len(validation)),
            "test_source_rows": int(len(test_all)),
            "test_final_engines": int(len(test)),
            "unit_id_handling": (
                "Numeric unit IDs across training and test datasets "
                "are not compared because they are separate namespaces."
            ),
        },
        "safety_constraints": {
            "max_missed_critical_validation_engines": (
                MAX_MISSED_CRITICAL_VALIDATION
            ),
            "max_missed_high_risk_validation_rate": (
                MAX_MISSED_HIGH_RISK_RATE
            ),
            "critical_rul_boundary": CRITICAL_RUL,
            "high_risk_rul_boundary": HIGH_RUL,
        },
        "candidate_thresholds": THRESHOLDS.tolist(),
        "cost_scenarios": COST_SCENARIOS,
        "selected_validation_policies": selected_report,
        "held_out_test_results": test_report,
        "limitations": [
            (
                "The validation set contains relatively few engines; "
                "selected thresholds may be unstable."
            ),
            (
                "If no threshold satisfies the safety constraints, "
                "the report marks the selection as infeasible rather "
                "than claiming that the constraints were met."
            ),
            (
                "Cost scenarios are illustrative assumptions, not "
                "empirically measured industrial costs."
            ),
            (
                "The risk boundaries are design choices and require "
                "domain validation before industrial deployment."
            ),
            (
                "This step performs threshold optimization, not "
                "reinforcement learning or unrestricted self-evolution."
            ),
            (
                "The analysis uses one final observed row per engine "
                "and is not a full multi-cycle maintenance simulation."
            ),
            (
                "Test metrics are for final evaluation only; thresholds "
                "are selected using validation data."
            ),
        ],
    }

    report_path = OUTPUT_DIR / f"step07_1_{SUBSET}_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, allow_nan=False)

    # --------------------------------------------------------
    # 5. Console summary.
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("HELD-OUT TEST RESULTS")
    print("=" * 72)

    display_columns = [
        "model",
        "cost_scenario",
        "threshold_cycles",
        "maintenance_actions",
        "missed_critical_engines",
        "missed_high_or_critical_engines",
        "false_alarm_actions",
        "total_simulated_cost",
        "selection_status",
        "validation_safety_feasible",
    ]

    print(
        test_results_df[display_columns].to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("STEP 7.1 COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print("Output directory:", OUTPUT_DIR)
    print("Research report:", report_path)
    print("Test labels were not used to select thresholds.")
    print("Steps 1–7 were not modified.")


if __name__ == "__main__":
    main()
