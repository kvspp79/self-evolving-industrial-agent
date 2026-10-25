
"""
NASA C-MAPSS FD001 — STEP 7
ADAPTIVE MAINTENANCE POLICY LEARNING AND DECISION COMPARISON

- Selects maintenance thresholds using validation data only.
- Evaluates frozen thresholds on held-out test engines.
- Does not compare engine IDs across training and test namespaces.
- Saves CSV results, plots, and a JSON report.
- Does not modify Steps 1–6.

IMPORTANT:
Costs and risk thresholds are illustrative assumptions.
This is deterministic threshold optimization, not reinforcement learning.
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
OUTPUT_DIR = ROOT / "outputs" / "step07"
SUBSET = "FD001"

CRITICAL_RUL = 15.0
HIGH_RUL = 30.0

MAINTENANCE_COST = 100.0
MISSED_CRITICAL_COST = 1000.0
FALSE_ALARM_COST = 50.0

CANDIDATE_THRESHOLDS = [
    5.0, 10.0, 15.0, 20.0, 25.0,
    30.0, 40.0, 50.0, 60.0, 80.0,
]

FIXED_THRESHOLD = 30.0

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
# INPUT VALIDATION
# ============================================================

def load_predictions(path, split_name):
    """Load and validate a Step 4 prediction CSV."""

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {split_name} prediction file:\n{path}\n\n"
            "Run Step 4 first and confirm its output filenames."
        )

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{split_name} file is missing columns: {missing}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    if df.empty:
        raise ValueError(f"{split_name} prediction file is empty.")

    df = df.copy()

    for col in [
        "actual_RUL",
        "baseline_predicted_RUL",
        "temporal_predicted_RUL",
    ]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    values = df[
        [
            "actual_RUL",
            "baseline_predicted_RUL",
            "temporal_predicted_RUL",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{split_name} data contains NaN or infinite values."
        )

    if "cycle" in df.columns:
        df["cycle"] = pd.to_numeric(df["cycle"], errors="raise")

    return df


def final_row_per_engine(df):
    """
    Return the last observed prediction for each engine.

    Validation may contain multiple cycles per engine. Test predictions
    are normally already one final-cycle row per test engine.
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
            "Could not produce one final row per engine."
        )

    return result


# ============================================================
# METRICS AND POLICY EVALUATION
# ============================================================

def regression_metrics(df, model_name):
    """Calculate RUL regression metrics for the selected rows."""

    prediction_col = PREDICTION_COLUMNS[model_name]

    actual = df["actual_RUL"].to_numpy(dtype=float)
    predicted = df[prediction_col].to_numpy(dtype=float)

    error = predicted - actual
    squared_error = error ** 2

    # NASA C-MAPSS asymmetric score:
    # over-prediction of RUL (late maintenance risk) is penalized
    # more strongly than under-prediction.
    d = predicted - actual
    score_terms = np.where(
        d < 0,
        np.expm1(np.clip(-d / 13.0, 0, 50)),
        np.expm1(np.clip(d / 10.0, 0, 50)),
    )

    return {
        "n": int(len(df)),
        "RMSE": float(np.sqrt(np.mean(squared_error))),
        "MAE": float(np.mean(np.abs(error))),
        "NASA_score": float(np.sum(score_terms)),
    }


def evaluate_policy(df, model_name, threshold):
    """
    Evaluate one threshold policy on a set of engine-level rows.

    Maintenance action:
        predicted RUL <= threshold

    Missed critical:
        actual RUL <= CRITICAL_RUL and no action was taken.

    Missed high-or-critical:
        actual RUL <= HIGH_RUL and no action was taken.

    False alarm:
        action was taken when actual RUL > HIGH_RUL.
    """

    prediction_col = PREDICTION_COLUMNS[model_name]

    actual = df["actual_RUL"].to_numpy(dtype=float)
    predicted = df[prediction_col].to_numpy(dtype=float)

    action = predicted <= float(threshold)

    critical = actual <= CRITICAL_RUL
    high_or_critical = actual <= HIGH_RUL

    missed_critical = critical & ~action
    missed_high_or_critical = high_or_critical & ~action
    false_alarm = action & (actual > HIGH_RUL)

    n = len(df)
    actions_n = int(action.sum())
    missed_critical_n = int(missed_critical.sum())
    missed_high_n = int(missed_high_or_critical.sum())
    false_alarm_n = int(false_alarm.sum())

    maintenance_cost = actions_n * MAINTENANCE_COST
    missed_cost = missed_critical_n * MISSED_CRITICAL_COST
    false_alarm_cost = false_alarm_n * FALSE_ALARM_COST

    total_cost = maintenance_cost + missed_cost + false_alarm_cost

    return {
        "model": model_name,
        "threshold_cycles": float(threshold),
        "n_engines": int(n),
        "maintenance_actions": actions_n,
        "maintenance_action_rate": float(actions_n / n) if n else 0.0,
        "actual_critical_engines": int(critical.sum()),
        "actual_high_or_critical_engines": int(
            high_or_critical.sum()
        ),
        "missed_critical_engines": missed_critical_n,
        "missed_high_or_critical_engines": missed_high_n,
        "false_alarm_actions": false_alarm_n,
        "maintenance_cost": float(maintenance_cost),
        "missed_critical_cost": float(missed_cost),
        "false_alarm_cost": float(false_alarm_cost),
        "total_simulated_cost": float(total_cost),
        "simulated_cost_per_engine": (
            float(total_cost / n) if n else 0.0
        ),
        "costs_are_illustrative": True,
    }


# ============================================================
# VALIDATION-ONLY THRESHOLD SELECTION
# ============================================================

def select_policy(validation, model_name):
    """
    Choose the threshold with the lowest validation simulated cost.

    Tie-breaking is deterministic:
    cost, missed critical engines, false alarms, action count,
    then threshold.

    Test data must never be passed to this function.
    """

    rows = [
        evaluate_policy(validation, model_name, threshold)
        for threshold in CANDIDATE_THRESHOLDS
    ]

    results = pd.DataFrame(rows).sort_values(
        [
            "total_simulated_cost",
            "missed_critical_engines",
            "false_alarm_actions",
            "maintenance_actions",
            "threshold_cycles",
        ],
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)

    return results.iloc[0].to_dict(), results


# ============================================================
# DECISION TABLES
# ============================================================

def make_decision_table(df, selected_policies, split_name):
    """Create per-engine decisions using frozen thresholds."""

    result = df.copy()
    result["evaluation_split"] = split_name

    for model_name, prediction_col in PREDICTION_COLUMNS.items():
        threshold = float(
            selected_policies[model_name]["threshold_cycles"]
        )

        action_col = f"{model_name}_maintenance_action"
        result[f"{model_name}_selected_threshold"] = threshold

        result[action_col] = result[prediction_col] <= threshold

        result[f"{model_name}_absolute_error"] = np.abs(
            result["actual_RUL"] - result[prediction_col]
        )

        result[f"{model_name}_missed_critical"] = (
            (result["actual_RUL"] <= CRITICAL_RUL)
            & ~result[action_col]
        )

        result[f"{model_name}_missed_high_or_critical"] = (
            (result["actual_RUL"] <= HIGH_RUL)
            & ~result[action_col]
        )

        result[f"{model_name}_false_alarm"] = (
            result[action_col]
            & (result["actual_RUL"] > HIGH_RUL)
        )

    return result


# ============================================================
# PLOTS
# ============================================================

def plot_validation_costs(policy_results, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, results in policy_results.items():
        ordered = results.sort_values("threshold_cycles")

        ax.plot(
            ordered["threshold_cycles"],
            ordered["total_simulated_cost"],
            marker="o",
            label=model_name.capitalize(),
        )

    ax.set_xlabel("Maintenance threshold (RUL cycles)")
    ax.set_ylabel("Validation simulated cost")
    ax.set_title("Validation-Only Maintenance Threshold Selection")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_test_decisions(test_results, output_path):
    labels = [
        "Maintenance actions",
        "Missed critical",
        "False alarms",
    ]

    baseline = [
        test_results["baseline"]["maintenance_actions"],
        test_results["baseline"]["missed_critical_engines"],
        test_results["baseline"]["false_alarm_actions"],
    ]

    temporal = [
        test_results["temporal"]["maintenance_actions"],
        test_results["temporal"]["missed_critical_engines"],
        test_results["temporal"]["false_alarm_actions"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, baseline, width, label="Baseline")
    ax.bar(x + width / 2, temporal, width, label="Temporal")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of test engines")
    ax.set_title("Held-Out Test Maintenance Decisions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_test_costs(test_results, output_path):
    labels = [
        "Maintenance",
        "Missed critical",
        "False alarm",
    ]

    baseline = [
        test_results["baseline"]["maintenance_cost"],
        test_results["baseline"]["missed_critical_cost"],
        test_results["baseline"]["false_alarm_cost"],
    ]

    temporal = [
        test_results["temporal"]["maintenance_cost"],
        test_results["temporal"]["missed_critical_cost"],
        test_results["temporal"]["false_alarm_cost"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, baseline, width, label="Baseline")
    ax.bar(x + width / 2, temporal, width, label="Temporal")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Illustrative cost units")
    ax.set_title("Held-Out Test Simulated Cost Components")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("NASA C-MAPSS — STEP 7")
    print("ADAPTIVE MAINTENANCE POLICY LEARNING")
    print(f"Subset: {SUBSET}")
    print("=" * 68)

    validation_path = (
        INPUT_DIR / f"validation_predictions_{SUBSET}.csv"
    )
    test_path = INPUT_DIR / f"test_predictions_{SUBSET}.csv"

    validation_all = load_predictions(
        validation_path,
        "validation",
    )
    test_all = load_predictions(
        test_path,
        "test",
    )

    # Use one final observed row per engine in both splits.
    validation = final_row_per_engine(validation_all)
    test = final_row_per_engine(test_all)

    # IMPORTANT:
    # Do NOT compare validation and test unit_id values.
    # C-MAPSS training-engine IDs and test-engine IDs are separate
    # namespaces and can legitimately overlap numerically.
    #
    # Leakage prevention is based on the Step 4 split construction,
    # not on whether the integer engine IDs happen to match.

    if len(validation) < 10:
        warnings.warn(
            "Fewer than 10 validation engines are available. "
            "Threshold selection may be unstable."
        )

    print("\nDATA SUMMARY")
    print("-" * 48)
    print("Validation prediction rows:", len(validation_all))
    print("Final validation engines:", len(validation))
    print("Test prediction rows:", len(test_all))
    print("Final test engines:", len(test))
    print("Cross-split numeric unit_id overlap is allowed.")

    print("\nILLUSTRATIVE COST ASSUMPTIONS")
    print("-" * 48)
    print("Critical RUL <=", CRITICAL_RUL)
    print("High-risk RUL <=", HIGH_RUL)
    print("Maintenance cost:", MAINTENANCE_COST)
    print("Missed-critical cost:", MISSED_CRITICAL_COST)
    print("False-alarm cost:", FALSE_ALARM_COST)

    # --------------------------------------------------------
    # A. Select thresholds using validation data only.
    # --------------------------------------------------------

    selected_policies = {}
    validation_policy_results = {}

    for model_name in PREDICTION_COLUMNS:
        selected, results = select_policy(
            validation,
            model_name,
        )

        selected_policies[model_name] = selected
        validation_policy_results[model_name] = results

        results.to_csv(
            OUTPUT_DIR
            / f"validation_policy_search_{model_name}_{SUBSET}.csv",
            index=False,
        )

        print(f"\n{model_name.upper()} SELECTED ON VALIDATION")
        print("-" * 48)
        print("Threshold:", selected["threshold_cycles"])
        print("Validation cost:", selected["total_simulated_cost"])
        print("Maintenance actions:", selected["maintenance_actions"])
        print(
            "Missed critical engines:",
            selected["missed_critical_engines"],
        )
        print("False alarms:", selected["false_alarm_actions"])

    combined_search = pd.concat(
        list(validation_policy_results.values()),
        ignore_index=True,
    )

    combined_search.to_csv(
        OUTPUT_DIR / f"validation_policy_search_all_{SUBSET}.csv",
        index=False,
    )

    # --------------------------------------------------------
    # B. Freeze thresholds and evaluate the test set.
    # --------------------------------------------------------

    test_results = {}

    for model_name in PREDICTION_COLUMNS:
        threshold = selected_policies[model_name]["threshold_cycles"]

        test_results[model_name] = evaluate_policy(
            test,
            model_name,
            threshold,
        )

    validation_decisions = make_decision_table(
        validation,
        selected_policies,
        "validation",
    )

    test_decisions = make_decision_table(
        test,
        selected_policies,
        "test",
    )

    validation_decisions.to_csv(
        OUTPUT_DIR
        / f"validation_adaptive_policy_decisions_{SUBSET}.csv",
        index=False,
    )

    test_decisions.to_csv(
        OUTPUT_DIR
        / f"test_adaptive_policy_decisions_{SUBSET}.csv",
        index=False,
    )

    # --------------------------------------------------------
    # C. Fixed 30-cycle reference policy.
    # This is a predeclared descriptive reference, not tuned on test.
    # --------------------------------------------------------

    fixed_test_results = {
        model_name: evaluate_policy(
            test,
            model_name,
            FIXED_THRESHOLD,
        )
        for model_name in PREDICTION_COLUMNS
    }

    pd.DataFrame(
        fixed_test_results.values()
    ).to_csv(
        OUTPUT_DIR / f"test_fixed_threshold_comparison_{SUBSET}.csv",
        index=False,
    )

    # --------------------------------------------------------
    # D. Figures.
    # --------------------------------------------------------

    plot_validation_costs(
        validation_policy_results,
        OUTPUT_DIR / f"validation_policy_costs_{SUBSET}.png",
    )

    plot_test_decisions(
        test_results,
        OUTPUT_DIR / f"test_policy_decision_comparison_{SUBSET}.png",
    )

    plot_test_costs(
        test_results,
        OUTPUT_DIR / f"test_policy_cost_components_{SUBSET}.png",
    )

    # --------------------------------------------------------
    # E. Research report.
    # --------------------------------------------------------

    report = {
        "dataset": "NASA C-MAPSS",
        "subset": SUBSET,
        "step": 7,
        "method": (
            "Deterministic threshold selection using validation engines "
            "followed by evaluation on held-out test engines."
        ),
        "validation_rows_in_source_file": int(len(validation_all)),
        "validation_final_engines": int(len(validation)),
        "test_rows_in_source_file": int(len(test_all)),
        "test_final_engines": int(len(test)),
        "cross_split_unit_id_overlap_check": (
            "Not applicable: training and test engine IDs are separate "
            "namespaces in C-MAPSS."
        ),
        "candidate_thresholds_cycles": CANDIDATE_THRESHOLDS,
        "selected_policies_from_validation": selected_policies,
        "held_out_test_results": test_results,
        "fixed_threshold_reference": {
            "threshold_cycles": FIXED_THRESHOLD,
            "results": fixed_test_results,
        },
        "cost_assumptions": {
            "maintenance_cost_per_action": MAINTENANCE_COST,
            "missed_critical_cost_per_engine": MISSED_CRITICAL_COST,
            "false_alarm_cost_per_action": FALSE_ALARM_COST,
            "currency": "unspecified illustrative units",
            "empirically_validated": False,
        },
        "risk_definitions": {
            "critical": f"actual RUL <= {CRITICAL_RUL}",
            "high_or_critical": f"actual RUL <= {HIGH_RUL}",
            "false_alarm": (
                f"action taken when actual RUL > {HIGH_RUL}"
            ),
        },
        "limitations": [
            (
                "Threshold optimization is not reinforcement learning "
                "or unrestricted self-evolving control."
            ),
            (
                "Thresholds depend on the chosen illustrative cost "
                "assumptions and risk definitions."
            ),
            (
                "The validation sample contains only a limited number "
                "of engines, so the selected thresholds may be unstable."
            ),
            (
                "Test metrics are a held-out evaluation, not a guarantee "
                "of future fleet performance."
            ),
            (
                "This is a final-observed-cycle decision comparison, "
                "not a multi-cycle maintenance simulation."
            ),
            (
                "Poor uncertainty interval coverage observed in Step 6 "
                "is not corrected by this step."
            ),
        ],
    }

    report_path = OUTPUT_DIR / f"step07_{SUBSET}_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, allow_nan=False)

    # --------------------------------------------------------
    # F. Final console summary.
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("HELD-OUT TEST RESULTS — VALIDATION-SELECTED POLICIES")
    print("=" * 68)

    for model_name, result in test_results.items():
        print(f"\n{model_name.upper()}")
        print("Frozen threshold:", result["threshold_cycles"])
        print("Test engines:", result["n_engines"])
        print("Maintenance actions:", result["maintenance_actions"])
        print(
            "Missed critical engines:",
            result["missed_critical_engines"],
        )
        print(
            "Missed high-or-critical engines:",
            result["missed_high_or_critical_engines"],
        )
        print("False alarms:", result["false_alarm_actions"])
        print("Maintenance cost:", result["maintenance_cost"])
        print("Missed-critical cost:", result["missed_critical_cost"])
        print("False-alarm cost:", result["false_alarm_cost"])
        print("Total simulated cost:", result["total_simulated_cost"])
        print(
            "Cost per engine:",
            round(result["simulated_cost_per_engine"], 4),
        )

    print("\n" + "=" * 68)
    print("STEP 7 COMPLETED SUCCESSFULLY")
    print("=" * 68)
    print("Output directory:", OUTPUT_DIR)
    print("Research report:", report_path)
    print("Thresholds selected on validation data only.")
    print("Test labels were not used to select thresholds.")
    print("Steps 1–6 were not modified.")


if __name__ == "__main__":
    main()
