
# ================================================================
# STEP 09: COST AND ROBUSTNESS EVALUATION
# Project:
# A Self-Evolving Explainable Rational Agent for Adaptive
# Industrial Decision Intelligence Using Statistical Trend Learning
#
# Input:
# outputs/step05_two_feature_test_predictions.csv
#
# Outputs:
#   outputs/step09_fixed_policy_baselines.csv
#   outputs/step09_repeated_episode_results.csv
#   outputs/step09_online_batch_feedback.csv
#   outputs/step09_online_decisions.csv
#   outputs/step09_robustness_summary.csv
#   outputs/step09_fixed_policy_costs.png
#   outputs/step09_online_cost_variability.png
#   outputs/step09_report.json
#
# Note:
# The cost matrix is a simulation, not measured industrial cost data.
# AI4I rows are shuffled for repeated episodes; they are not a true
# chronological history of an individual machine.
# ================================================================

import os
import sys
import json
import re
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------
# 1. PATHS AND REPRODUCIBILITY
# ------------------------------------------------

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

INPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "step05_two_feature_test_predictions.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEEDS = [11, 23, 42, 57, 81, 101, 202, 303, 404, 505]

BATCH_SIZES = [50, 100, 200]

EPSILON = 0.15


# ------------------------------------------------
# 2. SIMULATED COST MATRIX
# ------------------------------------------------
# Each action has a cost depending on whether a failure
# actually occurred.
#
# These values are illustrative assumptions and must be
# justified or calibrated before making industrial claims.

COST_MATRIX = {
    "Normal": {
        "failure": 1000.0,
        "nonfailure": 0.0
    },
    "Monitor": {
        "failure": 600.0,
        "nonfailure": 20.0
    },
    "Schedule Maintenance": {
        "failure": 250.0,
        "nonfailure": 80.0
    },
    "Critical": {
        "failure": 100.0,
        "nonfailure": 200.0
    }
}


# ------------------------------------------------
# 3. CANDIDATE POLICIES
# ------------------------------------------------
# Thresholds are in ascending order:
# (monitor threshold, scheduled maintenance threshold,
#  critical threshold).
#
# IMPORTANT:
# For a valid comparison with Step 8, these thresholds
# should match the policies used in Step 8 exactly.

POLICIES = {
    "Initial_Step6": (0.02, 0.08, 0.20),
    "Higher_Thresholds": (0.05, 0.15, 0.30),
    "Lower_Thresholds": (0.01, 0.05, 0.15),
    "Maintenance_Focused": (0.01, 0.04, 0.12),
    "Conservative": (0.08, 0.20, 0.40)
}


# ------------------------------------------------
# 4. GENERAL UTILITIES
# ------------------------------------------------

def normalize_name(name):
    """Normalize a column name for matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column_by_alias(columns, aliases):
    """Find a column using normalized aliases."""
    normalized = {}

    for column in columns:
        normalized[normalize_name(column)] = column

    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]

    return None


def identify_actual_column(data):
    """Identify the ground-truth failure label."""
    aliases = [
        "Actual_Failure",
        "actual_failure",
        "Actual",
        "y_true",
        "y_test",
        "True_Label",
        "Actual_Label",
        "Failure",
        "Machine_Failure",
        "Target",
        "Failure_Target",
        "Failure_Actual",
        "Target_Label"
    ]

    column = find_column_by_alias(data.columns, aliases)

    if column is not None:
        return column

    # Try common target names after normalization.
    for column in data.columns:
        name = normalize_name(column)

        if name in [
            "machinefailure",
            "failuretarget",
            "actualfailure",
            "failureactual",
            "targetlabel",
            "ytrue"
        ]:
            return column

    return None


def identify_probability_column(data):
    """Identify the predicted failure-probability column."""
    aliases = [
        "Predicted_Probability",
        "Failure_Probability",
        "Calibrated_Probability",
        "Predicted_Failure_Probability",
        "Probability",
        "probability",
        "y_probability",
        "y_prob",
        "Failure_Prob",
        "Predicted_Prob",
        "Two_Feature_Probability",
        "XGBoost_Probability",
        "Calibrated_Prob"
    ]

    column = find_column_by_alias(data.columns, aliases)

    if column is not None:
        return column

    # Look for probability-like columns, excluding actual labels.
    candidates = []

    for column in data.columns:
        name = normalize_name(column)

        if (
            "prob" in name
            or name.endswith("score")
            or name in ["yprob", "predictionprobability"]
        ):
            if not any(
                token in name
                for token in ["actual", "true", "label"]
            ):
                candidates.append(column)

    # Prefer a column whose values lie in [0, 1].
    for column in candidates:
        values = pd.to_numeric(data[column], errors="coerce")
        valid = values.dropna()

        if len(valid) > 0:
            if valid.min() >= 0.0 and valid.max() <= 1.0:
                return column

    return None


def identify_columns(data):
    """Identify actual labels and predicted probabilities."""
    actual_column = identify_actual_column(data)
    probability_column = identify_probability_column(data)

    print("\nInput columns found:")
    for column in data.columns:
        print("  - {0}".format(column))

    if actual_column is None:
        raise ValueError(
            "Could not identify the actual failure-label column. "
            "Review the printed input columns and update "
            "identify_actual_column() with the correct column name."
        )

    if probability_column is None:
        raise ValueError(
            "Could not identify the predicted probability column. "
            "Review the printed input columns and update "
            "identify_probability_column() with the correct column name."
        )

    return actual_column, probability_column


# ------------------------------------------------
# 5. ACTION SELECTION AND COST CALCULATION
# ------------------------------------------------

def select_actions(probabilities, thresholds):
    """Map failure probabilities to maintenance actions."""
    monitor_threshold, maintenance_threshold, critical_threshold = (
        thresholds
    )

    actions = np.select(
        [
            probabilities >= critical_threshold,
            probabilities >= maintenance_threshold,
            probabilities >= monitor_threshold
        ],
        [
            "Critical",
            "Schedule Maintenance",
            "Monitor"
        ],
        default="Normal"
    )

    return actions.astype(str)


def calculate_costs(actual, actions):
    """Calculate simulated row-level costs."""
    actual = np.asarray(actual).astype(int)
    actions = np.asarray(actions).astype(str)

    costs = np.zeros(len(actual), dtype=float)

    for index in range(len(actual)):
        action = actions[index]

        if action not in COST_MATRIX:
            raise ValueError(
                "Unknown action encountered: {0}".format(action)
            )

        if actual[index] == 1:
            costs[index] = COST_MATRIX[action]["failure"]
        else:
            costs[index] = COST_MATRIX[action]["nonfailure"]

    return costs


def evaluate_actions(actual, actions):
    """Return standard cost and action metrics."""
    actual = np.asarray(actual).astype(int)
    actions = np.asarray(actions).astype(str)

    row_costs = calculate_costs(actual, actions)

    failure_mask = actual == 1
    nonfailure_mask = actual == 0

    missed_failures = int(
        np.sum(failure_mask & (actions == "Normal"))
    )

    maintenance_actions = int(
        np.sum(actions == "Schedule Maintenance")
    )

    critical_actions = int(
        np.sum(actions == "Critical")
    )

    monitor_actions = int(
        np.sum(actions == "Monitor")
    )

    normal_actions = int(
        np.sum(actions == "Normal")
    )

    return {
        "rows": int(len(actual)),
        "actual_failures": int(np.sum(failure_mask)),
        "failure_rate": float(np.mean(actual)),
        "total_cost": float(np.sum(row_costs)),
        "mean_cost_per_row": float(np.mean(row_costs)),
        "cost_std_per_row": float(np.std(row_costs)),
        "missed_failures": missed_failures,
        "maintenance_actions": maintenance_actions,
        "critical_actions": critical_actions,
        "monitor_actions": monitor_actions,
        "normal_actions": normal_actions,
        "failure_action_rate": float(
            np.mean(actions[failure_mask] != "Normal")
        ) if np.any(failure_mask) else float("nan"),
        "nonfailure_maintenance_rate": float(
            np.mean(
                np.isin(
                    actions[nonfailure_mask],
                    ["Schedule Maintenance", "Critical"]
                )
            )
        ) if np.any(nonfailure_mask) else float("nan")
    }


# ------------------------------------------------
# 6. FIXED POLICY AND SIMPLE BASELINE EVALUATION
# ------------------------------------------------

def evaluate_fixed_policies(actual, probabilities):
    """Evaluate every fixed policy on the same rows."""
    results = []

    for policy_name, thresholds in POLICIES.items():
        actions = select_actions(probabilities, thresholds)
        metrics = evaluate_actions(actual, actions)

        row = {
            "policy": policy_name,
            "monitor_threshold": thresholds[0],
            "maintenance_threshold": thresholds[1],
            "critical_threshold": thresholds[2]
        }

        row.update(metrics)
        results.append(row)

    # Simple action-only baselines.
    for action_name in [
        "Normal",
        "Monitor",
        "Schedule Maintenance",
        "Critical"
    ]:
        actions = np.full(
            len(actual),
            action_name,
            dtype=object
        )

        metrics = evaluate_actions(actual, actions)

        row = {
            "policy": "Always_" + action_name.replace(" ", "_"),
            "monitor_threshold": np.nan,
            "maintenance_threshold": np.nan,
            "critical_threshold": np.nan
        }

        row.update(metrics)
        results.append(row)

    return pd.DataFrame(results)


# ------------------------------------------------
# 7. REPEATED RANDOMIZED EPISODES
# ------------------------------------------------

def run_repeated_episodes(actual, probabilities):
    """Evaluate policies over repeated shuffled episodes."""
    records = []

    number_of_rows = len(actual)

    for seed in RANDOM_SEEDS:
        rng = np.random.RandomState(seed)
        order = rng.permutation(number_of_rows)

        shuffled_actual = actual[order]
        shuffled_probabilities = probabilities[order]

        for policy_name, thresholds in POLICIES.items():
            actions = select_actions(
                shuffled_probabilities,
                thresholds
            )

            metrics = evaluate_actions(shuffled_actual, actions)

            record = {
                "seed": int(seed),
                "policy": policy_name
            }

            record.update(metrics)
            records.append(record)

    return pd.DataFrame(records)


# ------------------------------------------------
# 8. ONLINE BATCH FEEDBACK EXPERIMENT
# ------------------------------------------------

def run_online_episode(actual, probabilities, batch_size, seed):
    """
    Select a policy for each next batch using feedback from
    previously completed batches only.

    This is a simple epsilon-greedy policy selector.
    It is not a full reinforcement-learning system.
    """
    rng = np.random.RandomState(seed)

    order = rng.permutation(len(actual))
    shuffled_actual = actual[order]
    shuffled_probabilities = probabilities[order]

    policy_names = list(POLICIES.keys())

    observed_costs = dict(
        (name, []) for name in policy_names
    )

    decision_records = []
    batch_records = []

    batch_number = 0

    for start in range(0, len(order), batch_size):
        end = min(start + batch_size, len(order))

        batch_actual = shuffled_actual[start:end]
        batch_probabilities = shuffled_probabilities[start:end]

        batch_number += 1

        # Explore randomly or exploit the policy with the
        # lowest mean cost observed in earlier batches.
        if (
            len(decision_records) == 0
            or rng.rand() < EPSILON
            or all(
                len(observed_costs[name]) == 0
                for name in policy_names
            )
        ):
            selected_policy = policy_names[
                rng.randint(0, len(policy_names))
            ]
            selection_type = "exploration"
        else:
            available = [
                name for name in policy_names
                if len(observed_costs[name]) > 0
            ]

            if len(available) == 0:
                selected_policy = policy_names[
                    rng.randint(0, len(policy_names))
                ]
                selection_type = "exploration"
            else:
                mean_costs = dict(
                    (
                        name,
                        float(np.mean(observed_costs[name]))
                    )
                    for name in available
                )

                best_cost = min(mean_costs.values())

                best_policies = [
                    name for name in available
                    if mean_costs[name] == best_cost
                ]

                selected_policy = best_policies[
                    rng.randint(0, len(best_policies))
                ]
                selection_type = "exploitation"

        thresholds = POLICIES[selected_policy]

        selected_actions = select_actions(
            batch_probabilities,
            thresholds
        )

        selected_metrics = evaluate_actions(
            batch_actual,
            selected_actions
        )

        # Feedback is recorded after the selected batch ends.
        observed_costs[selected_policy].append(
            selected_metrics["mean_cost_per_row"]
        )

        # Counterfactual comparisons on the same batch.
        # These are offline evaluations, not actions actually taken.
        counterfactual_costs = {}

        for policy_name, policy_thresholds in POLICIES.items():
            counterfactual_actions = select_actions(
                batch_probabilities,
                policy_thresholds
            )

            counterfactual_metrics = evaluate_actions(
                batch_actual,
                counterfactual_actions
            )

            counterfactual_costs[policy_name] = (
                counterfactual_metrics["mean_cost_per_row"]
            )

        batch_record = {
            "seed": int(seed),
            "batch_size": int(batch_size),
            "batch_number": int(batch_number),
            "start_index": int(start),
            "end_index_exclusive": int(end),
            "selected_policy": selected_policy,
            "selection_type": selection_type,
            "batch_rows": int(len(batch_actual)),
            "batch_actual_failures": int(np.sum(batch_actual)),
            "batch_total_cost": selected_metrics["total_cost"],
            "batch_mean_cost": selected_metrics["mean_cost_per_row"],
            "batch_cost_std": selected_metrics["cost_std_per_row"],
            "historical_selected_policy_mean_cost": float(
                np.mean(observed_costs[selected_policy])
            )
        }

        for policy_name in policy_names:
            batch_record[
                "counterfactual_mean_cost_" + policy_name
            ] = counterfactual_costs[policy_name]

        batch_records.append(batch_record)

        decision_records.append({
            "seed": int(seed),
            "batch_size": int(batch_size),
            "batch_number": int(batch_number),
            "selected_policy": selected_policy,
            "selection_type": selection_type,
            "batch_mean_cost": selected_metrics["mean_cost_per_row"],
            "cumulative_cost": float(
                sum(
                    item["batch_total_cost"]
                    for item in batch_records
                )
            )
        })

    episode_total_cost = float(
        sum(
            item["batch_total_cost"]
            for item in batch_records
        )
    )

    episode_mean_cost = episode_total_cost / float(len(order))

    episode_record = {
        "seed": int(seed),
        "batch_size": int(batch_size),
        "rows": int(len(order)),
        "total_cost": episode_total_cost,
        "mean_cost_per_row": episode_mean_cost,
        "number_of_batches": int(len(batch_records)),
        "exploration_batches": int(
            sum(
                item["selection_type"] == "exploration"
                for item in batch_records
            )
        ),
        "exploitation_batches": int(
            sum(
                item["selection_type"] == "exploitation"
                for item in batch_records
            )
        )
    }

    for policy_name in policy_names:
        episode_record[
            "selected_batches_" + policy_name
        ] = int(
            sum(
                item["selected_policy"] == policy_name
                for item in batch_records
            )
        )

    return episode_record, batch_records, decision_records


# ------------------------------------------------
# 9. ROBUSTNESS SUMMARY
# ------------------------------------------------

def build_robustness_summary(repeated_results):
    """Summarize mean and variation across randomized seeds."""
    summary = (
        repeated_results
        .groupby("policy")
        .agg(
            mean_total_cost=("total_cost", "mean"),
            std_total_cost=("total_cost", "std"),
            min_total_cost=("total_cost", "min"),
            max_total_cost=("total_cost", "max"),
            mean_cost_per_row=("mean_cost_per_row", "mean"),
            std_cost_per_row=("mean_cost_per_row", "std"),
            mean_missed_failures=("missed_failures", "mean"),
            mean_maintenance_actions=("maintenance_actions", "mean"),
            mean_critical_actions=("critical_actions", "mean")
        )
        .reset_index()
    )

    summary = summary.fillna(0.0)

    return summary.sort_values(
        "mean_cost_per_row",
        ascending=True
    )


# ------------------------------------------------
# 10. PLOTS
# ------------------------------------------------

def save_fixed_policy_plot(fixed_results):
    """Save a cost comparison chart."""
    selected = fixed_results[
        ~fixed_results["policy"].str.startswith("Always_")
    ].copy()

    selected = selected.sort_values(
        "mean_cost_per_row",
        ascending=True
    )

    plt.figure(figsize=(11, 6))
    plt.bar(
        selected["policy"],
        selected["mean_cost_per_row"]
    )
    plt.title("Step 9: Fixed Policy Simulated Cost")
    plt.xlabel("Policy")
    plt.ylabel("Mean Simulated Cost per Row")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "step09_fixed_policy_costs.png"
    )

    plt.savefig(output_path, dpi=160)
    plt.close()

    print("Saved plot: {0}".format(output_path))


def save_online_variability_plot(online_results):
    """Plot mean online cost by batch size."""
    grouped = (
        online_results
        .groupby("batch_size")["mean_cost_per_row"]
        .agg(["mean", "std"])
        .reset_index()
    )

    grouped["std"] = grouped["std"].fillna(0.0)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        grouped["batch_size"],
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        capsize=5
    )
    plt.title("Step 9: Online Agent Cost Variability")
    plt.xlabel("Batch Size")
    plt.ylabel("Mean Simulated Cost per Row")
    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "step09_online_cost_variability.png"
    )

    plt.savefig(output_path, dpi=160)
    plt.close()

    print("Saved plot: {0}".format(output_path))


# ------------------------------------------------
# 11. MAIN
# ------------------------------------------------

def main():
    print("=" * 68)
    print("STEP 9: COST AND ROBUSTNESS EVALUATION")
    print("=" * 68)

    print("Python executable: {0}".format(sys.executable))
    print("Python version: {0}".format(sys.version.split()[0]))
    print("Project root: {0}".format(ROOT_DIR))
    print("Input file: {0}".format(INPUT_FILE))

    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(
            "Required input CSV was not found: {0}".format(INPUT_FILE)
        )

    data = pd.read_csv(INPUT_FILE)

    if data.empty:
        raise ValueError("The input CSV contains no rows.")

    actual_column, probability_column = identify_columns(data)

    print("\nDetected actual-label column: {0}".format(actual_column))
    print(
        "Detected probability column: {0}".format(probability_column)
    )

    actual_series = pd.to_numeric(
        data[actual_column],
        errors="coerce"
    )

    probability_series = pd.to_numeric(
        data[probability_column],
        errors="coerce"
    )

    valid_mask = (
        actual_series.notna()
        & probability_series.notna()
    )

    data = data.loc[valid_mask].copy()

    actual_series = actual_series.loc[valid_mask]
    probability_series = probability_series.loc[valid_mask]

    if len(data) == 0:
        raise ValueError(
            "No rows have valid numeric labels and probabilities."
        )

    actual_values = actual_series.to_numpy(dtype=float)
    probability_values = probability_series.to_numpy(dtype=float)

    # Ensure actual labels are binary.
    unique_labels = set(np.unique(actual_values).tolist())

    if not unique_labels.issubset(set([0.0, 1.0])):
        raise ValueError(
            "Actual labels must be encoded as 0 and 1. "
            "Observed labels: {0}".format(sorted(unique_labels))
        )

    # Ensure probabilities are valid.
    if (
        np.any(probability_values < 0.0)
        or np.any(probability_values > 1.0)
    ):
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    actual = actual_values.astype(int)
    probabilities = probability_values.astype(float)

    print("\nValid rows: {0}".format(len(actual)))
    print("Actual failures: {0}".format(int(np.sum(actual))))
    print(
        "Failure rate: {0:.3f}%".format(
            100.0 * float(np.mean(actual))
        )
    )

    if len(np.unique(actual)) < 2:
        print(
            "WARNING: Only one actual-label class is present. "
            "Cost metrics can be calculated, but the sample may not "
            "be representative."
        )

    # A. Fixed policies and always-action baselines.
    fixed_results = evaluate_fixed_policies(
        actual,
        probabilities
    )

    fixed_path = os.path.join(
        OUTPUT_DIR,
        "step09_fixed_policy_baselines.csv"
    )
    fixed_results.to_csv(fixed_path, index=False)

    print("\nFixed policy cost comparison:")
    display_columns = [
        "policy",
        "total_cost",
        "mean_cost_per_row",
        "missed_failures"
    ]

    print(
        fixed_results[display_columns]
        .sort_values("mean_cost_per_row")
        .to_string(index=False)
    )

    # B. Repeated randomized episodes.
    repeated_results = run_repeated_episodes(
        actual,
        probabilities
    )

    repeated_path = os.path.join(
        OUTPUT_DIR,
        "step09_repeated_episode_results.csv"
    )
    repeated_results.to_csv(repeated_path, index=False)

    # C. Online batch-feedback experiments.
    online_episode_records = []
    online_batch_records = []
    online_decision_records = []

    for batch_size in BATCH_SIZES:
        for seed in RANDOM_SEEDS:
            episode, batches, decisions = run_online_episode(
                actual,
                probabilities,
                batch_size,
                seed
            )

            online_episode_records.append(episode)
            online_batch_records.extend(batches)
            online_decision_records.extend(decisions)

    online_results = pd.DataFrame(online_episode_records)
    online_batches = pd.DataFrame(online_batch_records)
    online_decisions = pd.DataFrame(online_decision_records)

    online_results_path = os.path.join(
        OUTPUT_DIR,
        "step09_online_batch_feedback.csv"
    )
    online_batches.to_csv(online_results_path, index=False)

    online_decisions_path = os.path.join(
        OUTPUT_DIR,
        "step09_online_decisions.csv"
    )
    online_decisions.to_csv(online_decisions_path, index=False)

    online_summary_path = os.path.join(
        OUTPUT_DIR,
        "step09_online_episode_summary.csv"
    )
    online_results.to_csv(online_summary_path, index=False)

    # D. Robustness summary across shuffled test-set orders.
    robustness_summary = build_robustness_summary(
        repeated_results
    )

    robustness_path = os.path.join(
        OUTPUT_DIR,
        "step09_robustness_summary.csv"
    )
    robustness_summary.to_csv(robustness_path, index=False)

    # E. Visualizations.
    save_fixed_policy_plot(fixed_results)
    save_online_variability_plot(online_results)

    # F. Machine-readable report.
    best_fixed = fixed_results.sort_values(
        "mean_cost_per_row",
        ascending=True
    ).iloc[0]

    online_grouped = (
        online_results
        .groupby("batch_size")["mean_cost_per_row"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    report = {
        "step": 9,
        "title": "Cost and Robustness Evaluation",
        "input_file": os.path.relpath(INPUT_FILE, ROOT_DIR),
        "actual_label_column": str(actual_column),
        "probability_column": str(probability_column),
        "valid_rows": int(len(actual)),
        "actual_failures": int(np.sum(actual)),
        "failure_rate": float(np.mean(actual)),
        "random_seeds": RANDOM_SEEDS,
        "batch_sizes": BATCH_SIZES,
        "epsilon": EPSILON,
        "policies": {
            name: list(thresholds)
            for name, thresholds in POLICIES.items()
        },
        "simulated_cost_matrix": COST_MATRIX,
        "best_fixed_policy_by_observed_mean_cost": str(
            best_fixed["policy"]
        ),
        "best_fixed_policy_mean_cost_per_row": float(
            best_fixed["mean_cost_per_row"]
        ),
        "online_batch_size_summary": online_grouped.to_dict(
            orient="records"
        ),
        "interpretation_limitations": [
            "Costs are simulated assumptions, not measured industrial costs.",
            "The AI4I dataset does not provide chronological histories "
            "for individual machines.",
            "Repeated episodes shuffle the available test rows and "
            "measure sensitivity to row order, not real temporal drift.",
            "The online selector chooses among predefined policies.",
            "The best observed cost does not establish statistical "
            "significance or real-world savings.",
            "For a valid Step 8 comparison, use the same cost matrix, "
            "policy thresholds, and evaluation rows."
        ]
    }

    report_path = os.path.join(
        OUTPUT_DIR,
        "step09_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(
            report,
            report_file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 68)
    print("STEP 9 COMPLETED")
    print("=" * 68)

    print("Fixed policy results: {0}".format(fixed_path))
    print("Repeated episodes: {0}".format(repeated_path))
    print("Online batch feedback: {0}".format(online_results_path))
    print("Online decisions: {0}".format(online_decisions_path))
    print("Online episode summary: {0}".format(online_summary_path))
    print("Robustness summary: {0}".format(robustness_path))
    print("JSON report: {0}".format(report_path))

    print("\nOnline episode summary by batch size:")
    print(online_grouped.to_string(index=False))

    print(
        "\nResearch note: this experiment evaluates a simulated "
        "cost-sensitive adaptive policy selector. It does not, by "
        "itself, demonstrate real industrial cost savings."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nSTEP 9 FAILED")
        print("Error type: {0}".format(type(error).__name__))
        print("Error message: {0}".format(error))
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
