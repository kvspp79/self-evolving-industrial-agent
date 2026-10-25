
"""
STEP 6: RATIONAL MAINTENANCE AGENT
Project: Self-Evolving Explainable Industrial Agent

Input:
    outputs/step05_two_feature_test_predictions.csv

Uses the all-six-feature model's held-out test probabilities to
make transparent maintenance decisions.

IMPORTANT:
    All maintenance costs below are illustrative simulated costs.
    They are not measured real-world industrial costs.
    The test labels are used only to evaluate decisions after they
    have been made; they do not determine the actions.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INPUT_PATH = OUTPUT_DIR / "step05_two_feature_test_predictions.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initial transparent probability thresholds.
# These are starting policy assumptions, not learned thresholds.
NORMAL_MAX = 0.02
MONITOR_MAX = 0.08
SCHEDULE_MAX = 0.20

# Simulated cost units.
# Each action has a cost depending on whether failure occurred.
#
# failure_cost: residual cost if failure occurs after the decision
# no_failure_cost: intervention/monitoring cost if failure does not occur
#
# These values must be replaced with domain-validated costs if available.
COST_MODEL = {
    "Normal": {
        "failure_cost": 1000.0,
        "no_failure_cost": 0.0,
    },
    "Monitor": {
        "failure_cost": 600.0,
        "no_failure_cost": 20.0,
    },
    "Schedule Maintenance": {
        "failure_cost": 250.0,
        "no_failure_cost": 80.0,
    },
    "Critical": {
        "failure_cost": 100.0,
        "no_failure_cost": 200.0,
    },
}


# ============================================================
# 2. LOAD AND VALIDATE STEP 5 PREDICTIONS
# ============================================================

def load_predictions():
    print("=" * 72)
    print("STEP 6: RATIONAL MAINTENANCE AGENT")
    print("=" * 72)

    print(f"\nProject folder: {PROJECT_ROOT}")
    print(f"Input file:     {INPUT_PATH}")

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            "\nRequired Step 5 prediction file was not found:\n"
            f"{INPUT_PATH}\n\n"
            "Run Step 5 successfully before running Step 6."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "Actual_Failure",
        "All_Features_Probability",
        "All_Features_Prediction",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "The Step 5 prediction file is missing required columns: "
            + ", ".join(missing)
        )

    df = df.copy()

    df["Actual_Failure"] = pd.to_numeric(
        df["Actual_Failure"], errors="coerce"
    )
    df["All_Features_Probability"] = pd.to_numeric(
        df["All_Features_Probability"], errors="coerce"
    )
    df["All_Features_Prediction"] = pd.to_numeric(
        df["All_Features_Prediction"], errors="coerce"
    )

    df = df.dropna(subset=required_columns).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid prediction records remain.")

    if not df["Actual_Failure"].isin([0, 1]).all():
        raise ValueError("Actual_Failure must contain only 0 and 1.")

    if not df["All_Features_Prediction"].isin([0, 1]).all():
        raise ValueError(
            "All_Features_Prediction must contain only 0 and 1."
        )

    probabilities = df["All_Features_Probability"]

    if not probabilities.between(0, 1).all():
        raise ValueError(
            "Failure probabilities must be between 0 and 1."
        )

    if df["Actual_Failure"].nunique() != 2:
        raise ValueError(
            "Evaluation requires both failure and non-failure cases."
        )

    print("\nPredictions loaded successfully.")
    print(f"Evaluation rows: {len(df)}")
    print(
        "Actual failures: "
        f"{int(df['Actual_Failure'].sum())}"
    )
    print(
        "Actual failure rate: "
        f"{df['Actual_Failure'].mean():.3%}"
    )

    return df


# ============================================================
# 3. TRANSPARENT DECISION POLICY
# ============================================================

def choose_action(probability):
    """
    Map a predicted failure probability to a maintenance action.

    The action depends only on the predicted probability.
    It does not use the actual test outcome.
    """

    if probability < NORMAL_MAX:
        return "Normal"

    if probability < MONITOR_MAX:
        return "Monitor"

    if probability < SCHEDULE_MAX:
        return "Schedule Maintenance"

    return "Critical"


def assign_actions(df):
    result = df.copy()

    result["Agent_Action"] = (
        result["All_Features_Probability"].apply(choose_action)
    )

    # This column is included for descriptive evaluation only.
    result["Decision_Threshold"] = result[
        "All_Features_Probability"
    ].apply(
        lambda p: (
            NORMAL_MAX if p < NORMAL_MAX
            else MONITOR_MAX if p < MONITOR_MAX
            else SCHEDULE_MAX if p < SCHEDULE_MAX
            else 1.0
        )
    )

    return result


# ============================================================
# 4. APPLY THE SIMULATED COST MODEL
# ============================================================

def calculate_cost(action, actual_failure):
    """
    Calculate simulated cost after the outcome is known.

    This function evaluates a previously selected action.
    It does not choose the action.
    """

    action_costs = COST_MODEL[action]

    if int(actual_failure) == 1:
        return float(action_costs["failure_cost"])

    return float(action_costs["no_failure_cost"])


def evaluate_agent(df):
    result = df.copy()

    result["Simulated_Cost"] = result.apply(
        lambda row: calculate_cost(
            row["Agent_Action"],
            row["Actual_Failure"],
        ),
        axis=1,
    )

    result["Correct_Failure_Alert"] = (
        (result["Agent_Action"] != "Normal")
        & (result["Actual_Failure"] == 1)
    ).astype(int)

    result["Failure_Missed_By_Agent"] = (
        (result["Agent_Action"] == "Normal")
        & (result["Actual_Failure"] == 1)
    ).astype(int)

    return result


# ============================================================
# 5. SUMMARIZE AGENT DECISIONS
# ============================================================

def summarize_actions(df):
    action_order = [
        "Normal",
        "Monitor",
        "Schedule Maintenance",
        "Critical",
    ]

    summary = (
        df.groupby("Agent_Action")
        .agg(
            Number_of_Rows=("Actual_Failure", "size"),
            Actual_Failures=("Actual_Failure", "sum"),
            Mean_Predicted_Probability=(
                "All_Features_Probability", "mean"
            ),
            Total_Simulated_Cost=("Simulated_Cost", "sum"),
            Mean_Simulated_Cost=("Simulated_Cost", "mean"),
        )
        .reindex(action_order, fill_value=0)
        .reset_index()
        .rename(columns={"Agent_Action": "Action"})
    )

    summary["Percentage_of_Rows"] = (
        summary["Number_of_Rows"] / len(df) * 100.0
    )

    summary["Observed_Failure_Rate"] = np.where(
        summary["Number_of_Rows"] > 0,
        summary["Actual_Failures"]
        / summary["Number_of_Rows"],
        np.nan,
    )

    return summary


# ============================================================
# 6. COMPARE WITH SIMPLE REFERENCE POLICIES
# ============================================================

def evaluate_reference_policy(df, policy_name, action):
    """
    Reference policies always choose one action.
    They are included only to provide a basic cost comparison.
    """

    costs = [
        calculate_cost(action, y)
        for y in df["Actual_Failure"]
    ]

    return {
        "Policy": policy_name,
        "Action": action,
        "Total_Simulated_Cost": float(np.sum(costs)),
        "Mean_Simulated_Cost_Per_Row": float(np.mean(costs)),
        "Rows": int(len(df)),
    }


def compare_policies(df):
    rows = [
        evaluate_reference_policy(
            df, "Always Normal", "Normal"
        ),
        evaluate_reference_policy(
            df, "Always Monitor", "Monitor"
        ),
        evaluate_reference_policy(
            df, "Always Schedule", "Schedule Maintenance"
        ),
        evaluate_reference_policy(
            df, "Always Critical", "Critical"
        ),
        {
            "Policy": "Probability-Based Agent",
            "Action": "Probability thresholds",
            "Total_Simulated_Cost": float(
                df["Simulated_Cost"].sum()
            ),
            "Mean_Simulated_Cost_Per_Row": float(
                df["Simulated_Cost"].mean()
            ),
            "Rows": int(len(df)),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# 7. PLOT THE DECISION DISTRIBUTION
# ============================================================

def create_action_chart(summary):
    chart_path = OUTPUT_DIR / "step06_action_distribution.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        summary["Action"],
        summary["Number_of_Rows"],
    )

    ax.set_title("Step 6: Maintenance Agent Action Distribution")
    ax.set_xlabel("Maintenance action")
    ax.set_ylabel("Number of test records")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)

    for bar, count in zip(
        bars, summary["Number_of_Rows"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(count)),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return chart_path


# ============================================================
# 8. SAVE A REPRODUCIBLE REPORT
# ============================================================

def save_report(decisions, summary, policy_comparison):
    decisions_path = OUTPUT_DIR / "step06_agent_decisions.csv"
    summary_path = OUTPUT_DIR / "step06_action_summary.csv"
    comparison_path = OUTPUT_DIR / "step06_policy_cost_comparison.csv"
    report_path = OUTPUT_DIR / "step06_report.json"

    decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    policy_comparison.to_csv(comparison_path, index=False)

    report = {
        "step": 6,
        "title": "Rational Maintenance Agent",
        "input_file": str(INPUT_PATH),
        "evaluation_rows": int(len(decisions)),
        "actual_failures": int(decisions["Actual_Failure"].sum()),
        "actual_failure_rate": float(
            decisions["Actual_Failure"].mean()
        ),
        "policy_type": "Fixed transparent probability thresholds",
        "thresholds": {
            "Normal": f"p < {NORMAL_MAX}",
            "Monitor": f"{NORMAL_MAX} <= p < {MONITOR_MAX}",
            "Schedule Maintenance": (
                f"{MONITOR_MAX} <= p < {SCHEDULE_MAX}"
            ),
            "Critical": f"p >= {SCHEDULE_MAX}",
        },
        "simulated_cost_model": COST_MODEL,
        "total_simulated_cost": float(
            decisions["Simulated_Cost"].sum()
        ),
        "mean_simulated_cost_per_row": float(
            decisions["Simulated_Cost"].mean()
        ),
        "missed_failures_under_policy": int(
            decisions["Failure_Missed_By_Agent"].sum()
        ),
        "action_counts": {
            str(row["Action"]): int(row["Number_of_Rows"])
            for _, row in summary.iterrows()
        },
        "limitations": [
            "The maintenance costs are illustrative assumptions, not measured costs.",
            "AI4I does not contain actual maintenance actions or intervention outcomes.",
            "The agent uses fixed thresholds and does not learn from feedback in this step.",
            "This is a held-out test-set evaluation, not a prospective industrial trial.",
            "The reference-policy costs use the same assumed cost model.",
            "Thresholds have not been optimized using an independent validation set.",
        ],
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return (
        decisions_path,
        summary_path,
        comparison_path,
        report_path,
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():
    predictions = load_predictions()

    print("\n" + "-" * 72)
    print("INITIAL MAINTENANCE POLICY")
    print("-" * 72)
    print(f"Normal:              probability < {NORMAL_MAX:.2f}")
    print(f"Monitor:             probability < {MONITOR_MAX:.2f}")
    print(f"Schedule Maintenance: probability < {SCHEDULE_MAX:.2f}")
    print(f"Critical:            probability >= {SCHEDULE_MAX:.2f}")

    print("\nSimulated cost assumptions:")
    for action, costs in COST_MODEL.items():
        print(
            f"{action}: failure cost={costs['failure_cost']:.2f}, "
            f"no-failure cost={costs['no_failure_cost']:.2f}"
        )

    decisions = assign_actions(predictions)
    decisions = evaluate_agent(decisions)

    summary = summarize_actions(decisions)
    policy_comparison = compare_policies(decisions)

    print("\n" + "=" * 72)
    print("MAINTENANCE ACTION SUMMARY")
    print("=" * 72)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 72)
    print("SIMULATED COST COMPARISON")
    print("=" * 72)

    print(
        policy_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    total_cost = float(decisions["Simulated_Cost"].sum())
    mean_cost = float(decisions["Simulated_Cost"].mean())
    missed = int(decisions["Failure_Missed_By_Agent"].sum())

    print("\nAgent evaluation:")
    print(f"Total simulated cost: {total_cost:.2f}")
    print(f"Mean simulated cost per row: {mean_cost:.4f}")
    print(f"Failures assigned Normal action: {missed}")

    chart_path = create_action_chart(summary)

    (
        decisions_path,
        summary_path,
        comparison_path,
        report_path,
    ) = save_report(decisions, summary, policy_comparison)

    print("\nSaved outputs:")
    print(f"  Decisions:       {decisions_path}")
    print(f"  Action summary:  {summary_path}")
    print(f"  Cost comparison: {comparison_path}")
    print(f"  Action chart:    {chart_path}")
    print(f"  JSON report:     {report_path}")

    print("\n" + "=" * 72)
    print("STEP 6 COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "!" * 72)
        print("STEP 6 FAILED")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        print("!" * 72)
        raise
