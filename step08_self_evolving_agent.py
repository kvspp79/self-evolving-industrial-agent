
"""
STEP 8: FEEDBACK-DRIVEN SELF-EVOLVING MAINTENANCE AGENT

Project:
    Self-Evolving Explainable Industrial Agent

Dataset:
    AI4I 2020 Predictive Maintenance

Core contribution:
    An online policy selector updates its policy preferences using
    simulated outcome feedback from PREVIOUS batches.

Leakage safeguards:
    1. A batch policy is selected before its actual labels are evaluated.
    2. Feedback from a batch is used only for later batches.
    3. All policies are evaluated on the same batch sequence.
    4. The costs are explicitly simulated assumptions, not real costs.

IMPORTANT:
    AI4I rows are not a natural chronological machine history.
    Batch order is randomized to simulate sequential feedback.
    This experiment demonstrates a learning mechanism, not actual
    temporal degradation or validated industrial deployment.
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

INPUT_PATH = (
    OUTPUT_DIR / "step05_two_feature_test_predictions.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BATCH_SIZE = 100
EPSILON = 0.15

# Simulated costs copied from Step 6.
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

ACTION_ORDER = [
    "Normal",
    "Monitor",
    "Schedule Maintenance",
    "Critical",
]

# Candidate policies.
# Each policy has three ascending probability thresholds:
# p < t1: Normal
# t1 <= p < t2: Monitor
# t2 <= p < t3: Schedule Maintenance
# p >= t3: Critical
#
# These are candidate policies, not known optimal thresholds.

POLICIES = {
    "Conservative": (0.01, 0.04, 0.12),
    "Initial_Step6": (0.02, 0.08, 0.20),
    "Lower_Thresholds": (0.005, 0.03, 0.10),
    "Higher_Thresholds": (0.04, 0.12, 0.30),
    "Maintenance_Focused": (0.01, 0.06, 0.15),
}


# ============================================================
# 2. LOAD AND VALIDATE INPUT
# ============================================================

def load_data():
    print("=" * 76)
    print("STEP 8: FEEDBACK-DRIVEN SELF-EVOLVING AGENT")
    print("=" * 76)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input file:   {INPUT_PATH}")

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_PATH}\n"
            "Run Step 5 successfully before running Step 8."
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        "Actual_Failure",
        "All_Features_Probability",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(missing)
        )

    df = df.copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    original_rows = len(df)

    df = df.dropna(subset=required).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid rows remain after cleaning.")

    if len(df) != original_rows:
        print(
            f"Rows removed due to missing values: "
            f"{original_rows - len(df)}"
        )

    if not df["Actual_Failure"].isin([0, 1]).all():
        raise ValueError(
            "Actual_Failure must contain only 0 and 1."
        )

    if not df["All_Features_Probability"].between(0, 1).all():
        raise ValueError(
            "Failure probabilities must be between 0 and 1."
        )

    if df["Actual_Failure"].nunique() != 2:
        raise ValueError(
            "Input must contain both failures and non-failures."
        )

    print(f"Valid rows:      {len(df)}")
    print(
        f"Actual failures: "
        f"{int(df['Actual_Failure'].sum())}"
    )
    print(
        f"Failure rate:    "
        f"{df['Actual_Failure'].mean():.3%}"
    )

    return df


# ============================================================
# 3. ACTION AND COST FUNCTIONS
# ============================================================

def choose_action(probability, thresholds):
    t1, t2, t3 = thresholds

    if probability < t1:
        return "Normal"
    elif probability < t2:
        return "Monitor"
    elif probability < t3:
        return "Schedule Maintenance"
    else:
        return "Critical"


def calculate_cost(action, actual_failure):
    """
    Evaluate the chosen action after the outcome is known.
    This function does not choose or change the action.
    """
    costs = COST_MODEL[action]

    if int(actual_failure) == 1:
        return float(costs["failure_cost"])

    return float(costs["no_failure_cost"])


def apply_policy(batch, policy_name):
    thresholds = POLICIES[policy_name]

    actions = [
        choose_action(float(p), thresholds)
        for p in batch["All_Features_Probability"]
    ]

    return actions


def evaluate_actions(batch, actions):
    """
    Evaluation only: actual labels are read after actions are fixed.
    """
    actual = batch["Actual_Failure"].to_numpy(dtype=int)

    costs = np.array([
        calculate_cost(action, y)
        for action, y in zip(actions, actual)
    ], dtype=float)

    return costs


# ============================================================
# 4. ONLINE POLICY SELECTOR
# ============================================================

class OnlinePolicyLearner:
    """
    Epsilon-greedy policy selector.

    It estimates each policy's reward from previous batches only.
    Reward is negative mean simulated cost, so higher is better.

    Exploration:
        With probability epsilon, try a random candidate policy.

    Exploitation:
        Otherwise, choose the policy with the best historical
        average reward.

    The learner receives feedback only AFTER a batch is evaluated.
    """

    def __init__(self, policy_names, epsilon=0.15, seed=42):
        self.policy_names = list(policy_names)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

        self.counts = {
            name: 0 for name in self.policy_names
        }

        self.mean_rewards = {
            name: 0.0 for name in self.policy_names
        }

        self.feedback_history = []

    def select_policy(self):
        # Ensure each candidate is tried before relying on estimates.
        untried = [
            name for name in self.policy_names
            if self.counts[name] == 0
        ]

        if untried:
            return str(self.rng.choice(untried))

        # Epsilon-greedy exploration.
        if self.rng.random() < self.epsilon:
            return str(self.rng.choice(self.policy_names))

        best_reward = max(self.mean_rewards.values())

        best_policies = [
            name for name in self.policy_names
            if np.isclose(
                self.mean_rewards[name], best_reward
            )
        ]

        return str(self.rng.choice(best_policies))

    def update(self, policy_name, mean_cost, batch_number):
        """
        Update after the selected policy's batch has been evaluated.
        Lower cost means higher reward.
        """
        reward = -float(mean_cost)

        old_count = self.counts[policy_name]
        new_count = old_count + 1

        old_mean = self.mean_rewards[policy_name]

        # Incremental sample mean; no need to store all outcomes.
        new_mean = old_mean + (reward - old_mean) / new_count

        self.counts[policy_name] = new_count
        self.mean_rewards[policy_name] = float(new_mean)

        self.feedback_history.append({
            "Batch": int(batch_number),
            "Policy_Updated": policy_name,
            "Batch_Mean_Cost": float(mean_cost),
            "Reward": float(reward),
            "Policy_Historical_Mean_Reward": float(new_mean),
            "Policy_Historical_Mean_Cost": float(-new_mean),
            "Policy_Selection_Count": int(new_count),
        })


# ============================================================
# 5. FIXED POLICY BASELINES
# ============================================================

def evaluate_fixed_policies(df):
    rows = []

    for policy_name in POLICIES:
        all_actions = apply_policy(df, policy_name)
        costs = evaluate_actions(df, all_actions)

        rows.append({
            "Policy": policy_name,
            "Rows": int(len(df)),
            "Total_Simulated_Cost": float(costs.sum()),
            "Mean_Simulated_Cost_Per_Row": float(costs.mean()),
        })

    # Always-action baselines.
    for action in ACTION_ORDER:
        costs = np.array([
            calculate_cost(action, y)
            for y in df["Actual_Failure"].to_numpy(dtype=int)
        ], dtype=float)

        rows.append({
            "Policy": f"Always_{action.replace(' ', '_')}",
            "Rows": int(len(df)),
            "Total_Simulated_Cost": float(costs.sum()),
            "Mean_Simulated_Cost_Per_Row": float(costs.mean()),
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. RUN SIMULATED ONLINE LEARNING
# ============================================================

def run_online_learning(df):
    print("\n" + "=" * 76)
    print("ONLINE FEEDBACK LEARNING")
    print("=" * 76)

    # AI4I does not provide natural chronological sequences.
    # Shuffle deterministically to create a reproducible simulation.
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(df))

    stream = df.iloc[order].reset_index(drop=True).copy()
    stream["Simulation_Row_ID"] = np.arange(len(stream))

    learner = OnlinePolicyLearner(
        policy_names=list(POLICIES.keys()),
        epsilon=EPSILON,
        seed=RANDOM_SEED + 1,
    )

    decision_records = []
    batch_records = []

    number_of_batches = int(
        np.ceil(len(stream) / BATCH_SIZE)
    )

    for batch_index, start in enumerate(
        range(0, len(stream), BATCH_SIZE), start=1
    ):
        end = min(start + BATCH_SIZE, len(stream))
        batch = stream.iloc[start:end].copy()

        # IMPORTANT:
        # Select the policy BEFORE accessing this batch's labels
        # for evaluation. The learner has only previous feedback.
        selected_policy = learner.select_policy()

        actions = apply_policy(batch, selected_policy)

        # Only now evaluate against actual outcomes.
        costs = evaluate_actions(batch, actions)

        mean_cost = float(costs.mean())
        total_cost = float(costs.sum())

        # Store decisions with outcomes for later audit.
        for row_index, (_, row) in enumerate(batch.iterrows()):
            decision_records.append({
                "Simulation_Row_ID": int(
                    row["Simulation_Row_ID"]
                ),
                "Batch": int(batch_index),
                "Policy_Selected_Before_Feedback": selected_policy,
                "Failure_Probability": float(
                    row["All_Features_Probability"]
                ),
                "Agent_Action": actions[row_index],
                "Actual_Failure": int(row["Actual_Failure"]),
                "Simulated_Cost": float(costs[row_index]),
                "Threshold_Normal_Max": POLICIES[selected_policy][0],
                "Threshold_Monitor_Max": POLICIES[selected_policy][1],
                "Threshold_Schedule_Max": POLICIES[selected_policy][2],
            })

        batch_records.append({
            "Batch": int(batch_index),
            "Batch_Size": int(len(batch)),
            "Selected_Policy": selected_policy,
            "Batch_Failure_Rate": float(
                batch["Actual_Failure"].mean()
            ),
            "Mean_Predicted_Probability": float(
                batch["All_Features_Probability"].mean()
            ),
            "Total_Simulated_Cost": total_cost,
            "Mean_Simulated_Cost": mean_cost,
            "Policy_Count_Before_Update": int(
                learner.counts[selected_policy]
            ),
        })

        # Feedback from this batch can affect ONLY later batches.
        learner.update(
            policy_name=selected_policy,
            mean_cost=mean_cost,
            batch_number=batch_index,
        )

        if (
            batch_index == 1
            or batch_index % 10 == 0
            or batch_index == number_of_batches
        ):
            print(
                f"Batch {batch_index:>3}/{number_of_batches}: "
                f"policy={selected_policy:<22} "
                f"mean_cost={mean_cost:>8.2f}"
            )

    decisions_df = pd.DataFrame(decision_records)
    batches_df = pd.DataFrame(batch_records)
    feedback_df = pd.DataFrame(learner.feedback_history)

    return decisions_df, batches_df, feedback_df, learner


# ============================================================
# 7. SUMMARY AND FAIR COST COMPARISON
# ============================================================

def build_summary(decisions, learner):
    total_cost = float(decisions["Simulated_Cost"].sum())
    mean_cost = float(decisions["Simulated_Cost"].mean())

    policy_counts = decisions[
        "Policy_Selected_Before_Feedback"
    ].value_counts().to_dict()

    return {
        "total_rows": int(len(decisions)),
        "total_batches": int(decisions["Batch"].nunique()),
        "total_simulated_cost": total_cost,
        "mean_simulated_cost_per_row": mean_cost,
        "actual_failures": int(decisions["Actual_Failure"].sum()),
        "actual_failure_rate": float(
            decisions["Actual_Failure"].mean()
        ),
        "epsilon": float(EPSILON),
        "batch_size": int(BATCH_SIZE),
        "random_seed": int(RANDOM_SEED),
        "candidate_policies": POLICIES,
        "policy_selection_counts": {
            str(k): int(v) for k, v in policy_counts.items()
        },
        "historical_mean_cost_by_policy": {
            name: (
                float(-learner.mean_rewards[name])
                if learner.counts[name] > 0 else None
            )
            for name in learner.policy_names
        },
        "feedback_update_counts": {
            name: int(learner.counts[name])
            for name in learner.policy_names
        },
        "cost_model": COST_MODEL,
        "limitations": [
            "All cost values are illustrative simulated assumptions.",
            "AI4I does not contain actual maintenance intervention feedback.",
            "AI4I rows are not chronological machine histories.",
            "The randomized batch stream is a simulation of sequential feedback.",
            "The policy selector learns among predefined candidate policies.",
            "This is not an unconstrained policy generator or proof of real-world deployment.",
            "The evaluation uses a held-out test prediction file; further independent "
            "evaluation is needed before making generalization claims.",
        ],
    }


def build_comparison(online_decisions, fixed_baselines):
    rows = fixed_baselines.to_dict(orient="records")

    rows.append({
        "Policy": "Feedback_Learning_Agent",
        "Rows": int(len(online_decisions)),
        "Total_Simulated_Cost": float(
            online_decisions["Simulated_Cost"].sum()
        ),
        "Mean_Simulated_Cost_Per_Row": float(
            online_decisions["Simulated_Cost"].mean()
        ),
    })

    return pd.DataFrame(rows).sort_values(
        "Mean_Simulated_Cost_Per_Row"
    ).reset_index(drop=True)


# ============================================================
# 8. VISUALIZATIONS
# ============================================================

def save_cost_chart(comparison):
    path = OUTPUT_DIR / "step08_policy_cost_comparison.png"

    plot_df = comparison.sort_values(
        "Mean_Simulated_Cost_Per_Row",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.barh(
        plot_df["Policy"],
        plot_df["Mean_Simulated_Cost_Per_Row"],
    )

    ax.set_title(
        "Step 8: Mean Simulated Cost — Online Agent vs Baselines"
    )
    ax.set_xlabel("Mean simulated cost per row")
    ax.set_ylabel("Policy")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return path


def save_learning_curve(batches):
    path = OUTPUT_DIR / "step08_online_learning_curve.png"

    batches = batches.copy()

    # Running average of realized batch costs.
    batches["Running_Mean_Cost"] = (
        batches["Mean_Simulated_Cost"].expanding().mean()
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        batches["Batch"],
        batches["Mean_Simulated_Cost"],
        marker="o",
        markersize=3,
        alpha=0.65,
        label="Batch mean cost",
    )

    ax.plot(
        batches["Batch"],
        batches["Running_Mean_Cost"],
        linewidth=2,
        label="Running mean cost",
    )

    ax.set_title("Step 8: Online Agent Feedback History")
    ax.set_xlabel("Simulated batch")
    ax.set_ylabel("Simulated cost per row")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return path


def save_policy_selection_chart(decisions):
    path = OUTPUT_DIR / "step08_policy_selection_counts.png"

    counts = (
        decisions["Policy_Selected_Before_Feedback"]
        .value_counts()
        .reindex(list(POLICIES.keys()), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(counts.index, counts.values)

    ax.set_title("Step 8: Online Policy Selection Counts")
    ax.set_xlabel("Candidate policy")
    ax.set_ylabel("Number of batches selected")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(value)),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return path


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

def save_outputs(
    decisions,
    batches,
    feedback,
    comparison,
    summary,
):
    paths = {
        "decisions": OUTPUT_DIR / "step08_online_agent_decisions.csv",
        "batches": OUTPUT_DIR / "step08_batch_feedback.csv",
        "feedback": OUTPUT_DIR / "step08_policy_learning_history.csv",
        "comparison": OUTPUT_DIR / "step08_policy_cost_comparison.csv",
        "report": OUTPUT_DIR / "step08_report.json",
    }

    decisions.to_csv(paths["decisions"], index=False)
    batches.to_csv(paths["batches"], index=False)
    feedback.to_csv(paths["feedback"], index=False)
    comparison.to_csv(paths["comparison"], index=False)

    with paths["report"].open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return paths


# ============================================================
# 10. MAIN
# ============================================================

def main():
    data = load_data()

    # Fixed-policy baseline evaluation uses the same full set of rows.
    fixed_baselines = evaluate_fixed_policies(data)

    (
        online_decisions,
        batches,
        feedback,
        learner,
    ) = run_online_learning(data)

    summary = build_summary(online_decisions, learner)

    comparison = build_comparison(
        online_decisions,
        fixed_baselines,
    )

    cost_chart = save_cost_chart(comparison)
    learning_chart = save_learning_curve(batches)
    selection_chart = save_policy_selection_chart(
        online_decisions
    )

    paths = save_outputs(
        online_decisions,
        batches,
        feedback,
        comparison,
        summary,
    )

    print("\n" + "=" * 76)
    print("ONLINE AGENT SUMMARY")
    print("=" * 76)
    print(f"Rows evaluated: {summary['total_rows']}")
    print(f"Batches: {summary['total_batches']}")
    print(
        "Online agent total simulated cost: "
        f"{summary['total_simulated_cost']:.2f}"
    )
    print(
        "Online agent mean simulated cost per row: "
        f"{summary['mean_simulated_cost_per_row']:.4f}"
    )

    print("\nPolicy selection counts:")
    for policy, count in summary[
        "policy_selection_counts"
    ].items():
        print(f"  {policy}: {count}")

    print("\nHistorical mean cost by selected policy:")
    for policy, mean_cost in summary[
        "historical_mean_cost_by_policy"
    ].items():
        if mean_cost is None:
            print(f"  {policy}: no feedback yet")
        else:
            print(f"  {policy}: {mean_cost:.4f}")

    print("\n" + "=" * 76)
    print("POLICY COST COMPARISON")
    print("=" * 76)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print(f"  cost chart:      {cost_chart}")
    print(f"  learning curve:  {learning_chart}")
    print(f"  selection chart: {selection_chart}")

    print("\nResearch interpretation:")
    print(
        "The learner updates policy preferences from prior-batch "
        "simulated cost feedback. A lower observed cost does not, by "
        "itself, prove statistically significant improvement."
    )
    print(
        "Because candidate policies are predefined and AI4I is not "
        "chronological, describe this as a simulated online adaptive "
        "policy-selection experiment."
    )

    print("\n" + "=" * 76)
    print("STEP 8 COMPLETED SUCCESSFULLY")
    print("=" * 76)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "!" * 76)
        print("STEP 8 FAILED")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        print("!" * 76)
        raise
