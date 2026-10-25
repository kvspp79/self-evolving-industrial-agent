
"""
NASA C-MAPSS FD001 — STEP 7.2
ROBUST POLICY VALIDATION

Research goals
--------------
1. Five-fold engine-level cross-validation for maintenance-policy selection.
2. Compare baseline and temporal RUL predictions.
3. Evaluate threshold stability across folds.
4. Evaluate missed critical/high-risk engines, maintenance actions,
   false alarms, and illustrative simulated costs.
5. Select final thresholds using all available validation engines.
6. Evaluate the frozen policies once on the held-out test set.

Important:
- No threshold is selected using test labels.
- Numeric engine IDs are not compared between validation and test.
- Each engine contributes only its final observed prediction row.
- This evaluates policy selection; it does not retrain the RUL models.
- Simulated costs are assumptions, not measured industrial costs.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "outputs" / "step04"
OUTPUT_DIR = ROOT / "outputs" / "step07_2"

SUBSET = "FD001"

VALIDATION_FILE = (
    INPUT_DIR / f"validation_predictions_{SUBSET}.csv"
)
TEST_FILE = INPUT_DIR / f"test_predictions_{SUBSET}.csv"

N_SPLITS = 5
RANDOM_STATE = 42

CRITICAL_RUL = 15.0
HIGH_RUL = 30.0

# Safety constraints applied on policy-training engines.
MAX_MISSED_CRITICAL = 0
MAX_HIGH_RISK_MISS_RATE = 0.10

# Candidate thresholds, in predicted RUL cycles.
THRESHOLDS = np.arange(0.0, 201.0, 5.0)

MODELS = {
    "baseline": "baseline_predicted_RUL",
    "temporal": "temporal_predicted_RUL",
}

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

REQUIRED_COLUMNS = [
    "unit_id",
    "actual_RUL",
    "baseline_predicted_RUL",
    "temporal_predicted_RUL",
]


# ============================================================
# DATA VALIDATION
# ============================================================

def load_predictions(path, split_name):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {split_name} prediction file:\n{path}\n\n"
            "Check that Step 4 completed and its CSV files exist."
        )

    df = pd.read_csv(path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{split_name} CSV is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
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
            f"{split_name} contains NaN or infinite values."
        )

    if "cycle" in df.columns:
        df["cycle"] = pd.to_numeric(df["cycle"], errors="raise")

    return df


def final_row_per_engine(df):
    """
    Use one final observed row per engine.

    For validation trajectories, this selects the last observed
    cycle of each validation engine.
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
            "Final-row selection did not produce unique engine IDs."
        )

    return result


# ============================================================
# POLICY METRICS
# ============================================================

def evaluate_policy(
    df,
    model_name,
    threshold,
    scenario_name,
):
    pred_col = MODELS[model_name]

    actual = df["actual_RUL"].to_numpy(dtype=float)
    predicted = df[pred_col].to_numpy(dtype=float)

    # Policy rule: schedule maintenance when predicted RUL is
    # at or below the selected threshold.
    action = predicted <= float(threshold)

    critical = actual <= CRITICAL_RUL
    high_risk = actual <= HIGH_RUL

    missed_critical = critical & ~action
    missed_high_risk = high_risk & ~action

    # False alarm: maintenance on an engine whose actual RUL
    # exceeds the predefined high-risk boundary.
    false_alarm = action & (actual > HIGH_RUL)

    n = len(df)
    critical_count = int(critical.sum())
    high_risk_count = int(high_risk.sum())

    missed_critical_count = int(missed_critical.sum())
    missed_high_count = int(missed_high_risk.sum())

    high_risk_miss_rate = (
        missed_high_count / high_risk_count
        if high_risk_count > 0
        else 0.0
    )

    actions = int(action.sum())
    false_alarms = int(false_alarm.sum())

    costs = COST_SCENARIOS[scenario_name]

    maintenance_cost = actions * costs["maintenance"]
    missed_cost = (
        missed_critical_count * costs["missed_critical"]
    )
    false_alarm_cost = false_alarms * costs["false_alarm"]

    total_cost = (
        maintenance_cost + missed_cost + false_alarm_cost
    )

    feasible = (
        missed_critical_count <= MAX_MISSED_CRITICAL
        and high_risk_miss_rate <= MAX_HIGH_RISK_MISS_RATE
    )

    return {
        "model": model_name,
        "cost_scenario": scenario_name,
        "threshold": float(threshold),
        "n_engines": int(n),
        "actual_critical_engines": critical_count,
        "actual_high_risk_engines": high_risk_count,
        "maintenance_actions": actions,
        "maintenance_action_rate": (
            float(actions / n) if n else 0.0
        ),
        "missed_critical_engines": missed_critical_count,
        "missed_high_risk_engines": missed_high_count,
        "high_risk_miss_rate": float(high_risk_miss_rate),
        "false_alarm_actions": false_alarms,
        "maintenance_cost": float(maintenance_cost),
        "missed_critical_cost": float(missed_cost),
        "false_alarm_cost": float(false_alarm_cost),
        "total_simulated_cost": float(total_cost),
        "cost_per_engine": float(total_cost / n) if n else 0.0,
        "safety_feasible": bool(feasible),
    }


# ============================================================
# SELECT A THRESHOLD USING ONLY POLICY-TRAINING ENGINES
# ============================================================

def select_threshold(policy_train, model_name, scenario_name):
    candidates = []

    for threshold in THRESHOLDS:
        result = evaluate_policy(
            policy_train,
            model_name,
            threshold,
            scenario_name,
        )
        candidates.append(result)

    table = pd.DataFrame(candidates)

    feasible = table[table["safety_feasible"]].copy()

    if not feasible.empty:
        # Choose the lowest-cost feasible threshold.
        selected = feasible.sort_values(
            [
                "total_simulated_cost",
                "missed_critical_engines",
                "missed_high_risk_engines",
                "maintenance_actions",
                "threshold",
            ],
            ascending=True,
            kind="stable",
        ).iloc[0].to_dict()

        status = "feasible_lowest_cost"

    else:
        # Explicit fallback if no candidate meets the safety
        # constraints. Do not label this policy as safe.
        selected = table.sort_values(
            [
                "missed_critical_engines",
                "missed_high_risk_engines",
                "total_simulated_cost",
                "threshold",
            ],
            ascending=True,
            kind="stable",
        ).iloc[0].to_dict()

        status = "no_feasible_threshold"

    selected["selection_status"] = status

    return selected, table


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(validation):
    """
    Cross-validate the decision policy using the final prediction
    row from each validation engine.

    Each fold selects a threshold on policy-training engines and
    evaluates the frozen threshold on held-out engines.

    The existing RUL predictions are not retrained in this step.
    """

    n_engines = len(validation)

    if n_engines < 5:
        raise ValueError(
            "At least five validation engines are required "
            "for five-fold cross-validation."
        )

    if validation["unit_id"].nunique() != n_engines:
        raise ValueError(
            "Cross-validation input must contain one row per engine."
        )

    splitter = KFold(
        n_splits=min(N_SPLITS, n_engines),
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []
    fold_thresholds = []
    threshold_search_rows = []

    for fold_number, (train_idx, heldout_idx) in enumerate(
        splitter.split(validation),
        start=1,
    ):
        policy_train = validation.iloc[train_idx].copy()
        heldout = validation.iloc[heldout_idx].copy()

        train_ids = set(policy_train["unit_id"].tolist())
        heldout_ids = set(heldout["unit_id"].tolist())

        if train_ids.intersection(heldout_ids):
            raise RuntimeError(
                f"Engine leakage detected in fold {fold_number}."
            )

        print(
            f"\nFold {fold_number}/{min(N_SPLITS, n_engines)}"
            f" | policy-training engines={len(policy_train)}"
            f" | held-out engines={len(heldout)}"
        )

        for model_name in MODELS:
            for scenario_name in COST_SCENARIOS:
                selected, search_table = select_threshold(
                    policy_train,
                    model_name,
                    scenario_name,
                )

                search_table["fold"] = fold_number
                search_table["policy_train_engines"] = len(policy_train)
                threshold_search_rows.append(search_table)

                heldout_metrics = evaluate_policy(
                    heldout,
                    model_name,
                    selected["threshold"],
                    scenario_name,
                )

                heldout_metrics.update({
                    "fold": fold_number,
                    "policy_train_engines": len(policy_train),
                    "heldout_engines": len(heldout),
                    "selected_threshold": float(
                        selected["threshold"]
                    ),
                    "selection_status": selected[
                        "selection_status"
                    ],
                    "training_policy_safety_feasible": bool(
                        selected["safety_feasible"]
                    ),
                })

                fold_results.append(heldout_metrics)

                fold_thresholds.append({
                    "fold": fold_number,
                    "model": model_name,
                    "cost_scenario": scenario_name,
                    "selected_threshold": float(
                        selected["threshold"]
                    ),
                    "selection_status": selected[
                        "selection_status"
                    ],
                    "training_policy_safety_feasible": bool(
                        selected["safety_feasible"]
                    ),
                    "policy_train_engines": len(policy_train),
                    "heldout_engines": len(heldout),
                })

    return (
        pd.DataFrame(fold_results),
        pd.DataFrame(fold_thresholds),
        pd.concat(threshold_search_rows, ignore_index=True),
    )


# ============================================================
# SUMMARY AND FINAL VALIDATION-SELECTED TEST EVALUATION
# ============================================================

def summarize_folds(fold_results):
    numeric_columns = [
        "selected_threshold",
        "maintenance_actions",
        "missed_critical_engines",
        "missed_high_risk_engines",
        "high_risk_miss_rate",
        "false_alarm_actions",
        "total_simulated_cost",
        "cost_per_engine",
    ]

    summary = (
        fold_results
        .groupby(["model", "cost_scenario"], as_index=False)[
            numeric_columns
        ]
        .agg(["mean", "std"])
    )

    summary.columns = [
        "_".join(str(part) for part in col if part != "")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    return summary


def evaluate_final_frozen_policies(validation, test):
    """
    Select one threshold per model/scenario using all validation
    engines, then evaluate it on test exactly once.

    Test outcomes do not alter thresholds.
    """

    selected_rows = []
    test_rows = []

    for model_name in MODELS:
        for scenario_name in COST_SCENARIOS:
            selected, _ = select_threshold(
                validation,
                model_name,
                scenario_name,
            )

            selected_record = dict(selected)
            selected_record["model"] = model_name
            selected_record["cost_scenario"] = scenario_name
            selected_rows.append(selected_record)

            test_metrics = evaluate_policy(
                test,
                model_name,
                selected["threshold"],
                scenario_name,
            )

            test_metrics["selection_status"] = selected[
                "selection_status"
            ]
            test_metrics["validation_safety_feasible"] = bool(
                selected["safety_feasible"]
            )

            test_rows.append(test_metrics)

    return pd.DataFrame(selected_rows), pd.DataFrame(test_rows)


# ============================================================
# FIGURES
# ============================================================

def plot_threshold_stability(thresholds, output_path):
    combinations = (
        thresholds[["model", "cost_scenario"]]
        .drop_duplicates()
        .to_dict("records")
    )

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 9),
        squeeze=False,
    )

    for model_name in MODELS:
        subset = thresholds[thresholds["model"] == model_name]

        labels = []
        data = []

        for scenario_name in COST_SCENARIOS:
            values = subset.loc[
                subset["cost_scenario"] == scenario_name,
                "selected_threshold",
            ].to_numpy(dtype=float)

            if len(values):
                labels.append(scenario_name)
                data.append(values)

        if data:
            ax = axes[0 if model_name == "baseline" else 1, 0]
            ax.boxplot(data, tick_labels=labels)
            ax.set_title(
                f"{model_name.capitalize()} threshold stability"
            )
            ax.set_ylabel("Selected threshold (cycles)")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cv_costs(fold_results, output_path):
    data = fold_results.copy()
    data["label"] = (
        data["model"] + "\n" + data["cost_scenario"]
    )

    labels = data["label"].drop_duplicates().tolist()
    values = [
        data.loc[
            data["label"] == label,
            "cost_per_engine",
        ].to_numpy(dtype=float)
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(values, tick_labels=labels)
    ax.set_title("Cross-validated held-out cost per engine")
    ax.set_ylabel("Illustrative simulated cost per engine")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_missed_critical(fold_results, output_path):
    data = fold_results.copy()
    data["label"] = (
        data["model"] + "\n" + data["cost_scenario"]
    )

    labels = data["label"].drop_duplicates().tolist()
    values = [
        data.loc[
            data["label"] == label,
            "missed_critical_engines",
        ].to_numpy(dtype=float)
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(values, tick_labels=labels)
    ax.set_title("Cross-validated missed critical engines")
    ax.set_ylabel("Missed critical engines per held-out fold")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# JSON-SAFE CONVERSION
# ============================================================

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if pd.isna(value):
        return None

    return value


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("NASA C-MAPSS — STEP 7.2")
    print("ROBUST POLICY VALIDATION")
    print(f"Subset: {SUBSET}")
    print("=" * 72)

    validation_all = load_predictions(
        VALIDATION_FILE,
        "validation",
    )
    test_all = load_predictions(TEST_FILE, "test")

    validation = final_row_per_engine(validation_all)
    test = final_row_per_engine(test_all)

    print("\nDATA SUMMARY")
    print("-" * 50)
    print("Validation source rows:", len(validation_all))
    print("Validation final engines:", len(validation))
    print("Test source rows:", len(test_all))
    print("Test final engines:", len(test))

    if len(validation) < 30:
        warnings.warn(
            "The number of validation engines is small. "
            "Fold-level safety and cost estimates may be unstable."
        )

    print("\nRunning engine-level policy cross-validation...")

    (
        fold_results,
        fold_thresholds,
        threshold_search,
    ) = run_cross_validation(validation)

    fold_results.to_csv(
        OUTPUT_DIR / f"cv_fold_results_{SUBSET}.csv",
        index=False,
    )

    fold_thresholds.to_csv(
        OUTPUT_DIR / f"cv_selected_thresholds_{SUBSET}.csv",
        index=False,
    )

    threshold_search.to_csv(
        OUTPUT_DIR / f"cv_threshold_search_{SUBSET}.csv",
        index=False,
    )

    summary = summarize_folds(fold_results)

    summary.to_csv(
        OUTPUT_DIR / f"cv_summary_{SUBSET}.csv",
        index=False,
    )

    print("\nCROSS-VALIDATION SUMMARY")
    print("-" * 72)
    print(summary.to_string(index=False))

    print("\nSelecting final policies using all validation engines...")
    print("Evaluating those frozen policies on test engines...")

    selected_final, test_results = evaluate_final_frozen_policies(
        validation,
        test,
    )

    selected_final.to_csv(
        OUTPUT_DIR / f"final_validation_selected_policies_{SUBSET}.csv",
        index=False,
    )

    test_results.to_csv(
        OUTPUT_DIR / f"final_frozen_test_results_{SUBSET}.csv",
        index=False,
    )

    plot_threshold_stability(
        fold_thresholds,
        OUTPUT_DIR / f"threshold_stability_{SUBSET}.png",
    )

    plot_cv_costs(
        fold_results,
        OUTPUT_DIR / f"cv_cost_per_engine_{SUBSET}.png",
    )

    plot_missed_critical(
        fold_results,
        OUTPUT_DIR / f"cv_missed_critical_{SUBSET}.png",
    )

    # Count how often each selected threshold occurred across folds.
    threshold_frequency = (
        fold_thresholds
        .groupby(
            ["model", "cost_scenario", "selected_threshold"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "fold_count"})
    )

    threshold_frequency.to_csv(
        OUTPUT_DIR / f"threshold_frequency_{SUBSET}.csv",
        index=False,
    )

    # Fold-level feasibility is based on the policy-training engines.
    training_feasibility = (
        fold_thresholds
        .groupby(["model", "cost_scenario"], as_index=False)
        .agg(
            folds=("fold", "count"),
            feasible_training_folds=(
                "training_policy_safety_feasible",
                "sum",
            ),
        )
    )

    training_feasibility["fraction_feasible_training_folds"] = (
        training_feasibility["feasible_training_folds"]
        / training_feasibility["folds"]
    )

    training_feasibility.to_csv(
        OUTPUT_DIR / f"cv_training_safety_feasibility_{SUBSET}.csv",
        index=False,
    )

    report = {
        "dataset": "NASA C-MAPSS",
        "subset": SUBSET,
        "step": "7.2",
        "method": (
            "Five-fold shuffled engine-level cross-validation "
            "of threshold selection using existing validation "
            "predictions. Thresholds are selected within each "
            "policy-training fold and evaluated on its held-out "
            "engines. Final thresholds are selected using all "
            "validation engines and then evaluated on test engines."
        ),
        "inputs": {
            "validation_file": str(VALIDATION_FILE),
            "test_file": str(TEST_FILE),
        },
        "data_counts": {
            "validation_source_rows": int(len(validation_all)),
            "validation_engines": int(len(validation)),
            "test_source_rows": int(len(test_all)),
            "test_engines": int(len(test)),
        },
        "cross_validation": {
            "folds": int(min(N_SPLITS, len(validation))),
            "random_state": RANDOM_STATE,
            "split_unit": "engine-level final observed row",
            "prediction_models_retrained": False,
            "safety_feasibility_by_training_fold": (
                training_feasibility.to_dict(orient="records")
            ),
            "fold_results": fold_results.to_dict(orient="records"),
            "summary": summary.to_dict(orient="records"),
            "threshold_frequency": (
                threshold_frequency.to_dict(orient="records")
            ),
        },
        "policy_configuration": {
            "critical_rul": CRITICAL_RUL,
            "high_risk_rul": HIGH_RUL,
            "max_missed_critical_engines": MAX_MISSED_CRITICAL,
            "max_high_risk_miss_rate": MAX_HIGH_RISK_MISS_RATE,
            "candidate_thresholds": THRESHOLDS.tolist(),
            "cost_scenarios": COST_SCENARIOS,
        },
        "final_validation_selected_policies": (
            selected_final.to_dict(orient="records")
        ),
        "final_frozen_test_results": (
            test_results.to_dict(orient="records")
        ),
        "limitations": [
            (
                "Only 20 validation engines are available in the "
                "current FD001 prediction file, so five-fold estimates "
                "will have high uncertainty."
            ),
            (
                "This cross-validation assesses threshold-selection "
                "stability using existing predictions. It does not "
                "retrain the RUL prediction models in each fold."
            ),
            (
                "The final test set is not used for threshold selection."
            ),
            (
                "Cost values and safety boundaries are illustrative "
                "research assumptions, not validated industrial costs."
            ),
            (
                "A feasible policy on a small validation fold does "
                "not establish real-world safety."
            ),
            (
                "The policy is a threshold rule, not reinforcement "
                "learning or proof of self-evolving behavior."
            ),
        ],
    }

    report_path = OUTPUT_DIR / f"step07_2_{SUBSET}_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(report),
            file,
            indent=4,
            allow_nan=False,
        )

    print("\nFINAL FROZEN TEST RESULTS")
    print("=" * 72)

    print(
        test_results[
            [
                "model",
                "cost_scenario",
                "threshold",
                "maintenance_actions",
                "missed_critical_engines",
                "missed_high_risk_engines",
                "false_alarm_actions",
                "total_simulated_cost",
                "selection_status",
                "validation_safety_feasible",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("STEP 7.2 COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print("Output directory:", OUTPUT_DIR)
    print("Research report:", report_path)
    print("Test labels were not used to select thresholds.")
    print("Steps 1–7.1 were not modified.")


if __name__ == "__main__":
    main()
