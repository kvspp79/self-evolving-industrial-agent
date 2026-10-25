
"""
NASA C-MAPSS FD001 — STEP 8
SELF-EVOLVING ADAPTIVE INDUSTRIAL DECISION AGENT

Research safeguards
-------------------
1. Uses validation engines to select the model and maintenance threshold.
2. Freezes the selected policy before test evaluation.
3. Never uses test labels to select or update the policy.
4. Compares baseline, temporal, and adaptive policies under identical costs.
5. Logs policy-selection rationale, safety metrics, and simulated costs.
6. Writes all artifacts to outputs/step08; earlier steps are not modified.

Important:
This is validation-driven adaptive policy selection with an auditable
policy-update record. It is not evidence of unrestricted online learning.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
SUBSET = "FD001"

INPUT_DIR = PROJECT_DIR / "outputs" / "step04"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "step08"

VALIDATION_FILE = INPUT_DIR / f"validation_predictions_{SUBSET}.csv"
TEST_FILE = INPUT_DIR / f"test_predictions_{SUBSET}.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# These are illustrative research assumptions, not real industrial costs.
CRITICAL_RUL = 15.0
HIGH_RISK_RUL = 30.0

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

# Safety constraints are evaluated on validation engines only.
MAX_MISSED_CRITICAL_ENGINES = 0
MAX_HIGH_RISK_MISS_RATE = 0.10

# A threshold is the predicted RUL at or below which maintenance is triggered.
THRESHOLDS = np.arange(0.0, 201.0, 5.0)


# ============================================================
# UTILITIES
# ============================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def save_json(data: dict, path: Path) -> None:
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialize {type(obj).__name__}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=convert, allow_nan=False)


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
    required: bool = True,
) -> str | None:
    """Find a column case-insensitively, first by exact name then aliases."""
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.lower()
        if key in lower_to_original:
            return lower_to_original[key]

    if required:
        raise ValueError(
            f"Could not identify the {description} column.\n"
            f"Tried: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def find_prediction_column(
    df: pd.DataFrame,
    model_name: str,
) -> str:
    """
    Identify the requested prediction column.
    Avoids accidentally treating actual_RUL as a prediction.
    """
    cols = list(df.columns)
    normalized = {
        str(c).strip().lower().replace("-", "_").replace(" ", "_"): c
        for c in cols
    }

    aliases = {
        "baseline": [
            "baseline_prediction",
            "baseline_pred",
            "baseline_predicted_rul",
            "baseline_rul_prediction",
            "baseline_rul_pred",
            "predicted_rul_baseline",
            "y_pred_baseline",
            "baseline",
        ],
        "temporal": [
            "temporal_prediction",
            "temporal_pred",
            "temporal_predicted_rul",
            "temporal_rul_prediction",
            "temporal_rul_pred",
            "predicted_rul_temporal",
            "y_pred_temporal",
            "temporal",
        ],
    }

    for name in aliases[model_name]:
        if name in normalized:
            return normalized[name]

    # Fallback: require both the model name and a prediction-like token.
    # Exclude actual/true/target columns.
    matches = []
    for col in cols:
        name = str(col).strip().lower().replace("-", "_").replace(" ", "_")
        if model_name not in name:
            continue
        if any(token in name for token in ("actual", "true", "target", "label")):
            continue
        if any(token in name for token in ("pred", "forecast", "estimate")):
            matches.append(col)

    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"Cannot safely identify the {model_name} prediction column.\n"
        f"Available columns: {cols}\n"
        "Edit find_prediction_column() to match the actual column name."
    )


def load_prediction_file(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Confirm that Step 4 completed and its prediction CSV files exist."
        )

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"{split_name} prediction file is empty: {path}")

    unit_col = find_column(
        raw,
        ["unit_id", "unit", "engine_id", "engine", "id"],
        "engine ID",
    )
    cycle_col = find_column(
        raw,
        ["cycle", "time_in_cycles", "time_cycle", "t"],
        "cycle",
        required=False,
    )
    actual_col = find_column(
        raw,
        ["actual_RUL", "actual_rul", "RUL", "rul", "true_RUL",
         "true_rul", "y_true", "target"],
        "actual RUL",
    )
    baseline_col = find_prediction_column(raw, "baseline")
    temporal_col = find_prediction_column(raw, "temporal")

    data = pd.DataFrame({
        "unit_id": pd.to_numeric(raw[unit_col], errors="coerce"),
        "actual_RUL": pd.to_numeric(raw[actual_col], errors="coerce"),
        "baseline_prediction": pd.to_numeric(
            raw[baseline_col], errors="coerce"
        ),
        "temporal_prediction": pd.to_numeric(
            raw[temporal_col], errors="coerce"
        ),
    })

    if cycle_col is not None:
        data["cycle"] = pd.to_numeric(raw[cycle_col], errors="coerce")
    else:
        # This fallback preserves row order within each engine.
        # Prefer a real cycle column whenever one is available.
        warnings.warn(
            f"{split_name}: no cycle column found; using input row order "
            "to identify each engine's final row."
        )
        data["cycle"] = data.groupby("unit_id").cumcount() + 1

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(
        subset=[
            "unit_id", "cycle", "actual_RUL",
            "baseline_prediction", "temporal_prediction",
        ]
    ).copy()

    if data.empty:
        raise ValueError(f"No valid rows remain in {split_name} data.")

    # RUL cannot be negative in the target representation used here.
    # Predictions are not clipped: negative predictions are retained
    # because they are useful diagnostic information.
    data["unit_id"] = data["unit_id"].astype(int)

    # Use the latest cycle for each engine to avoid counting an engine
    # multiple times in maintenance-policy evaluation.
    data = (
        data.sort_values(["unit_id", "cycle"])
        .groupby("unit_id", as_index=False, sort=True)
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    if data["unit_id"].duplicated().any():
        raise RuntimeError(f"{split_name} final-cycle data has duplicate engines.")

    data["actual_RUL"] = data["actual_RUL"].clip(lower=0.0)

    print(f"{split_name} source: {path}")
    print(f"{split_name} final-cycle engines: {len(data)}")

    return data


def nasa_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """NASA C-MAPSS asymmetric RUL score."""
    error = predicted - actual
    score = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )
    # Guard against overflow in extreme predictions.
    score = np.nan_to_num(score, nan=0.0, posinf=1e300, neginf=0.0)
    return float(np.sum(score))


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    error = predicted - actual
    return {
        "n_engines": int(len(actual)),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "NASA_score": nasa_score(actual, predicted),
        "mean_error_predicted_minus_actual": float(np.mean(error)),
    }


# ============================================================
# MAINTENANCE POLICY EVALUATION
# ============================================================

def evaluate_policy(
    df: pd.DataFrame,
    prediction_column: str,
    threshold: float,
    costs: dict,
) -> dict:
    actual = df["actual_RUL"].to_numpy(dtype=float)
    predicted = df[prediction_column].to_numpy(dtype=float)

    maintenance = predicted <= threshold
    actual_critical = actual <= CRITICAL_RUL
    actual_high_or_critical = actual <= HIGH_RISK_RUL

    missed_critical = (~maintenance) & actual_critical
    missed_high_or_critical = (~maintenance) & actual_high_or_critical

    # A false alarm is maintenance when actual RUL is above the
    # defined high-risk range. This is an explicit simulation assumption.
    false_alarm = maintenance & (~actual_high_or_critical)

    n_engines = len(df)
    n_high_or_critical = int(actual_high_or_critical.sum())

    missed_high_count = int(missed_high_or_critical.sum())
    high_risk_miss_rate = (
        missed_high_count / n_high_or_critical
        if n_high_or_critical > 0
        else 0.0
    )

    actions = int(maintenance.sum())
    missed_critical_count = int(missed_critical.sum())
    false_alarm_count = int(false_alarm.sum())

    maintenance_cost = actions * costs["maintenance_cost"]
    missed_cost = missed_critical_count * costs["missed_critical_cost"]
    false_alarm_cost = false_alarm_count * costs["false_alarm_cost"]
    total_cost = maintenance_cost + missed_cost + false_alarm_cost

    safe = (
        missed_critical_count <= MAX_MISSED_CRITICAL_ENGINES
        and high_risk_miss_rate <= MAX_HIGH_RISK_MISS_RATE
    )

    return {
        "threshold": float(threshold),
        "n_engines": int(n_engines),
        "maintenance_actions": actions,
        "missed_critical_engines": missed_critical_count,
        "missed_high_or_critical_engines": missed_high_count,
        "actual_high_or_critical_engines": n_high_or_critical,
        "high_risk_miss_rate": float(high_risk_miss_rate),
        "false_alarm_actions": false_alarm_count,
        "maintenance_cost": float(maintenance_cost),
        "missed_critical_cost": float(missed_cost),
        "false_alarm_cost": float(false_alarm_cost),
        "total_simulated_cost": float(total_cost),
        "cost_per_engine": float(total_cost / n_engines),
        "safety_constraints_satisfied": bool(safe),
    }


def select_policy_on_validation(
    validation: pd.DataFrame,
    prediction_column: str,
    model_name: str,
    scenario_name: str,
    costs: dict,
) -> tuple[dict, pd.DataFrame]:
    rows = []

    for threshold in THRESHOLDS:
        result = evaluate_policy(
            validation,
            prediction_column,
            float(threshold),
            costs,
        )
        result["model"] = model_name
        result["cost_scenario"] = scenario_name
        rows.append(result)

    table = pd.DataFrame(rows)
    feasible = table[table["safety_constraints_satisfied"]].copy()

    if not feasible.empty:
        # First minimize cost, then prefer fewer missed high-risk engines,
        # fewer false alarms, and finally a lower threshold.
        feasible = feasible.sort_values(
            [
                "total_simulated_cost",
                "missed_high_or_critical_engines",
                "false_alarm_actions",
                "threshold",
            ],
            ascending=True,
        )
        selected = feasible.iloc[0].to_dict()
        status = "feasible_lowest_validation_cost"
    else:
        # Do not pretend an unsafe policy is feasible. If no candidate
        # satisfies constraints, select the most safety-conservative
        # candidate lexicographically and flag the violation.
        table = table.sort_values(
            [
                "missed_critical_engines",
                "high_risk_miss_rate",
                "missed_high_or_critical_engines",
                "total_simulated_cost",
                "threshold",
            ],
            ascending=True,
        )
        selected = table.iloc[0].to_dict()
        status = "no_feasible_candidate_safety_priority_fallback"

    selected["selection_status"] = status
    selected["selection_basis"] = (
        "Validation final-cycle engines only; test labels not used."
    )
    return selected, table


# ============================================================
# PLOTS
# ============================================================

def plot_cost_by_threshold(
    validation: pd.DataFrame,
    prediction_column: str,
    model_name: str,
    costs: dict,
    scenario_name: str,
    path: Path,
) -> None:
    records = [
        evaluate_policy(validation, prediction_column, float(t), costs)
        for t in THRESHOLDS
    ]
    curve = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        curve["threshold"],
        curve["total_simulated_cost"],
        marker="o",
        markersize=3,
        label="Validation simulated cost",
    )

    feasible = curve[curve["safety_constraints_satisfied"]]
    if not feasible.empty:
        best = feasible.loc[feasible["total_simulated_cost"].idxmin()]
        ax.axvline(
            best["threshold"],
            linestyle="--",
            label=f"Lowest-cost feasible threshold = {best['threshold']:.0f}",
        )

    ax.set_title(f"{model_name.title()} policy — {scenario_name}")
    ax.set_xlabel("Maintenance threshold (predicted RUL, cycles)")
    ax.set_ylabel("Simulated validation cost")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_test_cost_comparison(test_results: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = (
        test_results["model"].astype(str)
        + " | "
        + test_results["cost_scenario"].astype(str)
    )
    ax.bar(labels, test_results["total_simulated_cost"])
    ax.set_title("Frozen policies evaluated on held-out test engines")
    ax.set_ylabel("Simulated test cost")
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_section("NASA C-MAPSS — STEP 8")
    print("SELF-EVOLVING ADAPTIVE INDUSTRIAL DECISION AGENT")
    print(f"Subset: {SUBSET}")
    print(f"UTC run time: {datetime.now(timezone.utc).isoformat()}")

    print_section("LOADING FINAL-CYCLE ENGINE PREDICTIONS")

    validation = load_prediction_file(VALIDATION_FILE, "Validation")
    test = load_prediction_file(TEST_FILE, "Test")

    # Numeric engine IDs in validation and test are identifiers local
    # to each split. Overlap is expected and is not data leakage.
    overlap = set(validation["unit_id"]) & set(test["unit_id"])
    print(f"Numeric engine-ID overlap across splits: {len(overlap)}")
    print("Validation and test are treated as separate dataset partitions.")

    # The input CSVs must represent the same prediction definitions.
    # Only validation labels are used for policy selection.
    print_section("RUL REGRESSION METRICS")

    regression_rows = []
    for split_name, frame in [
        ("validation", validation),
        ("test_descriptive", test),
    ]:
        actual = frame["actual_RUL"].to_numpy(dtype=float)

        for model_name, col in [
            ("baseline", "baseline_prediction"),
            ("temporal", "temporal_prediction"),
        ]:
            metrics = regression_metrics(
                actual,
                frame[col].to_numpy(dtype=float),
            )
            metrics["split"] = split_name
            metrics["model"] = model_name
            regression_rows.append(metrics)

    regression_table = pd.DataFrame(regression_rows)
    regression_table.to_csv(
        OUTPUT_DIR / "step08_rul_regression_metrics.csv",
        index=False,
    )
    print(regression_table.to_string(index=False))

    print_section("VALIDATION-ONLY POLICY SELECTION")

    selection_rows = []
    all_threshold_rows = []
    frozen_policies = []

    for scenario_name, costs in COST_SCENARIOS.items():
        for model_name, prediction_column in [
            ("baseline", "baseline_prediction"),
            ("temporal", "temporal_prediction"),
        ]:
            selected, threshold_table = select_policy_on_validation(
                validation=validation,
                prediction_column=prediction_column,
                model_name=model_name,
                scenario_name=scenario_name,
                costs=costs,
            )

            selected["prediction_column"] = prediction_column
            selection_rows.append(selected)

            threshold_table["prediction_column"] = prediction_column
            all_threshold_rows.append(threshold_table)

            frozen_policies.append({
                "model": model_name,
                "prediction_column": prediction_column,
                "cost_scenario": scenario_name,
                "costs": costs,
                "threshold": float(selected["threshold"]),
                "selection_status": selected["selection_status"],
                "selection_basis": selected["selection_basis"],
            })

            plot_cost_by_threshold(
                validation=validation,
                prediction_column=prediction_column,
                model_name=model_name,
                costs=costs,
                scenario_name=scenario_name,
                path=OUTPUT_DIR
                / f"validation_cost_curve_{model_name}_{scenario_name}.png",
            )

            print(
                f"{model_name.upper()} | {scenario_name} | "
                f"threshold={selected['threshold']:.1f} | "
                f"validation cost={selected['total_simulated_cost']:.1f} | "
                f"missed critical={selected['missed_critical_engines']} | "
                f"high-risk miss rate={selected['high_risk_miss_rate']:.3f} | "
                f"status={selected['selection_status']}"
            )

    selection_table = pd.DataFrame(selection_rows)
    threshold_table_all = pd.concat(all_threshold_rows, ignore_index=True)

    selection_table.to_csv(
        OUTPUT_DIR / "step08_validation_policy_selection.csv",
        index=False,
    )
    threshold_table_all.to_csv(
        OUTPUT_DIR / "step08_all_validation_threshold_candidates.csv",
        index=False,
    )

    # Choose the adaptive agent's primary policy using the standard
    # scenario. This selection is validation-only.
    standard_candidates = selection_table[
        selection_table["cost_scenario"] == "standard"
    ].copy()

    feasible_standard = standard_candidates[
        standard_candidates["safety_constraints_satisfied"]
    ].copy()

    if not feasible_standard.empty:
        feasible_standard = feasible_standard.sort_values(
            [
                "total_simulated_cost",
                "missed_high_or_critical_engines",
                "false_alarm_actions",
                "threshold",
            ],
            ascending=True,
        )
        primary = feasible_standard.iloc[0].to_dict()
        adaptive_selection_status = "feasible_lowest_validation_cost"
    else:
        standard_candidates = standard_candidates.sort_values(
            [
                "missed_critical_engines",
                "high_risk_miss_rate",
                "missed_high_or_critical_engines",
                "total_simulated_cost",
            ],
            ascending=True,
        )
        primary = standard_candidates.iloc[0].to_dict()
        adaptive_selection_status = (
            "no_feasible_model_threshold_pair_safety_priority_fallback"
        )

    adaptive_model = str(primary["model"])
    adaptive_prediction_column = (
        "baseline_prediction"
        if adaptive_model == "baseline"
        else "temporal_prediction"
    )
    adaptive_threshold = float(primary["threshold"])

    # Auditable policy update event: this records how the initial policy
    # was chosen. It does not claim that the model weights were retrained.
    policy_update_log = pd.DataFrame([{
        "update_index": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "previous_policy": "initial_unselected_policy",
        "selected_model": adaptive_model,
        "selected_prediction_column": adaptive_prediction_column,
        "selected_threshold": adaptive_threshold,
        "cost_scenario": "standard",
        "selection_status": adaptive_selection_status,
        "trigger": "initial_validation_policy_selection",
        "decision_rule": (
            "Minimize validation simulated cost among safety-feasible "
            "model-threshold pairs; if none are feasible, apply the "
            "documented safety-priority fallback."
        ),
        "data_used": "validation final-cycle engines only",
        "test_labels_used_for_selection": False,
    }])

    policy_update_log.to_csv(
        OUTPUT_DIR / "step08_policy_update_log.csv",
        index=False,
    )

    print_section("FROZEN ADAPTIVE POLICY")
    print(f"Selected model: {adaptive_model}")
    print(f"Selected threshold: {adaptive_threshold:.1f} cycles")
    print(f"Selection status: {adaptive_selection_status}")
    print("Selection data: validation final-cycle engines only.")
    print("The policy is now frozen before test evaluation.")

    # Save validation decisions for auditability.
    validation_decisions = validation.copy()
    validation_decisions["adaptive_model"] = adaptive_model
    validation_decisions["adaptive_threshold"] = adaptive_threshold
    validation_decisions["maintenance_action"] = (
        validation_decisions[adaptive_prediction_column]
        <= adaptive_threshold
    )
    validation_decisions.to_csv(
        OUTPUT_DIR / "step08_validation_adaptive_decisions.csv",
        index=False,
    )

    print_section("HELD-OUT TEST EVALUATION OF FROZEN POLICIES")

    test_rows = []
    test_decision_frames = []

    for scenario_name, costs in COST_SCENARIOS.items():
        # Compare baseline and temporal policies using thresholds chosen
        # independently on validation data for this same cost scenario.
        scenario_selections = selection_table[
            selection_table["cost_scenario"] == scenario_name
        ]

        for _, policy in scenario_selections.iterrows():
            model_name = str(policy["model"])
            prediction_column = (
                "baseline_prediction"
                if model_name == "baseline"
                else "temporal_prediction"
            )
            threshold = float(policy["threshold"])

            result = evaluate_policy(
                test,
                prediction_column,
                threshold,
                costs,
            )
            result.update({
                "model": model_name,
                "cost_scenario": scenario_name,
                "selection_status": policy["selection_status"],
                "threshold_selected_on": "validation",
                "test_labels_used_for_selection": False,
            })
            test_rows.append(result)

            decisions = test[[
                "unit_id", "cycle", "actual_RUL",
                "baseline_prediction", "temporal_prediction",
            ]].copy()
            decisions["model"] = model_name
            decisions["cost_scenario"] = scenario_name
            decisions["frozen_threshold"] = threshold
            decisions["maintenance_action"] = (
                decisions[prediction_column] <= threshold
            )
            decisions["actual_critical"] = (
                decisions["actual_RUL"] <= CRITICAL_RUL
            )
            decisions["actual_high_or_critical"] = (
                decisions["actual_RUL"] <= HIGH_RISK_RUL
            )
            test_decision_frames.append(decisions)

    # Evaluate the single adaptive policy under every cost scenario.
    for scenario_name, costs in COST_SCENARIOS.items():
        result = evaluate_policy(
            test,
            adaptive_prediction_column,
            adaptive_threshold,
            costs,
        )
        result.update({
            "model": "adaptive_agent",
            "cost_scenario": scenario_name,
            "selection_status": adaptive_selection_status,
            "threshold_selected_on": "validation_standard_scenario",
            "test_labels_used_for_selection": False,
        })
        test_rows.append(result)

        decisions = test[[
            "unit_id", "cycle", "actual_RUL",
            "baseline_prediction", "temporal_prediction",
        ]].copy()
        decisions["model"] = "adaptive_agent"
        decisions["cost_scenario"] = scenario_name
        decisions["frozen_threshold"] = adaptive_threshold
        decisions["maintenance_action"] = (
            decisions[adaptive_prediction_column] <= adaptive_threshold
        )
        decisions["actual_critical"] = (
            decisions["actual_RUL"] <= CRITICAL_RUL
        )
        decisions["actual_high_or_critical"] = (
            decisions["actual_RUL"] <= HIGH_RISK_RUL
        )
        test_decision_frames.append(decisions)

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(
        OUTPUT_DIR / "step08_frozen_test_policy_results.csv",
        index=False,
    )

    if test_decision_frames:
        pd.concat(test_decision_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "step08_test_engine_decisions.csv",
            index=False,
        )

    print(test_results[[
        "model",
        "cost_scenario",
        "threshold",
        "maintenance_actions",
        "missed_critical_engines",
        "missed_high_or_critical_engines",
        "false_alarm_actions",
        "total_simulated_cost",
        "cost_per_engine",
        "safety_constraints_satisfied",
    ]].to_string(index=False))

    plot_test_cost_comparison(
        test_results,
        OUTPUT_DIR / "step08_frozen_test_cost_comparison.png",
    )

    # ========================================================
    # ADAPTATION AUDIT AND RESEARCH REPORT
    # ========================================================

    standard_test = test_results[
        test_results["cost_scenario"] == "standard"
    ].copy()

    baseline_test = standard_test[
        standard_test["model"] == "baseline"
    ].iloc[0]
    temporal_test = standard_test[
        standard_test["model"] == "temporal"
    ].iloc[0]
    adaptive_test = standard_test[
        standard_test["model"] == "adaptive_agent"
    ].iloc[0]

    if baseline_test["total_simulated_cost"] != 0:
        adaptive_cost_reduction_pct = (
            (
                baseline_test["total_simulated_cost"]
                - adaptive_test["total_simulated_cost"]
            )
            / baseline_test["total_simulated_cost"]
        ) * 100.0
    else:
        adaptive_cost_reduction_pct = None

    report = {
        "project": "Self-Evolving Explainable Rational Agent for Adaptive Industrial Decision Intelligence Using Statistical Trend Learning",
        "dataset": "NASA C-MAPSS",
        "subset": SUBSET,
        "step": "8",
        "method": "Validation-driven adaptive model and maintenance-threshold selection",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "validation_predictions": str(VALIDATION_FILE),
            "test_predictions": str(TEST_FILE),
        },
        "data_summary": {
            "validation_final_cycle_engines": int(len(validation)),
            "test_final_cycle_engines": int(len(test)),
            "validation_engine_ids": validation["unit_id"].tolist(),
            "test_engine_ids": test["unit_id"].tolist(),
            "cross_split_numeric_id_overlap_is_expected": True,
        },
        "assumptions": {
            "critical_rul_cycles": CRITICAL_RUL,
            "high_risk_rul_cycles": HIGH_RISK_RUL,
            "cost_scenarios": COST_SCENARIOS,
            "threshold_grid_cycles": THRESHOLDS.tolist(),
            "false_alarm_definition": (
                "Maintenance action when actual RUL is greater than "
                f"{HIGH_RISK_RUL} cycles."
            ),
            "simulated_costs_are_illustrative_not_real_cost_estimates": True,
        },
        "safety_constraints": {
            "maximum_missed_critical_validation_engines":
                MAX_MISSED_CRITICAL_ENGINES,
            "maximum_high_risk_miss_rate": MAX_HIGH_RISK_MISS_RATE,
        },
        "adaptive_policy": {
            "selected_model": adaptive_model,
            "prediction_column": adaptive_prediction_column,
            "selected_threshold_cycles": adaptive_threshold,
            "selection_status": adaptive_selection_status,
            "selection_data": "validation only",
            "test_labels_used_for_selection": False,
            "policy_update_mechanism": (
                "Auditable initial policy selection from candidate "
                "model-threshold pairs using validation safety constraints "
                "and simulated costs. This step does not retrain model "
                "weights or demonstrate online learning."
            ),
        },
        "standard_scenario_test_comparison": {
            "baseline": {
                "simulated_cost": float(
                    baseline_test["total_simulated_cost"]
                ),
                "missed_critical_engines": int(
                    baseline_test["missed_critical_engines"]
                ),
                "missed_high_or_critical_engines": int(
                    baseline_test["missed_high_or_critical_engines"]
                ),
            },
            "temporal": {
                "simulated_cost": float(
                    temporal_test["total_simulated_cost"]
                ),
                "missed_critical_engines": int(
                    temporal_test["missed_critical_engines"]
                ),
                "missed_high_or_critical_engines": int(
                    temporal_test["missed_high_or_critical_engines"]
                ),
            },
            "adaptive_agent": {
                "simulated_cost": float(
                    adaptive_test["total_simulated_cost"]
                ),
                "missed_critical_engines": int(
                    adaptive_test["missed_critical_engines"]
                ),
                "missed_high_or_critical_engines": int(
                    adaptive_test["missed_high_or_critical_engines"]
                ),
            },
            "adaptive_vs_baseline_test_cost_reduction_percent":
                adaptive_cost_reduction_pct,
        },
        "limitations": [
            "Only the FD001 subset is evaluated in this step.",
            "Policy constraints are estimated using a small validation set; "
            "20 validation engines may not support stable safety estimates.",
            "Maintenance costs are simulated assumptions and require "
            "industrial data for deployment claims.",
            "The final-cycle engine-level policy evaluation does not model "
            "full maintenance scheduling, spare parts, repair duration, "
            "or repeated interventions.",
            "This step performs policy selection and logs the selection; "
            "it does not prove continuous online self-evolution.",
            "Test metrics are for final reporting only and must not be used "
            "to tune the policy after inspection.",
        ],
        "artifacts": {
            "regression_metrics_csv":
                str(OUTPUT_DIR / "step08_rul_regression_metrics.csv"),
            "validation_policy_selection_csv":
                str(OUTPUT_DIR / "step08_validation_policy_selection.csv"),
            "all_threshold_candidates_csv":
                str(OUTPUT_DIR / "step08_all_validation_threshold_candidates.csv"),
            "policy_update_log_csv":
                str(OUTPUT_DIR / "step08_policy_update_log.csv"),
            "validation_decisions_csv":
                str(OUTPUT_DIR / "step08_validation_adaptive_decisions.csv"),
            "test_policy_results_csv":
                str(OUTPUT_DIR / "step08_frozen_test_policy_results.csv"),
            "test_engine_decisions_csv":
                str(OUTPUT_DIR / "step08_test_engine_decisions.csv"),
            "test_cost_plot":
                str(OUTPUT_DIR / "step08_frozen_test_cost_comparison.png"),
        },
    }

    save_json(report, OUTPUT_DIR / f"step08_{SUBSET}_report.json")

    print_section("STEP 8 COMPLETED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Research report: {OUTPUT_DIR / f'step08_{SUBSET}_report.json'}")
    print("Validation labels were used for policy selection.")
    print("Test labels were not used to select the policy.")
    print("The selected policy was frozen before test evaluation.")
    print("Steps 1–7.2 were not modified.")


if __name__ == "__main__":
    main()
