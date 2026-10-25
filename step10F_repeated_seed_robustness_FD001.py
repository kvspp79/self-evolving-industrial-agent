
"""
STEP 10F: Repeated-Seed Robustness Evaluation
Project: Self-Evolving Explainable Industrial Agent
Dataset: NASA C-MAPSS FD001

Purpose:
1. Repeat engine-level train/adaptation/gate/test splits across fixed seeds.
2. Apply the same predefined synthetic sensor shift.
3. Compare a frozen model, unconditional retraining, and safety-gated adaptation.
4. Keep final held-out engines out of training and gate decisions.
5. Summarize performance across seeds, including variability.

Fixed gate:
- No monitored validation metric may worsen by more than 2%.
- At least one monitored validation metric must improve by at least 1%.

Important:
- Synthetic shift is not evidence of real-world industrial drift.
- The NASA score is asymmetric and may be dominated by large errors.
- Final results are descriptive; repeated seeds are not independent real-world
  industrial deployments.
"""

from pathlib import Path
import json
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. CONFIGURATION — FIX BEFORE RUNNING
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS"
)

INPUT_FILE = PROJECT_ROOT / "outputs" / "step01" / "train_FD001_processed.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step10F_repeated_seed_robustness"

# Ten predetermined random seeds.
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

# Engine split proportions: 55 / 15 / 15 / 15 for 100 engines.
BASE_FRACTION = 0.55
ADAPTATION_FRACTION = 0.15
GATE_FRACTION = 0.15

# Synthetic shift, expressed in standard deviations estimated from base engines.
SHIFT_CONFIG = {
    "sensor_04": 0.60,
    "sensor_07": -0.60,
    "sensor_11": 0.80,
    "sensor_12": -0.70,
    "sensor_15": 0.65,
}

# Safety gate criteria. Do not tune these after inspecting final results.
MAX_ALLOWED_RELATIVE_WORSENING = 0.02
MIN_REQUIRED_RELATIVE_IMPROVEMENT = 0.01

# Candidate adaptation strategy carried forward from Step 10E:
# adaptation engines receive 2x total weight per engine relative to a base engine.
ADAPTATION_ENGINE_WEIGHT_MULTIPLIER = 2.0

# Fixed model configuration for all seeds and both model types.
MODEL_CONFIG = {
    "n_estimators": 200,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

REQUIRED_COLUMNS = ["unit_id", "cycle", "RUL"]


# ============================================================
# 2. DATA AND METRIC UTILITIES
# ============================================================

def find_project_root():
    """Find the configured project root or a compatible parent."""
    if INPUT_FILE.exists():
        return PROJECT_ROOT

    candidates = [
        Path.cwd(),
        Path.cwd().parent,
    ]

    for candidate in candidates:
        possible = candidate / "outputs" / "step01" / "train_FD001_processed.csv"
        if possible.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find outputs/step01/train_FD001_processed.csv.\n"
        f"Configured path: {INPUT_FILE}\n"
        "Check PROJECT_ROOT near the top of this script."
    )


def nasa_score_sum(y_true, y_pred):
    """
    Standard asymmetric NASA PHM-style score.

    Error = prediction - actual.
    Negative errors (underprediction) use divisor 13.
    Positive errors (overprediction) use divisor 10.

    Returns a sum over rows. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    penalties = np.where(
        error < 0,
        np.exp(np.clip(-error / 13.0, -50, 50)) - 1.0,
        np.exp(np.clip(error / 10.0, -50, 50)) - 1.0,
    )
    return float(np.sum(penalties))


def engine_metrics(frame, predictions, model_name, seed):
    """Calculate metrics per engine, then summarize equally across engines."""
    temp = frame[["unit_id", "RUL"]].copy()
    temp["prediction"] = np.asarray(predictions, dtype=float)
    temp["error"] = temp["prediction"] - temp["RUL"]

    rows = []

    for engine_id, group in temp.groupby("unit_id", sort=True):
        y_true = group["RUL"].to_numpy(dtype=float)
        y_pred = group["prediction"].to_numpy(dtype=float)

        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        nasa = nasa_score_sum(y_true, y_pred)

        rows.append({
            "seed": int(seed),
            "model": model_name,
            "unit_id": int(engine_id),
            "n_rows": int(len(group)),
            "engine_RMSE": rmse,
            "engine_MAE": mae,
            "engine_NASA_score": nasa,
        })

    engine_df = pd.DataFrame(rows)

    summary = {
        "seed": int(seed),
        "model": model_name,
        "n_engines": int(engine_df["unit_id"].nunique()),
        "mean_engine_RMSE": float(engine_df["engine_RMSE"].mean()),
        "mean_engine_MAE": float(engine_df["engine_MAE"].mean()),
        "mean_engine_NASA_score": float(engine_df["engine_NASA_score"].mean()),
        "median_engine_RMSE": float(engine_df["engine_RMSE"].median()),
        "median_engine_MAE": float(engine_df["engine_MAE"].median()),
        "median_engine_NASA_score": float(engine_df["engine_NASA_score"].median()),
    }

    return summary, engine_df


def relative_improvement(baseline, candidate):
    """
    Positive value means candidate improved over baseline.
    Negative value means candidate worsened.
    """
    if not np.isfinite(baseline) or baseline <= 0:
        return np.nan
    return float((baseline - candidate) / baseline)


def apply_synthetic_shift(frame, base_sensor_std):
    """Apply the fixed synthetic shift without modifying the original frame."""
    shifted = frame.copy()

    for sensor, multiplier in SHIFT_CONFIG.items():
        if sensor not in shifted.columns:
            raise ValueError(
                f"Required shifted sensor '{sensor}' is missing from the dataset."
            )

        std_value = float(base_sensor_std[sensor])
        if not np.isfinite(std_value):
            raise ValueError(f"Invalid base standard deviation for {sensor}.")

        shifted[sensor] = (
            shifted[sensor].astype(float) + multiplier * std_value
        )

    return shifted


def build_engine_balanced_weights(base_frame, adaptation_frame):
    """
    Give each base engine total weight 1 and each adaptation engine total
    weight ADAPTATION_ENGINE_WEIGHT_MULTIPLIER.

    Normalize the resulting row weights to mean 1 for numerical stability.
    """
    base_counts = base_frame.groupby("unit_id").size()
    adaptation_counts = adaptation_frame.groupby("unit_id").size()

    base_weights = (
        base_frame["unit_id"]
        .map((1.0 / base_counts).to_dict())
        .to_numpy(dtype=float)
    )

    adaptation_weights = (
        adaptation_frame["unit_id"]
        .map(
            (
                ADAPTATION_ENGINE_WEIGHT_MULTIPLIER / adaptation_counts
            ).to_dict()
        )
        .to_numpy(dtype=float)
    )

    weights = np.concatenate([base_weights, adaptation_weights])

    mean_weight = float(np.mean(weights))
    if mean_weight <= 0 or not np.isfinite(mean_weight):
        raise ValueError("Invalid sample weights generated.")

    return weights / mean_weight


def fit_model(X, y, seed, sample_weight=None):
    """Fit a reproducible random forest."""
    config = dict(MODEL_CONFIG)
    config["random_state"] = int(seed)

    model = RandomForestRegressor(**config)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def evaluate_gate(baseline_summary, candidate_summary):
    """
    Apply the fixed gate to the three monitored metrics.
    Uses only gate-validation results.
    """
    metric_map = {
        "mean_engine_RMSE": "RMSE",
        "mean_engine_MAE": "MAE",
        "mean_engine_NASA_score": "NASA_score",
    }

    details = {}
    relative_changes = []

    no_metric_worsened_too_much = True
    at_least_one_improved_enough = False

    for metric, short_name in metric_map.items():
        baseline_value = float(baseline_summary[metric])
        candidate_value = float(candidate_summary[metric])

        improvement = relative_improvement(
            baseline_value, candidate_value
        )

        details[metric] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "relative_improvement": improvement,
        }

        if not np.isfinite(improvement):
            no_metric_worsened_too_much = False
            continue

        relative_changes.append(improvement)

        # A negative improvement indicates worsening.
        if improvement < -MAX_ALLOWED_RELATIVE_WORSENING:
            no_metric_worsened_too_much = False

        if improvement >= MIN_REQUIRED_RELATIVE_IMPROVEMENT:
            at_least_one_improved_enough = True

    accepted = (
        no_metric_worsened_too_much
        and at_least_one_improved_enough
    )

    return {
        "accepted": bool(accepted),
        "no_metric_worsened_more_than_2_percent": bool(
            no_metric_worsened_too_much
        ),
        "at_least_one_metric_improved_1_percent": bool(
            at_least_one_improved_enough
        ),
        "metric_comparisons": details,
    }


# ============================================================
# 3. ONE COMPLETE REPEATED-SEED EXPERIMENT
# ============================================================

def run_one_seed(data, feature_columns, seed, output_dir):
    print("\n" + "=" * 72)
    print(f"RUNNING SEED {seed}")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    engine_ids = np.array(sorted(data["unit_id"].unique()))
    rng.shuffle(engine_ids)

    n_engines = len(engine_ids)
    n_base = int(round(n_engines * BASE_FRACTION))
    n_adaptation = int(round(n_engines * ADAPTATION_FRACTION))
    n_gate = int(round(n_engines * GATE_FRACTION))

    # Keep the final group as the remainder to avoid rounding gaps.
    base_ids = engine_ids[:n_base]
    adaptation_ids = engine_ids[n_base:n_base + n_adaptation]
    gate_start = n_base + n_adaptation
    gate_ids = engine_ids[gate_start:gate_start + n_gate]
    final_ids = engine_ids[gate_start + n_gate:]

    if min(len(base_ids), len(adaptation_ids), len(gate_ids), len(final_ids)) < 1:
        raise ValueError("One or more engine split groups are empty.")

    assert len(set(base_ids) & set(adaptation_ids)) == 0
    assert len(set(base_ids) & set(gate_ids)) == 0
    assert len(set(base_ids) & set(final_ids)) == 0
    assert len(set(adaptation_ids) & set(gate_ids)) == 0
    assert len(set(adaptation_ids) & set(final_ids)) == 0
    assert len(set(gate_ids) & set(final_ids)) == 0

    base = data[data["unit_id"].isin(base_ids)].copy()
    adaptation_original = data[data["unit_id"].isin(adaptation_ids)].copy()
    gate_original = data[data["unit_id"].isin(gate_ids)].copy()
    final_original = data[data["unit_id"].isin(final_ids)].copy()

    print(
        "Engine split:",
        f"base={len(base_ids)},",
        f"adaptation={len(adaptation_ids)},",
        f"gate={len(gate_ids)},",
        f"final={len(final_ids)}",
    )

    # Estimate shift scales using base engines only.
    base_sensor_std = base[list(SHIFT_CONFIG)].std(ddof=0)

    # Shift adaptation and both evaluation subsets using the fixed configuration.
    adaptation = apply_synthetic_shift(adaptation_original, base_sensor_std)
    gate = apply_synthetic_shift(gate_original, base_sensor_std)
    final = apply_synthetic_shift(final_original, base_sensor_std)

    # Base training rows remain unshifted.
    X_base = base[feature_columns]
    y_base = base["RUL"].to_numpy(dtype=float)

    X_adaptation = adaptation[feature_columns]
    y_adaptation = adaptation["RUL"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Frozen baseline
    # --------------------------------------------------------
    print("Training frozen baseline ...")
    frozen_model = fit_model(X_base, y_base, seed=seed)

    # --------------------------------------------------------
    # Candidate update: engine-balanced adaptation 2x
    # --------------------------------------------------------
    print("Training engine-balanced adaptation candidate ...")

    combined = pd.concat([base, adaptation], ignore_index=True)
    X_combined = combined[feature_columns]
    y_combined = combined["RUL"].to_numpy(dtype=float)

    sample_weights = build_engine_balanced_weights(base, adaptation)

    candidate_model = fit_model(
        X_combined,
        y_combined,
        seed=seed,
        sample_weight=sample_weights,
    )

    # --------------------------------------------------------
    # Gate evaluation — final held-out data is not accessed here
    # --------------------------------------------------------
    print("Evaluating safety gate on gate-validation engines ...")

    frozen_gate_predictions = frozen_model.predict(gate[feature_columns])
    candidate_gate_predictions = candidate_model.predict(gate[feature_columns])

    frozen_gate_summary, _ = engine_metrics(
        gate, frozen_gate_predictions, "frozen_baseline", seed
    )
    candidate_gate_summary, _ = engine_metrics(
        gate, candidate_gate_predictions, "candidate_engine_balanced_2x", seed
    )

    gate_result = evaluate_gate(
        frozen_gate_summary, candidate_gate_summary
    )

    if gate_result["accepted"]:
        selected_model = candidate_model
        selected_name = "safety_gated_model"
        print("Gate decision: ACCEPT candidate")
    else:
        selected_model = frozen_model
        selected_name = "safety_gated_model"
        print("Gate decision: REJECT candidate; retain frozen model")

    print("Gate comparison:")
    for metric, result in gate_result["metric_comparisons"].items():
        change = result["relative_improvement"]
        change_text = (
            f"{change * 100:.2f}% improvement"
            if np.isfinite(change)
            else "not calculable"
        )
        print(f"  {metric}: {change_text}")

    # --------------------------------------------------------
    # Final held-out evaluation
    # No model choice is made using these results.
    # --------------------------------------------------------
    print("Evaluating on final held-out engines ...")

    frozen_final_predictions = frozen_model.predict(final[feature_columns])
    candidate_final_predictions = candidate_model.predict(final[feature_columns])
    gated_final_predictions = selected_model.predict(final[feature_columns])

    summaries = []
    engine_tables = []

    for model_name, predictions in [
        ("frozen_baseline", frozen_final_predictions),
        ("unconditional_retraining", candidate_final_predictions),
        ("safety_gated_model", gated_final_predictions),
    ]:
        summary, per_engine = engine_metrics(
            final, predictions, model_name, seed
        )
        summaries.append(summary)
        engine_tables.append(per_engine)

    print("\nFinal held-out results for seed", seed)
    for summary in summaries:
        print(
            f"{summary['model']}: "
            f"RMSE={summary['mean_engine_RMSE']:.4f}, "
            f"MAE={summary['mean_engine_MAE']:.4f}, "
            f"NASA={summary['mean_engine_NASA_score']:.4f}"
        )

    # Save engine assignments for reproducibility.
    split_record = {
        "seed": int(seed),
        "n_base_engines": int(len(base_ids)),
        "n_adaptation_engines": int(len(adaptation_ids)),
        "n_gate_engines": int(len(gate_ids)),
        "n_final_engines": int(len(final_ids)),
        "base_engine_ids": [int(x) for x in sorted(base_ids)],
        "adaptation_engine_ids": [int(x) for x in sorted(adaptation_ids)],
        "gate_engine_ids": [int(x) for x in sorted(gate_ids)],
        "final_engine_ids": [int(x) for x in sorted(final_ids)],
        "gate_result": gate_result,
    }

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    with open(seed_dir / "split_and_gate_record.json", "w", encoding="utf-8") as f:
        json.dump(split_record, f, indent=2)

    return summaries, engine_tables, split_record


# ============================================================
# 4. MAIN
# ============================================================

def main():
    global PROJECT_ROOT, INPUT_FILE, OUTPUT_DIR

    PROJECT_ROOT = find_project_root()
    INPUT_FILE = PROJECT_ROOT / "outputs" / "step01" / "train_FD001_processed.csv"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step10F_repeated_seed_robustness"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print("=" * 72)
    print("STEP 10F — REPEATED-SEED ROBUSTNESS EVALUATION")
    print("=" * 72)
    print("Project root:", PROJECT_ROOT)
    print("Input:", INPUT_FILE)
    print("Output:", OUTPUT_DIR)
    print("Seeds:", SEEDS)
    print("Synthetic shift:", SHIFT_CONFIG)
    print("Fixed gate: no metric worsens >2%; at least one improves >=1%")

    data = pd.read_csv(INPUT_FILE)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["unit_id"] = pd.to_numeric(data["unit_id"], errors="raise").astype(int)
    data["RUL"] = pd.to_numeric(data["RUL"], errors="raise")
    data["cycle"] = pd.to_numeric(data["cycle"], errors="raise")

    if data["RUL"].isna().any():
        raise ValueError("RUL contains missing values.")
    if not np.isfinite(data["RUL"]).all():
        raise ValueError("RUL contains non-finite values.")

    # Exclude identifier and target from predictors. Cycle is retained.
    feature_columns = [
        col for col in data.select_dtypes(include=[np.number]).columns
        if col not in ["unit_id", "RUL"]
    ]

    missing_sensors = [s for s in SHIFT_CONFIG if s not in feature_columns]
    if missing_sensors:
        raise ValueError(
            f"Shift sensors are missing or non-numeric: {missing_sensors}"
        )

    if not feature_columns:
        raise ValueError("No numeric model features were found.")

    # Ensure all model features are numeric and finite.
    data[feature_columns] = data[feature_columns].apply(
        pd.to_numeric, errors="raise"
    )

    if not np.isfinite(data[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError(
            "Model features contain missing or non-finite values. "
            "Resolve them before running this experiment."
        )

    print("Rows:", len(data))
    print("Engines:", data["unit_id"].nunique())
    print("Numeric model features:", len(feature_columns))
    print("Features:", feature_columns)

    all_summaries = []
    all_engine_tables = []
    all_split_records = []
    failures = []

    for seed in SEEDS:
        try:
            summaries, engine_tables, split_record = run_one_seed(
                data, feature_columns, seed, OUTPUT_DIR
            )
            all_summaries.extend(summaries)
            all_engine_tables.extend(engine_tables)
            all_split_records.append(split_record)

        except Exception as exc:
            failures.append({
                "seed": int(seed),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"\nERROR on seed {seed}: {exc}")
            traceback.print_exc()

    if not all_summaries:
        raise RuntimeError(
            "No seed completed successfully. Inspect the errors above."
        )

    # --------------------------------------------------------
    # Save per-seed final results
    # --------------------------------------------------------
    results_df = pd.DataFrame(all_summaries)
    results_path = OUTPUT_DIR / "step10F_per_seed_final_metrics.csv"
    results_df.to_csv(results_path, index=False)

    engine_df = pd.concat(all_engine_tables, ignore_index=True)
    engine_path = OUTPUT_DIR / "step10F_final_engine_metrics.csv"
    engine_df.to_csv(engine_path, index=False)

    # Gate outcomes and engine splits.
    gate_rows = []
    for record in all_split_records:
        gate_rows.append({
            "seed": record["seed"],
            "n_base_engines": record["n_base_engines"],
            "n_adaptation_engines": record["n_adaptation_engines"],
            "n_gate_engines": record["n_gate_engines"],
            "n_final_engines": record["n_final_engines"],
            "candidate_accepted": record["gate_result"]["accepted"],
            "all_metrics_within_2_percent": record["gate_result"][
                "no_metric_worsened_more_than_2_percent"
            ],
            "at_least_one_metric_improved_1_percent": record["gate_result"][
                "at_least_one_metric_improved_1_percent"
            ],
        })

    gate_df = pd.DataFrame(gate_rows)
    gate_path = OUTPUT_DIR / "step10F_gate_decisions.csv"
    gate_df.to_csv(gate_path, index=False)

    # --------------------------------------------------------
    # Across-seed mean and standard deviation
    # --------------------------------------------------------
    metric_columns = [
        "mean_engine_RMSE",
        "mean_engine_MAE",
        "mean_engine_NASA_score",
    ]

    aggregate_rows = []
    for model_name, group in results_df.groupby("model", sort=True):
        row = {
            "model": model_name,
            "successful_seed_runs": int(group["seed"].nunique()),
        }

        for metric in metric_columns:
            row[f"{metric}_mean_across_seeds"] = float(group[metric].mean())
            row[f"{metric}_std_across_seeds"] = (
                float(group[metric].std(ddof=1))
                if group["seed"].nunique() > 1
                else 0.0
            )
            row[f"{metric}_median_across_seeds"] = float(group[metric].median())

        aggregate_rows.append(row)

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_path = OUTPUT_DIR / "step10F_across_seed_summary.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    # --------------------------------------------------------
    # Paired per-seed differences vs frozen baseline
    # --------------------------------------------------------
    wide = results_df.pivot(index="seed", columns="model", values=metric_columns)
    paired_rows = []

    for seed in wide.index:
        row = {"seed": int(seed)}

        for metric in metric_columns:
            baseline = wide.loc[seed, (metric, "frozen_baseline")]

            for comparison_model in [
                "unconditional_retraining",
                "safety_gated_model",
            ]:
                candidate = wide.loc[seed, (metric, comparison_model)]
                improvement = relative_improvement(baseline, candidate)

                row[f"{comparison_model}_{metric}_relative_improvement"] = (
                    improvement
                )

        paired_rows.append(row)

    paired_df = pd.DataFrame(paired_rows)
    paired_path = OUTPUT_DIR / "step10F_paired_seed_improvements.csv"
    paired_df.to_csv(paired_path, index=False)

    # Count whether each model improves relative to frozen baseline per seed.
    improvement_counts = {}
    for model_name in ["unconditional_retraining", "safety_gated_model"]:
        model_counts = {}

        for metric in metric_columns:
            col = f"{model_name}_{metric}_relative_improvement"
            model_counts[metric] = {
                "seeds_improved": int((paired_df[col] > 0).sum()),
                "successful_seeds": int(paired_df[col].notna().sum()),
            }

        improvement_counts[model_name] = model_counts

    report = {
        "project": "Self-Evolving Explainable Industrial Agent",
        "experiment": "Step 10F repeated-seed robustness evaluation",
        "dataset": str(INPUT_FILE),
        "seeds_requested": SEEDS,
        "successful_seeds": sorted(results_df["seed"].unique().tolist()),
        "failed_runs": failures,
        "n_rows": int(len(data)),
        "n_engines": int(data["unit_id"].nunique()),
        "feature_columns": feature_columns,
        "synthetic_shift": SHIFT_CONFIG,
        "model_config": MODEL_CONFIG,
        "fixed_gate": {
            "max_allowed_relative_worsening": MAX_ALLOWED_RELATIVE_WORSENING,
            "minimum_required_relative_improvement": MIN_REQUIRED_RELATIVE_IMPROVEMENT,
            "candidate_strategy": "engine-balanced adaptation 2x",
        },
        "gate_acceptance_count": int(gate_df["candidate_accepted"].sum()),
        "gate_rejection_count": int((~gate_df["candidate_accepted"]).sum()),
        "improvement_counts_vs_frozen_baseline": improvement_counts,
        "interpretation_limitations": [
            "The sensor shift is synthetic, not observed real-world drift.",
            "Only the reported seeds and split protocol were evaluated.",
            "Repeated seeds reuse the same dataset and are not independent deployments.",
            "The gate-validation set selects acceptance; it is not the final test set.",
            "Final held-out results must not be used to revise the gate or choose a strategy.",
            "NASA score is asymmetric and can be dominated by large prediction errors.",
            "This experiment alone does not establish causal explanations or safe real-world autonomy.",
        ],
        "output_files": {
            "per_seed_final_metrics": str(results_path),
            "final_engine_metrics": str(engine_path),
            "gate_decisions": str(gate_path),
            "across_seed_summary": str(aggregate_path),
            "paired_seed_improvements": str(paired_path),
        },
    }

    report_path = OUTPUT_DIR / "step10F_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("STEP 10F COMPLETE")
    print("=" * 72)

    print("\nAcross-seed summary:")
    print(aggregate_df.to_string(index=False))

    print("\nGate decisions:")
    print(gate_df.to_string(index=False))

    print("\nImprovement counts vs frozen baseline:")
    print(json.dumps(improvement_counts, indent=2))

    print("\nSuccessful seeds:", results_df["seed"].nunique(), "/", len(SEEDS))
    print("Failed runs:", len(failures))
    print("Results directory:", OUTPUT_DIR)
    print("Report:", report_path)

    if failures:
        print(
            "\nWARNING: Some seeds failed. Do not interpret the completed "
            "runs as the full planned experiment until failures are resolved."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nFATAL ERROR:", exc)
        traceback.print_exc()
        sys.exit(1)
