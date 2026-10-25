"""
STEP 10C — CONTROLLED SELF-EVOLUTION EXPERIMENT
Dataset: NASA C-MAPSS FD001
Python: 3.13+

Research objective:
Compare a frozen RUL prediction model with a model that updates
from sequentially revealed, labelled adaptation data.

Important limitations:
- The sensor shift is deliberately simulated.
- This does not prove real-world industrial drift.
- Engines are separated before model training and evaluation.
- The official NASA test file is not used.
- Results must be reported honestly, even if adaptation performs worse.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# Keep these settings fixed before evaluating the final engines.
# ============================================================

RANDOM_SEED = 20261025
SHIFT_SEED = 4107

BASE_ENGINE_FRACTION = 0.70
ADAPT_ENGINE_FRACTION = 0.15

N_ADAPTATION_ROUNDS = 3
N_TREES = 100
MIN_SAMPLES_LEAF = 2
MAX_DEPTH = 18

TARGET = "RUL"
GROUP = "unit_id"

NON_FEATURES = {TARGET, GROUP}
REQUIRED_COLUMNS = {GROUP, "cycle", TARGET}


# ============================================================
# 2. PREDEFINED SYNTHETIC SENSOR SHIFT
#
# Shift magnitudes are expressed in standard deviations
# calculated using base-training engines only.
#
# This is a controlled simulation, not observed real-world drift.
# ============================================================

SHIFT_SPEC = {
    "sensor_04": +0.60,
    "sensor_07": -0.60,
    "sensor_11": +0.80,
    "sensor_12": -0.70,
    "sensor_15": +0.65,
}


# ============================================================
# 3. FIND THE NASA PROJECT DIRECTORY
# ============================================================

def find_project_root() -> Path:
    """Find the project root containing the processed FD001 data."""

    candidates = [
        Path(__file__).resolve().parents[1],
        Path.cwd(),
    ]

    for candidate in candidates:
        dataset_path = (
            candidate
            / "outputs"
            / "step01"
            / "train_FD001_processed.csv"
        )

        if dataset_path.exists():
            return candidate

    raise FileNotFoundError(
        "\nCould not locate the processed NASA dataset.\n"
        "Expected location:\n"
        "outputs\\step01\\train_FD001_processed.csv\n\n"
        "Save this script inside the project's src folder and "
        "check that Step 1 has generated the processed dataset."
    )


# ============================================================
# 4. EVALUATION METRICS
# ============================================================

def calculate_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Root mean squared error. Lower is better."""

    return float(
        np.sqrt(mean_squared_error(actual, predicted))
    )


def calculate_nasa_score(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Standard asymmetric NASA PHM score.

    Overestimating RUL is penalized more strongly than
    underestimating RUL under this scoring convention.

    Lower is better.
    """

    error = np.asarray(predicted) - np.asarray(actual)

    penalties = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )

    return float(np.sum(penalties))


def evaluate_model(
    model_name: str,
    evaluation_data: pd.DataFrame,
    predictions: np.ndarray,
    evaluation_unit: str,
) -> dict:
    """Calculate evaluation metrics for one model."""

    actual = evaluation_data[TARGET].to_numpy(dtype=float)
    predicted = np.asarray(predictions, dtype=float)

    return {
        "model": model_name,
        "evaluation_unit": evaluation_unit,
        "n_rows_or_engines": int(len(evaluation_data)),
        "RMSE": calculate_rmse(actual, predicted),
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "NASA_PHM_score_lower_is_better": calculate_nasa_score(
            actual,
            predicted,
        ),
        "mean_prediction_error_pred_minus_actual": float(
            np.mean(predicted - actual)
        ),
    }


# ============================================================
# 5. APPLY THE CONTROLLED SYNTHETIC SENSOR SHIFT
# ============================================================

def apply_synthetic_shift(
    frame: pd.DataFrame,
    feature_columns: list[str],
    base_means: pd.Series,
    base_stds: pd.Series,
) -> pd.DataFrame:
    """
    Apply the predefined shift to a copy of the data.

    The target RUL and engine identifiers are not modified.
    """

    shifted = frame.copy()

    for sensor, shift_units in SHIFT_SPEC.items():

        if sensor not in feature_columns:
            raise ValueError(
                f"Required shifted feature not found: {sensor}"
            )

        scale = float(base_stds[sensor])

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Invalid training scale for {sensor}: {scale}"
            )

        shifted[sensor] = (
            shifted[sensor] + shift_units * scale
        )

    return shifted


# ============================================================
# 6. SELECT THE FINAL CYCLE OF EACH ENGINE
# ============================================================

def final_cycle_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one final observed cycle per engine.

    This is the primary engine-level evaluation because
    individual cycles from the same engine are dependent.
    """

    indices = (
        frame.groupby(GROUP, sort=False)["cycle"].idxmax()
    )

    return (
        frame.loc[indices]
        .sort_values(GROUP)
        .reset_index(drop=True)
    )


# ============================================================
# 7. CREATE THE FIXED MODEL CONFIGURATION
# ============================================================

def create_model() -> RandomForestRegressor:
    """Create a model with fixed hyperparameters."""

    return RandomForestRegressor(
        n_estimators=N_TREES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


# ============================================================
# 8. MAIN SELF-EVOLUTION EXPERIMENT
# ============================================================

def main() -> None:

    print("\n" + "=" * 65)
    print("STEP 10C: CONTROLLED SELF-EVOLUTION EXPERIMENT")
    print("=" * 65)

    # --------------------------------------------------------
    # 8.1 Locate input and output directories
    # --------------------------------------------------------

    project_root = find_project_root()

    input_path = (
        project_root
        / "outputs"
        / "step01"
        / "train_FD001_processed.csv"
    )

    output_dir = (
        project_root
        / "outputs"
        / "step10C_self_evolution"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nProject root:\n{project_root}")
    print(f"\nInput dataset:\n{input_path}")
    print(f"\nOutput directory:\n{output_dir}")

    # --------------------------------------------------------
    # 8.2 Load and validate the processed dataset
    # --------------------------------------------------------

    data = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Required columns missing from dataset: "
            f"{sorted(missing_columns)}"
        )

    if data[GROUP].isna().any():
        raise ValueError(
            "The engine identifier column contains missing values."
        )

    if data[TARGET].isna().any():
        raise ValueError(
            "The RUL target contains missing values."
        )

    if not pd.api.types.is_numeric_dtype(data[TARGET]):
        raise TypeError(
            "RUL target must be numeric."
        )

    feature_columns = [
        column
        for column in data.columns
        if (
            column not in NON_FEATURES
            and pd.api.types.is_numeric_dtype(data[column])
        )
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric model features were found."
        )

    for sensor in SHIFT_SPEC:
        if sensor not in feature_columns:
            raise ValueError(
                f"Shift feature missing from dataset: {sensor}"
            )

    if data[feature_columns].isna().any().any():
        raise ValueError(
            "Numeric features contain missing values. "
            "Resolve missing values before this experiment."
        )

    print(f"\nTotal rows: {len(data)}")
    print(f"Total engines: {data[GROUP].nunique()}")
    print(f"Numeric features: {len(feature_columns)}")

    # --------------------------------------------------------
    # 8.3 Split engines BEFORE separating their rows
    #
    # No engine can belong to more than one group.
    # --------------------------------------------------------

    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    engines = np.array(
        sorted(data[GROUP].unique())
    )

    if len(engines) < 20:
        raise ValueError(
            "Too few engines for this experiment. "
            f"Found {len(engines)}."
        )

    shuffled_engines = random_generator.permutation(
        engines
    )

    number_base_engines = int(
        math.floor(
            len(engines) * BASE_ENGINE_FRACTION
        )
    )

    number_adaptation_engines = int(
        math.floor(
            len(engines) * ADAPT_ENGINE_FRACTION
        )
    )

    number_adaptation_engines = max(
        N_ADAPTATION_ROUNDS,
        number_adaptation_engines,
    )

    if (
        number_base_engines
        + number_adaptation_engines
        >= len(engines)
    ):
        raise ValueError(
            "The engine split leaves no final evaluation engines."
        )

    base_engine_ids = set(
        shuffled_engines[:number_base_engines]
    )

    adaptation_engine_ids = list(
        shuffled_engines[
            number_base_engines:
            number_base_engines + number_adaptation_engines
        ]
    )

    evaluation_engine_ids = set(
        shuffled_engines[
            number_base_engines + number_adaptation_engines:
        ]
    )

    base_data = data[
        data[GROUP].isin(base_engine_ids)
    ].copy()

    adaptation_raw = data[
        data[GROUP].isin(adaptation_engine_ids)
    ].copy()

    final_evaluation_raw = data[
        data[GROUP].isin(evaluation_engine_ids)
    ].copy()

    print("\nEngine-level split")
    print("-" * 40)
    print(f"Base training engines: {len(base_engine_ids)}")
    print(
        "Adaptation engines: "
        f"{len(adaptation_engine_ids)}"
    )
    print(
        "Final held-out engines: "
        f"{len(evaluation_engine_ids)}"
    )

    # --------------------------------------------------------
    # 8.4 Estimate sensor scales from base-training engines only
    # --------------------------------------------------------

    base_means = base_data[feature_columns].mean()

    base_stds = base_data[feature_columns].std(
        ddof=0
    )

    # Constant non-shifted columns are permitted.
    # The shifted sensors are checked separately.
    base_stds = base_stds.where(
        np.isfinite(base_stds) & (base_stds > 0),
        1.0,
    )

    # --------------------------------------------------------
    # 8.5 Apply the same predefined shift to adaptation
    # and final held-out evaluation data.
    #
    # The final evaluation data is never used for updating.
    # --------------------------------------------------------

    adaptation_shifted = apply_synthetic_shift(
        adaptation_raw,
        feature_columns,
        base_means,
        base_stds,
    )

    final_evaluation_shifted = apply_synthetic_shift(
        final_evaluation_raw,
        feature_columns,
        base_means,
        base_stds,
    )

    print("\nSynthetic shift configuration:")
    for sensor, magnitude in SHIFT_SPEC.items():
        print(f"  {sensor}: {magnitude:+.2f} standard deviations")

    # --------------------------------------------------------
    # 8.6 Train the frozen baseline
    # --------------------------------------------------------

    print("\nTraining frozen baseline model...")

    frozen_model = create_model()

    frozen_model.fit(
        base_data[feature_columns],
        base_data[TARGET],
    )

    print("Frozen baseline training complete.")

    # --------------------------------------------------------
    # 8.7 Sequential feedback and model updates
    #
    # Each round reveals labelled data from new adaptation
    # engines. The model is retrained using replay data:
    #
    # original base data + all newly labelled batches so far.
    #
    # The final evaluation engines remain unavailable.
    # --------------------------------------------------------

    adaptation_rng = np.random.default_rng(
        SHIFT_SEED
    )

    adaptation_engine_order = adaptation_rng.permutation(
        np.array(adaptation_engine_ids)
    )

    adaptation_rounds = np.array_split(
        adaptation_engine_order,
        N_ADAPTATION_ROUNDS,
    )

    replay_data = base_data.copy()

    update_history = []
    previously_seen_adaptation_engines = []

    evolved_model = None

    print("\nBeginning sequential adaptation...")

    for round_number, engine_group in enumerate(
        adaptation_rounds,
        start=1,
    ):

        round_engine_set = set(
            engine_group.tolist()
        )

        newly_labelled_batch = adaptation_shifted[
            adaptation_shifted[GROUP].isin(
                round_engine_set
            )
        ].copy()

        if newly_labelled_batch.empty:
            raise RuntimeError(
                f"Adaptation round {round_number} has no rows."
            )

        # Simulate newly available labelled outcomes.
        replay_data = pd.concat(
            [
                replay_data,
                newly_labelled_batch,
            ],
            ignore_index=True,
        )

        previously_seen_adaptation_engines.extend(
            int(engine_id)
            for engine_id in engine_group.tolist()
        )

        # Update the predictive model.
        evolved_model = create_model()

        evolved_model.fit(
            replay_data[feature_columns],
            replay_data[TARGET],
        )

        update_record = {
            "adaptation_round": round_number,
            "newly_observed_engines": len(
                round_engine_set
            ),
            "cumulative_adaptation_engines": len(
                previously_seen_adaptation_engines
            ),
            "newly_labelled_rows": int(
                len(newly_labelled_batch)
            ),
            "cumulative_replay_rows": int(
                len(replay_data)
            ),
            "update_method": (
                "Refit fixed-configuration RandomForest "
                "on original replay data plus newly labelled "
                "shifted adaptation batches"
            ),
            "heldout_evaluation_used_for_update": False,
        }

        update_history.append(update_record)

        print(
            f"Round {round_number}/{N_ADAPTATION_ROUNDS}: "
            f"new engines={len(round_engine_set)}, "
            f"new rows={len(newly_labelled_batch)}, "
            f"replay rows={len(replay_data)}"
        )

    if evolved_model is None:
        raise RuntimeError(
            "The adaptation procedure did not produce a model."
        )

    print("Sequential adaptation complete.")

    # --------------------------------------------------------
    # 8.8 Evaluate on untouched held-out engines
    # --------------------------------------------------------

    print("\nEvaluating frozen and evolved models...")

    final_features = final_evaluation_shifted[
        feature_columns
    ]

    frozen_predictions = frozen_model.predict(
        final_features
    )

    evolved_predictions = evolved_model.predict(
        final_features
    )

    metrics_records = []

    # Secondary analysis: all cycles.
    metrics_records.append(
        evaluate_model(
            "frozen_baseline",
            final_evaluation_shifted,
            frozen_predictions,
            "all_cycles_rows",
        )
    )

    metrics_records.append(
        evaluate_model(
            "self_evolved_after_3_updates",
            final_evaluation_shifted,
            evolved_predictions,
            "all_cycles_rows",
        )
    )

    # Primary analysis: one final cycle per held-out engine.
    final_engine_rows = final_cycle_rows(
        final_evaluation_shifted
    )

    final_engine_features = final_engine_rows[
        feature_columns
    ]

    frozen_final_predictions = frozen_model.predict(
        final_engine_features
    )

    evolved_final_predictions = evolved_model.predict(
        final_engine_features
    )

    metrics_records.append(
        evaluate_model(
            "frozen_baseline",
            final_engine_rows,
            frozen_final_predictions,
            "one_final_cycle_per_heldout_engine",
        )
    )

    metrics_records.append(
        evaluate_model(
            "self_evolved_after_3_updates",
            final_engine_rows,
            evolved_final_predictions,
            "one_final_cycle_per_heldout_engine",
        )
    )

    metrics_df = pd.DataFrame(
        metrics_records
    )

    # --------------------------------------------------------
    # 8.9 Compare frozen versus evolved performance
    # --------------------------------------------------------

    comparison_records = []

    for evaluation_unit in metrics_df[
        "evaluation_unit"
    ].unique():

        subset = metrics_df[
            metrics_df["evaluation_unit"]
            == evaluation_unit
        ]

        frozen_row = subset[
            subset["model"] == "frozen_baseline"
        ].iloc[0]

        evolved_row = subset[
            subset["model"]
            == "self_evolved_after_3_updates"
        ].iloc[0]

        for metric in [
            "RMSE",
            "MAE",
            "NASA_PHM_score_lower_is_better",
        ]:

            frozen_value = float(
                frozen_row[metric]
            )

            evolved_value = float(
                evolved_row[metric]
            )

            difference = (
                evolved_value - frozen_value
            )

            relative_change = (
                100.0 * difference / abs(frozen_value)
                if frozen_value != 0
                else None
            )

            comparison_records.append(
                {
                    "evaluation_unit": evaluation_unit,
                    "metric": metric,
                    "frozen_baseline": frozen_value,
                    "self_evolved": evolved_value,
                    "change_evolved_minus_frozen": difference,
                    "relative_change_percent_vs_frozen": (
                        relative_change
                    ),
                    "interpretation": (
                        "Negative change favours the evolved model; "
                        "positive change favours the frozen model."
                    ),
                }
            )

    comparison_df = pd.DataFrame(
        comparison_records
    )

    # --------------------------------------------------------
    # 8.10 Inspect changes in global feature importance
    #
    # Random Forest feature importance is not causal.
    # It is not a substitute for local SHAP explanations.
    # --------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "frozen_model_feature_importance": (
                frozen_model.feature_importances_
            ),
            "evolved_model_feature_importance": (
                evolved_model.feature_importances_
            ),
        }
    )

    importance_df[
        "importance_change_evolved_minus_frozen"
    ] = (
        importance_df[
            "evolved_model_feature_importance"
        ]
        - importance_df[
            "frozen_model_feature_importance"
        ]
    )

    importance_df = importance_df.sort_values(
        "evolved_model_feature_importance",
        ascending=False,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 8.11 Save all experiment outputs
    # --------------------------------------------------------

    metrics_path = (
        output_dir / "step10C_final_metrics.csv"
    )

    updates_path = (
        output_dir / "step10C_update_history.csv"
    )

    comparison_path = (
        output_dir / "step10C_frozen_vs_evolved.csv"
    )

    importance_path = (
        output_dir / "step10C_feature_importance_change.csv"
    )

    report_path = (
        output_dir / "step10C_report.json"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    pd.DataFrame(
        update_history
    ).to_csv(
        updates_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    report = {
        "experiment": (
            "controlled_self_evolution_under_synthetic_sensor_shift"
        ),
        "status": "completed",
        "dataset": str(input_path),
        "target": TARGET,
        "random_seed": RANDOM_SEED,
        "base_engine_count": len(base_engine_ids),
        "adaptation_engine_count": len(
            adaptation_engine_ids
        ),
        "final_heldout_engine_count": len(
            evaluation_engine_ids
        ),
        "adaptation_rounds": N_ADAPTATION_ROUNDS,
        "adaptation_rounds_completed": len(
            update_history
        ),
        "base_training_rows": int(
            len(base_data)
        ),
        "adaptation_rows_total": int(
            len(adaptation_shifted)
        ),
        "final_heldout_rows": int(
            len(final_evaluation_shifted)
        ),
        "features": feature_columns,
        "synthetic_shift_standard_deviation_units": (
            SHIFT_SPEC
        ),
        "shift_statistics_reference": (
            "base training engines only"
        ),
        "update_method": (
            "Refit fixed-configuration RandomForest on original "
            "replay data plus newly labelled shifted adaptation batches"
        ),
        "final_evaluation_engine_overlap_with_training_or_adaptation": (
            False
        ),
        "final_evaluation_used_for_model_updates_or_selection": (
            False
        ),
        "official_NASA_test_file_used": False,
        "claims_limitations": [
            (
                "Sensor shift is intentionally synthetic and is not "
                "evidence of real operational drift."
            ),
            (
                "The experiment does not establish performance in "
                "a deployed industrial environment."
            ),
            (
                "Rows within an engine are dependent. Final-cycle-per-"
                "engine metrics are the primary engine-level summary."
            ),
            (
                "Random Forest feature importance is not causal and "
                "does not replace local SHAP explanations."
            ),
            (
                "Self-evolution is demonstrated as a controlled "
                "feedback/update loop, not unrestricted autonomous learning."
            ),
            (
                "Improvement in one metric does not imply improvement "
                "in every metric."
            ),
        ],
        "outputs": [
            str(metrics_path),
            str(updates_path),
            str(comparison_path),
            str(importance_path),
            str(report_path),
        ],
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 8.12 Print a concise research results summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("STEP 10C RESULTS")
    print("=" * 65)

    print(
        "\nSequential model updates completed: "
        f"{len(update_history)}"
    )

    print("\nPRIMARY RESULTS: FINAL CYCLE PER HELD-OUT ENGINE")

    primary_results = metrics_df[
        metrics_df["evaluation_unit"]
        == "one_final_cycle_per_heldout_engine"
    ][
        [
            "model",
            "n_rows_or_engines",
            "RMSE",
            "MAE",
            "NASA_PHM_score_lower_is_better",
        ]
    ]

    print(
        primary_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        "\nSECONDARY RESULTS: ALL HELD-OUT CYCLES"
    )

    secondary_results = metrics_df[
        metrics_df["evaluation_unit"]
        == "all_cycles_rows"
    ][
        [
            "model",
            "n_rows_or_engines",
            "RMSE",
            "MAE",
            "NASA_PHM_score_lower_is_better",
        ]
    ]

    print(
        secondary_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nFROZEN VERSUS EVOLVED COMPARISON")

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nOUTPUT FILES")

    for path in [
        metrics_path,
        updates_path,
        comparison_path,
        importance_path,
        report_path,
    ]:
        print(path)

    print("\nExperiment finished.")
    print(
        "Report the measured results honestly. "
        "The evolved model may improve some metrics and worsen others."
    )


# ============================================================
# 9. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )
        raise