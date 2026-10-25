
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# NASA C-MAPSS — STEP 4
# TEMPORAL FEATURE ENGINEERING AND CONTROLLED COMPARISON
# Subset: FD001
#
# Model A: original sensor and operating-condition features
# Model B: original features + past-only temporal features
#
# Both models use the same engine-level validation split.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "outputs" / "step01"
OUTPUT_DIR = ROOT / "outputs" / "step04"
MODEL_DIR = OUTPUT_DIR / "models"

SUBSET = "FD001"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 250

ROLLING_WINDOWS = (5, 10, 20)

SENSORS = [f"sensor_{i:02d}" for i in range(1, 22)]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]

BASE_FEATURES = ["cycle"] + SETTINGS + SENSORS


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

def make_model():
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ------------------------------------------------------------
# NASA ASYMMETRIC SCORING FUNCTION
# Lower score is better.
# ------------------------------------------------------------

def nasa_score(y_true, y_pred):
    error = np.asarray(y_pred, dtype=float) - np.asarray(
        y_true, dtype=float
    )

    exponent = np.where(
        error < 0,
        -error / 13.0,
        error / 10.0,
    )

    return float(
        np.sum(np.exp(np.clip(exponent, 0, 50)) - 1.0)
    )


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "n": int(len(y_true)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "NASA_score": nasa_score(y_true, y_pred),
    }


# ------------------------------------------------------------
# TEMPORAL FEATURE ENGINEERING
#
# All rolling statistics use current/past observations only.
# Features are calculated separately for each engine.
# New columns are assembled together to avoid fragmentation.
# ------------------------------------------------------------

def add_temporal_features(df):
    result = (
        df.sort_values(["unit_id", "cycle"])
        .copy()
        .reset_index(drop=True)
    )

    generated = {}
    feature_names = list(BASE_FEATURES)

    for sensor in SENSORS:
        values = result[sensor]
        grouped = result.groupby("unit_id", sort=False)[sensor]

        # One-cycle change.
        diff_name = f"{sensor}_diff1"
        generated[diff_name] = grouped.diff().fillna(0.0)
        feature_names.append(diff_name)

        # Trailing rolling means and standard deviations.
        for window in ROLLING_WINDOWS:
            mean_name = f"{sensor}_mean_{window}"
            std_name = f"{sensor}_std_{window}"

            rolling = grouped.transform(
                lambda series, w=window: series.rolling(
                    window=w,
                    min_periods=1,
                ).mean()
            )
            generated[mean_name] = rolling

            rolling_std = grouped.transform(
                lambda series, w=window: series.rolling(
                    window=w,
                    min_periods=1,
                ).std(ddof=0)
            )
            generated[std_name] = rolling_std

            feature_names.extend([mean_name, std_name])

        # Change per cycle over a nine-cycle difference span.
        # When fewer than 10 observations exist, use zero.
        slope_name = f"{sensor}_slope10"
        generated[slope_name] = (
            grouped.diff(9) / 9.0
        ).fillna(0.0)
        feature_names.append(slope_name)

    temporal_frame = pd.DataFrame(generated, index=result.index)

    result = pd.concat([result, temporal_frame], axis=1).copy()

    feature_values = result[feature_names].replace(
        [np.inf, -np.inf], np.nan
    )

    if feature_values.isna().any().any():
        missing = feature_values.columns[
            feature_values.isna().any()
        ].tolist()
        raise ValueError(
            f"Missing/non-finite feature values found: {missing[:10]}"
        )

    return result, feature_names


# ------------------------------------------------------------
# FIT AND VALIDATE
# ------------------------------------------------------------

def train_and_evaluate(train_fit, validation, features):
    model = make_model()

    model.fit(
        train_fit[features],
        train_fit["RUL"],
    )

    predictions = np.maximum(
        model.predict(validation[features]),
        0.0,
    )

    metrics = calculate_metrics(
        validation["RUL"],
        predictions,
    )

    return model, predictions, metrics


# ------------------------------------------------------------
# PLOT METRICS
# ------------------------------------------------------------

def plot_metric_comparison(
    baseline_metrics,
    temporal_metrics,
    title,
    output_path,
):
    # Internal dictionary keys must match exactly.
    metric_keys = ["RMSE", "MAE", "NASA_score"]
    display_labels = ["RMSE", "MAE", "NASA Score"]

    baseline_values = [
        baseline_metrics[key] for key in metric_keys
    ]
    temporal_values = [
        temporal_metrics[key] for key in metric_keys
    ]

    x = np.arange(len(metric_keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Baseline",
    )
    ax.bar(
        x + width / 2,
        temporal_values,
        width,
        label="Temporal features",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_yscale("log")
    ax.set_ylabel("Metric value (log scale)")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------------------------

def main():
    print("=" * 68)
    print("NASA C-MAPSS — STEP 4: TEMPORAL FEATURE ENGINEERING")
    print(f"Subset: {SUBSET}")
    print("=" * 68)

    train_path = INPUT_DIR / f"train_{SUBSET}_processed.csv"
    test_path = INPUT_DIR / f"test_{SUBSET}_processed.csv"
    rul_path = INPUT_DIR / f"RUL_{SUBSET}_reference.csv"

    for path in [train_path, test_path, rul_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found:\n{path}\n"
                "Check Step 1 output filenames."
            )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    rul_reference = pd.read_csv(rul_path)

    required_train = set(
        ["unit_id", "cycle", "RUL"] + SETTINGS + SENSORS
    )
    required_test = set(
        ["unit_id", "cycle"] + SETTINGS + SENSORS
    )

    missing_train = required_train - set(train.columns)
    missing_test = required_test - set(test.columns)

    if missing_train:
        raise ValueError(
            f"Training data missing columns: {sorted(missing_train)}"
        )

    if missing_test:
        raise ValueError(
            f"Test data missing columns: {sorted(missing_test)}"
        )

    if "unit_id" not in rul_reference.columns:
        raise ValueError("RUL reference is missing unit_id.")

    if "RUL_after_last_test_cycle" not in rul_reference.columns:
        raise ValueError(
            "RUL reference is missing RUL_after_last_test_cycle."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Generate features.
    # --------------------------------------------------------

    print("\nGenerating past-only temporal features...")

    train_features, temporal_features = add_temporal_features(train)
    test_features, test_feature_names = add_temporal_features(test)

    if temporal_features != test_feature_names:
        raise ValueError(
            "Training and test feature lists do not match."
        )

    print(f"Original features: {len(BASE_FEATURES)}")
    print(
        "Temporal features added:",
        len(temporal_features) - len(BASE_FEATURES),
    )
    print(f"Total enhanced features: {len(temporal_features)}")

    # --------------------------------------------------------
    # 2. Engine-level validation split.
    # No engine can appear in both partitions.
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    fit_idx, val_idx = next(
        splitter.split(
            train_features[BASE_FEATURES],
            train_features["RUL"],
            groups=train_features["unit_id"],
        )
    )

    fit_data = train_features.iloc[fit_idx].copy()
    val_data = train_features.iloc[val_idx].copy()

    fit_engines = set(fit_data["unit_id"].unique())
    val_engines = set(val_data["unit_id"].unique())

    if fit_engines.intersection(val_engines):
        raise RuntimeError("Engine leakage detected in validation split.")

    print("\nENGINE-LEVEL VALIDATION")
    print(f"Training engines: {len(fit_engines)}")
    print(f"Validation engines: {len(val_engines)}")
    print(f"Training rows: {len(fit_data):,}")
    print(f"Validation rows: {len(val_data):,}")

    # --------------------------------------------------------
    # 3. Model A: baseline.
    # --------------------------------------------------------

    print("\nTraining Model A: baseline features...")

    baseline_model, baseline_val_pred, baseline_metrics = (
        train_and_evaluate(
            fit_data,
            val_data,
            BASE_FEATURES,
        )
    )

    # --------------------------------------------------------
    # 4. Model B: baseline + temporal features.
    # --------------------------------------------------------

    print("Training Model B: temporal features...")

    temporal_model, temporal_val_pred, temporal_metrics = (
        train_and_evaluate(
            fit_data,
            val_data,
            temporal_features,
        )
    )

    # --------------------------------------------------------
    # 5. Final-cycle test evaluation.
    #
    # The test RUL reference represents RUL after each engine's
    # last observed test cycle. It is not the RUL at every row.
    # --------------------------------------------------------

    test_last = (
        test_features.sort_values(["unit_id", "cycle"])
        .groupby("unit_id", sort=True)
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    reference = (
        rul_reference.sort_values("unit_id")
        .reset_index(drop=True)
    )

    if test_last["unit_id"].tolist() != reference["unit_id"].tolist():
        raise ValueError(
            "Test engine IDs do not match the RUL reference order."
        )

    actual_test_rul = reference[
        "RUL_after_last_test_cycle"
    ].to_numpy(dtype=float)

    baseline_test_pred = np.maximum(
        baseline_model.predict(test_last[BASE_FEATURES]),
        0.0,
    )

    temporal_test_pred = np.maximum(
        temporal_model.predict(test_last[temporal_features]),
        0.0,
    )

    baseline_test_metrics = calculate_metrics(
        actual_test_rul,
        baseline_test_pred,
    )

    temporal_test_metrics = calculate_metrics(
        actual_test_rul,
        temporal_test_pred,
    )

    # --------------------------------------------------------
    # 6. Compare metrics.
    # Positive improvement means the metric decreased.
    # --------------------------------------------------------

    comparison_rows = []

    evaluations = [
        (
            "validation_all_cycles",
            baseline_metrics,
            temporal_metrics,
        ),
        (
            "test_final_cycle",
            baseline_test_metrics,
            temporal_test_metrics,
        ),
    ]

    for evaluation_name, base, enhanced in evaluations:
        for metric in ["RMSE", "MAE", "NASA_score"]:
            base_value = float(base[metric])
            enhanced_value = float(enhanced[metric])

            improvement = (
                (base_value - enhanced_value) / base_value * 100.0
                if base_value != 0
                else 0.0
            )

            comparison_rows.append({
                "evaluation": evaluation_name,
                "metric": metric,
                "baseline": base_value,
                "temporal_model": enhanced_value,
                "improvement_percent": improvement,
            })

    comparison = pd.DataFrame(comparison_rows)

    # --------------------------------------------------------
    # 7. Save predictions.
    # --------------------------------------------------------

    validation_predictions = pd.DataFrame({
        "unit_id": val_data["unit_id"].to_numpy(),
        "cycle": val_data["cycle"].to_numpy(),
        "actual_RUL": val_data["RUL"].to_numpy(),
        "baseline_predicted_RUL": baseline_val_pred,
        "temporal_predicted_RUL": temporal_val_pred,
    })

    validation_predictions["baseline_error"] = (
        validation_predictions["baseline_predicted_RUL"]
        - validation_predictions["actual_RUL"]
    )

    validation_predictions["temporal_error"] = (
        validation_predictions["temporal_predicted_RUL"]
        - validation_predictions["actual_RUL"]
    )

    test_predictions = pd.DataFrame({
        "unit_id": test_last["unit_id"].to_numpy(),
        "last_observed_cycle": test_last["cycle"].to_numpy(),
        "actual_RUL": actual_test_rul,
        "baseline_predicted_RUL": baseline_test_pred,
        "temporal_predicted_RUL": temporal_test_pred,
    })

    test_predictions["baseline_error"] = (
        test_predictions["baseline_predicted_RUL"]
        - test_predictions["actual_RUL"]
    )

    test_predictions["temporal_error"] = (
        test_predictions["temporal_predicted_RUL"]
        - test_predictions["actual_RUL"]
    )

    validation_predictions.to_csv(
        OUTPUT_DIR / f"validation_predictions_{SUBSET}.csv",
        index=False,
    )

    test_predictions.to_csv(
        OUTPUT_DIR / f"test_predictions_{SUBSET}.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / f"model_comparison_{SUBSET}.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 8. Save models and feature importance.
    # --------------------------------------------------------

    joblib.dump(
        baseline_model,
        MODEL_DIR / f"baseline_same_split_{SUBSET}.joblib",
    )

    joblib.dump(
        temporal_model,
        MODEL_DIR / f"temporal_random_forest_{SUBSET}.joblib",
    )

    importance = pd.DataFrame({
        "feature": temporal_features,
        "importance": temporal_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        OUTPUT_DIR / f"temporal_feature_importance_{SUBSET}.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 9. Save comparison plots.
    # --------------------------------------------------------

    plot_metric_comparison(
        baseline_metrics,
        temporal_metrics,
        "Model Comparison — Validation, All Cycles",
        OUTPUT_DIR / "validation_metric_comparison.png",
    )

    plot_metric_comparison(
        baseline_test_metrics,
        temporal_test_metrics,
        "Model Comparison — Test, Final Cycle",
        OUTPUT_DIR / "test_metric_comparison.png",
    )

    top_features = importance.head(20).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        top_features["feature"],
        top_features["importance"],
    )
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("Top 20 Temporal Model Features — FD001")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"top_temporal_features_{SUBSET}.png",
        dpi=160,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 10. Research report.
    # --------------------------------------------------------

    report = {
        "dataset": "NASA C-MAPSS",
        "subset": SUBSET,
        "experiment": (
            "Baseline features versus baseline plus past-only "
            "temporal sensor features"
        ),
        "random_state": RANDOM_STATE,
        "model_configuration": {
            "model": "RandomForestRegressor",
            "n_estimators": N_ESTIMATORS,
            "max_depth": 18,
            "min_samples_leaf": 2,
            "max_features": 0.8,
        },
        "feature_design": {
            "baseline_feature_count": len(BASE_FEATURES),
            "temporal_feature_count": (
                len(temporal_features) - len(BASE_FEATURES)
            ),
            "total_enhanced_feature_count": len(temporal_features),
            "rolling_windows": list(ROLLING_WINDOWS),
            "features": [
                "one-cycle sensor difference",
                "trailing rolling means",
                "trailing rolling standard deviations",
                "nine-cycle-span difference slope",
            ],
            "future_observations_used": False,
        },
        "validation_split": {
            "method": "GroupShuffleSplit by engine",
            "random_state": RANDOM_STATE,
            "training_engines": len(fit_engines),
            "validation_engines": len(val_engines),
            "training_rows": len(fit_data),
            "validation_rows": len(val_data),
        },
        "baseline_validation_all_cycles": baseline_metrics,
        "temporal_validation_all_cycles": temporal_metrics,
        "baseline_test_final_cycle": baseline_test_metrics,
        "temporal_test_final_cycle": temporal_test_metrics,
        "comparison": comparison.to_dict(orient="records"),
        "top_20_features": importance.head(20).to_dict(
            orient="records"
        ),
        "interpretation_notes": [
            "Validation engines are held out from model fitting.",
            "Validation metrics cover every row of held-out engines.",
            "Test metrics cover one final observed row per test engine.",
            "Test data is evaluated, not used for model selection.",
            "Feature importance is predictive, not causal.",
            "Results apply to FD001 and require further subset testing.",
            "NASA score is an asymmetric error score; lower is better.",
        ],
    }

    report_path = OUTPUT_DIR / f"step04_{SUBSET}_report.json"

    report_path.write_text(
        json.dumps(report, indent=4, allow_nan=False),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 11. Print results.
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("VALIDATION RESULTS — ALL CYCLES")
    print("=" * 68)
    print(
        pd.DataFrame({
            "Baseline": baseline_metrics,
            "Temporal": temporal_metrics,
        }).to_string()
    )

    print("\n" + "=" * 68)
    print("TEST RESULTS — FINAL CYCLE OF EACH ENGINE")
    print("=" * 68)
    print(
        pd.DataFrame({
            "Baseline": baseline_test_metrics,
            "Temporal": temporal_test_metrics,
        }).to_string()
    )

    print("\n" + "=" * 68)
    print("METRIC COMPARISON")
    print("Positive improvement (%) means the metric decreased.")
    print("=" * 68)
    print(comparison.to_string(index=False))

    print("\nTOP 10 TEMPORAL MODEL FEATURES")
    print(importance.head(10).to_string(index=False))

    print("\nSTEP 4 COMPLETED SUCCESSFULLY")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Report: {report_path}")
    print("Models, predictions, metrics, and plots saved.")


if __name__ == "__main__":
    main()
