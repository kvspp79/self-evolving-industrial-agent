
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# STEP 8.5 — CONTROLLED RUL CALIBRATION
#
# Dataset: NASA C-MAPSS FD001
#
# Calibration is fitted only on out-of-fold predictions from
# training engines. Validation labels are evaluation-only.
# Test data is not used.
#
# This is a calibration experiment, not online learning or
# automatic deployment.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "outputs" / "step01"
STEP04_DIR = ROOT / "outputs" / "step04"
OUTPUT_DIR = ROOT / "outputs" / "step08_5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSET = "FD001"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 250
N_OOF_SPLITS = 5

ROLLING_WINDOWS = (5, 10, 20)
SENSORS = [f"sensor_{i:02d}" for i in range(1, 22)]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
BASE_FEATURES = ["cycle"] + SETTINGS + SENSORS


def make_model():
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def nasa_score(y_true, y_pred):
    error = (
        np.asarray(y_pred, dtype=float)
        - np.asarray(y_true, dtype=float)
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
        "n_rows": int(len(y_true)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "NASA_score": nasa_score(y_true, y_pred),
        "mean_error_predicted_minus_actual": float(
            np.mean(y_pred - y_true)
        ),
    }


def add_temporal_features(df):
    result = (
        df.sort_values(["unit_id", "cycle"])
        .copy()
        .reset_index(drop=True)
    )

    generated = {}
    feature_names = list(BASE_FEATURES)

    for sensor in SENSORS:
        grouped = result.groupby("unit_id", sort=False)[sensor]

        diff_name = f"{sensor}_diff1"
        generated[diff_name] = grouped.diff().fillna(0.0)
        feature_names.append(diff_name)

        for window in ROLLING_WINDOWS:
            mean_name = f"{sensor}_mean_{window}"
            std_name = f"{sensor}_std_{window}"

            generated[mean_name] = grouped.transform(
                lambda s, w=window: s.rolling(
                    window=w, min_periods=1
                ).mean()
            )

            generated[std_name] = grouped.transform(
                lambda s, w=window: s.rolling(
                    window=w, min_periods=1
                ).std(ddof=0)
            )

            feature_names.extend([mean_name, std_name])

        slope_name = f"{sensor}_slope10"
        generated[slope_name] = (
            grouped.diff(9) / 9.0
        ).fillna(0.0)
        feature_names.append(slope_name)

    temporal_frame = pd.DataFrame(
        generated, index=result.index
    )
    result = pd.concat(
        [result, temporal_frame], axis=1
    ).copy()

    values = result[feature_names].replace(
        [np.inf, -np.inf], np.nan
    )

    if values.isna().any().any():
        missing = values.columns[
            values.isna().any()
        ].tolist()
        raise ValueError(
            f"Non-finite feature values: {missing[:10]}"
        )

    return result, feature_names


def fit_oof_predictions(data, features, n_splits):
    """
    Generate out-of-fold predictions with engines grouped.
    Every prediction is made by a model that did not train
    on that prediction's engine.
    """
    engine_count = data["unit_id"].nunique()

    if engine_count < n_splits:
        raise ValueError(
            f"Need at least {n_splits} training engines; "
            f"found {engine_count}."
        )

    splitter = GroupKFold(n_splits=n_splits)

    oof_predictions = np.full(len(data), np.nan, dtype=float)
    fold_ids = np.full(len(data), -1, dtype=int)

    for fold, (fit_idx, holdout_idx) in enumerate(
        splitter.split(
            data[features],
            data["RUL"],
            groups=data["unit_id"],
        ),
        start=1,
    ):
        fit_engines = set(
            data.iloc[fit_idx]["unit_id"].unique()
        )
        holdout_engines = set(
            data.iloc[holdout_idx]["unit_id"].unique()
        )

        if fit_engines.intersection(holdout_engines):
            raise RuntimeError(
                f"Engine leakage detected in OOF fold {fold}."
            )

        model = make_model()
        model.fit(
            data.iloc[fit_idx][features],
            data.iloc[fit_idx]["RUL"],
        )

        oof_predictions[holdout_idx] = np.maximum(
            model.predict(data.iloc[holdout_idx][features]),
            0.0,
        )
        fold_ids[holdout_idx] = fold

        print(
            f"  Fold {fold}/{n_splits}: "
            f"fit engines={len(fit_engines)}, "
            f"held-out engines={len(holdout_engines)}"
        )

    if np.isnan(oof_predictions).any() or (fold_ids < 0).any():
        raise RuntimeError(
            "Some rows did not receive out-of-fold predictions."
        )

    return oof_predictions, fold_ids


def fit_calibrator(oof_predictions, actual_rul):
    """
    Fit a one-variable affine calibration:
        calibrated_RUL = intercept + slope * predicted_RUL

    This is intentionally simple and interpretable.
    Predictions are clipped to non-negative RUL.
    """
    calibrator = LinearRegression()
    calibrator.fit(
        np.asarray(oof_predictions).reshape(-1, 1),
        np.asarray(actual_rul, dtype=float),
    )

    return calibrator


def main():
    print("=" * 72)
    print("NASA C-MAPSS FD001 — STEP 8.5")
    print("CONTROLLED TRAINING-ONLY RUL CALIBRATION")
    print("=" * 72)

    train_path = (
        INPUT_DIR / f"train_{SUBSET}_processed.csv"
    )
    validation_path = (
        STEP04_DIR / f"validation_predictions_{SUBSET}.csv"
    )

    for path in [train_path, validation_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)

    required_train = (
        ["unit_id", "cycle", "RUL"] + SETTINGS + SENSORS
    )
    missing = [
        col for col in required_train
        if col not in train.columns
    ]
    if missing:
        raise ValueError(
            f"Training data is missing columns: {missing}"
        )

    required_validation = [
        "unit_id",
        "cycle",
        "actual_RUL",
        "baseline_predicted_RUL",
        "temporal_predicted_RUL",
    ]
    missing = [
        col for col in required_validation
        if col not in validation.columns
    ]
    if missing:
        raise ValueError(
            f"Validation prediction file is missing columns: {missing}"
        )

    # --------------------------------------------------------
    # 1. Recreate Step 4 features and the exact engine split.
    # --------------------------------------------------------
    print("\nGenerating Step 4-compatible temporal features...")
    featured_train, temporal_features = add_temporal_features(train)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    fit_idx, val_idx = next(
        splitter.split(
            featured_train[BASE_FEATURES],
            featured_train["RUL"],
            groups=featured_train["unit_id"],
        )
    )

    fit_data = (
        featured_train.iloc[fit_idx]
        .copy()
        .reset_index(drop=True)
    )
    split_validation = (
        featured_train.iloc[val_idx]
        .copy()
        .reset_index(drop=True)
    )

    fit_engines = set(fit_data["unit_id"].unique())
    validation_engines = set(
        split_validation["unit_id"].unique()
    )

    if fit_engines.intersection(validation_engines):
        raise RuntimeError(
            "Training/validation engine overlap detected."
        )

    # Verify that the saved Step 4 validation predictions match
    # the validation engine/cycle rows reconstructed from the split.
    saved_keys = validation[
        ["unit_id", "cycle"]
    ].copy()

    expected_keys = split_validation[
        ["unit_id", "cycle"]
    ].copy()

    saved_keys["unit_id"] = saved_keys["unit_id"].astype(int)
    saved_keys["cycle"] = saved_keys["cycle"].astype(int)
    expected_keys["unit_id"] = expected_keys["unit_id"].astype(int)
    expected_keys["cycle"] = expected_keys["cycle"].astype(int)

    saved_key_set = set(
        map(tuple, saved_keys.to_numpy())
    )
    expected_key_set = set(
        map(tuple, expected_keys.to_numpy())
    )

    if saved_key_set != expected_key_set:
        raise ValueError(
            "Saved Step 4 validation rows do not match the "
            "reconstructed GroupShuffleSplit. Stop rather than "
            "calibrating against a potentially different split."
        )

    # Confirm labels agree with the reconstructed split.
    expected_labels = split_validation[
        ["unit_id", "cycle", "RUL"]
    ].rename(columns={"RUL": "expected_RUL"})

    merged_labels = validation.merge(
        expected_labels,
        on=["unit_id", "cycle"],
        how="left",
        validate="one_to_one",
    )

    if merged_labels["expected_RUL"].isna().any():
        raise ValueError(
            "Some validation rows could not be matched to training data."
        )

    if not np.allclose(
        merged_labels["actual_RUL"].to_numpy(dtype=float),
        merged_labels["expected_RUL"].to_numpy(dtype=float),
    ):
        raise ValueError(
            "Saved validation labels differ from reconstructed labels."
        )

    # Use a stable row order and ensure unique engine-cycle pairs.
    if validation.duplicated(["unit_id", "cycle"]).any():
        raise ValueError(
            "Duplicate unit_id/cycle rows in validation predictions."
        )

    print(f"Training engines for OOF: {len(fit_engines)}")
    print(f"Validation engines: {len(validation_engines)}")
    print(f"Training rows for OOF: {len(fit_data):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Baseline feature count: {len(BASE_FEATURES)}")
    print(f"Temporal feature count: {len(temporal_features)}")

    # --------------------------------------------------------
    # 2. Generate OOF predictions on training engines only.
    # --------------------------------------------------------
    print("\nGenerating baseline out-of-fold predictions...")
    baseline_oof, baseline_folds = fit_oof_predictions(
        fit_data,
        BASE_FEATURES,
        N_OOF_SPLITS,
    )

    print("\nGenerating temporal out-of-fold predictions...")
    temporal_oof, temporal_folds = fit_oof_predictions(
        fit_data,
        temporal_features,
        N_OOF_SPLITS,
    )

    oof = pd.DataFrame({
        "unit_id": fit_data["unit_id"].to_numpy(),
        "cycle": fit_data["cycle"].to_numpy(),
        "actual_RUL": fit_data["RUL"].to_numpy(dtype=float),
        "baseline_oof_predicted_RUL": baseline_oof,
        "baseline_oof_fold": baseline_folds,
        "temporal_oof_predicted_RUL": temporal_oof,
        "temporal_oof_fold": temporal_folds,
    })

    oof.to_csv(
        OUTPUT_DIR / "training_oof_predictions_FD001.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Fit calibrators using OOF predictions only.
    # --------------------------------------------------------
    baseline_calibrator = fit_calibrator(
        baseline_oof,
        fit_data["RUL"].to_numpy(dtype=float),
    )

    temporal_calibrator = fit_calibrator(
        temporal_oof,
        fit_data["RUL"].to_numpy(dtype=float),
    )

    # --------------------------------------------------------
    # 4. Apply frozen calibrators to held-out validation rows.
    # No validation labels are used to fit the calibrators.
    # --------------------------------------------------------
    validation = validation.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    actual = validation["actual_RUL"].to_numpy(dtype=float)
    baseline_raw = np.maximum(
        validation["baseline_predicted_RUL"].to_numpy(dtype=float),
        0.0,
    )
    temporal_raw = np.maximum(
        validation["temporal_predicted_RUL"].to_numpy(dtype=float),
        0.0,
    )

    baseline_calibrated = np.maximum(
        baseline_calibrator.predict(
            baseline_raw.reshape(-1, 1)
        ),
        0.0,
    )

    temporal_calibrated = np.maximum(
        temporal_calibrator.predict(
            temporal_raw.reshape(-1, 1)
        ),
        0.0,
    )

    # --------------------------------------------------------
    # 5. Compare all four validation prediction variants.
    # --------------------------------------------------------
    variants = {
        "baseline_raw": baseline_raw,
        "baseline_calibrated": baseline_calibrated,
        "temporal_raw": temporal_raw,
        "temporal_calibrated": temporal_calibrated,
    }

    metric_rows = []
    for name, predictions in variants.items():
        row = {"model_variant": name}
        row.update(calculate_metrics(actual, predictions))
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(
        OUTPUT_DIR / "validation_calibration_metrics_FD001.csv",
        index=False,
    )

    print("\nVALIDATION METRICS")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # 6. Save calibration parameters and predictions.
    # --------------------------------------------------------
    parameters = {
        "dataset": "NASA C-MAPSS FD001",
        "method": "one-dimensional linear calibration",
        "fit_data": "training-partition out-of-fold predictions only",
        "random_state": RANDOM_STATE,
        "oof_splits": N_OOF_SPLITS,
        "training_engines": int(len(fit_engines)),
        "validation_engines": int(len(validation_engines)),
        "baseline_calibrator": {
            "intercept": float(baseline_calibrator.intercept_),
            "slope": float(baseline_calibrator.coef_[0]),
        },
        "temporal_calibrator": {
            "intercept": float(temporal_calibrator.intercept_),
            "slope": float(temporal_calibrator.coef_[0]),
        },
        "limitations": [
            "Calibration is a candidate post-processing method, not model retraining.",
            "OOF predictions are generated at the engine level.",
            "Validation labels are evaluation-only.",
            "The calibration may not generalize to other datasets or subsets.",
            "Only FD001 is assessed in this step.",
            "No test labels are read or used.",
            "No maintenance policy is modified or deployed.",
        ],
    }

    params_path = (
        OUTPUT_DIR / "calibration_parameters_FD001.json"
    )
    params_path.write_text(
        json.dumps(parameters, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    prediction_output = validation[
        ["unit_id", "cycle", "actual_RUL"]
    ].copy()

    prediction_output["baseline_raw_RUL"] = baseline_raw
    prediction_output["baseline_calibrated_RUL"] = baseline_calibrated
    prediction_output["temporal_raw_RUL"] = temporal_raw
    prediction_output["temporal_calibrated_RUL"] = temporal_calibrated

    prediction_output.to_csv(
        OUTPUT_DIR / "validation_calibrated_predictions_FD001.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. Visualize validation metrics.
    # --------------------------------------------------------
    metric_names = ["RMSE", "MAE", "NASA_score"]
    labels = [
        "Baseline raw",
        "Baseline calibrated",
        "Temporal raw",
        "Temporal calibrated",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, metric_names):
        values = [
            float(
                metrics_df.loc[
                    metrics_df["model_variant"] == name,
                    metric,
                ].iloc[0]
            )
            for name in variants
        ]

        ax.bar(labels, values)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Step 8.5 — Raw vs Calibrated Validation Predictions"
    )
    fig.tight_layout()

    plot_path = (
        OUTPUT_DIR / "calibration_validation_comparison_FD001.png"
    )
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 8. Record results without automatically selecting or
    # deploying a new model or maintenance policy.
    # --------------------------------------------------------
    report = {
        "step": "8.5",
        "dataset": "NASA C-MAPSS FD001",
        "method": "training-only out-of-fold linear calibration",
        "training_oof_rows": int(len(oof)),
        "training_engines": int(len(fit_engines)),
        "validation_rows": int(len(validation)),
        "validation_engines": int(len(validation_engines)),
        "oof_splits": N_OOF_SPLITS,
        "calibration_parameters": parameters,
        "validation_metrics": metric_rows,
        "interpretation": (
            "Compare calibrated variants with their corresponding raw "
            "variants. Calibration is a candidate only; do not assume "
            "it improves performance unless the validation metrics "
            "support that conclusion."
        ),
        "test_set_used": False,
        "model_retrained_for_deployment": False,
        "policy_changed": False,
    }

    report_path = (
        OUTPUT_DIR / "step08_5_FD001_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nSTEP 8.5 COMPLETED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"OOF predictions: {OUTPUT_DIR / 'training_oof_predictions_FD001.csv'}")
    print(f"Metrics: {OUTPUT_DIR / 'validation_calibration_metrics_FD001.csv'}")
    print(f"Predictions: {OUTPUT_DIR / 'validation_calibrated_predictions_FD001.csv'}")
    print(f"Parameters: {params_path}")
    print(f"Plot: {plot_path}")
    print(f"Report: {report_path}")
    print("\nNo test data was used.")
    print("No maintenance policy was modified or deployed.")


if __name__ == "__main__":
    main()
