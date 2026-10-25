
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# STEP 2: BASELINE REMAINING USEFUL LIFE (RUL) PREDICTION
# Dataset: NASA C-MAPSS FD001
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "outputs" / "step01"
OUTPUT_DIR = ROOT / "outputs" / "step02"
MODEL_DIR = OUTPUT_DIR / "models"

SUBSET = "FD001"
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nasa_score(y_true, y_pred):
    """
    Standard asymmetric NASA C-MAPSS score.

    Prediction error = predicted RUL - actual RUL.
    Overestimating RUL receives the steeper penalty.
    Lower scores are better.
    """
    error = np.asarray(y_pred) - np.asarray(y_true)

    penalties = np.where(
        error < 0,
        np.exp(np.clip(-error / 13.0, 0, 50)) - 1,
        np.exp(np.clip(error / 10.0, 0, 50)) - 1,
    )

    return float(np.sum(penalties))


def evaluate(y_true, y_pred):
    return {
        "rows_evaluated": int(len(y_true)),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE": round(
            float(mean_absolute_error(y_true, y_pred)), 4
        ),
        "NASA_score": round(nasa_score(y_true, y_pred), 4),
    }


def main():
    print("=" * 65)
    print("NASA C-MAPSS — STEP 2: BASELINE RUL PREDICTION")
    print(f"Subset: {SUBSET}")
    print("=" * 65)

    train_path = INPUT_DIR / f"train_{SUBSET}_processed.csv"
    test_path = INPUT_DIR / f"test_{SUBSET}_processed.csv"
    rul_path = INPUT_DIR / f"RUL_{SUBSET}_reference.csv"

    for path in [train_path, test_path, rul_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run Step 1 successfully before running Step 2."
            )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    rul_reference = pd.read_csv(rul_path)

    sensor_columns = [
        f"sensor_{i:02d}" for i in range(1, 22)
    ]
    setting_columns = [
        f"setting_{i}" for i in range(1, 4)
    ]

    # Cycle is included as the engine's observed operating age.
    feature_columns = (
        ["cycle"] + setting_columns + sensor_columns
    )

    required_train = feature_columns + ["unit_id", "RUL"]
    required_test = feature_columns + ["unit_id", "RUL"]

    if not set(required_train).issubset(train.columns):
        raise ValueError("Training CSV is missing required columns.")

    if not set(required_test).issubset(test.columns):
        raise ValueError("Test CSV is missing required columns.")

    if train[feature_columns + ["RUL"]].isna().any().any():
        raise ValueError("Missing values found in training features or RUL.")

    if test[feature_columns].isna().any().any():
        raise ValueError("Missing values found in test features.")

    # --------------------------------------------------------
    # 1. Split by engine, not by individual rows.
    # This prevents cycles from the same engine appearing in
    # both training and validation sets.
    # --------------------------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    fit_idx, val_idx = next(
        splitter.split(
            train[feature_columns],
            train["RUL"],
            groups=train["unit_id"],
        )
    )

    fit_data = train.iloc[fit_idx].copy()
    val_data = train.iloc[val_idx].copy()

    print("\nENGINE-LEVEL VALIDATION SPLIT")
    print(f"Training engines:   {fit_data['unit_id'].nunique()}")
    print(f"Validation engines: {val_data['unit_id'].nunique()}")
    print(f"Training rows:      {len(fit_data):,}")
    print(f"Validation rows:    {len(val_data):,}")

    # --------------------------------------------------------
    # 2. Train a reproducible baseline model.
    # --------------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining Random Forest baseline...")
    model.fit(
        fit_data[feature_columns],
        fit_data["RUL"],
    )

    # --------------------------------------------------------
    # 3. Evaluate on all cycles of held-out training engines.
    # These are validation results, not official test results.
    # --------------------------------------------------------
    val_predictions = np.maximum(
        model.predict(val_data[feature_columns]), 0
    )

    validation_results = evaluate(
        val_data["RUL"].to_numpy(),
        val_predictions,
    )

    validation_output = val_data[
        ["unit_id", "cycle", "RUL"]
    ].copy()
    validation_output["predicted_RUL"] = val_predictions
    validation_output["prediction_error"] = (
        validation_output["predicted_RUL"]
        - validation_output["RUL"]
    )

    # --------------------------------------------------------
    # 4. Refit on all training engines.
    # --------------------------------------------------------
    final_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nRefitting baseline on all training engines...")
    final_model.fit(
        train[feature_columns],
        train["RUL"],
    )

    # --------------------------------------------------------
    # 5. Official-style test evaluation:
    # predict RUL at each test engine's final observed cycle.
    # --------------------------------------------------------
    test_last_rows = (
        test.sort_values(["unit_id", "cycle"])
        .groupby("unit_id", sort=True)
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    reference = (
        rul_reference
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    if test_last_rows["unit_id"].tolist() != reference["unit_id"].tolist():
        raise ValueError(
            "Test engine IDs do not match the RUL reference IDs."
        )

    test_predictions = np.maximum(
        final_model.predict(test_last_rows[feature_columns]), 0
    )

    actual_test_rul = reference[
        "RUL_after_last_test_cycle"
    ].to_numpy()

    test_results = evaluate(actual_test_rul, test_predictions)

    test_output = pd.DataFrame({
        "unit_id": test_last_rows["unit_id"].to_numpy(),
        "last_observed_cycle": test_last_rows["cycle"].to_numpy(),
        "actual_RUL": actual_test_rul,
        "predicted_RUL": test_predictions,
    })

    test_output["prediction_error"] = (
        test_output["predicted_RUL"] - test_output["actual_RUL"]
    )

    # --------------------------------------------------------
    # 6. Save model, predictions, metrics, and feature importance.
    # --------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"random_forest_{SUBSET}.joblib"
    validation_csv = OUTPUT_DIR / f"validation_predictions_{SUBSET}.csv"
    test_csv = OUTPUT_DIR / f"test_predictions_{SUBSET}.csv"
    importance_csv = OUTPUT_DIR / f"feature_importance_{SUBSET}.csv"
    report_path = OUTPUT_DIR / f"step02_{SUBSET}_report.json"

    joblib.dump(final_model, model_path)

    validation_output.to_csv(validation_csv, index=False)
    test_output.to_csv(test_csv, index=False)

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(importance_csv, index=False)

    report = {
        "dataset": "NASA C-MAPSS",
        "subset": SUBSET,
        "model": "RandomForestRegressor",
        "features": feature_columns,
        "validation_split": {
            "method": "GroupShuffleSplit by engine",
            "random_state": RANDOM_STATE,
            "training_engines": int(
                fit_data["unit_id"].nunique()
            ),
            "validation_engines": int(
                val_data["unit_id"].nunique()
            ),
        },
        "validation_metrics_all_cycles": validation_results,
        "official_test_metrics_final_cycle_only": test_results,
        "test_evaluation_note": (
            "Evaluation uses supplied RUL references for the final "
            "observed cycle of each test engine. This is the "
            "benchmark-style test evaluation."
        ),
        "model_path": str(model_path.relative_to(ROOT)),
        "validation_predictions": str(validation_csv.relative_to(ROOT)),
        "test_predictions": str(test_csv.relative_to(ROOT)),
        "feature_importance": str(importance_csv.relative_to(ROOT)),
    }

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 7. Print concise results.
    # --------------------------------------------------------
    print("\nVALIDATION RESULTS — ALL CYCLES")
    for key, value in validation_results.items():
        print(f"{key}: {value}")

    print("\nTEST RESULTS — FINAL CYCLE OF EACH ENGINE")
    for key, value in test_results.items():
        print(f"{key}: {value}")

    print("\nTOP 10 FEATURES")
    print(importance.head(10).to_string(index=False))

    print("\nSTEP 2 COMPLETED")
    print(f"Model:  {model_path}")
    print(f"Report: {report_path}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
