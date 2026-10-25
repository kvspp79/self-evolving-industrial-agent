from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# NASA C-MAPSS FD001 — STEP 5
# EXPLAINABILITY AND ERROR ANALYSIS
#
# Loads existing Step 4 models and predictions.
# Does not retrain models or modify Steps 1-4.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

STEP01_DIR = ROOT / "outputs" / "step01"
STEP04_DIR = ROOT / "outputs" / "step04"
OUTPUT_DIR = ROOT / "outputs" / "step05"
MODEL_DIR = STEP04_DIR / "models"

SUBSET = "FD001"
RANDOM_STATE = 42

SHAP_BACKGROUND_SIZE = 100
SHAP_EXPLAIN_SIZE = 300

ROLLING_WINDOWS = (5, 10, 20)

SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i:02d}" for i in range(1, 22)]
BASE_FEATURES = ["cycle"] + SETTINGS + SENSORS


# ============================================================
# 1. REPRODUCE STEP 4 TEMPORAL FEATURES
# ============================================================

def add_temporal_features(df):
    """
    Reproduce the Step 4 feature generation.

    Features:
    - Baseline operating settings and sensors
    - One-cycle sensor differences
    - Rolling means and standard deviations
    - Nine-interval sensor differences divided by nine
    """

    result = (
        df.sort_values(["unit_id", "cycle"])
        .copy()
        .reset_index(drop=True)
    )

    required = ["unit_id", "cycle", "RUL"] + BASE_FEATURES
    missing = [c for c in required if c not in result.columns]

    if missing:
        raise ValueError(
            f"Processed training data is missing columns: {missing}"
        )

    generated = {}
    feature_names = list(BASE_FEATURES)

    for sensor in SENSORS:
        grouped = result.groupby("unit_id", sort=False)[sensor]

        # One-cycle difference.
        diff_name = f"{sensor}_diff1"
        generated[diff_name] = grouped.diff().fillna(0.0)
        feature_names.append(diff_name)

        # Rolling statistics.
        for window in ROLLING_WINDOWS:
            mean_name = f"{sensor}_mean_{window}"
            std_name = f"{sensor}_std_{window}"

            generated[mean_name] = grouped.transform(
                lambda s, w=window:
                s.rolling(window=w, min_periods=1).mean()
            )

            generated[std_name] = grouped.transform(
                lambda s, w=window:
                s.rolling(window=w, min_periods=1).std(ddof=0)
            )

            feature_names.extend([mean_name, std_name])

        # Nine-interval slope.
        slope_name = f"{sensor}_slope10"
        generated[slope_name] = (
            grouped.diff(9) / 9.0
        ).fillna(0.0)

        feature_names.append(slope_name)

    temporal_frame = pd.DataFrame(
        generated,
        index=result.index
    )

    result = pd.concat(
        [result, temporal_frame],
        axis=1
    )

    values = result[feature_names].replace(
        [np.inf, -np.inf],
        np.nan
    )

    if values.isna().any().any():
        bad = values.columns[values.isna().any()].tolist()
        raise ValueError(
            f"Non-finite values found in temporal features: {bad[:10]}"
        )

    return result, feature_names


# ============================================================
# 2. METRICS AND FILE UTILITIES
# ============================================================

def nasa_score(y_true, y_pred):
    """NASA asymmetric RUL score. Lower is better."""

    error = (
        np.asarray(y_pred, dtype=float)
        - np.asarray(y_true, dtype=float)
    )

    exponent = np.where(
        error < 0,
        -error / 13.0,
        error / 10.0
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
        "NASA_score": nasa_score(y_true, y_pred)
    }


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n"
            "Check that Steps 1 and 4 completed successfully."
        )

    return pd.read_csv(path)


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=4, allow_nan=False),
        encoding="utf-8"
    )


def validate_predictions(df, filename):
    required = [
        "unit_id",
        "actual_RUL",
        "baseline_predicted_RUL",
        "temporal_predicted_RUL"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            f"{filename} contains missing prediction values."
        )


# ============================================================
# 3. ERROR ANALYSIS
# ============================================================

def make_comparison_table(predictions):
    actual = predictions["actual_RUL"].to_numpy(dtype=float)
    baseline = predictions[
        "baseline_predicted_RUL"
    ].to_numpy(dtype=float)
    temporal = predictions[
        "temporal_predicted_RUL"
    ].to_numpy(dtype=float)

    result = pd.DataFrame({
        "unit_id": predictions["unit_id"].to_numpy(),
        "actual_RUL": actual,
        "baseline_predicted_RUL": baseline,
        "temporal_predicted_RUL": temporal
    })

    result["baseline_signed_error"] = baseline - actual
    result["temporal_signed_error"] = temporal - actual

    result["baseline_absolute_error"] = np.abs(
        result["baseline_signed_error"]
    )

    result["temporal_absolute_error"] = np.abs(
        result["temporal_signed_error"]
    )

    # Positive means temporal model has lower absolute error.
    result["absolute_error_reduction"] = (
        result["baseline_absolute_error"]
        - result["temporal_absolute_error"]
    )

    result["temporal_model_result"] = np.select(
        [
            result["absolute_error_reduction"] > 0,
            result["absolute_error_reduction"] < 0
        ],
        [
            "Temporal model better",
            "Baseline better"
        ],
        default="Tie"
    )

    return result


def plot_prediction_comparison(data, output_path, title):
    actual = data["actual_RUL"].to_numpy(dtype=float)
    baseline = data["baseline_predicted_RUL"].to_numpy(dtype=float)
    temporal = data["temporal_predicted_RUL"].to_numpy(dtype=float)

    upper = max(
        float(np.max(actual)),
        float(np.max(baseline)),
        float(np.max(temporal))
    ) + 5

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(actual, baseline, alpha=0.6, label="Baseline")
    ax.scatter(actual, temporal, alpha=0.6, label="Temporal")
    ax.plot(
        [0, upper],
        [0, upper],
        linestyle="--",
        label="Perfect prediction"
    )

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_error_distribution(comparison, output_path, title):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        comparison["baseline_absolute_error"],
        bins=25,
        alpha=0.6,
        label="Baseline"
    )

    ax.hist(
        comparison["temporal_absolute_error"],
        bins=25,
        alpha=0.6,
        label="Temporal"
    )

    ax.set_xlabel("Absolute RUL prediction error")
    ax.set_ylabel("Number of observations")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_shap_importance(importance, output_path):
    top = (
        importance.head(20)
        .sort_values("mean_absolute_SHAP")
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        top["feature"],
        top["mean_absolute_SHAP"]
    )

    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Top 20 SHAP Features — Temporal RUL Model")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ============================================================
# 4. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("NASA C-MAPSS — STEP 5")
    print("EXPLAINABILITY AND ERROR ANALYSIS")
    print(f"Subset: {SUBSET}")
    print("=" * 68)

    # --------------------------------------------------------
    # A. Load existing Step 4 artifacts.
    # --------------------------------------------------------

    baseline_model_path = (
        MODEL_DIR / f"baseline_same_split_{SUBSET}.joblib"
    )

    temporal_model_path = (
        MODEL_DIR / f"temporal_random_forest_{SUBSET}.joblib"
    )

    if not baseline_model_path.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {baseline_model_path}"
        )

    if not temporal_model_path.exists():
        raise FileNotFoundError(
            f"Temporal model not found: {temporal_model_path}"
        )

    baseline_model = joblib.load(baseline_model_path)
    temporal_model = joblib.load(temporal_model_path)

    validation = load_csv(
        STEP04_DIR / f"validation_predictions_{SUBSET}.csv"
    )

    test = load_csv(
        STEP04_DIR / f"test_predictions_{SUBSET}.csv"
    )

    train = load_csv(
        STEP01_DIR / f"train_{SUBSET}_processed.csv"
    )

    feature_importance = load_csv(
        STEP04_DIR
        / f"temporal_feature_importance_{SUBSET}.csv"
    )

    validate_predictions(validation, "Validation predictions")
    validate_predictions(test, "Test predictions")

    if "feature" not in feature_importance.columns:
        raise ValueError(
            "Step 4 feature-importance CSV must contain 'feature'."
        )

    if "importance" not in feature_importance.columns:
        raise ValueError(
            "Step 4 feature-importance CSV must contain 'importance'."
        )

    if not hasattr(temporal_model, "feature_names_in_"):
        raise ValueError(
            "Temporal model has no feature_names_in_. "
            "Cannot safely verify feature compatibility."
        )

    if not hasattr(baseline_model, "feature_names_in_"):
        raise ValueError(
            "Baseline model has no feature_names_in_. "
            "Cannot safely verify feature compatibility."
        )

    baseline_features = list(baseline_model.feature_names_in_)
    temporal_features = list(temporal_model.feature_names_in_)

    # --------------------------------------------------------
    # B. Reconstruct Step 4 features and validate schema.
    # --------------------------------------------------------

    print("\nReconstructing temporal features from Step 1 data...")

    train_features, generated_features = add_temporal_features(train)

    if generated_features != temporal_features:
        generated_set = set(generated_features)
        model_set = set(temporal_features)

        raise ValueError(
            "Feature schema mismatch between Step 4 and Step 5.\n"
            f"Generated feature count: {len(generated_features)}\n"
            f"Model feature count: {len(temporal_features)}\n"
            f"Missing features: {sorted(model_set - generated_set)[:15]}\n"
            f"Extra features: {sorted(generated_set - model_set)[:15]}\n"
            "Do not bypass this check. Match the feature pipeline first."
        )

    missing_baseline = [
        c for c in baseline_features
        if c not in train_features.columns
    ]

    if missing_baseline:
        raise ValueError(
            f"Baseline model features missing from data: {missing_baseline}"
        )

    print(f"Training rows: {len(train_features):,}")
    print(f"Training engines: {train_features['unit_id'].nunique()}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Baseline features: {len(baseline_features)}")
    print(f"Temporal features: {len(temporal_features)}")

    # --------------------------------------------------------
    # C. Validation and test error analysis.
    # --------------------------------------------------------

    print("\nAnalyzing prediction errors...")

    evaluation_results = {}
    comparisons = {}

    for split_name, data in [
        ("validation", validation),
        ("test", test)
    ]:
        comparison = make_comparison_table(data)
        comparisons[split_name] = comparison

        comparison.to_csv(
            OUTPUT_DIR / f"{split_name}_error_analysis_{SUBSET}.csv",
            index=False
        )

        baseline_metrics = calculate_metrics(
            data["actual_RUL"],
            data["baseline_predicted_RUL"]
        )

        temporal_metrics = calculate_metrics(
            data["actual_RUL"],
            data["temporal_predicted_RUL"]
        )

        temporal_better = int(
            (comparison["absolute_error_reduction"] > 0).sum()
        )

        baseline_better = int(
            (comparison["absolute_error_reduction"] < 0).sum()
        )

        ties = int(
            (comparison["absolute_error_reduction"] == 0).sum()
        )

        unit_description = (
            "validation cycle rows"
            if split_name == "validation"
            else "final observed cycle per test engine"
        )

        evaluation_results[split_name] = {
            "unit_of_analysis": unit_description,
            "baseline_metrics": baseline_metrics,
            "temporal_metrics": temporal_metrics,
            "observations_temporal_better": temporal_better,
            "observations_baseline_better": baseline_better,
            "ties": ties,
            "mean_absolute_error_reduction": float(
                comparison["absolute_error_reduction"].mean()
            ),
            "median_absolute_error_reduction": float(
                comparison["absolute_error_reduction"].median()
            )
        }

        plot_prediction_comparison(
            data,
            OUTPUT_DIR / f"{split_name}_prediction_comparison.png",
            f"{split_name.title()} Actual vs Predicted RUL"
        )

        plot_error_distribution(
            comparison,
            OUTPUT_DIR / f"{split_name}_absolute_error_distribution.png",
            f"{split_name.title()} Absolute Error Distribution"
        )

        print(f"\n{split_name.upper()} RESULTS")
        print("-" * 48)
        print("Baseline:", baseline_metrics)
        print("Temporal:", temporal_metrics)
        print("Temporal better:", temporal_better)
        print("Baseline better:", baseline_better)
        print("Ties:", ties)

    # --------------------------------------------------------
    # D. Largest test-engine improvements and regressions.
    # --------------------------------------------------------

    test_comparison = comparisons["test"]

    largest_improvements = (
        test_comparison.sort_values(
            "absolute_error_reduction",
            ascending=False
        ).head(10)
    )

    largest_regressions = (
        test_comparison.sort_values(
            "absolute_error_reduction",
            ascending=True
        ).head(10)
    )

    largest_improvements.to_csv(
        OUTPUT_DIR / f"top_10_test_engine_improvements_{SUBSET}.csv",
        index=False
    )

    largest_regressions.to_csv(
        OUTPUT_DIR / f"top_10_test_engine_regressions_{SUBSET}.csv",
        index=False
    )

    # --------------------------------------------------------
    # E. Prepare leakage-controlled SHAP data.
    #
    # Validation engines are excluded from the SHAP samples.
    # Test engines are not used to construct SHAP background data.
    # --------------------------------------------------------

    print("\nPreparing SHAP samples...")

    rng = np.random.default_rng(RANDOM_STATE)

    validation_engine_ids = set(validation["unit_id"].unique())
    training_engine_ids = set(train_features["unit_id"].unique())

    unknown_ids = validation_engine_ids - training_engine_ids

    if unknown_ids:
        raise ValueError(
            "Validation predictions contain engine IDs not found "
            f"in training data: {sorted(unknown_ids)}"
        )

    shap_source = train_features.loc[
        ~train_features["unit_id"].isin(validation_engine_ids)
    ].copy()

    if shap_source.empty:
        raise ValueError("No training engines remain for SHAP analysis.")

    overlap = validation_engine_ids.intersection(
        set(shap_source["unit_id"].unique())
    )

    if overlap:
        raise RuntimeError(
            f"Validation engine leakage detected: {sorted(overlap)}"
        )

    background_n = min(SHAP_BACKGROUND_SIZE, len(shap_source))
    explain_n = min(SHAP_EXPLAIN_SIZE, len(shap_source))

    background_idx = rng.choice(
        len(shap_source),
        size=background_n,
        replace=False
    )

    explain_idx = rng.choice(
        len(shap_source),
        size=explain_n,
        replace=False
    )

    background_rows = shap_source.iloc[background_idx].copy()
    explain_rows = shap_source.iloc[explain_idx].copy()

    background = background_rows[temporal_features].copy()
    explain_data = explain_rows[temporal_features].copy()

    if list(explain_data.columns) != temporal_features:
        raise ValueError("SHAP feature order differs from the model feature order.")

    print("Training engines used for SHAP:", shap_source["unit_id"].nunique())
    print("Background rows sampled:", len(background))
    print("Rows to explain:", len(explain_data))

    # --------------------------------------------------------
    # F. Calculate SHAP values.
    #
    # Tree-path-dependent SHAP avoids repeatedly evaluating
    # the model against a 100-row interventional background.
    # The attribution method is explicitly recorded in the report.
    # --------------------------------------------------------

    print("\nComputing tree-path-dependent SHAP values...")

    explainer = shap.TreeExplainer(
        temporal_model,
        feature_perturbation="tree_path_dependent",
        model_output="raw"
    )

    shap_result = explainer.shap_values(
        explain_data,
        check_additivity=False
    )

    shap_values = np.asarray(shap_result)

    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 1:
            shap_values = shap_values[:, :, 0]
        else:
            raise ValueError(
                f"Unexpected multi-output SHAP shape: {shap_values.shape}"
            )

    expected_shape = (len(explain_data), len(temporal_features))

    if shap_values.shape != expected_shape:
        raise ValueError(
            f"Unexpected SHAP shape {shap_values.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(shap_values).all():
        raise ValueError("SHAP values contain non-finite values.")

    print("SHAP computation completed.")

    # --------------------------------------------------------
    # G. Global SHAP feature importance.
    # --------------------------------------------------------

    mean_absolute_shap = np.abs(shap_values).mean(axis=0)

    shap_importance = pd.DataFrame({
        "feature": temporal_features,
        "mean_absolute_SHAP": mean_absolute_shap
    }).sort_values(
        "mean_absolute_SHAP",
        ascending=False
    )

    shap_importance.to_csv(
        OUTPUT_DIR / f"shap_feature_importance_{SUBSET}.csv",
        index=False
    )

    plot_shap_importance(
        shap_importance,
        OUTPUT_DIR / f"shap_global_importance_{SUBSET}.png"
    )

    # --------------------------------------------------------
    # H. Save SHAP values.
    #
    # FIX: prefix attribution columns with SHAP_.
    # This prevents a duplicate "cycle" column because cycle
    # is both model input and row metadata.
    # --------------------------------------------------------

    shap_column_names = [
        f"SHAP_{feature}"
        for feature in temporal_features
    ]

    shap_value_table = pd.DataFrame(
        shap_values,
        columns=shap_column_names
    )

    shap_value_table.insert(
        0,
        "unit_id",
        explain_rows["unit_id"].to_numpy()
    )

    shap_value_table.insert(
        1,
        "cycle",
        explain_rows["cycle"].to_numpy()
    )

    shap_value_table.insert(
        2,
        "actual_RUL",
        explain_rows["RUL"].to_numpy()
    )

    shap_value_table.to_csv(
        OUTPUT_DIR / f"shap_values_sample_{SUBSET}.csv",
        index=False
    )

    print("Saved per-row SHAP values.")

    # --------------------------------------------------------
    # I. SHAP beeswarm summary plot.
    # --------------------------------------------------------

    print("Generating SHAP summary plot...")

    try:
        shap.summary_plot(
            shap_values,
            explain_data,
            feature_names=temporal_features,
            max_display=20,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"shap_summary_{SUBSET}.png",
            dpi=160,
            bbox_inches="tight"
        )

        plt.close("all")

    except Exception as exc:
        plt.close("all")

        warnings.warn(
            "SHAP values were calculated and saved, but the summary "
            f"plot failed: {exc}"
        )

    # --------------------------------------------------------
    # J. Compare Random Forest and SHAP importance.
    # --------------------------------------------------------

    importance_comparison = feature_importance.merge(
        shap_importance,
        on="feature",
        how="outer"
    ).fillna(0.0)

    importance_comparison["random_forest_rank"] = (
        importance_comparison["importance"]
        .rank(method="min", ascending=False)
    )

    importance_comparison["shap_rank"] = (
        importance_comparison["mean_absolute_SHAP"]
        .rank(method="min", ascending=False)
    )

    importance_comparison.to_csv(
        OUTPUT_DIR / f"feature_importance_comparison_{SUBSET}.csv",
        index=False
    )

    # --------------------------------------------------------
    # K. Save research report.
    # --------------------------------------------------------

    report = {
        "dataset": "NASA C-MAPSS",
        "subset": SUBSET,
        "step": 5,
        "purpose": (
            "Explain temporal RUL predictions and analyze "
            "prediction errors."
        ),
        "models_loaded_from_step04": {
            "baseline_model": str(baseline_model_path),
            "temporal_model": str(temporal_model_path)
        },
        "feature_validation": {
            "baseline_feature_count": len(baseline_features),
            "temporal_feature_count": len(temporal_features),
            "temporal_feature_schema_matches_model": True
        },
        "explainability": {
            "method": "SHAP TreeExplainer (tree-path-dependent)",
            "background_sample_size": len(background),
            "explained_sample_size": len(explain_data),
            "training_engines_used": int(
                shap_source["unit_id"].nunique()
            ),
            "validation_engines_excluded": int(
                len(validation_engine_ids)
            ),
            "explained_model": "Temporal Random Forest",
            "top_20_features": (
                shap_importance.head(20).to_dict(orient="records")
            ),
            "interpretation_caution": (
                "SHAP values quantify feature attribution under the "
                "chosen tree-path-dependent method. They do not prove "
                "physical causation. Correlated sensor features may "
                "share or redistribute attribution."
            )
        },
        "error_analysis": evaluation_results,
        "largest_test_engine_improvements": (
            largest_improvements.to_dict(orient="records")
        ),
        "largest_test_engine_regressions": (
            largest_regressions.to_dict(orient="records")
        ),
        "methodology_notes": [
            "Step 4 models are loaded without retraining.",
            "Temporal features are reconstructed and checked against "
            "the saved model feature names and order.",
            "Validation engines are excluded from SHAP background "
            "and explanation samples.",
            "SHAP values use tree-path-dependent attribution.",
            "Validation metrics are calculated over validation rows.",
            "Test metrics represent final observed cycles per test engine.",
            "Test results are descriptive and must not be repeatedly "
            "used to tune model hyperparameters.",
            "SHAP attribution does not establish physical causality."
        ]
    }

    report_path = OUTPUT_DIR / f"step05_{SUBSET}_report.json"
    save_json(report_path, report)

    # --------------------------------------------------------
    # L. Console summary.
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("TOP 15 SHAP FEATURES")
    print("=" * 68)
    print(shap_importance.head(15).to_string(index=False))

    print("\n" + "=" * 68)
    print("TOP 10 TEST ENGINE IMPROVEMENTS")
    print("=" * 68)

    print(
        largest_improvements[
            [
                "unit_id",
                "actual_RUL",
                "baseline_absolute_error",
                "temporal_absolute_error",
                "absolute_error_reduction"
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 68)
    print("TOP 10 TEST ENGINE REGRESSIONS")
    print("=" * 68)

    print(
        largest_regressions[
            [
                "unit_id",
                "actual_RUL",
                "baseline_absolute_error",
                "temporal_absolute_error",
                "absolute_error_reduction"
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 68)
    print("STEP 5 COMPLETED SUCCESSFULLY")
    print("=" * 68)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Research report: {report_path}")
    print("Error tables, SHAP values, rankings, and plots saved.")


if __name__ == "__main__":
    main()