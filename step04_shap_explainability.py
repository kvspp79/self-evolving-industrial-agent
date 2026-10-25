
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


# ============================================================
# STEP 4: SHAP EXPLAINABILITY
# AI4I 2020 Predictive Maintenance
# ============================================================

RANDOM_STATE = 42
MAX_BACKGROUND_ROWS = 100
MAX_EXPLANATION_ROWS = 2000

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

MODEL_PATH = MODEL_DIR / "XGBoost.joblib"
X_TRAIN_PATH = OUTPUT_DIR / "X_train.csv"
X_TEST_PATH = OUTPUT_DIR / "X_test.csv"
Y_TEST_PATH = OUTPUT_DIR / "y_test.csv"

TARGET = "Machine failure"


def load_inputs():
    """Load the trained pipeline and saved train/test datasets."""

    required_files = [
        MODEL_PATH,
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
    ]

    missing = [str(p) for p in required_files if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(missing)
            + "\nRun Step 1 and Step 2 first."
        )

    model = joblib.load(MODEL_PATH)
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test_df = pd.read_csv(Y_TEST_PATH)

    if TARGET not in y_test_df.columns:
        raise ValueError(
            "Target column {!r} not found in {}.".format(
                TARGET, Y_TEST_PATH
            )
        )

    y_test = y_test_df[TARGET].astype(int)

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test have different row counts.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train columns do not match X_test columns.")

    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing dataset is empty.")

    if not hasattr(model, "named_steps"):
        raise TypeError(
            "The saved model is not an sklearn Pipeline."
        )

    if "preprocessor" not in model.named_steps:
        raise KeyError("Pipeline has no 'preprocessor' step.")

    if "classifier" not in model.named_steps:
        raise KeyError("Pipeline has no 'classifier' step.")

    print("Model and datasets loaded successfully.")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Failures in test set:", int(y_test.sum()))

    return model, X_train, X_test, y_test


def transform_data(model, X_train, X_test):
    """Apply the already-fitted Step 2 preprocessing pipeline."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    train_transformed = preprocessor.transform(X_train)
    test_transformed = preprocessor.transform(X_test)

    if hasattr(train_transformed, "toarray"):
        train_transformed = train_transformed.toarray()

    if hasattr(test_transformed, "toarray"):
        test_transformed = test_transformed.toarray()

    # Use NumPy arrays during SHAP calculations. This prevents
    # original names containing [, ], or < from being passed
    # as XGBoost DMatrix feature names.
    X_train_array = np.asarray(train_transformed, dtype=np.float32)
    X_test_array = np.asarray(test_transformed, dtype=np.float32)

    if not np.isfinite(X_train_array).all():
        raise ValueError("Transformed training data has invalid values.")

    if not np.isfinite(X_test_array).all():
        raise ValueError("Transformed test data has invalid values.")

    try:
        original_names = [
            str(name)
            for name in preprocessor.get_feature_names_out()
        ]
    except (AttributeError, ValueError):
        original_names = [
            "feature_{}".format(i)
            for i in range(X_test_array.shape[1])
        ]

    if len(original_names) != X_test_array.shape[1]:
        raise ValueError("Feature names and transformed columns do not match.")

    # Safe names are used for calculations and saved output.
    # Keep original names in a separate mapping for interpretation.
    safe_names = [
        "feature_{:02d}".format(i)
        for i in range(len(original_names))
    ]

    feature_map = pd.DataFrame({
        "Safe_Feature": safe_names,
        "Original_Feature": original_names,
    })

    print("Transformed training shape:", X_train_array.shape)
    print("Transformed test shape:", X_test_array.shape)
    print("Transformed features:", original_names)

    return (
        classifier,
        X_train_array,
        X_test_array,
        safe_names,
        original_names,
        feature_map,
    )


def normalize_shap_values(values):
    """Normalize SHAP outputs from common binary-classifier formats."""

    if isinstance(values, list):
        values = np.asarray(values[-1] if len(values) == 2 else values[0])
    else:
        values = np.asarray(values)

        if values.ndim == 3:
            if values.shape[-1] == 2:
                values = values[:, :, 1]
            else:
                values = values[:, :, 0]

    if values.ndim != 2:
        raise ValueError(
            "Unexpected SHAP dimensions: {}".format(values.shape)
        )

    return values


def main():
    print("=" * 72)
    print("STEP 4: SHAP EXPLAINABILITY")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and model.
    model, X_train, X_test, y_test = load_inputs()

    # 2. Transform using the existing fitted preprocessing.
    (
        classifier,
        X_train_array,
        X_test_array,
        safe_names,
        original_names,
        feature_map,
    ) = transform_data(model, X_train, X_test)

    feature_map_path = OUTPUT_DIR / "step04_feature_name_mapping.csv"
    feature_map.to_csv(feature_map_path, index=False)

    # 3. Select a background sample from training data only.
    rng = np.random.default_rng(RANDOM_STATE)
    background_size = min(MAX_BACKGROUND_ROWS, len(X_train_array))
    background_indices = rng.choice(
        len(X_train_array),
        size=background_size,
        replace=False,
    )
    X_background = X_train_array[background_indices]

    print("\nSHAP background rows:", len(X_background))

    # 4. Build TreeExplainer.
    # tree_path_dependent avoids passing the background matrix
    # into TreeExplainer. NumPy input avoids invalid feature names.
    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(
        classifier,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    explain_size = min(MAX_EXPLANATION_ROWS, len(X_test_array))
    X_explain = X_test_array[:explain_size]

    print("Rows being explained:", explain_size)
    print("Calculating SHAP values...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_values = explainer.shap_values(X_explain)

    shap_values = normalize_shap_values(raw_values)

    if shap_values.shape != X_explain.shape:
        raise ValueError(
            "SHAP shape {} does not match input shape {}.".format(
                shap_values.shape, X_explain.shape
            )
        )

    if not np.isfinite(shap_values).all():
        raise ValueError("SHAP values contain NaN or infinite values.")

    print("SHAP calculation successful:", shap_values.shape)

    # 5. Global importance.
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Safe_Feature": safe_names,
        "Feature": original_names,
        "Mean_Absolute_SHAP": mean_abs_shap,
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(0, "Rank", np.arange(1, len(importance_df) + 1))

    importance_path = OUTPUT_DIR / "step04_shap_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    print("\nGLOBAL SHAP FEATURE IMPORTANCE")
    print(importance_df.to_string(index=False))

    # 6. Global importance plot.
    plot_df = importance_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=True,
    )

    plt.figure(figsize=(11, max(6, 0.5 * len(plot_df))))
    plt.barh(
        plot_df["Feature"],
        plot_df["Mean_Absolute_SHAP"],
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Global SHAP Feature Importance")
    plt.tight_layout()

    bar_path = OUTPUT_DIR / "step04_shap_importance_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 7. SHAP summary plot using safe feature names internally.
    # Replace displayed labels with the original names afterward.
    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=safe_names,
        max_display=min(20, len(safe_names)),
        show=False,
    )

    ax = plt.gca()
    displayed_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    display_map = dict(zip(safe_names, original_names))

    ax.set_yticklabels([
        display_map.get(label, label)
        for label in displayed_labels
    ])

    plt.title("SHAP Summary Plot")
    plt.tight_layout()

    summary_path = OUTPUT_DIR / "step04_shap_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 8. Calculate probabilities with the complete original pipeline.
    probabilities = model.predict_proba(X_test)[:, 1]

    if len(probabilities) != len(X_test):
        raise ValueError("Model returned an unexpected number of predictions.")

    # Explain the highest-risk row among the explained test rows.
    local_position = int(np.argmax(probabilities[:explain_size]))

    local_df = pd.DataFrame({
        "Safe_Feature": safe_names,
        "Feature": original_names,
        "Transformed_Value": X_explain[local_position],
        "SHAP_Value": shap_values[local_position],
    })

    local_df["Absolute_SHAP"] = local_df["SHAP_Value"].abs()
    local_df = local_df.sort_values(
        "Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    local_df.insert(0, "Rank", np.arange(1, len(local_df) + 1))

    local_path = OUTPUT_DIR / "step04_shap_highest_risk_example.csv"
    local_df.to_csv(local_path, index=False)

    # 9. Local explanation plot.
    local_plot_df = local_df.head(min(15, len(local_df))).sort_values(
        "SHAP_Value",
        ascending=True,
    )

    colors = [
        "firebrick" if value > 0 else "steelblue"
        for value in local_plot_df["SHAP_Value"]
    ]

    plt.figure(figsize=(11, max(5, 0.45 * len(local_plot_df))))
    plt.barh(
        local_plot_df["Feature"],
        local_plot_df["SHAP_Value"],
        color=colors,
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("SHAP value (raw model output)")
    plt.ylabel("Feature")
    plt.title("Local SHAP Explanation: Highest-Risk Test Row")
    plt.tight_layout()

    local_plot_path = OUTPUT_DIR / "step04_shap_local_explanation.png"
    plt.savefig(local_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 10. Save row-level SHAP values.
    shap_output = pd.DataFrame(
        shap_values,
        columns=safe_names,
    )

    # Add original feature names as a separate mapping file;
    # safe column names are retained to ensure compatibility.
    shap_output.insert(0, "Test_Row_Position", np.arange(explain_size))
    shap_output.insert(
        1,
        "Actual_Failure",
        y_test.iloc[:explain_size].to_numpy(),
    )
    shap_output.insert(
        2,
        "Predicted_Failure_Probability",
        probabilities[:explain_size],
    )

    shap_values_path = OUTPUT_DIR / "step04_shap_values.csv"
    shap_output.to_csv(shap_values_path, index=False)

    # 11. Save report.
    report = {
        "step": 4,
        "method": "SHAP TreeExplainer",
        "dataset": "AI4I 2020 Predictive Maintenance",
        "model_path": str(MODEL_PATH),
        "background_rows_sampled": int(background_size),
        "explained_test_rows": int(explain_size),
        "transformed_feature_count": int(len(safe_names)),
        "highest_risk_test_row_position": int(local_position),
        "highest_risk_predicted_failure_probability": float(
            probabilities[local_position]
        ),
        "actual_failure_for_highest_risk_example": int(
            y_test.iloc[local_position]
        ),
        "top_10_global_features": importance_df.head(10).to_dict(
            orient="records"
        ),
        "interpretation_notes": [
            "Global importance is the mean absolute SHAP value.",
            "Positive SHAP values push the raw model output higher; negative values push it lower.",
            "Raw-output SHAP values are not percentage-point changes in failure probability.",
            "One-hot encoded Type categories may appear as separate features.",
            "SHAP explains model behavior and does not establish causation.",
            "The local example is the highest predicted-risk row among the explained test rows.",
            "The background sample is selected from training data. Tree-path-dependent SHAP uses the model's learned tree paths as its reference mechanism.",
        ],
    }

    report_path = OUTPUT_DIR / "step04_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # 12. Completion summary.
    print("\nHighest-risk explained test row position:", local_position)
    print(
        "Predicted failure probability:",
        round(float(probabilities[local_position]), 6),
    )
    print("Actual failure:", int(y_test.iloc[local_position]))

    print("\nSaved outputs:")
    for path in [
        feature_map_path,
        importance_path,
        bar_path,
        summary_path,
        local_path,
        local_plot_path,
        shap_values_path,
        report_path,
    ]:
        print(path)

    print("\n" + "=" * 72)
    print("STEP 4 COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    main()
