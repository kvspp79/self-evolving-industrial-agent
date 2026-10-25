
"""
STEP 5: TWO-FEATURE EXPERIMENT
Project: Self-Evolving Explainable Industrial Agent

Model A: All six candidate features
Model B: Torque and Tool Wear only

Categorical feature handling:
- "Type" is one-hot encoded using OneHotEncoder.
- Numeric features pass through unchanged.
- The preprocessing pipeline is fitted on training data only.

Run from the project root:
    py -3.13 src/step05_two_feature_experiment.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50

TARGET = "Machine failure"

ALL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TWO_FEATURES = [
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = ["Type"]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_dataset():
    print("=" * 72)
    print("STEP 5: TWO-FEATURE EXPERIMENT")
    print("=" * 72)

    print(f"\nProject folder: {PROJECT_ROOT}")
    print(f"Dataset path:   {DATA_PATH}")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            "\nAI4I dataset was not found.\n"
            f"Expected file:\n{DATA_PATH}\n\n"
            "Make sure ai4i2020.csv is inside the project's data folder."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = ALL_FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    df = df[required_columns].copy()

    # Convert numeric inputs and target explicitly.
    for column in NUMERIC_FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows with missing values in required columns.
    before = len(df)
    df = df.dropna(subset=required_columns).copy()
    removed = before - len(df)

    # Validate target values.
    target_values = set(df[TARGET].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Expected binary target values 0 and 1, got {target_values}"
        )

    df[TARGET] = df[TARGET].astype(int)
    df["Type"] = df["Type"].astype(str).str.strip()

    if df.empty:
        raise ValueError("No valid rows remain after data cleaning.")

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Both failure classes (0 and 1) are required for this experiment."
        )

    print(f"\nRows loaded: {len(df)}")
    print(f"Rows removed due to missing values: {removed}")
    print(f"Failure cases: {(df[TARGET] == 1).sum()}")
    print(f"Non-failure cases: {(df[TARGET] == 0).sum()}")
    print(f"Failure rate: {df[TARGET].mean():.3%}")

    return df


# ============================================================
# 3. CREATE ONE SHARED STRATIFIED SPLIT
# ============================================================

def create_split(df):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n" + "-" * 72)
    print("SHARED TRAIN/TEST SPLIT")
    print("-" * 72)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")
    print(f"Training failure rate: {y_train.mean():.3%}")
    print(f"Testing failure rate:  {y_test.mean():.3%}")
    print(f"Random state: {RANDOM_STATE}")

    return X_train, X_test, y_train, y_test


# ============================================================
# 4. BUILD A SAFE PREPROCESSING PIPELINE
# ============================================================

def build_pipeline(selected_features, scale_pos_weight):
    categorical = [
        col for col in selected_features
        if col in CATEGORICAL_FEATURES
    ]

    numeric = [
        col for col in selected_features
        if col in NUMERIC_FEATURES
    ]

    transformers = []

    if numeric:
        # Numeric values pass through without changing their scale.
        transformers.append(
            ("numeric", "passthrough", numeric)
        )

    if categorical:
        # Converts L/M/H into numeric indicator columns.
        # sparse_output=False supports modern scikit-learn versions.
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=2,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


# ============================================================
# 5. EVALUATE A MODEL
# ============================================================

def evaluate_model(
    model_name,
    selected_features,
    X_train,
    X_test,
    y_train,
    y_test,
):
    print("\n" + "=" * 72)
    print(f"TRAINING {model_name}")
    print("=" * 72)
    print(f"Input features: {selected_features}")

    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()

    # Calculate imbalance weight from training data only.
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    if positives == 0:
        raise ValueError("Training data contains no failure cases.")

    scale_pos_weight = negatives / positives

    print(f"Training non-failures: {negatives}")
    print(f"Training failures:     {positives}")
    print(f"scale_pos_weight:      {scale_pos_weight:.4f}")

    model = build_pipeline(
        selected_features=selected_features,
        scale_pos_weight=scale_pos_weight,
    )

    # The pipeline fits the encoder on training data only.
    model.fit(X_train_selected, y_train)

    probabilities = model.predict_proba(X_test_selected)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "Model": model_name,
        "Feature_Count": len(selected_features),
        "Features": ", ".join(selected_features),
        "Threshold": THRESHOLD,
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "F1": float(f1_score(y_test, predictions, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_test, probabilities)),
        "PR_AUC_Average_Precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "Brier_Score": float(
            brier_score_loss(y_test, probabilities)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Predicted_Failures": int(predictions.sum()),
        "Actual_Failures": int(np.asarray(y_test).sum()),
    }

    print(f"\nMetrics at threshold {THRESHOLD:.2f}:")
    for key, value in metrics.items():
        if key in {"Model", "Features"}:
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Failure", "Failure"],
            zero_division=0,
        )
    )

    predictions_df = pd.DataFrame({
        "Test_Row_Position": np.arange(len(y_test)),
        "Actual_Failure": np.asarray(y_test, dtype=int),
        f"{model_name}_Failure_Probability": probabilities,
        f"{model_name}_Prediction": predictions,
    })

    # Save the fitted pipeline, including its encoder and classifier.
    model_path = (
        OUTPUT_DIR
        / "models"
        / f"step05_{model_name.lower().replace(' ', '_')}.joblib"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(model, model_path)

    print(f"Saved model: {model_path}")

    return model, metrics, predictions_df


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================

def main():
    df = load_dataset()

    X_train, X_test, y_train, y_test = create_split(df)

    # Model A: all six features.
    full_model, full_metrics, full_predictions = evaluate_model(
        model_name="All Six Features",
        selected_features=ALL_FEATURES,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # Model B: Torque and Tool Wear only.
    two_model, two_metrics, two_predictions = evaluate_model(
        model_name="Torque and Tool Wear",
        selected_features=TWO_FEATURES,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # --------------------------------------------------------
    # 7. SAVE COMPARISON TABLE
    # --------------------------------------------------------

    comparison = pd.DataFrame([full_metrics, two_metrics])

    comparison_path = (
        OUTPUT_DIR / "step05_two_feature_comparison.csv"
    )
    comparison.to_csv(comparison_path, index=False)

    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)

    display_columns = [
        "Model",
        "Feature_Count",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC_Average_Precision",
        "Brier_Score",
        "FP",
        "FN",
    ]

    print(
        comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # 8. SAVE ALIGNED TEST PREDICTIONS
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({
        "Test_Row_Position": np.arange(len(y_test)),
        "Actual_Failure": np.asarray(y_test, dtype=int),
        "All_Features_Probability": full_predictions[
            "All Six Features_Failure_Probability"
        ].to_numpy(),
        "All_Features_Prediction": full_predictions[
            "All Six Features_Prediction"
        ].to_numpy(),
        "Two_Features_Probability": two_predictions[
            "Torque and Tool Wear_Failure_Probability"
        ].to_numpy(),
        "Two_Features_Prediction": two_predictions[
            "Torque and Tool Wear_Prediction"
        ].to_numpy(),
    })

    predictions_path = (
        OUTPUT_DIR / "step05_two_feature_test_predictions.csv"
    )
    predictions_df.to_csv(predictions_path, index=False)

    # --------------------------------------------------------
    # 9. PLOT PERFORMANCE COMPARISON
    # --------------------------------------------------------

    plot_metrics = [
        ("Accuracy", "Accuracy"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1", "F1 score"),
        ("ROC_AUC", "ROC-AUC"),
        ("PR_AUC_Average_Precision", "Average precision"),
    ]

    labels = [item[1] for item in plot_metrics]
    full_values = [
        full_metrics[item[0]] for item in plot_metrics
    ]
    two_values = [
        two_metrics[item[0]] for item in plot_metrics
    ]

    positions = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        positions - width / 2,
        full_values,
        width,
        label="All six features",
    )
    ax.bar(
        positions + width / 2,
        two_values,
        width,
        label="Torque + Tool Wear",
    )

    ax.set_title("Step 5: Full-Feature vs Two-Feature Model")
    ax.set_ylabel("Score")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    chart_path = (
        OUTPUT_DIR / "step05_two_feature_comparison.png"
    )
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 10. WRITE A MACHINE-READABLE REPORT
    # --------------------------------------------------------

    report = {
        "step": 5,
        "title": "Two-Feature Experiment",
        "dataset": str(DATA_PATH),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "threshold": THRESHOLD,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "training_failure_rate": float(y_train.mean()),
        "testing_failure_rate": float(y_test.mean()),
        "model_a": full_metrics,
        "model_b": two_metrics,
        "interpretation_note": (
            "The two-feature model is a feature-reduction experiment. "
            "A lower score does not automatically mean it is inferior "
            "for every industrial use case. Compare failure recall, "
            "precision, false negatives, false positives, and operating "
            "costs as well as overall accuracy."
        ),
        "limitations": [
            "The split is a random stratified split, not a chronological split.",
            "AI4I 2020 does not provide a real maintenance-action history.",
            "Results depend on the split and model configuration.",
            "The default threshold of 0.50 is not necessarily cost-optimal.",
        ],
    }

    report_path = OUTPUT_DIR / "step05_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # --------------------------------------------------------
    # 11. COMPLETION
    # --------------------------------------------------------

    print("\nSaved outputs:")
    print(f"  Comparison CSV:  {comparison_path}")
    print(f"  Test predictions: {predictions_path}")
    print(f"  Comparison chart: {chart_path}")
    print(f"  Report JSON:     {report_path}")

    print("\n" + "=" * 72)
    print("STEP 5 COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "!" * 72)
        print("STEP 5 FAILED")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        print("!" * 72)
        raise
