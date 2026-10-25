
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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


# ============================================================
# STEP 2: MACHINE FAILURE PREDICTION
# Project: Self-Evolving Explainable Rational Agent
# Dataset: AI4I 2020 Predictive Maintenance
# ============================================================

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.50

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Machine failure"


def load_data():
    """Load the fixed train/test split created in Step 1."""

    required_files = [
        OUTPUT_DIR / "X_train.csv",
        OUTPUT_DIR / "X_test.csv",
        OUTPUT_DIR / "y_train.csv",
        OUTPUT_DIR / "y_test.csv",
    ]

    missing_files = [
        str(path) for path in required_files if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required Step 1 files are missing:\n"
            + "\n".join(missing_files)
            + "\nRun Step 1 before running Step 2."
        )

    X_train = pd.read_csv(OUTPUT_DIR / "X_train.csv")
    X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv")

    y_train = pd.read_csv(OUTPUT_DIR / "y_train.csv")[TARGET]
    y_test = pd.read_csv(OUTPUT_DIR / "y_test.csv")[TARGET]

    # Keep training and testing columns in the same order.
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing data is empty.")

    if y_train.nunique() != 2:
        raise ValueError("Training target must contain both classes.")

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Training failures:", int(y_train.sum()))
    print("Testing failures:", int(y_test.sum()))

    return X_train, X_test, y_train, y_test


def build_preprocessor(X):
    """Impute missing values, scale numeric features, encode categories."""

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_cols = [
        col for col in X.columns if col not in categorical_cols
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate metrics using a specified probability threshold."""

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "F1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "ROC_AUC": float(roc_auc_score(y_true, probabilities)),
        "PR_AUC_Average_Precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "Brier_Score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    return metrics, predictions


def main():
    print("=" * 70)
    print("STEP 2: MACHINE FAILURE PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    print("\nInput features:")
    print(X_train.columns.tolist())

    # Calculate class imbalance from the training set only.
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    if positive_count == 0:
        raise ValueError("No failure examples in training data.")

    scale_pos_weight = negative_count / positive_count

    print("\nTraining class distribution:")
    print("Non-failures:", negative_count)
    print("Failures:", positive_count)
    print("XGBoost scale_pos_weight:", round(scale_pos_weight, 3))

    # --------------------------------------------------------
    # 2. BUILD TWO MODELS
    # --------------------------------------------------------

    models = {
        "Logistic_Regression": Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "XGBoost": Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.90,
                    colsample_bytree=0.90,
                    min_child_weight=2,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
    }

    # --------------------------------------------------------
    # 3. TRAIN AND EVALUATE
    # --------------------------------------------------------

    results = []
    predictions_output = pd.DataFrame({
        "Actual": y_test.to_numpy()
    })

    probability_output = {}

    for model_name, model in models.items():
        print("\n" + "-" * 70)
        print("Training:", model_name)
        print("-" * 70)

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        metrics, predictions = calculate_metrics(
            y_test,
            probabilities,
            DECISION_THRESHOLD,
        )

        metrics["Model"] = model_name
        results.append(metrics)

        predictions_output[model_name + "_Prediction"] = predictions
        predictions_output[model_name + "_Probability"] = probabilities

        probability_output[model_name] = probabilities

        print("\nMetrics at threshold {:.2f}:".format(
            DECISION_THRESHOLD
        ))

        for metric_name, value in metrics.items():
            if metric_name != "Model":
                print("{}: {}".format(metric_name, value))

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

        # Save the complete preprocessing/model pipeline.
        model_path = MODEL_DIR / (model_name + ".joblib")
        joblib.dump(model, model_path)
        print("Saved model:", model_path)

    # --------------------------------------------------------
    # 4. SAVE METRICS AND PREDICTIONS
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    preferred_columns = [
        "Model",
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC_Average_Precision",
        "Brier_Score",
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    results_df = results_df[preferred_columns]

    results_path = OUTPUT_DIR / "step02_model_metrics.csv"
    predictions_path = OUTPUT_DIR / "step02_test_predictions.csv"

    results_df.to_csv(results_path, index=False)
    predictions_output.to_csv(predictions_path, index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # --------------------------------------------------------
    # 5. PLOT ROC CURVES
    # --------------------------------------------------------

    from sklearn.metrics import roc_curve

    plt.figure(figsize=(8, 6))

    for model_name, probabilities in probability_output.items():
        fpr, tpr, _ = roc_curve(y_test, probabilities)

        auc_value = roc_auc_score(y_test, probabilities)

        plt.plot(
            fpr,
            tpr,
            label="{} (AUC = {:.4f})".format(
                model_name, auc_value
            ),
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random baseline",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Step 2: ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()

    roc_path = OUTPUT_DIR / "step02_roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 6. PLOT PRECISION-RECALL CURVES
    # --------------------------------------------------------

    from sklearn.metrics import precision_recall_curve

    plt.figure(figsize=(8, 6))

    for model_name, probabilities in probability_output.items():
        precision_values, recall_values, _ = (
            precision_recall_curve(y_test, probabilities)
        )

        ap_value = average_precision_score(y_test, probabilities)

        plt.plot(
            recall_values,
            precision_values,
            label="{} (AP = {:.4f})".format(
                model_name, ap_value
            ),
        )

    # Positive-class prevalence is the random-ranking baseline.
    baseline_prevalence = float(y_test.mean())

    plt.axhline(
        baseline_prevalence,
        linestyle="--",
        label="Positive prevalence = {:.3f}".format(
            baseline_prevalence
        ),
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Step 2: Precision-Recall Curve Comparison")
    plt.legend()
    plt.tight_layout()

    pr_path = OUTPUT_DIR / "step02_precision_recall_curves.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 7. SAVE SUMMARY REPORT
    # --------------------------------------------------------

    report = {
        "step": 2,
        "dataset": "AI4I 2020 Predictive Maintenance",
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "training_failure_count": positive_count,
        "training_nonfailure_count": negative_count,
        "testing_failure_count": int(y_test.sum()),
        "testing_nonfailure_count": int((y_test == 0).sum()),
        "decision_threshold": DECISION_THRESHOLD,
        "scale_pos_weight": float(scale_pos_weight),
        "models": results,
        "interpretation_note": (
            "Threshold-based metrics use 0.50. "
            "Threshold selection and probability calibration "
            "will be investigated in later steps. "
            "The held-out test set should not be used to tune "
            "the model or select a threshold."
        ),
    }

    report_path = OUTPUT_DIR / "step02_report.json"

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 8. COMPLETION
    # --------------------------------------------------------

    print("\nSaved files:")
    print(results_path)
    print(predictions_path)
    print(roc_path)
    print(pr_path)
    print(report_path)
    print("Saved models in:", MODEL_DIR)

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
