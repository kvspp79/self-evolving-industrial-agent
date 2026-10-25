
from pathlib import Path
import json
import inspect
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import sklearn
from sklearn.base import clone
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.utils.validation import check_is_fitted


# ============================================================
# STEP 3: PROBABILITY CALIBRATION
# Project: A Self-Evolving Explainable Rational Agent
# Dataset: AI4I 2020 Predictive Maintenance
# ============================================================

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.50
CALIBRATION_FOLDS = 5
CALIBRATION_METHODS = ["sigmoid", "isotonic"]
NUMBER_OF_BINS = 10

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Machine failure"


# ------------------------------------------------------------
# 1. LOAD AND VALIDATE INPUT FILES
# ------------------------------------------------------------

def load_data():
    required_files = {
        "X_train": OUTPUT_DIR / "X_train.csv",
        "X_test": OUTPUT_DIR / "X_test.csv",
        "y_train": OUTPUT_DIR / "y_train.csv",
        "y_test": OUTPUT_DIR / "y_test.csv",
        "xgboost_model": MODEL_DIR / "XGBoost.joblib",
    }

    missing = [
        str(path)
        for path in required_files.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "The following required files are missing:\n"
            + "\n".join(missing)
            + "\nRun Step 1 and Step 2 successfully before Step 3."
        )

    X_train = pd.read_csv(required_files["X_train"])
    X_test = pd.read_csv(required_files["X_test"])

    y_train_df = pd.read_csv(required_files["y_train"])
    y_test_df = pd.read_csv(required_files["y_test"])

    if TARGET not in y_train_df.columns:
        raise ValueError(
            "Column '{}' was not found in y_train.csv. "
            "Check the target column name saved by Step 1.".format(TARGET)
        )

    if TARGET not in y_test_df.columns:
        raise ValueError(
            "Column '{}' was not found in y_test.csv. "
            "Check the target column name saved by Step 1.".format(TARGET)
        )

    y_train = y_train_df[TARGET].astype(int)
    y_test = y_test_df[TARGET].astype(int)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "X_train and X_test have different feature columns."
        )

    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing data is empty.")

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "Both training and testing targets must contain "
            "failure and non-failure examples."
        )

    if not set(X_train.columns).issubset(set(X_test.columns)):
        raise ValueError("Test features do not match training features.")

    original_model = joblib.load(required_files["xgboost_model"])

    # Confirm that the loaded model is fitted.
    check_is_fitted(original_model)

    print("Dataset and model loaded successfully.")
    print("scikit-learn version:", sklearn.__version__)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Training failures:", int(y_train.sum()))
    print("Testing failures:", int(y_test.sum()))
    print("Features:", X_train.columns.tolist())

    return X_train, X_test, y_train, y_test, original_model


# ------------------------------------------------------------
# 2. METRIC HELPERS
# ------------------------------------------------------------

def expected_calibration_error(y_true, probabilities, n_bins=10):
    """
    Calculate a weighted expected calibration error (ECE).

    Each bin contributes its proportion of samples multiplied
    by the absolute difference between average confidence and
    observed failure frequency.
    """

    y_array = np.asarray(y_true, dtype=int)
    p_array = np.asarray(probabilities, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (p_array >= lower) & (p_array <= upper)
        else:
            mask = (p_array >= lower) & (p_array < upper)

        if np.any(mask):
            average_probability = float(p_array[mask].mean())
            observed_frequency = float(y_array[mask].mean())
            bin_weight = float(mask.mean())

            ece += bin_weight * abs(
                average_probability - observed_frequency
            )

    return float(ece)


def calculate_metrics(y_true, probabilities, model_name):
    """Evaluate ranking, probability quality and threshold decisions."""

    probabilities = np.asarray(probabilities, dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "{} produced non-finite probabilities.".format(model_name)
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "{} produced probabilities outside [0, 1].".format(model_name)
        )

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    return {
        "Model": model_name,
        "Threshold": DECISION_THRESHOLD,
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
        "Log_Loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
        "ECE_10_Bins": expected_calibration_error(
            y_true, probabilities, NUMBER_OF_BINS
        ),
        "Predicted_Failures": int(predictions.sum()),
    }


# ------------------------------------------------------------
# 3. CREATE CALIBRATOR COMPATIBLY ACROSS SKLEARN VERSIONS
# ------------------------------------------------------------

def make_calibrator(base_estimator, method):
    """
    Construct CalibratedClassifierCV using the parameter name
    supported by the installed scikit-learn version.
    """

    parameters = inspect.signature(
        CalibratedClassifierCV
    ).parameters

    if "estimator" in parameters:
        return CalibratedClassifierCV(
            estimator=clone(base_estimator),
            method=method,
            cv=CALIBRATION_FOLDS,
        )

    if "base_estimator" in parameters:
        return CalibratedClassifierCV(
            base_estimator=clone(base_estimator),
            method=method,
            cv=CALIBRATION_FOLDS,
        )

    raise RuntimeError(
        "Could not identify the estimator parameter for "
        "CalibratedClassifierCV in this scikit-learn version."
    )


# ------------------------------------------------------------
# 4. SAVE CALIBRATION PLOT
# ------------------------------------------------------------

def save_calibration_plot(y_test, probability_sets):
    plt.figure(figsize=(9, 7))

    for model_name, probabilities in probability_sets.items():
        observed_rate, mean_probability = calibration_curve(
            y_test,
            probabilities,
            n_bins=NUMBER_OF_BINS,
            strategy="quantile",
        )

        plt.plot(
            mean_probability,
            observed_rate,
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel("Mean predicted failure probability")
    plt.ylabel("Observed failure frequency")
    plt.title("Step 3: Probability Calibration Curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "step03_calibration_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


# ------------------------------------------------------------
# 5. SAVE BRIER SCORE COMPARISON
# ------------------------------------------------------------

def save_brier_plot(results_df):
    plt.figure(figsize=(9, 6))

    plt.bar(
        results_df["Model"],
        results_df["Brier_Score"],
    )

    plt.ylabel("Brier score (lower is better)")
    plt.title("Step 3: Brier Score Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "step03_brier_score_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


# ------------------------------------------------------------
# 6. MAIN EXPERIMENT
# ------------------------------------------------------------

def main():
    print("=" * 72)
    print("STEP 3: PROBABILITY CALIBRATION")
    print("=" * 72)

    X_train, X_test, y_train, y_test, original_model = load_data()

    # The original model is already fitted using the training set.
    # Do not fit or tune it on the test data.
    original_probabilities = original_model.predict_proba(
        X_test
    )[:, 1]

    probability_sets = {
        "Original XGBoost": original_probabilities
    }

    calibrated_models = {}
    calibration_errors = {}

    # --------------------------------------------------------
    # Fit calibration using training data only.
    # The five-fold CV prevents calibration from simply fitting
    # probabilities on the same rows used to fit each fold model.
    # --------------------------------------------------------

    for method in CALIBRATION_METHODS:
        print("\n" + "-" * 72)
        print("Fitting {} calibration".format(method.upper()))
        print("-" * 72)

        try:
            calibrator = make_calibrator(original_model, method)

            calibrator.fit(X_train, y_train)

            probabilities = calibrator.predict_proba(
                X_test
            )[:, 1]

            probability_sets[
                "{} calibration".format(method.title())
            ] = probabilities

            calibrated_models[method] = calibrator

            model_path = (
                MODEL_DIR / "XGBoost_{}_calibrated.joblib".format(method)
            )

            joblib.dump(calibrator, model_path)

            print("Calibration completed.")
            print("Saved calibrated model:", model_path)

        except Exception as exc:
            calibration_errors[method] = str(exc)

            print(
                "WARNING: {} calibration failed: {}".format(
                    method, exc
                )
            )

    if len(probability_sets) < 2:
        raise RuntimeError(
            "Both calibration methods failed. "
            "Review the error messages above before continuing."
        )

    # --------------------------------------------------------
    # Evaluate all probability sets on the same held-out test set.
    # The test labels are not used to fit the calibrators.
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 72)

    result_rows = []

    for model_name, probabilities in probability_sets.items():
        metrics = calculate_metrics(
            y_test,
            probabilities,
            model_name,
        )

        result_rows.append(metrics)

        print("\n{}".format(model_name))

        for name, value in metrics.items():
            if name != "Model":
                print("{}: {}".format(name, value))

    results_df = pd.DataFrame(result_rows)

    results_path = OUTPUT_DIR / "step03_calibration_metrics.csv"
    results_df.to_csv(results_path, index=False)

    # --------------------------------------------------------
    # Save all test probabilities for future steps.
    # Keep the original probabilities for fair comparisons.
    # --------------------------------------------------------

    probability_df = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Original_XGBoost_Probability": probability_sets[
            "Original XGBoost"
        ],
    })

    if "Sigmoid calibration" in probability_sets:
        probability_df["Sigmoid_Probability"] = probability_sets[
            "Sigmoid calibration"
        ]

    if "Isotonic calibration" in probability_sets:
        probability_df["Isotonic_Probability"] = probability_sets[
            "Isotonic calibration"
        ]

    probability_path = OUTPUT_DIR / "step03_calibrated_probabilities.csv"
    probability_df.to_csv(probability_path, index=False)

    # --------------------------------------------------------
    # Plot reliability curves and Brier scores.
    # --------------------------------------------------------

    calibration_plot_path = save_calibration_plot(
        y_test, probability_sets
    )

    brier_plot_path = save_brier_plot(results_df)

    # --------------------------------------------------------
    # Create summary report.
    # --------------------------------------------------------

    best_brier_row = results_df.loc[
        results_df["Brier_Score"].idxmin()
    ]

    report = {
        "step": 3,
        "project": (
            "A Self-Evolving Explainable Rational Agent for "
            "Adaptive Industrial Decision Intelligence Using "
            "Statistical Trend Learning"
        ),
        "dataset": "AI4I 2020 Predictive Maintenance",
        "scikit_learn_version": sklearn.__version__,
        "calibration_methods_attempted": CALIBRATION_METHODS,
        "calibration_folds": CALIBRATION_FOLDS,
        "decision_threshold": DECISION_THRESHOLD,
        "number_of_bins_for_ece": NUMBER_OF_BINS,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "test_failure_count": int(y_test.sum()),
        "test_failure_rate": float(y_test.mean()),
        "metrics": result_rows,
        "lowest_test_brier_score_model": str(best_brier_row["Model"]),
        "lowest_test_brier_score": float(best_brier_row["Brier_Score"]),
        "calibration_errors": calibration_errors,
        "interpretation_note": (
            "Calibration models were fitted using training data "
            "with cross-validation. The held-out test set was used "
            "for evaluation only. The lowest test Brier score is "
            "descriptive and must not be used to repeatedly tune "
            "methods against this same test set. ECE depends on "
            "the chosen binning scheme. Calibration quality should "
            "be interpreted alongside discrimination and decision costs."
        ),
    }

    report_path = OUTPUT_DIR / "step03_report.json"

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("CALIBRATION COMPARISON")
    print("=" * 72)

    print(
        results_df[
            [
                "Model",
                "Brier_Score",
                "Log_Loss",
                "ECE_10_Bins",
                "ROC_AUC",
                "PR_AUC_Average_Precision",
                "F1",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs:")
    print(results_path)
    print(probability_path)
    print(calibration_plot_path)
    print(brier_plot_path)
    print(report_path)

    print("\nSaved calibrated models:")
    for method in calibrated_models:
        print(
            MODEL_DIR
            / "XGBoost_{}_calibrated.joblib".format(method)
        )

    if calibration_errors:
        print("\nCalibration warnings:")
        for method, error in calibration_errors.items():
            print("{}: {}".format(method, error))

    print("\n" + "=" * 72)
    print("STEP 3 COMPLETED")
    print("=" * 72)


if __name__ == "__main__":
    main()
