from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


# ============================================================
# STEP 1: AI4I DATASET UNDERSTANDING AND PREPROCESSING
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Project folder is one level above src.
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "ai4i2020.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():

    print("=" * 60)
    print("STEP 1: AI4I DATASET PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------------

    print("\nDataset location:")
    print(DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset file was not found. Expected location: "
            + str(DATA_PATH)
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    print("\nDataset loaded successfully.")
    print("Dataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    # --------------------------------------------------------
    # 2. VALIDATE TARGET COLUMN
    # --------------------------------------------------------

    target = "Machine failure"

    if target not in df.columns:
        raise ValueError(
            "Machine failure column is missing. Columns found: "
            + str(df.columns.tolist())
        )

    df[target] = pd.to_numeric(df[target], errors="raise")

    if df[target].isna().any():
        raise ValueError("The target contains missing values.")

    if not set(df[target].unique()).issubset({0, 1}):
        raise ValueError("The target must contain only 0 and 1.")

    # --------------------------------------------------------
    # 3. CHECK DATA QUALITY
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("DATA QUALITY")
    print("-" * 60)

    missing = df.isnull().sum()

    print("\nMissing values by column:")
    print(missing.to_string())

    print("\nTotal missing values:", int(missing.sum()))
    print("Exact duplicate rows:", int(df.duplicated().sum()))

    # We preserve the original data and do not delete rows
    # automatically.

    # --------------------------------------------------------
    # 4. TARGET CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("MACHINE FAILURE DISTRIBUTION")
    print("-" * 60)

    counts = df[target].value_counts().sort_index()

    summary = pd.DataFrame({
        "Count": counts,
        "Percentage": (counts / len(df) * 100).round(3)
    })

    print(summary.to_string())

    failure_rate = df[target].mean() * 100
    print("\nFailure rate: {:.3f}%".format(failure_rate))

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(data=df, x=target, ax=ax)

    ax.set_title("AI4I Machine Failure Distribution")
    ax.set_xlabel("Machine Failure (0 = No, 1 = Yes)")
    ax.set_ylabel("Number of Records")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "step01_target_distribution.png",
        dpi=300
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 5. FAILURE MODE SUMMARY
    # --------------------------------------------------------

    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    available_modes = [
        column for column in failure_modes
        if column in df.columns
    ]

    if available_modes:

        mode_counts = df[available_modes].sum().sort_values(
            ascending=False
        )

        print("\nFailure-mode counts:")
        print(mode_counts.to_string())

        fig, ax = plt.subplots(figsize=(8, 4))

        mode_counts.plot(kind="bar", ax=ax)

        ax.set_title("AI4I Failure-Mode Counts")
        ax.set_xlabel("Failure Mode")
        ax.set_ylabel("Number of Positive Labels")

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / "step01_failure_modes.png",
            dpi=300
        )
        plt.close(fig)

    # --------------------------------------------------------
    # 6. SENSOR STATISTICS AND DISTRIBUTIONS
    # --------------------------------------------------------

    sensor_candidates = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    sensor_cols = [
        column for column in sensor_candidates
        if column in df.columns
    ]

    if sensor_cols:

        print("\nSensor descriptive statistics:")
        print(df[sensor_cols].describe().round(3).to_string())

        df[sensor_cols].hist(
            figsize=(12, 8),
            bins=30,
            edgecolor="black"
        )

        plt.suptitle("AI4I Sensor Feature Distributions")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "step01_sensor_distributions.png",
            dpi=300
        )

        plt.close()

    else:
        print("\nWarning: Expected sensor columns were not found.")

    # --------------------------------------------------------
    # 7. SENSOR CORRELATION
    # --------------------------------------------------------

    excluded_from_correlation = [
        "UDI",
        "UID",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    correlation_cols = [
        column for column in numeric_cols
        if column not in excluded_from_correlation
    ]

    if len(correlation_cols) >= 2:

        correlation = df[correlation_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax
        )

        ax.set_title("Correlation Between Operating Features")

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR / "step01_sensor_correlation.png",
            dpi=300
        )

        plt.close(fig)

    # --------------------------------------------------------
    # 8. PREPARE INPUT FEATURES
    # --------------------------------------------------------

    # Exclude failure labels to prevent target leakage.
    outcome_cols = [
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    # Identifiers are excluded from predictive features.
    identifier_cols = [
        "UDI",
        "UID",
        "Product ID"
    ]

    excluded_cols = [
        column
        for column in outcome_cols + identifier_cols
        if column in df.columns
    ]

    X = df.drop(columns=excluded_cols).copy()
    y = df[target].astype(int).copy()

    if X.empty:
        raise ValueError("No input features remain.")

    if X.isnull().any().any():
        raise ValueError(
            "Input features contain missing values. "
            "Investigate them before proceeding."
        )

    if y.nunique() != 2:
        raise ValueError("Both target classes must be present.")

    print("\nExcluded columns:")
    print(excluded_cols)

    print("\nCandidate input features:")
    print(X.columns.tolist())

    print("\nFeature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    # --------------------------------------------------------
    # 9. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining feature shape:", X_train.shape)
    print("Testing feature shape:", X_test.shape)

    print(
        "Training failure rate: {:.3f}%".format(
            y_train.mean() * 100
        )
    )

    print(
        "Testing failure rate: {:.3f}%".format(
            y_test.mean() * 100
        )
    )

    # --------------------------------------------------------
    # 10. SAVE DATA SPLITS
    # --------------------------------------------------------

    X_train.to_csv(
        OUTPUT_DIR / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        OUTPUT_DIR / "X_test.csv",
        index=False
    )

    y_train.to_frame(name=target).to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False
    )

    y_test.to_frame(name=target).to_csv(
        OUTPUT_DIR / "y_test.csv",
        index=False
    )

    # --------------------------------------------------------
    # 11. SAVE STEP 1 REPORT
    # --------------------------------------------------------

    report_lines = [
        "AI4I STEP 1 REPORT",
        "=" * 50,
        "Dataset rows: " + str(len(df)),
        "Dataset columns: " + str(len(df.columns)),
        "Missing values: " + str(int(df.isnull().sum().sum())),
        "Exact duplicate rows: " + str(int(df.duplicated().sum())),
        "Failure records: " + str(int(y.sum())),
        "Non-failure records: " + str(int((y == 0).sum())),
        "Failure rate: {:.3f}%".format(y.mean() * 100),
        "Excluded columns: " + str(excluded_cols),
        "Input features: " + str(X.columns.tolist()),
        "Training rows: " + str(len(X_train)),
        "Testing rows: " + str(len(X_test)),
        "",
        "Limitation: AI4I is a synthetic benchmark dataset.",
        "UDI order is not genuine per-machine degradation history."
    ]

    report_path = OUTPUT_DIR / "step01_report.txt"

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("Outputs saved in:", OUTPUT_DIR)
    print("Report saved as:", report_path.name)


if __name__ == "__main__":
    main()