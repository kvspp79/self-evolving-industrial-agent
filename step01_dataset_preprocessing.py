
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# STEP 1: NASA C-MAPSS DATA INSPECTION AND PREPROCESSING
# Initial subset: FD001
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "CMAPSSData"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step01"

SUBSET = "FD001"

# C-MAPSS column definitions:
# 2 identifiers + 3 operating settings + 21 sensor measurements
COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)


def read_cmapss_file(file_path):
    """Read a whitespace-separated C-MAPSS text file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset file was not found:\n{file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    # Some distributed C-MAPSS files contain trailing empty columns.
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{file_path.name} has {df.shape[1]} columns; "
            f"expected {len(COLUMNS)}. "
            "Check that you are using the original C-MAPSS files."
        )

    df.columns = COLUMNS
    return df


def main():
    print("=" * 65)
    print("NASA C-MAPSS — STEP 1: DATA PREPROCESSING")
    print(f"Subset: {SUBSET}")
    print("=" * 65)

    train_path = DATA_DIR / f"train_{SUBSET}.txt"
    test_path = DATA_DIR / f"test_{SUBSET}.txt"
    rul_path = DATA_DIR / f"RUL_{SUBSET}.txt"

    # 1. Load the raw files
    train_df = read_cmapss_file(train_path)
    test_df = read_cmapss_file(test_path)

    if not rul_path.exists():
        raise FileNotFoundError(f"RUL file not found: {rul_path}")

    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)
    rul_values = pd.to_numeric(rul_df.iloc[:, 0], errors="raise")

    if len(rul_values) != test_df["unit_id"].nunique():
        raise ValueError(
            "The number of RUL values does not match the number "
            "of test engines."
        )

    # 2. Basic validation
    for name, df in [("training", train_df), ("test", test_df)]:
        if df.isna().any().any():
            raise ValueError(f"Missing values found in {name} data.")

        if (df["cycle"] < 1).any():
            raise ValueError(f"Invalid cycle numbers found in {name} data.")

        if df.duplicated(["unit_id", "cycle"]).any():
            raise ValueError(
                f"Duplicate unit_id/cycle rows found in {name} data."
            )

    if train_df["unit_id"].nunique() != train_df["unit_id"].max():
        raise ValueError("Training unit IDs are not the expected 1..N range.")

    # 3. Calculate training RUL labels.
    # Training engines run to their recorded end-of-life cycle.
    max_cycle = train_df.groupby("unit_id")["cycle"].transform("max")
    train_df["RUL"] = max_cycle - train_df["cycle"]

    # 4. Attach provided RUL values to test engines.
    # The provided value is RUL AFTER each engine's final test cycle.
    test_last_cycle = test_df.groupby("unit_id")["cycle"].transform("max")
    test_df["RUL_at_last_test_cycle"] = test_df["unit_id"].map(
        dict(enumerate(rul_values.tolist(), start=1))
    )

    test_df["RUL"] = (
        test_df["RUL_at_last_test_cycle"]
        + test_last_cycle
        - test_df["cycle"]
    )

    test_df = test_df.drop(columns=["RUL_at_last_test_cycle"])

    # 5. Save processed datasets without modifying raw files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_output = OUTPUT_DIR / f"train_{SUBSET}_processed.csv"
    test_output = OUTPUT_DIR / f"test_{SUBSET}_processed.csv"
    rul_output = OUTPUT_DIR / f"RUL_{SUBSET}_reference.csv"
    report_output = OUTPUT_DIR / f"step01_{SUBSET}_report.json"

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    pd.DataFrame({
        "unit_id": range(1, len(rul_values) + 1),
        "RUL_after_last_test_cycle": rul_values,
    }).to_csv(rul_output, index=False)

    # 6. Create a compact dataset report.
    report = {
        "dataset": "NASA C-MAPSS",
        "subset": SUBSET,
        "training_rows": int(len(train_df)),
        "training_engines": int(train_df["unit_id"].nunique()),
        "test_rows": int(len(test_df)),
        "test_engines": int(test_df["unit_id"].nunique()),
        "number_of_features_before_RUL": 26,
        "number_of_sensor_features": 21,
        "training_missing_values": int(train_df.isna().sum().sum()),
        "test_missing_values": int(test_df.isna().sum().sum()),
        "training_RUL_min": int(train_df["RUL"].min()),
        "training_RUL_max": int(train_df["RUL"].max()),
        "test_RUL_min": int(test_df["RUL"].min()),
        "test_RUL_max": int(test_df["RUL"].max()),
        "outputs": [
            str(train_output.relative_to(PROJECT_ROOT)),
            str(test_output.relative_to(PROJECT_ROOT)),
            str(rul_output.relative_to(PROJECT_ROOT)),
        ],
    }

    report_output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # 7. Print summary.
    print("\nPREPROCESSING SUCCESSFUL")
    print("-" * 40)
    print(f"Training rows:       {len(train_df):,}")
    print(f"Training engines:    {train_df['unit_id'].nunique():,}")
    print(f"Test rows:           {len(test_df):,}")
    print(f"Test engines:        {test_df['unit_id'].nunique():,}")
    print(f"Training RUL range:  {train_df['RUL'].min()} to {train_df['RUL'].max()}")
    print(f"Test RUL range:      {test_df['RUL'].min()} to {test_df['RUL'].max()}")
    print(f"\nOutput folder: {OUTPUT_DIR}")
    print(f"Report: {report_output.name}")


if __name__ == "__main__":
    main()
