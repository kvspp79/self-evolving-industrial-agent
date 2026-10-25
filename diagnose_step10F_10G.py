import pandas as pd
from pathlib import Path

root = Path("outputs")

files = {
    "Step 10F metrics": root / "step10F_repeated_seed_robustness" / "step10F_per_seed_final_metrics.csv",
    "Step 10F gate decisions": root / "step10F_repeated_seed_robustness" / "step10F_gate_decisions.csv",
    "Step 10G paired statistics": root / "step10G_statistical_robustness" / "step10G_paired_statistical_analysis.csv",
}

for label, path in files.items():
    print(f"\n{'=' * 15} {label} {'=' * 15}")
    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        continue

    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if label == "Step 10F metrics":
        print("\nCounts by model:")
        print(df.groupby("model", dropna=False).size().to_string())

        if {"seed", "model"}.issubset(df.columns):
            print("\nDuplicate seed/model rows:")
            duplicates = df[df.duplicated(["seed", "model"], keep=False)]
            print(duplicates.to_string(index=False) if len(duplicates) else "None")

            print("\nSeed counts by model:")
            print(df.groupby("model")["seed"].nunique().to_string())

            print("\nSeed/model combinations missing:")
            seeds = sorted(df["seed"].dropna().unique())
            models = sorted(df["model"].dropna().unique())
            expected = pd.MultiIndex.from_product(
                [seeds, models], names=["seed", "model"]
            ).to_frame(index=False)
            actual = df[["seed", "model"]].drop_duplicates()
            missing = expected.merge(
                actual, on=["seed", "model"], how="left", indicator=True
            )
            missing = missing[missing["_merge"] == "left_only"]
            print(missing[["seed", "model"]].to_string(index=False)
                  if len(missing) else "None")

    elif label == "Step 10F gate decisions":
        print(df.to_string(index=False))

    else:
        print(df.to_string(index=False))
