from pathlib import Path
import pandas as pd

root = Path(".")
metrics_path = root / "outputs" / "step10F_repeated_seed_robustness" / "step10F_per_seed_final_metrics.csv"

df = pd.read_csv(metrics_path)
df.columns = df.columns.str.strip()

print("SOURCE COLUMNS:")
print(df.columns.tolist())

required = {
    "seed", "model", "mean_engine_RMSE",
    "mean_engine_MAE", "mean_engine_NASA_score"
}
missing = required - set(df.columns)

if missing:
    print("\nSTOP: Required columns not found:", sorted(missing))
    print("Please share the printed column names.")
    raise SystemExit(1)

df["model"] = df["model"].astype(str).str.strip()

models = ["unconditional_retraining", "safety_gated_model"]
metrics = [
    "mean_engine_RMSE",
    "mean_engine_MAE",
    "mean_engine_NASA_score",
]

for metric in metrics:
    print(f"\n{'=' * 20} {metric} {'=' * 20}")

    wide = df.pivot(index="seed", columns="model", values=metric)

    for model in models:
        if "frozen_baseline" not in wide.columns or model not in wide.columns:
            print(f"{model}: required model column missing")
            continue

        pair = wide[["frozen_baseline", model]].dropna()
        # Positive improvement means the candidate is better.
        improvement = pair["frozen_baseline"] - pair[model]

        improved = int((improvement > 1e-10).sum())
        worsened = int((improvement < -1e-10).sum())
        tied = int((improvement.abs() <= 1e-10).sum())

        print(f"\nCandidate: {model}")
        print("Paired seeds:", len(pair))
        print("Improved:", improved)
        print("Worsened:", worsened)
        print("Tied:", tied)
        print("Count total:", improved + worsened + tied)
        print("Mean absolute improvement:", improvement.mean())
        print("Median absolute improvement:", improvement.median())

        print("\nPer-seed differences (baseline - candidate):")
        print(improvement.to_string())

stats_path = (
    root / "outputs" / "step10G_statistical_robustness"
    / "step10G_paired_statistical_analysis.csv"
)

if stats_path.exists():
    stats = pd.read_csv(stats_path)
    print("\n\nSTORED STEP 10G COUNTS:")
    cols = [
        c for c in [
            "metric", "comparison_model", "n_paired_seeds",
            "seeds_improved", "seeds_worsened", "seeds_tied",
            "sign_test_positive", "sign_test_negative", "sign_test_ties"
        ] if c in stats.columns
    ]
    print(stats[cols].to_string(index=False))
