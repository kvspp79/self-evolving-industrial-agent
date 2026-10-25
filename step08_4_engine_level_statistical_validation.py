
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 8.4
# Engine-level paired bootstrap validation for NASA C-MAPSS FD001
#
# Uses existing validation predictions only.
# Does not retrain models, use test labels for model selection,
# change maintenance thresholds, or deploy any policy.
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS"
)

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "step04"
    / "validation_predictions_FD001.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "step08_4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
CONFIDENCE_LEVEL = 0.95

ENGINE_COL = "unit_id"
ACTUAL_COL = "actual_RUL"
BASELINE_COL = "baseline_predicted_RUL"
TEMPORAL_COL = "temporal_predicted_RUL"


def nasa_score(actual, predicted):
    """
    Standard asymmetric C-MAPSS score.

    Error = predicted RUL - actual RUL.
    Positive error means RUL was overestimated.
    """
    error = np.asarray(predicted) - np.asarray(actual)

    penalties = np.where(
        error < 0,
        np.expm1(-error / 13.0),
        np.expm1(error / 10.0),
    )

    return float(np.sum(penalties))


def calculate_metrics(frame):
    actual = frame[ACTUAL_COL].to_numpy(dtype=float)
    baseline = frame[BASELINE_COL].to_numpy(dtype=float)
    temporal = frame[TEMPORAL_COL].to_numpy(dtype=float)

    return {
        "n_rows": int(len(frame)),
        "n_engines": int(frame[ENGINE_COL].nunique()),
        "baseline_RMSE": float(
            np.sqrt(np.mean((baseline - actual) ** 2))
        ),
        "temporal_RMSE": float(
            np.sqrt(np.mean((temporal - actual) ** 2))
        ),
        "baseline_MAE": float(
            np.mean(np.abs(baseline - actual))
        ),
        "temporal_MAE": float(
            np.mean(np.abs(temporal - actual))
        ),
        "baseline_NASA_score": nasa_score(actual, baseline),
        "temporal_NASA_score": nasa_score(actual, temporal),
    }


def metrics_for_sampled_engines(sampled_frames):
    """
    Concatenate all rows belonging to each sampled engine.
    Repeated engine draws are retained, as required by cluster
    bootstrap resampling.
    """
    sample = pd.concat(sampled_frames, ignore_index=True)

    actual = sample[ACTUAL_COL].to_numpy(dtype=float)
    baseline = sample[BASELINE_COL].to_numpy(dtype=float)
    temporal = sample[TEMPORAL_COL].to_numpy(dtype=float)

    baseline_rmse = np.sqrt(np.mean((baseline - actual) ** 2))
    temporal_rmse = np.sqrt(np.mean((temporal - actual) ** 2))

    baseline_mae = np.mean(np.abs(baseline - actual))
    temporal_mae = np.mean(np.abs(temporal - actual))

    baseline_nasa = nasa_score(actual, baseline)
    temporal_nasa = nasa_score(actual, temporal)

    return {
        "RMSE_temporal_minus_baseline":
            float(temporal_rmse - baseline_rmse),
        "MAE_temporal_minus_baseline":
            float(temporal_mae - baseline_mae),
        "NASA_temporal_minus_baseline":
            float(temporal_nasa - baseline_nasa),
    }


def main():
    print("=" * 72)
    print("NASA C-MAPSS FD001 - STEP 8.4")
    print("ENGINE-LEVEL STATISTICAL VALIDATION")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required validation prediction file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        ENGINE_COL,
        ACTUAL_COL,
        BASELINE_COL,
        TEMPORAL_COL,
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Actual columns: {list(df.columns)}"
        )

    df = df[required].copy()

    for column in [ACTUAL_COL, BASELINE_COL, TEMPORAL_COL]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)

    if df.empty:
        raise ValueError("No valid prediction rows remain after cleaning.")

    if (df[ACTUAL_COL] < 0).any():
        raise ValueError(
            "Negative actual RUL values detected. Check the input file."
        )

    engine_ids = df[ENGINE_COL].drop_duplicates().to_numpy()
    n_engines = len(engine_ids)

    if n_engines < 10:
        warnings.warn(
            "Very few validation engines are available. "
            "Bootstrap intervals may be highly uncertain."
        )

    print(f"Input: {INPUT_FILE}")
    print(f"Valid rows: {len(df):,}")
    print(f"Validation engines: {n_engines}")
    print(f"Bootstrap resamples: {N_BOOTSTRAP}")
    print("Test predictions are not used in this analysis.")

    # --------------------------------------------------------
    # 1. Original validation metrics
    # --------------------------------------------------------
    point_metrics = calculate_metrics(df)

    point_df = pd.DataFrame(
        [
            {
                "model": "baseline",
                "n_rows": point_metrics["n_rows"],
                "n_engines": point_metrics["n_engines"],
                "RMSE": point_metrics["baseline_RMSE"],
                "MAE": point_metrics["baseline_MAE"],
                "NASA_score": point_metrics["baseline_NASA_score"],
            },
            {
                "model": "temporal",
                "n_rows": point_metrics["n_rows"],
                "n_engines": point_metrics["n_engines"],
                "RMSE": point_metrics["temporal_RMSE"],
                "MAE": point_metrics["temporal_MAE"],
                "NASA_score": point_metrics["temporal_NASA_score"],
            },
        ]
    )

    point_df.to_csv(
        OUTPUT_DIR / "validation_metrics_FD001.csv",
        index=False,
    )

    print("\nPoint estimates:")
    print(point_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --------------------------------------------------------
    # 2. Paired cluster bootstrap over engines
    # --------------------------------------------------------
    # Each sampled engine contributes all of its validation rows.
    # The same sampled engines are used for both models.
    rng = np.random.default_rng(RANDOM_SEED)

    engine_frames = {
        engine_id: group.copy()
        for engine_id, group in df.groupby(ENGINE_COL, sort=False)
    }

    bootstrap_results = []

    for iteration in range(N_BOOTSTRAP):
        sampled_ids = rng.choice(
            engine_ids,
            size=n_engines,
            replace=True,
        )

        sampled_frames = [
            engine_frames[engine_id]
            for engine_id in sampled_ids
        ]

        result = metrics_for_sampled_engines(sampled_frames)
        result["bootstrap_iteration"] = iteration + 1
        bootstrap_results.append(result)

    boot = pd.DataFrame(bootstrap_results)

    boot.to_csv(
        OUTPUT_DIR / "bootstrap_differences_FD001.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Confidence intervals for paired differences
    # --------------------------------------------------------
    alpha = 1.0 - CONFIDENCE_LEVEL

    metric_columns = [
        "RMSE_temporal_minus_baseline",
        "MAE_temporal_minus_baseline",
        "NASA_temporal_minus_baseline",
    ]

    point_differences = {
        "RMSE_temporal_minus_baseline":
            point_metrics["temporal_RMSE"] - point_metrics["baseline_RMSE"],
        "MAE_temporal_minus_baseline":
            point_metrics["temporal_MAE"] - point_metrics["baseline_MAE"],
        "NASA_temporal_minus_baseline":
            point_metrics["temporal_NASA_score"]
            - point_metrics["baseline_NASA_score"],
    }

    summary_rows = []

    for column in metric_columns:
        values = boot[column].to_numpy(dtype=float)

        lower = float(np.quantile(values, alpha / 2.0))
        upper = float(np.quantile(values, 1.0 - alpha / 2.0))
        mean_difference = float(np.mean(values))
        observed_difference = float(point_differences[column])

        # Negative difference favors the temporal model.
        # Positive difference favors the baseline.
        if lower > 0:
            interpretation = "CI favors baseline"
        elif upper < 0:
            interpretation = "CI favors temporal"
        else:
            interpretation = "inconclusive: CI includes zero"

        summary_rows.append({
            "metric": column,
            "observed_temporal_minus_baseline": observed_difference,
            "bootstrap_mean_difference": mean_difference,
            "CI_lower": lower,
            "CI_upper": upper,
            "confidence_level": CONFIDENCE_LEVEL,
            "interpretation": interpretation,
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "bootstrap_CI_summary_FD001.csv",
        index=False,
    )

    print("\nPaired engine-bootstrap results:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --------------------------------------------------------
    # 4. Plot the confidence intervals
    # --------------------------------------------------------
    labels = ["RMSE", "MAE", "NASA score"]
    x_positions = np.arange(len(summary_df))

    observed = summary_df[
        "observed_temporal_minus_baseline"
    ].to_numpy()

    lower = summary_df["CI_lower"].to_numpy()
    upper = summary_df["CI_upper"].to_numpy()

    # Keep a metric-specific scale in the CSV; plot each metric's
    # difference and interval in separate panels for interpretability.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for i, ax in enumerate(axes):
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.errorbar(
            observed[i],
            0,
            xerr=np.array([
                [observed[i] - lower[i]],
                [upper[i] - observed[i]],
            ]),
            fmt="o",
            capsize=5,
        )
        ax.set_title(labels[i])
        ax.set_xlabel("Temporal minus baseline")
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Step 8.4 - Paired Engine-Level Bootstrap Confidence Intervals"
    )
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "bootstrap_CI_FD001.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 5. Research report
    # --------------------------------------------------------
    report = {
        "step": "8.4",
        "dataset": "NASA C-MAPSS FD001",
        "analysis": "paired engine-level cluster bootstrap",
        "input_file": str(INPUT_FILE),
        "validation_rows": int(len(df)),
        "validation_engines": int(n_engines),
        "bootstrap_resamples": int(N_BOOTSTRAP),
        "random_seed": RANDOM_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "point_metrics": point_metrics,
        "difference_definition":
            "temporal metric minus baseline metric; lower is better",
        "bootstrap_summary": summary_rows,
        "limitations": [
            "Bootstrap resamples validation engines, not individual rows.",
            "Only the existing validation split is analyzed.",
            "A small number of validation engines yields uncertain intervals.",
            "Confidence intervals are exploratory and are not a substitute "
            "for independent external validation.",
            "The test set is not used for model selection.",
            "No model was retrained and no policy was changed.",
        ],
    }

    report_path = OUTPUT_DIR / "step08_4_FD001_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print("\nSTEP 8.4 COMPLETED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metrics: {OUTPUT_DIR / 'validation_metrics_FD001.csv'}")
    print(f"Bootstrap draws: {OUTPUT_DIR / 'bootstrap_differences_FD001.csv'}")
    print(f"Confidence intervals: {OUTPUT_DIR / 'bootstrap_CI_summary_FD001.csv'}")
    print(f"Plot: {plot_path}")
    print(f"Report: {report_path}")
    print("\nNo model was retrained.")
    print("No maintenance policy was modified or deployed.")


if __name__ == "__main__":
    main()
