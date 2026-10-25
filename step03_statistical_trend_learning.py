
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# STEP 3: STATISTICAL SENSOR-TREND LEARNING
# NASA C-MAPSS FD001
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "outputs" / "step01"
OUTPUT_DIR = ROOT / "outputs" / "step03"
SUBSET = "FD001"

SENSORS = [f"sensor_{i:02d}" for i in range(1, 22)]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]


def safe_correlation(x, y, method="spearman"):
    """Return a correlation, or NaN when it is undefined."""
    x = pd.Series(x)
    y = pd.Series(y)
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan

    return float(x.corr(y, method=method))


def calculate_engine_slopes(df, columns):
    """
    Calculate a linear sensor-versus-cycle slope for each engine.
    Slopes describe observed trends; they do not prove causation.
    """
    records = []

    for unit_id, group in df.groupby("unit_id"):
        group = group.sort_values("cycle")
        cycles = group["cycle"].to_numpy(dtype=float)

        record = {"unit_id": int(unit_id)}

        for sensor in columns:
            values = group[sensor].to_numpy(dtype=float)

            if len(values) >= 2 and np.ptp(cycles) > 0:
                slope = np.polyfit(cycles, values, 1)[0]
            else:
                slope = np.nan

            record[f"{sensor}_slope"] = float(slope)

        records.append(record)

    return pd.DataFrame(records)


def main():
    print("=" * 65)
    print("NASA C-MAPSS — STEP 3: STATISTICAL SENSOR-TREND LEARNING")
    print(f"Subset: {SUBSET}")
    print("=" * 65)

    train_path = INPUT_DIR / f"train_{SUBSET}_processed.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Processed training data not found: {train_path}\n"
            "Run Step 1 first."
        )

    df = pd.read_csv(train_path)

    required = {"unit_id", "cycle", "RUL", *SENSORS, *SETTINGS}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Global sensor statistics
    # --------------------------------------------------------
    statistics = df[SENSORS].agg(
        ["mean", "std", "min", "max"]
    ).T
    statistics.index.name = "sensor"
    statistics = statistics.reset_index()

    statistics["range"] = statistics["max"] - statistics["min"]
    statistics["coefficient_of_variation"] = (
        statistics["std"] / statistics["mean"].abs().replace(0, np.nan)
    )

    # --------------------------------------------------------
    # 2. Association between each sensor and RUL
    # These are observational associations across training rows.
    # They are not causal effects.
    # --------------------------------------------------------
    associations = []

    for sensor in SENSORS:
        associations.append({
            "sensor": sensor,
            "pearson_correlation_with_RUL": safe_correlation(
                df[sensor], df["RUL"], "pearson"
            ),
            "spearman_correlation_with_RUL": safe_correlation(
                df[sensor], df["RUL"], "spearman"
            ),
            "unique_values": int(df[sensor].nunique()),
            "standard_deviation": float(df[sensor].std()),
        })

    association_df = pd.DataFrame(associations)

    # --------------------------------------------------------
    # 3. Per-engine linear trends over operating cycles
    # --------------------------------------------------------
    slope_df = calculate_engine_slopes(df, SENSORS)

    slope_records = []

    for sensor in SENSORS:
        slope_column = f"{sensor}_slope"
        slopes = slope_df[slope_column].dropna()

        slope_records.append({
            "sensor": sensor,
            "mean_slope_per_cycle": float(slopes.mean()),
            "median_slope_per_cycle": float(slopes.median()),
            "slope_std_across_engines": float(slopes.std()),
            "engines_with_positive_slope": int((slopes > 0).sum()),
            "engines_with_negative_slope": int((slopes < 0).sum()),
            "engines_with_zero_slope": int((slopes == 0).sum()),
            "engines_analyzed": int(len(slopes)),
        })

    slope_summary = pd.DataFrame(slope_records)

    # --------------------------------------------------------
    # 4. Rank sensors using several descriptive indicators.
    # This is a screening ranking, not proof of predictive value.
    # --------------------------------------------------------
    ranking = (
        association_df
        .merge(slope_summary, on="sensor")
        .merge(
            statistics[["sensor", "mean", "std", "range",
                        "coefficient_of_variation"]],
            on="sensor",
        )
    )

    ranking["absolute_spearman_RUL_correlation"] = (
        ranking["spearman_correlation_with_RUL"].abs()
    )

    ranking = ranking.sort_values(
        "absolute_spearman_RUL_correlation",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    ranking.insert(0, "association_rank", np.arange(1, len(ranking) + 1))

    # --------------------------------------------------------
    # 5. Plot average sensor trends over normalized engine life.
    # Each engine is mapped to 0–100% of its recorded training life.
    # This visualization is descriptive, not a deployable forecast.
    # --------------------------------------------------------
    normalized_parts = []

    for unit_id, group in df.groupby("unit_id"):
        group = group.sort_values("cycle").copy()
        max_cycle = group["cycle"].max()

        if max_cycle <= 1:
            continue

        group["life_percent"] = (
            (group["cycle"] - 1) / (max_cycle - 1) * 100
        )

        group["life_bin"] = (
            group["life_percent"] // 5
        ).clip(upper=19).astype(int)

        grouped = group.groupby("life_bin")[SENSORS].mean()
        normalized_parts.append(grouped)

    if not normalized_parts:
        raise ValueError("Unable to calculate normalized life trends.")

    # Equal engine weighting: average each engine's bin mean first.
    normalized = pd.concat(
        normalized_parts,
        keys=range(len(normalized_parts)),
        names=["engine_index", "life_bin"],
    )

    life_trends = normalized.groupby("life_bin").mean()
    life_trends.index = life_trends.index * 5 + 2.5
    life_trends.index.name = "normalized_life_percent"

    # --------------------------------------------------------
    # 6. Save outputs
    # --------------------------------------------------------
    stats_path = OUTPUT_DIR / f"sensor_statistics_{SUBSET}.csv"
    association_path = OUTPUT_DIR / f"sensor_RUL_associations_{SUBSET}.csv"
    slopes_path = OUTPUT_DIR / f"sensor_engine_slopes_{SUBSET}.csv"
    slope_summary_path = OUTPUT_DIR / f"sensor_trend_summary_{SUBSET}.csv"
    ranking_path = OUTPUT_DIR / f"sensor_trend_ranking_{SUBSET}.csv"
    life_trend_path = OUTPUT_DIR / f"normalized_life_trends_{SUBSET}.csv"

    statistics.to_csv(stats_path, index=False)
    association_df.to_csv(association_path, index=False)
    slope_df.to_csv(slopes_path, index=False)
    slope_summary.to_csv(slope_summary_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    life_trends.to_csv(life_trend_path)

    # Plot 1: top sensors by absolute Spearman association with RUL.
    top = ranking.head(10).sort_values(
        "spearman_correlation_with_RUL"
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        top["sensor"],
        top["spearman_correlation_with_RUL"],
    )
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Spearman correlation with RUL")
    plt.ylabel("Sensor")
    plt.title("Top 10 Sensor–RUL Associations (FD001)")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"top_sensor_RUL_associations_{SUBSET}.png",
        dpi=160,
    )
    plt.close()

    # Plot 2: normalized life trends for the top five sensors.
    top_sensors = ranking.head(5)["sensor"].tolist()

    plt.figure(figsize=(10, 6))
    for sensor in top_sensors:
        plt.plot(
            life_trends.index,
            life_trends[sensor],
            label=sensor,
        )

    plt.xlabel("Normalized recorded engine life (%)")
    plt.ylabel("Mean sensor measurement")
    plt.title("Sensor Trends Across Normalized Training Life (FD001)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"top_sensor_life_trends_{SUBSET}.png",
        dpi=160,
    )
    plt.close()

    # Plot 3: distribution of per-engine slopes for the top sensors.
    slope_plot_columns = [
        f"{sensor}_slope" for sensor in top_sensors
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [slope_df[column].dropna() for column in slope_plot_columns],
        tick_labels=top_sensors,
    )
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Sensor")
    plt.ylabel("Linear slope per cycle")
    plt.title("Per-Engine Sensor Trend Variability (FD001)")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"sensor_slope_distributions_{SUBSET}.png",
        dpi=160,
    )
    plt.close()

    # --------------------------------------------------------
    # 7. Research summary
    # --------------------------------------------------------
    top_records = ranking.head(10).replace(
        {np.nan: None}
    ).to_dict(orient="records")

    report = {
        "dataset": "NASA C-MAPSS",
        "subset": SUBSET,
        "training_rows": int(len(df)),
        "training_engines": int(df["unit_id"].nunique()),
        "sensor_count": len(SENSORS),
        "methods": [
            "Descriptive sensor statistics",
            "Pearson and Spearman sensor-RUL associations",
            "Per-engine linear trend slopes",
            "Equal-engine-weighted normalized-life trend summaries",
        ],
        "top_10_sensors_by_absolute_spearman_association": top_records,
        "interpretation_limitations": [
            "Sensor-RUL associations are observational and not causal.",
            "Rows from the same engine are not independent observations.",
            "Trend slopes summarize recorded training trajectories; they are not forecasts.",
            "Normalized-life plots align engines by recorded lifetime and are descriptive.",
            "Sensor ranking here is exploratory and must be validated in later predictive experiments.",
        ],
        "outputs": [
            str(path.relative_to(ROOT))
            for path in [
                stats_path, association_path, slopes_path,
                slope_summary_path, ranking_path, life_trend_path,
            ]
        ],
    }

    report_path = OUTPUT_DIR / f"step03_{SUBSET}_report.json"
    report_path.write_text(
        json.dumps(report, indent=4, allow_nan=False, default=str),
        encoding="utf-8",
    )

    print("\nSTEP 3 COMPLETED SUCCESSFULLY")
    print("-" * 45)
    print(f"Training rows:   {len(df):,}")
    print(f"Training engines:{df['unit_id'].nunique():>5}")
    print("\nTOP 10 SENSOR-RUL ASSOCIATIONS")
    print(
        ranking[
            ["sensor", "spearman_correlation_with_RUL",
             "mean_slope_per_cycle", "slope_std_across_engines"]
        ].head(10).to_string(index=False)
    )

    print(f"\nOutput folder: {OUTPUT_DIR}")
    print(f"Report: {report_path.name}")
    print("Three diagnostic plots saved.")


if __name__ == "__main__":
    main()
