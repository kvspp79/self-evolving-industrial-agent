"""
STEP 7: STATISTICAL CONDITION-RISK TREND LEARNING
Project: Self-Evolving Explainable Rational Agent for
         Adaptive Industrial Decision Intelligence

Dataset: AI4I 2020 Predictive Maintenance

IMPORTANT:
This step studies statistical associations between sensor conditions
and machine failures. AI4I does not provide chronological sensor
histories for individual machines, so UDI order is NOT treated as time.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("default")

# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_BINS = 10
MIN_BIN_ROWS = 30
BOOTSTRAP_REPS = 1000

SENSOR_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"
TYPE_COL = "Type"


# ============================================================
# 2. FIND PROJECT AND OUTPUT DIRECTORIES
# ============================================================

def find_project_root():
    script_dir = Path(__file__).resolve().parent

    candidates = [
        script_dir.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]

    for candidate in candidates:
        if (candidate / "data" / "ai4i2020.csv").is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate data/ai4i2020.csv.\n"
        "Expected structure:\n"
        "Self_Evolving_Industrial_Agent/\n"
        "    data/ai4i2020.csv\n"
        "    src/step07_statistical_trend_learning.py"
    )


# ============================================================
# 3. STATISTICAL HELPERS
# ============================================================

def wilson_interval(successes, total):
    """Calculate a 95% Wilson confidence interval for a proportion."""

    if total <= 0:
        return np.nan, np.nan

    z = 1.959963984540054
    p = successes / total

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator

    margin = (
        z
        * np.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def calculate_spearman(x, y):
    """Return descriptive Spearman correlation and p-value when available."""

    clean = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()

    if len(clean) < 3:
        return np.nan, np.nan

    if clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return np.nan, np.nan

    try:
        from scipy.stats import spearmanr

        result = spearmanr(
            clean["x"].to_numpy(),
            clean["y"].to_numpy()
        )

        return float(result.statistic), float(result.pvalue)

    except ImportError:
        # Correlation can still be calculated without SciPy.
        ranked_x = clean["x"].rank(method="average")
        ranked_y = clean["y"].rank(method="average")

        return float(ranked_x.corr(ranked_y)), np.nan


def bootstrap_risk_difference(
    low_labels,
    high_labels,
    repetitions=BOOTSTRAP_REPS,
    seed=RANDOM_STATE,
):
    """
    Estimate high-group minus low-group failure risk.

    Bootstrap resamples are drawn independently within each group.
    The interval describes uncertainty under this resampling scheme.
    """

    low = np.asarray(low_labels, dtype=int)
    high = np.asarray(high_labels, dtype=int)

    if len(low) == 0 or len(high) == 0:
        return np.nan, np.nan, np.nan

    observed_difference = float(high.mean() - low.mean())

    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions)

    for i in range(repetitions):
        low_sample = rng.choice(
            low, size=len(low), replace=True
        )
        high_sample = rng.choice(
            high, size=len(high), replace=True
        )

        differences[i] = high_sample.mean() - low_sample.mean()

    lower, upper = np.quantile(
        differences, [0.025, 0.975]
    )

    return observed_difference, float(lower), float(upper)


# ============================================================
# 4. SENSOR CONDITION RISK ANALYSIS
# ============================================================

def analyze_sensor_bins(df, feature):
    """
    Divide sensor measurements into quantile bins and calculate
    observed failure rate and Wilson confidence intervals.
    """

    work = df[[feature, TARGET]].copy()

    work[feature] = pd.to_numeric(
        work[feature], errors="coerce"
    )

    work = work.dropna()

    if work.empty or work[feature].nunique() < 2:
        return pd.DataFrame()

    try:
        work["Condition_Bin"] = pd.qcut(
            work[feature],
            q=min(N_BINS, work[feature].nunique()),
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()

    rows = []

    for interval, group in work.groupby(
        "Condition_Bin",
        observed=True,
        sort=True,
    ):
        total = len(group)
        failures = int(group[TARGET].sum())
        rate = failures / total

        lower, upper = wilson_interval(
            failures, total
        )

        rows.append({
            "Feature": feature,
            "Condition_Bin": str(interval),
            "Bin_Lower": float(interval.left),
            "Bin_Upper": float(interval.right),
            "Rows": int(total),
            "Failures": failures,
            "Failure_Rate": rate,
            "Failure_Rate_Percent": 100 * rate,
            "Wilson_95CI_Lower": lower,
            "Wilson_95CI_Upper": upper,
            "Wilson_95CI_Lower_Percent": 100 * lower,
            "Wilson_95CI_Upper_Percent": 100 * upper,
            "Meets_Minimum_Bin_Size": total >= MIN_BIN_ROWS,
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. FAILURE RISK BY MACHINE TYPE
# ============================================================

def analyze_machine_type(df):
    rows = []

    for machine_type, group in df.groupby(
        TYPE_COL, dropna=False
    ):
        total = len(group)
        failures = int(group[TARGET].sum())
        rate = failures / total

        lower, upper = wilson_interval(
            failures, total
        )

        rows.append({
            "Machine_Type": str(machine_type),
            "Rows": int(total),
            "Failures": failures,
            "Failure_Rate": rate,
            "Failure_Rate_Percent": 100 * rate,
            "Wilson_95CI_Lower": lower,
            "Wilson_95CI_Upper": upper,
            "Wilson_95CI_Lower_Percent": 100 * lower,
            "Wilson_95CI_Upper_Percent": 100 * upper,
        })

    return pd.DataFrame(rows).sort_values(
        "Failure_Rate", ascending=False
    )


# ============================================================
# 6. TORQUE AND TOOL-WEAR RISK GRID
# ============================================================

def analyze_torque_toolwear(df):
    work = df[
        ["Torque [Nm]", "Tool wear [min]", TARGET]
    ].dropna().copy()

    work["Torque_Band"] = pd.qcut(
        work["Torque [Nm]"],
        q=5,
        duplicates="drop",
    )

    work["Tool_Wear_Band"] = pd.qcut(
        work["Tool wear [min]"],
        q=5,
        duplicates="drop",
    )

    grouped = (
        work.groupby(
            ["Torque_Band", "Tool_Wear_Band"],
            observed=True,
        )
        .agg(
            Rows=(TARGET, "size"),
            Failures=(TARGET, "sum"),
            Failure_Rate=(TARGET, "mean"),
        )
        .reset_index()
    )

    grouped["Failure_Rate_Percent"] = (
        100 * grouped["Failure_Rate"]
    )

    grouped["Meets_Minimum_Bin_Size"] = (
        grouped["Rows"] >= MIN_BIN_ROWS
    )

    grouped["Torque_Band"] = (
        grouped["Torque_Band"].astype(str)
    )

    grouped["Tool_Wear_Band"] = (
        grouped["Tool_Wear_Band"].astype(str)
    )

    return grouped


# ============================================================
# 7. LOW-QUARTILE VS HIGH-QUARTILE COMPARISON
# ============================================================

def analyze_low_high_groups(df):
    rows = []

    for index, feature in enumerate(SENSOR_FEATURES):
        q1 = float(df[feature].quantile(0.25))
        q3 = float(df[feature].quantile(0.75))

        low_group = df.loc[
            df[feature] <= q1, TARGET
        ]

        high_group = df.loc[
            df[feature] >= q3, TARGET
        ]

        difference, lower, upper = (
            bootstrap_risk_difference(
                low_group,
                high_group,
                seed=RANDOM_STATE + index,
            )
        )

        rows.append({
            "Feature": feature,
            "Q1_Cutoff": q1,
            "Q3_Cutoff": q3,
            "Low_Group_Rows": int(len(low_group)),
            "Low_Group_Failures": int(low_group.sum()),
            "Low_Group_Failure_Rate": float(
                low_group.mean()
            ),
            "High_Group_Rows": int(len(high_group)),
            "High_Group_Failures": int(high_group.sum()),
            "High_Group_Failure_Rate": float(
                high_group.mean()
            ),
            "High_Minus_Low_Risk_Difference": difference,
            "Bootstrap_95CI_Lower": lower,
            "Bootstrap_95CI_Upper": upper,
            "Bootstrap_Replicates": BOOTSTRAP_REPS,
        })

    return pd.DataFrame(rows)


# ============================================================
# 8. SENSOR RISK CHART
# ============================================================

def plot_sensor_risk(sensor_table, output_path):
    features = sensor_table["Feature"].unique()

    if len(features) == 0:
        print("No sensor-bin data available for plotting.")
        return

    fig, axes = plt.subplots(
        len(features),
        1,
        figsize=(12, 3 * len(features)),
        squeeze=False,
    )

    for ax, feature in zip(axes[:, 0], features):
        part = (
            sensor_table[
                sensor_table["Feature"] == feature
            ]
            .reset_index(drop=True)
        )

        x = np.arange(len(part))

        ax.plot(
            x,
            part["Failure_Rate_Percent"],
            marker="o",
            linewidth=1.5,
        )

        ax.fill_between(
            x,
            part["Wilson_95CI_Lower_Percent"].to_numpy(),
            part["Wilson_95CI_Upper_Percent"].to_numpy(),
            alpha=0.2,
        )

        labels = [
            f"{low:.1f}–{high:.1f}"
            for low, high in zip(
                part["Bin_Lower"],
                part["Bin_Upper"],
            )
        ]

        ax.set_title(feature)
        ax.set_ylabel("Failure rate (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, rotation=35, ha="right"
        )
        ax.grid(True, alpha=0.25)

    axes[-1, 0].set_xlabel(
        "Sensor-value quantile range"
    )

    fig.suptitle(
        "AI4I: Failure Rate Across Sensor Conditions",
        y=1.002,
    )

    fig.tight_layout()
    fig.savefig(
        output_path, dpi=180, bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# 9. TORQUE / TOOL-WEAR HEATMAP
# ============================================================

def plot_risk_heatmap(grid, output_path):
    if grid.empty:
        print("Risk grid empty; heatmap skipped.")
        return

    pivot = grid.pivot(
        index="Tool_Wear_Band",
        columns="Torque_Band",
        values="Failure_Rate_Percent",
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    image = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        pivot.columns, rotation=35, ha="right"
    )

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Torque band")
    ax.set_ylabel("Tool-wear band")

    ax.set_title(
        "Observed Failure Rate by Torque and Tool Wear (%)"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Observed failure rate (%)",
    )

    fig.tight_layout()
    fig.savefig(
        output_path, dpi=180, bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():
    np.random.seed(RANDOM_STATE)

    project_root = find_project_root()
    data_path = project_root / "data" / "ai4i2020.csv"
    output_dir = project_root / "outputs"

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    print("=" * 72)
    print("STEP 7: STATISTICAL CONDITION-RISK TREND LEARNING")
    print("=" * 72)

    print(f"Project root: {project_root}")
    print(f"Dataset: {data_path}")

    df_original = pd.read_csv(data_path)

    required_columns = [
        TYPE_COL,
        TARGET,
        *SENSOR_FEATURES,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df_original.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    original_rows = len(df_original)

    df = df_original[required_columns].copy()

    for feature in SENSOR_FEATURES:
        df[feature] = pd.to_numeric(
            df[feature], errors="coerce"
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="coerce"
    )

    df = df.dropna(
        subset=required_columns
    ).copy()

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(
            "Machine failure must contain only 0 and 1."
        )

    df[TARGET] = df[TARGET].astype(int)

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Both failure and non-failure cases are required."
        )

    rows_used = len(df)
    failures = int(df[TARGET].sum())
    non_failures = rows_used - failures
    baseline_rate = float(df[TARGET].mean())

    print(f"\nRows loaded: {original_rows}")
    print(f"Rows analyzed: {rows_used}")
    print(f"Rows dropped: {original_rows - rows_used}")
    print(f"Failures: {failures}")
    print(f"Non-failures: {non_failures}")
    print(f"Overall failure rate: {baseline_rate:.3%}")

    print(
        "\nNOTE: These are statistical condition-risk patterns, "
        "not chronological degradation trends or causal effects."
    )

    # A. Sensor-condition risk bins
    sensor_tables = [
        analyze_sensor_bins(df, feature)
        for feature in SENSOR_FEATURES
    ]

    sensor_tables = [
        table for table in sensor_tables
        if not table.empty
    ]

    sensor_risk = pd.concat(
        sensor_tables, ignore_index=True
    )

    sensor_risk_path = (
        output_dir / "step07_sensor_condition_risk.csv"
    )

    sensor_risk.to_csv(
        sensor_risk_path, index=False
    )

    # B. Sensor associations
    association_rows = []

    for feature in SENSOR_FEATURES:
        rho, p_value = calculate_spearman(
            df[feature], df[TARGET]
        )

        association_rows.append({
            "Feature": feature,
            "Spearman_Rho": rho,
            "P_Value_If_SciPy_Available": p_value,
            "Absolute_Rho": (
                abs(rho) if np.isfinite(rho) else np.nan
            ),
            "Interpretation": (
                "Descriptive association; not causal evidence"
            ),
        })

    associations = pd.DataFrame(
        association_rows
    ).sort_values(
        "Absolute_Rho",
        ascending=False,
        na_position="last",
    )

    associations_path = (
        output_dir / "step07_sensor_associations.csv"
    )

    associations.to_csv(
        associations_path, index=False
    )

    # C. Risk by machine type
    type_risk = analyze_machine_type(df)

    type_risk_path = (
        output_dir / "step07_failure_risk_by_type.csv"
    )

    type_risk.to_csv(
        type_risk_path, index=False
    )

    # D. Joint torque / tool-wear analysis
    risk_grid = analyze_torque_toolwear(df)

    risk_grid_path = (
        output_dir / "step07_torque_toolwear_risk_grid.csv"
    )

    risk_grid.to_csv(
        risk_grid_path, index=False
    )

    # E. Low/high quartile contrasts
    contrasts = analyze_low_high_groups(df)

    contrasts_path = (
        output_dir / "step07_low_high_risk_contrasts.csv"
    )

    contrasts.to_csv(
        contrasts_path, index=False
    )

    # F. Plots
    sensor_plot_path = (
        output_dir / "step07_sensor_condition_risk.png"
    )

    heatmap_path = (
        output_dir / "step07_torque_toolwear_heatmap.png"
    )

    plot_sensor_risk(
        sensor_risk, sensor_plot_path
    )

    plot_risk_heatmap(
        risk_grid, heatmap_path
    )

    # G. JSON report
    strongest_associations = (
        associations.dropna(
            subset=["Absolute_Rho"]
        )
        .head(3)
        .to_dict(orient="records")
    )

    report = {
        "step": 7,
        "title": "Statistical Condition-Risk Trend Learning",
        "dataset": str(data_path),
        "rows_loaded": int(original_rows),
        "rows_analyzed": int(rows_used),
        "rows_dropped": int(original_rows - rows_used),
        "failures": int(failures),
        "non_failures": int(non_failures),
        "baseline_failure_rate": baseline_rate,
        "features_analyzed": SENSOR_FEATURES,
        "strongest_descriptive_associations": (
            strongest_associations
        ),
        "methodology": {
            "sensor_binning": (
                f"Up to {N_BINS} quantile bins per sensor"
            ),
            "confidence_intervals": (
                "95% Wilson intervals for binomial proportions"
            ),
            "quartile_comparison": (
                "Q1 versus Q3 with bootstrap risk-difference interval"
            ),
            "bootstrap_repetitions": BOOTSTRAP_REPS,
            "random_state": RANDOM_STATE,
            "minimum_bin_rows_flag": MIN_BIN_ROWS,
        },
        "outputs": {
            "sensor_condition_risk_csv": str(sensor_risk_path),
            "sensor_associations_csv": str(associations_path),
            "failure_risk_by_type_csv": str(type_risk_path),
            "torque_toolwear_grid_csv": str(risk_grid_path),
            "low_high_contrasts_csv": str(contrasts_path),
            "sensor_risk_plot": str(sensor_plot_path),
            "torque_toolwear_heatmap": str(heatmap_path),
        },
        "limitations": [
            "Associations do not establish causation.",
            "AI4I does not provide chronological per-machine histories.",
            "UDI row order must not be treated as degradation time.",
            "Actual maintenance costs and post-maintenance outcomes are unavailable.",
            "Rare failures can make small-bin rates unstable.",
            "Exploratory p-values are not adjusted for multiple comparisons.",
        ],
    }

    report_path = output_dir / "step07_report.json"

    with report_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report, file, indent=2, allow_nan=False
        )

    # H. Console summaries
    print("\n" + "=" * 72)
    print("TOP SENSOR ASSOCIATIONS")
    print("=" * 72)

    print(
        associations.to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\n" + "=" * 72)
    print("FAILURE RISK BY MACHINE TYPE")
    print("=" * 72)

    print(
        type_risk.to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\n" + "=" * 72)
    print("LOW-QUARTILE VS HIGH-QUARTILE RISK")
    print("=" * 72)

    display_columns = [
        "Feature",
        "Low_Group_Failure_Rate",
        "High_Group_Failure_Rate",
        "High_Minus_Low_Risk_Difference",
        "Bootstrap_95CI_Lower",
        "Bootstrap_95CI_Upper",
    ]

    print(
        contrasts[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\n" + "=" * 72)
    print("SAVED OUTPUTS")
    print("=" * 72)

    for key, value in report["outputs"].items():
        print(f"{key}: {value}")

    print(f"JSON report: {report_path}")

    print("\nSTEP 7 COMPLETED SUCCESSFULLY")
    print(
        "Interpret results as condition-risk associations, "
        "not chronological degradation or causal effects."
    )


if __name__ == "__main__":
    main()