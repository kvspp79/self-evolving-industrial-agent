
"""
NASA C-MAPSS FD001 — STEP 8.2
Statistical Trend Drift Detection and Controlled Policy Adaptation

Research safeguards:
1. Uses training sensor distributions as the reference.
2. Uses test sensor values only for unsupervised shift detection.
3. Does not use test RUL labels to tune thresholds or policies.
4. Reports statistical distribution shift, not proven causal degradation.
5. Produces a reassessment recommendation; it does not automatically
   retrain or deploy a changed maintenance policy.

Input files:
    outputs/step01/train_FD001_processed.csv
    outputs/step01/test_FD001_processed.csv

Output:
    outputs/step08_2/
"""

from pathlib import Path
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT / "outputs" / "step01" / "train_FD001_processed.csv"
)
TEST_PATH = (
    PROJECT_ROOT / "outputs" / "step01" / "test_FD001_processed.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step08_2"

RANDOM_SEED = 42
MAX_ROWS_PER_PARTITION = 10000
MIN_ROWS_PER_COMPARISON = 30

SIGNIFICANCE_LEVEL = 0.05
PSI_WARNING_THRESHOLD = 0.10
PSI_MAJOR_THRESHOLD = 0.20
SMD_WARNING_THRESHOLD = 0.50

# Require several affected sensors before recommending reassessment.
MIN_DRIFTED_SENSORS = 3
MIN_DRIFTED_SENSOR_FRACTION = 0.20

# Quantile bins used to estimate PSI from the training reference.
PSI_BINS = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITIES
# ============================================================

def print_section(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def load_csv(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}\n"
            "Check the project folder and Step 1 output filenames."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{description} is empty: {path}")

    print(f"{description}: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def find_sensor_columns(train, test):
    train_sensors = [
        c for c in train.columns
        if str(c).lower().startswith("sensor_")
    ]
    test_sensors = [
        c for c in test.columns
        if str(c).lower().startswith("sensor_")
    ]

    common = sorted(set(train_sensors).intersection(test_sensors))

    if not common:
        raise ValueError(
            "No common sensor_* columns were found in the processed "
            "training and test files. Inspect their column names."
        )

    # Exclude sensors that contain no usable numeric variation.
    usable = []

    for col in common:
        train_values = pd.to_numeric(
            train[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        test_values = pd.to_numeric(
            test[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if (
            len(train_values) >= MIN_ROWS_PER_COMPARISON
            and len(test_values) >= MIN_ROWS_PER_COMPARISON
            and train_values.nunique() > 1
        ):
            usable.append(col)

    if not usable:
        raise ValueError(
            "No sensor columns have enough usable numeric observations."
        )

    return usable


def numeric_sample(series, max_rows, rng):
    values = pd.to_numeric(
        series, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    if len(values) > max_rows:
        indices = rng.choice(
            len(values), size=max_rows, replace=False
        )
        values = values[indices]

    return values


def calculate_psi(reference, comparison, bins=10):
    """
    Population Stability Index.

    Bin edges are learned from the reference sample only.
    Small probabilities are clipped to avoid log(0).
    """
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)

    reference = reference[np.isfinite(reference)]
    comparison = comparison[np.isfinite(comparison)]

    if len(reference) < MIN_ROWS_PER_COMPARISON:
        return np.nan

    if len(comparison) < MIN_ROWS_PER_COMPARISON:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))

    if len(edges) < 3:
        # Constant or nearly constant reference distribution.
        return 0.0 if np.allclose(
            np.mean(reference), np.mean(comparison)
        ) else np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cmp_counts, _ = np.histogram(comparison, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cmp_pct = cmp_counts / max(cmp_counts.sum(), 1)

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cmp_pct = np.clip(cmp_pct, epsilon, None)

    return float(
        np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct))
    )


def standardized_mean_difference(reference, comparison):
    """
    Absolute difference in means divided by pooled standard deviation.
    """
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)

    reference = reference[np.isfinite(reference)]
    comparison = comparison[np.isfinite(comparison)]

    if len(reference) < 2 or len(comparison) < 2:
        return np.nan

    var_ref = np.var(reference, ddof=1)
    var_cmp = np.var(comparison, ddof=1)

    pooled_var = (
        ((len(reference) - 1) * var_ref)
        + ((len(comparison) - 1) * var_cmp)
    ) / max(len(reference) + len(comparison) - 2, 1)

    if pooled_var <= 1e-12:
        return 0.0 if np.isclose(
            np.mean(reference), np.mean(comparison)
        ) else np.nan

    return float(
        abs(np.mean(comparison) - np.mean(reference))
        / np.sqrt(pooled_var)
    )


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg false-discovery-rate adjusted p-values.
    NaN entries remain NaN.
    """
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)

    valid_indices = np.where(np.isfinite(p_values))[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = p_values[valid_indices]
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    n = len(sorted_p)
    sorted_adjusted = sorted_p * n / np.arange(1, n + 1)

    # Enforce monotonicity from largest rank to smallest.
    sorted_adjusted = np.minimum.accumulate(
        sorted_adjusted[::-1]
    )[::-1]

    sorted_adjusted = np.clip(sorted_adjusted, 0, 1)

    adjusted[valid_indices[order]] = sorted_adjusted

    return adjusted


def drift_strength(psi, smd, q_value):
    statistically_significant = (
        np.isfinite(q_value) and q_value < SIGNIFICANCE_LEVEL
    )

    if (
        statistically_significant
        and np.isfinite(psi)
        and psi >= PSI_MAJOR_THRESHOLD
    ):
        return "major_shift"

    if (
        statistically_significant
        and (
            (np.isfinite(psi) and psi >= PSI_WARNING_THRESHOLD)
            or (np.isfinite(smd) and smd >= SMD_WARNING_THRESHOLD)
        )
    ):
        return "moderate_shift"

    if statistically_significant:
        return "statistical_difference_small_effect"

    if (
        (np.isfinite(psi) and psi >= PSI_MAJOR_THRESHOLD)
        or (np.isfinite(smd) and smd >= SMD_WARNING_THRESHOLD)
    ):
        return "effect_size_elevated_not_significant"

    return "no_clear_shift"


# ============================================================
# ANALYSIS
# ============================================================

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print_section(
        "NASA C-MAPSS — STEP 8.2\n"
        "STATISTICAL TREND DRIFT DETECTION AND CONTROLLED ADAPTATION"
    )
    print("Subset: FD001")
    print(
        "UTC run time:",
        datetime.now(timezone.utc).isoformat()
    )

    train = load_csv(TRAIN_PATH, "Processed training data")
    test = load_csv(TEST_PATH, "Processed test data")

    sensors = find_sensor_columns(train, test)

    print_section("DATA SUMMARY")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Common usable sensor features: {len(sensors)}")

    if "unit_id" in train.columns:
        print(
            "Training engines:",
            train["unit_id"].nunique()
        )

    if "unit_id" in test.columns:
        print(
            "Test engines:",
            test["unit_id"].nunique()
        )

    # Sample observations for bounded runtime. This is a random sample
    # of rows, not a claim that rows are independent engine observations.
    train_sample = train.sample(
        n=min(len(train), MAX_ROWS_PER_PARTITION),
        random_state=RANDOM_SEED
    )
    test_sample = test.sample(
        n=min(len(test), MAX_ROWS_PER_PARTITION),
        random_state=RANDOM_SEED
    )

    print_section("GLOBAL SENSOR-DISTRIBUTION COMPARISON")
    print(
        "Reference: sampled training sensor rows\n"
        "Comparison: sampled test sensor rows\n"
        "Test RUL labels are not loaded or used for this analysis."
    )

    records = []

    for sensor in sensors:
        ref = numeric_sample(
            train_sample[sensor], MAX_ROWS_PER_PARTITION, rng
        )
        cmp = numeric_sample(
            test_sample[sensor], MAX_ROWS_PER_PARTITION, rng
        )

        if (
            len(ref) < MIN_ROWS_PER_COMPARISON
            or len(cmp) < MIN_ROWS_PER_COMPARISON
        ):
            continue

        ks_result = ks_2samp(ref, cmp, alternative="two-sided")

        psi = calculate_psi(ref, cmp, bins=PSI_BINS)
        smd = standardized_mean_difference(ref, cmp)

        records.append({
            "sensor": sensor,
            "training_n": int(len(ref)),
            "test_n": int(len(cmp)),
            "training_mean": float(np.mean(ref)),
            "test_mean": float(np.mean(cmp)),
            "training_std": float(np.std(ref, ddof=1)),
            "test_std": float(np.std(cmp, ddof=1)),
            "ks_statistic": float(ks_result.statistic),
            "ks_p_value": float(ks_result.pvalue),
            "psi": psi,
            "absolute_standardized_mean_difference": smd,
        })

    if not records:
        raise ValueError(
            "No sensors had enough valid values for comparison."
        )

    results = pd.DataFrame(records)

    results["ks_q_value"] = benjamini_hochberg(
        results["ks_p_value"].to_numpy()
    )

    results["drift_status"] = results.apply(
        lambda row: drift_strength(
            row["psi"],
            row["absolute_standardized_mean_difference"],
            row["ks_q_value"],
        ),
        axis=1,
    )

    results = results.sort_values(
        by=[
            "ks_q_value",
            "psi",
            "absolute_standardized_mean_difference",
        ],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    results_path = (
        OUTPUT_DIR / "sensor_distribution_shift_FD001.csv"
    )
    results.to_csv(results_path, index=False)

    print("\nTop sensor shifts:")
    display_cols = [
        "sensor",
        "ks_statistic",
        "ks_p_value",
        "ks_q_value",
        "psi",
        "absolute_standardized_mean_difference",
        "drift_status",
    ]
    print(results[display_cols].head(15).to_string(index=False))

    # --------------------------------------------------------
    # Drift decision and controlled reassessment recommendation
    # --------------------------------------------------------

    drifted = results[
        results["drift_status"].isin(
            ["moderate_shift", "major_shift"]
        )
    ]

    n_sensors = len(results)
    n_drifted = len(drifted)
    drift_fraction = n_drifted / max(n_sensors, 1)

    major_count = int(
        (results["drift_status"] == "major_shift").sum()
    )

    reassessment_triggered = (
        n_drifted >= MIN_DRIFTED_SENSORS
        or drift_fraction >= MIN_DRIFTED_SENSOR_FRACTION
    )

    if reassessment_triggered:
        recommendation = (
            "TRIGGER_POLICY_REASSESSMENT_REVIEW: sensor distribution "
            "shift exceeded the configured screening rule. Review "
            "feature stability, model calibration, and maintenance "
            "policy performance using appropriately labeled validation "
            "data before approving any change."
        )
    else:
        recommendation = (
            "CONTINUE_MONITORING: the configured sensor-shift screening "
            "rule was not met. This does not prove that the model or "
            "policy remains reliable."
        )

    print_section("CONTROLLED ADAPTATION DECISION")
    print(f"Sensors tested: {n_sensors}")
    print(f"Moderate-or-major shifted sensors: {n_drifted}")
    print(f"Major-shift sensors: {major_count}")
    print(f"Shifted-sensor fraction: {drift_fraction:.3f}")
    print(
        "Minimum shifted-sensor count:",
        MIN_DRIFTED_SENSORS
    )
    print(
        "Minimum shifted-sensor fraction:",
        MIN_DRIFTED_SENSOR_FRACTION
    )
    print(
        "Reassessment review triggered:",
        reassessment_triggered
    )
    print("Recommendation:", recommendation)

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    plot_df = results.sort_values("psi", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot_df["sensor"], plot_df["psi"])
    ax.axvline(
        PSI_WARNING_THRESHOLD,
        linestyle="--",
        label=f"PSI warning ({PSI_WARNING_THRESHOLD})",
    )
    ax.axvline(
        PSI_MAJOR_THRESHOLD,
        linestyle=":",
        label=f"PSI major ({PSI_MAJOR_THRESHOLD})",
    )
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_ylabel("Sensor")
    ax.set_title(
        "FD001 Training-to-Test Sensor Distribution Shift"
    )
    ax.legend()
    fig.tight_layout()

    plot_path = (
        OUTPUT_DIR / "sensor_distribution_shift_FD001.png"
    )
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    # Mean comparison plot for the most shifted sensors.
    plot_df = results.assign(
        absolute_smd=results[
            "absolute_standardized_mean_difference"
        ].abs()
    ).sort_values(
        "absolute_smd", ascending=False
    ).head(15)

    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        plot_df["training_mean"],
        width,
        label="Training",
    )
    ax.bar(
        x + width / 2,
        plot_df["test_mean"],
        width,
        label="Test",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["sensor"], rotation=45, ha="right"
    )
    ax.set_ylabel("Mean sensor value")
    ax.set_title(
        "Training vs Test Means for Sensors with Largest Standardized Shifts"
    )
    ax.legend()
    fig.tight_layout()

    means_plot_path = (
        OUTPUT_DIR / "sensor_mean_comparison_FD001.png"
    )
    fig.savefig(means_plot_path, dpi=160)
    plt.close(fig)

    # --------------------------------------------------------
    # Research report
    # --------------------------------------------------------

    report = {
        "step": "8.2",
        "project": "NASA C-MAPSS",
        "subset": "FD001",
        "run_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "method": {
            "reference_distribution": "processed training sensor rows",
            "comparison_distribution": "processed test sensor rows",
            "tests": [
                "two-sample Kolmogorov-Smirnov test",
                "Benjamini-Hochberg false-discovery-rate correction",
                "Population Stability Index",
                "absolute standardized mean difference",
            ],
            "psi_bins": PSI_BINS,
            "significance_level": SIGNIFICANCE_LEVEL,
            "psi_warning_threshold": PSI_WARNING_THRESHOLD,
            "psi_major_threshold": PSI_MAJOR_THRESHOLD,
            "smd_warning_threshold": SMD_WARNING_THRESHOLD,
            "max_rows_per_partition": MAX_ROWS_PER_PARTITION,
            "random_seed": RANDOM_SEED,
        },
        "data": {
            "training_rows": int(len(train)),
            "test_rows": int(len(test)),
            "training_engines": (
                int(train["unit_id"].nunique())
                if "unit_id" in train.columns else None
            ),
            "test_engines": (
                int(test["unit_id"].nunique())
                if "unit_id" in test.columns else None
            ),
            "sensors_tested": int(n_sensors),
        },
        "shift_summary": {
            "moderate_or_major_shifted_sensors": int(n_drifted),
            "major_shift_sensors": int(major_count),
            "shifted_sensor_fraction": float(drift_fraction),
            "shifted_sensor_names": drifted["sensor"].tolist(),
        },
        "controlled_adaptation": {
            "reassessment_review_triggered": bool(
                reassessment_triggered
            ),
            "recommendation": recommendation,
            "automatic_model_retraining": False,
            "automatic_policy_deployment": False,
            "test_rul_labels_used_for_selection": False,
        },
        "limitations": [
            "A training-to-test distribution difference is not proof "
            "of temporal concept drift or causal degradation.",
            "The test partition may differ in engine lifecycle composition "
            "and observation length.",
            "Rows from the same engine are not independent observations; "
            "row-level KS p-values can overstate evidence.",
            "This is a screening experiment. A deployment system should "
            "also monitor engine-level summaries and use time-aware "
            "validation and operational feedback.",
            "The reassessment rule is a predefined screening rule, not "
            "a learned optimal maintenance policy.",
            "No test RUL labels were used to select thresholds or policies.",
        ],
        "files": {
            "sensor_results_csv": str(results_path),
            "psi_plot": str(plot_path),
            "mean_comparison_plot": str(means_plot_path),
        },
    }

    report_path = (
        OUTPUT_DIR / "step08_2_FD001_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print_section("STEP 8.2 COMPLETED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Sensor results: {results_path}")
    print(f"PSI plot: {plot_path}")
    print(f"Mean comparison plot: {means_plot_path}")
    print(f"Research report: {report_path}")
    print(
        "\nTest RUL labels were not used for drift detection, "
        "threshold selection, or policy adaptation."
    )
    print(
        "No model was retrained and no maintenance policy was "
        "automatically deployed."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nSTEP 8.2 FAILED")
        print(f"{type(exc).__name__}: {exc}")
        raise
