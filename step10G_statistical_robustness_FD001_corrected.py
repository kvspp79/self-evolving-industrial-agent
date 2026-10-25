"""
STEP 10G: Statistical Robustness and Safety-Gate Audit
Project: Self-Evolving Explainable Industrial Agent
Dataset: NASA C-MAPSS FD001

Uses existing Step 10F outputs. Does not retrain models or modify gate criteria.

Analyses:
- Paired differences across seeds
- Bootstrap and paired-t 95% confidence intervals
- Wilcoxon signed-rank tests
- Exact sign tests
- Floating-point-aware improvement, worsening, and tie counts
- Safety-gate outcome audit
- CSV tables, plots, and JSON report

Scientific limitations:
- Seeds reuse the same dataset; they are not independent fleets.
- Ten seeds provide limited inferential precision.
- Synthetic shifts do not establish robustness to real industrial drift.
- A validation gate cannot guarantee held-out safety.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
except ImportError as exc:
    raise ImportError(
        "SciPy is required. Install it with: "
        "py -3.13 -m pip install scipy"
    ) from exc


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "step10F_repeated_seed_robustness"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "step10G_statistical_robustness"
)

PER_SEED_FILE = INPUT_DIR / "step10F_per_seed_final_metrics.csv"
GATE_FILE = INPUT_DIR / "step10F_gate_decisions.csv"

METRICS = [
    "mean_engine_RMSE",
    "mean_engine_MAE",
    "mean_engine_NASA_score",
]

METRIC_LABELS = {
    "mean_engine_RMSE": "Mean engine RMSE",
    "mean_engine_MAE": "Mean engine MAE",
    "mean_engine_NASA_score": "Mean engine NASA score",
}

BASELINE_MODEL = "frozen_baseline"
CANDIDATE_MODEL = "unconditional_retraining"
GATED_MODEL = "safety_gated_model"

BOOTSTRAP_REPETITIONS = 20_000
BOOTSTRAP_SEED = 202610
CONFIDENCE_LEVEL = 0.95

# Differences are compared on their original metric scale.
# The tolerance scales with the magnitude of the compared values.
NUMERICAL_TOLERANCE = 1e-10


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def require_file(path):
    """Stop early if a required input is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Step 10F file not found:\n{path}\n\n"
            "Check PROJECT_ROOT and ensure Step 10F has completed."
        )


def json_safe(value):
    """Convert NumPy/Pandas values into JSON-compatible values."""
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if value is None:
        return None

    if pd.isna(value):
        return None

    return value


def paired_difference(baseline, comparison):
    """
    Calculate baseline - comparison.

    Positive means the comparison model is better because all three
    metrics are loss/error scores where lower is better.

    Differences within a scale-aware floating-point tolerance are
    normalized to exactly zero.
    """
    baseline = np.asarray(baseline, dtype=float)
    comparison = np.asarray(comparison, dtype=float)

    if baseline.shape != comparison.shape:
        raise ValueError("Paired metric arrays have different shapes.")

    if not (
        np.isfinite(baseline).all()
        and np.isfinite(comparison).all()
    ):
        raise ValueError("Non-finite values found in paired metrics.")

    difference = baseline - comparison

    tolerance = NUMERICAL_TOLERANCE * np.maximum(
        1.0,
        np.maximum(np.abs(baseline), np.abs(comparison)),
    )

    difference[np.abs(difference) <= tolerance] = 0.0

    return difference


def relative_improvement(baseline, comparison, differences=None):
    """
    Return relative improvement as a fraction.

    Positive = lower/better comparison metric.
    Negative = higher/worse comparison metric.
    Zero = tie within the configured numerical tolerance.
    """
    baseline = np.asarray(baseline, dtype=float)
    comparison = np.asarray(comparison, dtype=float)

    if differences is None:
        differences = paired_difference(baseline, comparison)

    if np.any(baseline == 0):
        raise ValueError(
            "Cannot calculate relative improvement when baseline is zero."
        )

    result = differences / np.abs(baseline)

    # Ensure numerical ties remain exact ties in the relative results.
    result = np.asarray(result, dtype=float)
    result[np.asarray(differences) == 0.0] = 0.0

    return result


def classify_difference(difference):
    """Classify a normalized paired difference."""
    if difference > 0:
        return "improved"

    if difference < 0:
        return "worsened"

    return "tied"


# ============================================================
# 3. STATISTICAL UTILITIES
# ============================================================

def bootstrap_mean_ci(
    values,
    repetitions=BOOTSTRAP_REPETITIONS,
    confidence=CONFIDENCE_LEVEL,
    seed=BOOTSTRAP_SEED,
):
    """
    Percentile bootstrap CI for the mean paired seed-level difference.

    Resampling unit is the seed-level paired difference, not individual
    engines or cycles. With ten seeds, the interval may be unstable.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(values)

    sampled_indices = rng.integers(
        low=0,
        high=n,
        size=(repetitions, n),
    )

    bootstrap_means = values[sampled_indices].mean(axis=1)

    alpha = 1.0 - confidence

    lower = float(np.quantile(bootstrap_means, alpha / 2))
    upper = float(np.quantile(bootstrap_means, 1 - alpha / 2))

    return lower, upper


def paired_t_ci(values, confidence=CONFIDENCE_LEVEL):
    """Paired t-based CI for the mean seed-level difference."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n < 2:
        return np.nan, np.nan

    mean_value = float(np.mean(values))
    sd_value = float(np.std(values, ddof=1))
    standard_error = sd_value / np.sqrt(n)

    critical = float(
        stats.t.ppf((1 + confidence) / 2, df=n - 1)
    )

    return (
        mean_value - critical * standard_error,
        mean_value + critical * standard_error,
    )


def safe_wilcoxon(values):
    """
    Two-sided Wilcoxon signed-rank test on paired seed differences.

    Exploratory only: ten seeds provide limited inferential precision.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return {
            "statistic": None,
            "p_value": None,
            "note": "Too few pairs",
        }

    # Differences have already been normalized for numerical ties.
    nonzero = values[values != 0.0]

    if len(nonzero) == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "note": "All paired differences are ties",
        }

    try:
        result = stats.wilcoxon(
            nonzero,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "note": "Exploratory paired signed-rank test",
        }

    except Exception as exc:
        return {
            "statistic": None,
            "p_value": None,
            "note": f"Test unavailable: {exc}",
        }


def sign_test_two_sided(values):
    """Exact two-sided sign test; normalized zero differences are ties."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    nonzero = values[values != 0.0]
    n = len(nonzero)

    positive = int(np.sum(nonzero > 0.0))
    negative = int(np.sum(nonzero < 0.0))
    ties = int(len(values) - n)

    if n == 0:
        return {
            "positive": 0,
            "negative": 0,
            "ties": ties,
            "p_value": 1.0,
        }

    p_value = min(
        1.0,
        2.0 * float(
            stats.binom.cdf(min(positive, negative), n, 0.5)
        ),
    )

    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "p_value": p_value,
    }


# ============================================================
# 4. LOAD AND VALIDATE INPUTS
# ============================================================

def load_inputs():
    require_file(PER_SEED_FILE)
    require_file(GATE_FILE)

    results = pd.read_csv(PER_SEED_FILE)
    gates = pd.read_csv(GATE_FILE)

    required_result_columns = {
        "seed",
        "model",
        *METRICS,
    }

    missing = required_result_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {PER_SEED_FILE.name}: "
            f"{sorted(missing)}"
        )

    required_gate_columns = {
        "seed",
        "candidate_accepted",
    }

    missing_gate = required_gate_columns - set(gates.columns)

    if missing_gate:
        raise ValueError(
            f"Missing columns in {GATE_FILE.name}: "
            f"{sorted(missing_gate)}"
        )

    results["seed"] = pd.to_numeric(
        results["seed"], errors="raise"
    ).astype(int)

    gates["seed"] = pd.to_numeric(
        gates["seed"], errors="raise"
    ).astype(int)

    results["model"] = results["model"].astype(str).str.strip()

    gates["candidate_accepted"] = (
        gates["candidate_accepted"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
        })
    )

    if gates["candidate_accepted"].isna().any():
        raise ValueError(
            "Could not parse candidate_accepted as True/False."
        )

    for metric in METRICS:
        results[metric] = pd.to_numeric(
            results[metric], errors="raise"
        )

    if results.duplicated(["seed", "model"]).any():
        raise ValueError("Duplicate seed/model rows detected.")

    if gates["seed"].duplicated().any():
        raise ValueError(
            "Duplicate seed rows detected in gate decisions."
        )

    if not np.isfinite(
        results[METRICS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite metric values found in Step 10F results."
        )

    result_seeds = set(results["seed"].unique())
    gate_seeds = set(gates["seed"].unique())

    if result_seeds != gate_seeds:
        raise ValueError(
            "Seed mismatch between metrics and gate decisions.\n"
            f"Results without gate decisions: "
            f"{sorted(result_seeds - gate_seeds)}\n"
            f"Gate decisions without final metrics: "
            f"{sorted(gate_seeds - result_seeds)}"
        )

    expected_models = {
        BASELINE_MODEL,
        CANDIDATE_MODEL,
        GATED_MODEL,
    }

    for seed, group in results.groupby("seed"):
        found_models = set(group["model"])

        if found_models != expected_models:
            raise ValueError(
                f"Seed {seed} model set mismatch.\n"
                f"Expected: {sorted(expected_models)}\n"
                f"Found: {sorted(found_models)}"
            )

    return results, gates


# ============================================================
# 5. PAIRED SEED-LEVEL ANALYSIS
# ============================================================

def paired_analysis(results):
    all_rows = []
    paired_data = {}

    for metric in METRICS:
        wide = results.pivot(
            index="seed",
            columns="model",
            values=metric,
        )

        for comparison_model in [
            CANDIDATE_MODEL,
            GATED_MODEL,
        ]:
            pair = wide[
                [BASELINE_MODEL, comparison_model]
            ].dropna()

            baseline = pair[BASELINE_MODEL].to_numpy(dtype=float)
            comparison = pair[comparison_model].to_numpy(dtype=float)

            differences = paired_difference(
                baseline,
                comparison,
            )

            relative = relative_improvement(
                baseline,
                comparison,
                differences,
            )

            bootstrap_low, bootstrap_high = bootstrap_mean_ci(
                differences,
                seed=BOOTSTRAP_SEED + len(all_rows),
            )

            t_low, t_high = paired_t_ci(differences)
            wilcoxon = safe_wilcoxon(differences)
            sign_test = sign_test_two_sided(differences)

            paired_data[(metric, comparison_model)] = {
                "seeds": pair.index.to_numpy(dtype=int),
                "baseline": baseline,
                "comparison": comparison,
                "difference": differences,
                "relative_improvement": relative,
            }

            all_rows.append({
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "comparison_model": comparison_model,
                "n_paired_seeds": int(len(differences)),
                "baseline_mean": float(np.mean(baseline)),
                "comparison_model_mean": float(np.mean(comparison)),
                "mean_absolute_improvement": float(
                    np.mean(differences)
                ),
                "median_absolute_improvement": float(
                    np.median(differences)
                ),
                "mean_relative_improvement_percent": float(
                    100.0 * np.mean(relative)
                ),
                "median_relative_improvement_percent": float(
                    100.0 * np.median(relative)
                ),
                "seeds_improved": int(
                    np.sum(differences > 0.0)
                ),
                "seeds_worsened": int(
                    np.sum(differences < 0.0)
                ),
                "seeds_tied": int(
                    np.sum(differences == 0.0)
                ),
                "bootstrap_95CI_lower_absolute": bootstrap_low,
                "bootstrap_95CI_upper_absolute": bootstrap_high,
                "paired_t_95CI_lower_absolute": t_low,
                "paired_t_95CI_upper_absolute": t_high,
                "wilcoxon_statistic": wilcoxon["statistic"],
                "wilcoxon_p_value": wilcoxon["p_value"],
                "wilcoxon_note": wilcoxon["note"],
                "sign_test_p_value": sign_test["p_value"],
                "sign_test_positive": sign_test["positive"],
                "sign_test_negative": sign_test["negative"],
                "sign_test_ties": sign_test["ties"],
            })

    return pd.DataFrame(all_rows), paired_data


# ============================================================
# 6. SAFETY-GATE OUTCOME AUDIT
# ============================================================

def gate_audit(results, gates):
    """
    Audit the actual safety_gated_model against frozen_baseline.

    Accepted seeds use the accepted update; rejected seeds retain baseline.

    The supplied Step 10F per-seed metrics do not preserve rejected-candidate
    counterfactual predictions. This audit therefore does not claim to measure
    what a rejected candidate would have done if it had been deployed.
    """
    wide = results.pivot(
        index="seed",
        columns="model",
        values=METRICS,
    )

    records = []

    for _, gate_row in gates.iterrows():
        seed = int(gate_row["seed"])
        accepted = bool(gate_row["candidate_accepted"])

        if seed not in wide.index:
            raise ValueError(
                f"Gate seed {seed} has no final metric results."
            )

        record = {
            "seed": seed,
            "candidate_accepted": accepted,
        }

        improvements = {}

        for metric in METRICS:
            baseline = float(
                wide.loc[seed, (metric, BASELINE_MODEL)]
            )

            gated_value = float(
                wide.loc[seed, (metric, GATED_MODEL)]
            )

            difference = paired_difference(
                np.array([baseline]),
                np.array([gated_value]),
            )[0]

            if baseline == 0.0:
                raise ValueError(
                    f"Baseline metric is zero for seed={seed}, "
                    f"metric={metric}."
                )

            improvement = float(
                difference / abs(baseline)
            )

            improvements[metric] = improvement

            record[
                f"{metric}_relative_improvement_percent"
            ] = 100.0 * improvement

            record[f"{metric}_improved"] = (
                improvement > 0.0
            )

            record[f"{metric}_worsened"] = (
                improvement < 0.0
            )

        any_improved = any(
            value > 0.0 for value in improvements.values()
        )

        any_worsened = any(
            value < 0.0 for value in improvements.values()
        )

        all_improved = all(
            value > 0.0 for value in improvements.values()
        )

        all_worsened = all(
            value < 0.0 for value in improvements.values()
        )

        record["gated_model_improved_any_metric"] = any_improved
        record["gated_model_worsened_any_metric"] = any_worsened
        record["gated_model_improved_all_metrics"] = all_improved
        record["gated_model_worsened_all_metrics"] = all_worsened

        record["accepted_then_any_metric_worsened"] = (
            accepted and any_worsened
        )

        record["accepted_then_all_metrics_improved"] = (
            accepted and all_improved
        )

        # For rejected seeds, the actual retained model is baseline.
        # Do not interpret these as rejected-candidate counterfactuals.
        record["rejected_actual_model_all_metrics_improved"] = (
            (not accepted) and all_improved
        )

        record["rejected_actual_model_any_metric_worsened"] = (
            (not accepted) and any_worsened
        )

        records.append(record)

    detail = pd.DataFrame(records)

    if detail.empty:
        raise ValueError("No matching seeds found for gate audit.")

    total = len(detail)
    accepted_count = int(detail["candidate_accepted"].sum())
    rejected_count = total - accepted_count

    accepted_rows = detail[detail["candidate_accepted"]]
    rejected_rows = detail[~detail["candidate_accepted"]]

    summary = {
        "total_seed_runs": total,
        "accepted_updates": accepted_count,
        "rejected_updates": rejected_count,
        "acceptance_rate_percent": (
            100.0 * accepted_count / total if total else None
        ),
        "accepted_then_any_metric_worsened_count": int(
            detail["accepted_then_any_metric_worsened"].sum()
        ),
        "accepted_then_all_metrics_improved_count": int(
            detail["accepted_then_all_metrics_improved"].sum()
        ),
        "rejected_actual_model_all_metrics_improved_count": int(
            detail["rejected_actual_model_all_metrics_improved"].sum()
        ),
        "rejected_actual_model_any_metric_worsened_count": int(
            detail["rejected_actual_model_any_metric_worsened"].sum()
        ),
        "accepted_update_worsening_rate_percent": (
            100.0
            * detail["accepted_then_any_metric_worsened"].sum()
            / accepted_count
            if accepted_count else None
        ),
        "accepted_update_all_metric_improvement_rate_percent": (
            100.0
            * detail["accepted_then_all_metrics_improved"].sum()
            / accepted_count
            if accepted_count else None
        ),
        "accepted_seed_ids": (
            accepted_rows["seed"].astype(int).tolist()
        ),
        "rejected_seed_ids": (
            rejected_rows["seed"].astype(int).tolist()
        ),
        "audited_model": GATED_MODEL,
        "rejected_candidate_counterfactual_outcomes_available": False,
        "definition": (
            "Held-out outcomes compare the actual safety_gated_model "
            "with frozen_baseline. Positive improvement means a lower "
            "metric. Differences within a scale-aware 1e-10 tolerance "
            "are treated as ties. Rejected-candidate counterfactuals "
            "cannot be inferred from retained-model metrics."
        ),
    }

    return detail, summary


# ============================================================
# 7. PLOTS
# ============================================================

def create_plots(results, gates, paired_data, output_dir):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: paired absolute improvement by seed.
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(9, 5))

        for model in [CANDIDATE_MODEL, GATED_MODEL]:
            entry = paired_data[(metric, model)]

            ax.plot(
                entry["seeds"],
                entry["difference"],
                marker="o",
                linestyle="-",
                label=model,
            )

        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Random seed")
        ax.set_ylabel("Baseline metric - comparison metric")
        ax.set_title(
            f"Step 10G: Paired improvement — {METRIC_LABELS[metric]}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig.savefig(
            plot_dir / f"paired_improvement_{metric}.png",
            dpi=160,
        )

        plt.close(fig)

    # Plot 2: held-out RMSE by seed.
    rmse_wide = results.pivot(
        index="seed",
        columns="model",
        values="mean_engine_RMSE",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    rmse_wide[
        [BASELINE_MODEL, CANDIDATE_MODEL, GATED_MODEL]
    ].plot(marker="o", ax=ax)

    ax.set_xlabel("Random seed")
    ax.set_ylabel("Mean engine RMSE")
    ax.set_title("Step 10G: Final held-out RMSE by seed")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "heldout_RMSE_by_seed.png", dpi=160)
    plt.close(fig)

    # Plot 3: gate decisions.
    gate_plot = gates.sort_values("seed").copy()
    gate_plot["decision_numeric"] = (
        gate_plot["candidate_accepted"].astype(int)
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        gate_plot["seed"].astype(str),
        gate_plot["decision_numeric"],
    )

    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1], labels=["Rejected", "Accepted"])
    ax.set_xlabel("Random seed")
    ax.set_title("Step 10G: Safety-gate decisions")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "gate_decisions.png", dpi=160)
    plt.close(fig)

    return plot_dir


# ============================================================
# 8. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STEP 10G — STATISTICAL ROBUSTNESS AND SAFETY-GATE AUDIT")
    print("=" * 72)
    print("Input directory:", INPUT_DIR)
    print("Output directory:", OUTPUT_DIR)

    results, gates = load_inputs()

    completed_seeds = sorted(
        results["seed"].unique().tolist()
    )

    gate_seeds = sorted(
        gates["seed"].unique().tolist()
    )

    print("Metric-result seeds:", completed_seeds)
    print("Gate-decision seeds:", gate_seeds)

    # Save normalized inputs used by this analysis.
    checked_input_path = (
        OUTPUT_DIR / "step10G_input_metrics_checked.csv"
    )

    results.to_csv(checked_input_path, index=False)

    # Paired statistical analysis.
    paired_table, paired_data = paired_analysis(results)

    paired_path = (
        OUTPUT_DIR / "step10G_paired_statistical_analysis.csv"
    )

    paired_table.to_csv(paired_path, index=False)

    # Audit actual safety-gated model outcomes.
    gate_detail, gate_summary = gate_audit(results, gates)

    gate_detail_path = (
        OUTPUT_DIR / "step10G_gate_outcome_audit.csv"
    )

    gate_detail.to_csv(gate_detail_path, index=False)

    # Generate plots.
    plot_dir = create_plots(
        results,
        gates,
        paired_data,
        OUTPUT_DIR,
    )

    # Console summary.
    print("\nPAIRED RESULTS: comparison versus frozen baseline")

    display_columns = [
        "metric",
        "comparison_model",
        "n_paired_seeds",
        "mean_relative_improvement_percent",
        "seeds_improved",
        "seeds_worsened",
        "seeds_tied",
        "bootstrap_95CI_lower_absolute",
        "bootstrap_95CI_upper_absolute",
        "paired_t_95CI_lower_absolute",
        "paired_t_95CI_upper_absolute",
        "wilcoxon_p_value",
        "sign_test_p_value",
    ]

    print(
        paired_table[display_columns].to_string(index=False)
    )

    print("\nSAFETY-GATE AUDIT")

    print(
        json.dumps(
            json_safe(gate_summary),
            indent=2,
        )
    )

    report = {
        "experiment": (
            "Step 10G statistical robustness and safety-gate audit"
        ),
        "source_directory": str(INPUT_DIR),
        "number_of_completed_seeds": len(completed_seeds),
        "seeds": completed_seeds,
        "models": [
            BASELINE_MODEL,
            CANDIDATE_MODEL,
            GATED_MODEL,
        ],
        "metrics": METRIC_LABELS,
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "paired_results": paired_table.to_dict(
            orient="records"
        ),
        "gate_audit": gate_summary,
        "interpretation_limitations": [
            "This step reuses Step 10F results and does not retrain models.",
            "Repeated seeds reuse the same dataset and are not independent fleets.",
            "Intervals describe variability across the tested seed-level splits.",
            "Ten seeds provide limited precision for inferential claims.",
            "Bootstrap intervals based on ten seed-level observations may be unstable.",
            "Statistical tests are exploratory and should not be overinterpreted.",
            "The gate is evaluated using validation data; final held-out outcomes are used retrospectively for audit.",
            "A validation gate cannot guarantee improved performance on unseen engines.",
            "Synthetic sensor shifts do not establish robustness to real industrial drift.",
            "NASA PHM score is asymmetric and should be reported separately.",
            "Do not revise gate criteria using these held-out results and then reuse the same results as an unbiased final evaluation.",
            "Rejected-candidate counterfactual outcomes cannot be inferred from the actual retained-model metrics.",
        ],
        "output_files": {
            "paired_statistical_analysis": str(paired_path),
            "gate_outcome_audit": str(gate_detail_path),
            "checked_input_metrics": str(checked_input_path),
            "plots_directory": str(plot_dir),
        },
    }

    report_path = OUTPUT_DIR / "step10G_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            json_safe(report),
            file,
            indent=2,
        )

    print("\nSTEP 10G COMPLETE")
    print("Output directory:", OUTPUT_DIR)
    print("Report:", report_path)
    print("Plots:", plot_dir)


if __name__ == "__main__":
    main()