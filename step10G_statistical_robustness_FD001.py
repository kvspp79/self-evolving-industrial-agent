
"""
STEP 10G: Statistical Robustness and Safety-Gate Audit
Project: Self-Evolving Explainable Industrial Agent
Dataset: NASA C-MAPSS FD001

Uses existing Step 10F outputs. Does not retrain models or modify gate criteria.

Analyses:
- Paired differences across seeds
- Bootstrap and paired-t 95% confidence intervals
- Wilcoxon signed-rank tests where computable
- Per-metric improvements and consistency counts
- Safety-gate acceptance versus final held-out outcomes
- CSV tables, plots, and JSON report

Interpretation:
Seeds reuse the same dataset and are not independent deployments.
Confidence intervals describe variability across the tested seed splits.
They do not establish real-world industrial generalizability.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
except ImportError as exc:
    raise ImportError(
        "SciPy is required. Install it with: py -3.13 -m pip install scipy"
    ) from exc


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS"
)

INPUT_DIR = PROJECT_ROOT / "outputs" / "step10F_repeated_seed_robustness"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step10G_statistical_robustness"

PER_SEED_FILE = INPUT_DIR / "step10F_per_seed_final_metrics.csv"
GATE_FILE = INPUT_DIR / "step10F_gate_decisions.csv"
ENGINE_FILE = INPUT_DIR / "step10F_final_engine_metrics.csv"

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

BOOTSTRAP_REPETITIONS = 20000
BOOTSTRAP_SEED = 202610
CONFIDENCE_LEVEL = 0.95


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required Step 10F file not found:\n{path}\n\n"
            "Check the project path and run Step 10F first if necessary."
        )


def json_safe(value):
    """Convert NumPy/Pandas values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def bootstrap_mean_ci(values, repetitions=BOOTSTRAP_REPETITIONS,
                      confidence=CONFIDENCE_LEVEL, seed=BOOTSTRAP_SEED):
    """
    Percentile bootstrap CI for the mean paired difference across seeds.

    Resampling unit is the seed-level paired difference, not individual rows
    or engines. With only ten seeds, this interval may be unstable.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(values)

    sampled_indices = rng.integers(
        low=0, high=n, size=(repetitions, n)
    )
    bootstrap_means = values[sampled_indices].mean(axis=1)

    alpha = 1.0 - confidence
    lower = float(np.quantile(bootstrap_means, alpha / 2))
    upper = float(np.quantile(bootstrap_means, 1 - alpha / 2))

    return lower, upper


def paired_t_ci(values, confidence=CONFIDENCE_LEVEL):
    """Paired t-based confidence interval for the mean difference."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)

    if n < 2:
        return np.nan, np.nan

    mean_value = float(np.mean(values))
    sd_value = float(np.std(values, ddof=1))
    standard_error = sd_value / np.sqrt(n)
    critical = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))

    return (
        mean_value - critical * standard_error,
        mean_value + critical * standard_error,
    )


def safe_wilcoxon(values):
    """
    Two-sided Wilcoxon signed-rank test on paired differences.

    The test assumes an appropriate paired-difference distribution.
    With only ten seeds, treat its result as exploratory.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return {"statistic": None, "p_value": None, "note": "Too few pairs"}

    if np.allclose(values, 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "note": "All paired differences are zero",
        }

    try:
        result = stats.wilcoxon(
            values,
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
    """Exact two-sided sign test, ignoring zero differences."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    nonzero = values[~np.isclose(values, 0)]
    n = len(nonzero)

    if n == 0:
        return {
            "positive": 0,
            "negative": 0,
            "ties": int(len(values)),
            "p_value": 1.0,
        }

    positive = int(np.sum(nonzero > 0))
    negative = int(np.sum(nonzero < 0))

    # Under the null, positive/negative signs each have probability 0.5.
    p_value = min(
        1.0,
        2.0 * float(stats.binom.cdf(min(positive, negative), n, 0.5)),
    )

    return {
        "positive": positive,
        "negative": negative,
        "ties": int(len(values) - n),
        "p_value": p_value,
    }


def relative_improvement(baseline, candidate):
    """
    Positive means candidate has a lower (better) metric than baseline.
    Negative means candidate has a higher (worse) metric.
    """
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)

    result = np.full_like(baseline, np.nan, dtype=float)
    valid = np.isfinite(baseline) & np.isfinite(candidate) & (baseline != 0)

    result[valid] = (
        (baseline[valid] - candidate[valid]) / np.abs(baseline[valid])
    )
    return result


# ============================================================
# 3. LOAD AND VALIDATE INPUTS
# ============================================================

def load_inputs():
    require_file(PER_SEED_FILE)
    require_file(GATE_FILE)

    results = pd.read_csv(PER_SEED_FILE)
    gates = pd.read_csv(GATE_FILE)

    required_result_columns = {
        "seed", "model", *METRICS
    }
    missing = required_result_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {PER_SEED_FILE.name}: {sorted(missing)}"
        )

    required_gate_columns = {"seed", "candidate_accepted"}
    missing_gate = required_gate_columns - set(gates.columns)

    if missing_gate:
        raise ValueError(
            f"Missing columns in {GATE_FILE.name}: {sorted(missing_gate)}"
        )

    results["seed"] = pd.to_numeric(results["seed"], errors="raise").astype(int)
    gates["seed"] = pd.to_numeric(gates["seed"], errors="raise").astype(int)

    results["model"] = results["model"].astype(str).str.strip()
    gates["candidate_accepted"] = (
        gates["candidate_accepted"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )

    if gates["candidate_accepted"].isna().any():
        raise ValueError("Could not parse candidate_accepted as True/False.")

    for metric in METRICS:
        results[metric] = pd.to_numeric(results[metric], errors="raise")

    if results.duplicated(["seed", "model"]).any():
        raise ValueError("Duplicate seed/model rows detected.")

    if gates["seed"].duplicated().any():
        raise ValueError("Duplicate seed rows detected in gate decisions.")

    return results, gates


# ============================================================
# 4. PAIRED SEED-LEVEL ANALYSIS
# ============================================================

def paired_analysis(results):
    all_rows = []
    paired_data = {}

    models = [BASELINE_MODEL, CANDIDATE_MODEL, GATED_MODEL]

    for metric in METRICS:
        wide = results.pivot(
            index="seed", columns="model", values=metric
        )

        missing_models = [m for m in models if m not in wide.columns]
        if missing_models:
            raise ValueError(
                f"Missing expected model(s) for {metric}: {missing_models}"
            )

        for comparison_model in [CANDIDATE_MODEL, GATED_MODEL]:
            pair = wide[[BASELINE_MODEL, comparison_model]].dropna()

            baseline = pair[BASELINE_MODEL].to_numpy(dtype=float)
            candidate = pair[comparison_model].to_numpy(dtype=float)

            # Positive differences mean improvement.
            differences = baseline - candidate
            relative = relative_improvement(baseline, candidate)

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
                "candidate": candidate,
                "difference": differences,
                "relative_improvement": relative,
            }

            all_rows.append({
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "comparison_model": comparison_model,
                "n_paired_seeds": int(len(differences)),
                "baseline_mean": float(np.mean(baseline)),
                "comparison_model_mean": float(np.mean(candidate)),
                "mean_absolute_improvement": float(np.mean(differences)),
                "median_absolute_improvement": float(np.median(differences)),
                "mean_relative_improvement_percent": float(
                    100 * np.mean(relative)
                ),
                "median_relative_improvement_percent": float(
                    100 * np.median(relative)
                ),
                "seeds_improved": int(np.sum(differences > 0)),
                "seeds_worsened": int(np.sum(differences < 0)),
                "seeds_tied": int(np.sum(np.isclose(differences, 0))),
                "bootstrap_95CI_lower_absolute": bootstrap_low,
                "bootstrap_95CI_upper_absolute": bootstrap_high,
                "paired_t_95CI_lower_absolute": t_low,
                "paired_t_95CI_upper_absolute": t_high,
                "wilcoxon_statistic": wilcoxon["statistic"],
                "wilcoxon_p_value": wilcoxon["p_value"],
                "sign_test_p_value": sign_test["p_value"],
                "sign_test_positive": sign_test["positive"],
                "sign_test_negative": sign_test["negative"],
                "sign_test_ties": sign_test["ties"],
            })

    return pd.DataFrame(all_rows), paired_data


# ============================================================
# 5. SAFETY-GATE AUDIT
# ============================================================

def gate_audit(results, gates):
    """
    The candidate model in Step 10F is the engine-balanced 2x candidate.

    An accepted update is classified as worsening if it is worse than the
    frozen baseline on at least one final held-out metric.

    This is an audit of observed outcomes, not proof of future safety.
    """
    wide = results.pivot(
        index="seed", columns="model", values=METRICS
    )

    records = []

    for _, gate_row in gates.iterrows():
        seed = int(gate_row["seed"])
        accepted = bool(gate_row["candidate_accepted"])

        if seed not in wide.index:
            warnings.warn(f"Seed {seed} has gate record but no metric results.")
            continue

        record = {
            "seed": seed,
            "candidate_accepted": accepted,
        }

        metric_improvements = {}

        for metric in METRICS:
            baseline = wide.loc[seed, (metric, BASELINE_MODEL)]
            candidate = wide.loc[seed, (metric, CANDIDATE_MODEL)]

            improvement = float(
                relative_improvement(
                    np.array([baseline]), np.array([candidate])
                )[0]
            )

            metric_improvements[metric] = improvement

            record[f"{metric}_relative_improvement_percent"] = (
                100 * improvement
            )

            record[f"{metric}_improved"] = bool(improvement > 0)
            record[f"{metric}_worsened"] = bool(improvement < 0)

        any_metric_improved = any(v > 0 for v in metric_improvements.values())
        any_metric_worsened = any(v < 0 for v in metric_improvements.values())
        all_metrics_improved = all(v > 0 for v in metric_improvements.values())
        all_metrics_worsened = all(v < 0 for v in metric_improvements.values())

        record["candidate_improved_any_metric"] = any_metric_improved
        record["candidate_worsened_any_metric"] = any_metric_worsened
        record["candidate_improved_all_metrics"] = all_metrics_improved
        record["candidate_worsened_all_metrics"] = all_metrics_worsened

        # Accepted update followed by any held-out metric worsening.
        record["accepted_then_any_metric_worsened"] = (
            accepted and any_metric_worsened
        )

        # Accepted update improved every reported held-out metric.
        record["accepted_then_all_metrics_improved"] = (
            accepted and all_metrics_improved
        )

        # Rejected candidate would have improved every held-out metric.
        # This is retrospective and must not be used to retune the gate.
        record["rejected_but_all_metrics_improved"] = (
            (not accepted) and all_metrics_improved
        )

        # Rejected candidate would have worsened at least one metric.
        record["rejected_and_any_metric_worsened"] = (
            (not accepted) and any_metric_worsened
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
            100 * accepted_count / total if total else None
        ),
        "accepted_then_any_metric_worsened_count": int(
            detail["accepted_then_any_metric_worsened"].sum()
        ),
        "accepted_then_all_metrics_improved_count": int(
            detail["accepted_then_all_metrics_improved"].sum()
        ),
        "rejected_but_all_metrics_improved_count": int(
            detail["rejected_but_all_metrics_improved"].sum()
        ),
        "rejected_and_any_metric_worsened_count": int(
            detail["rejected_and_any_metric_worsened"].sum()
        ),
        "accepted_update_worsening_rate_percent": (
            100 * detail["accepted_then_any_metric_worsened"].sum()
            / accepted_count
            if accepted_count else None
        ),
        "accepted_update_all_metric_improvement_rate_percent": (
            100 * detail["accepted_then_all_metrics_improved"].sum()
            / accepted_count
            if accepted_count else None
        ),
        "rejected_candidate_all_metric_improvement_rate_percent": (
            100 * detail["rejected_but_all_metrics_improved"].sum()
            / rejected_count
            if rejected_count else None
        ),
        "accepted_seed_ids": accepted_rows["seed"].astype(int).tolist(),
        "rejected_seed_ids": rejected_rows["seed"].astype(int).tolist(),
        "definition": (
            "Held-out worsening means the candidate was worse than frozen "
            "baseline on at least one of RMSE, MAE, or NASA score."
        ),
    }

    return detail, summary


# ============================================================
# 6. PLOTS
# ============================================================

def create_plots(results, gates, paired_data, output_dir):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: paired absolute improvement for each metric.
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(9, 5))

        for model in [CANDIDATE_MODEL, GATED_MODEL]:
            entry = paired_data[(metric, model)]
            seeds = entry["seeds"]
            differences = entry["difference"]

            ax.plot(
                seeds,
                differences,
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
        index="seed", columns="model", values="mean_engine_RMSE"
    ).sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    rmse_wide[[BASELINE_MODEL, CANDIDATE_MODEL, GATED_MODEL]].plot(
        marker="o", ax=ax
    )
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Mean engine RMSE")
    ax.set_title("Step 10G: Final held-out RMSE by seed")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "heldout_RMSE_by_seed.png", dpi=160)
    plt.close(fig)

    # Plot 3: gate decisions.
    gate_plot = gates.sort_values("seed").copy()
    gate_plot["decision_numeric"] = gate_plot["candidate_accepted"].astype(int)

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
# 7. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STEP 10G — STATISTICAL ROBUSTNESS AND SAFETY-GATE AUDIT")
    print("=" * 72)
    print("Input directory:", INPUT_DIR)
    print("Output directory:", OUTPUT_DIR)

    results, gates = load_inputs()

    completed_seeds = sorted(results["seed"].unique().tolist())
    gate_seeds = sorted(gates["seed"].unique().tolist())

    print("Metric-result seeds:", completed_seeds)
    print("Gate-decision seeds:", gate_seeds)

    # Check that all expected models exist for each completed seed.
    for seed in completed_seeds:
        available = set(results.loc[results["seed"] == seed, "model"])
        expected = {BASELINE_MODEL, CANDIDATE_MODEL, GATED_MODEL}
        if not expected.issubset(available):
            raise ValueError(
                f"Seed {seed} is missing model results. "
                f"Expected {expected}, found {available}"
            )

    # Save normalized copies used by this analysis.
    results.to_csv(OUTPUT_DIR / "step10G_input_metrics_checked.csv", index=False)

    # Paired statistical analysis.
    paired_table, paired_data = paired_analysis(results)
    paired_path = OUTPUT_DIR / "step10G_paired_statistical_analysis.csv"
    paired_table.to_csv(paired_path, index=False)

    # Gate audit.
    gate_detail, gate_summary = gate_audit(results, gates)
    gate_detail_path = OUTPUT_DIR / "step10G_gate_outcome_audit.csv"
    gate_detail.to_csv(gate_detail_path, index=False)

    # Generate plots.
    plot_dir = create_plots(results, gates, paired_data, OUTPUT_DIR)

    # Concise console output.
    print("\nPAIRED RESULTS: comparison versus frozen baseline")
    display_columns = [
        "metric",
        "comparison_model",
        "n_paired_seeds",
        "mean_relative_improvement_percent",
        "seeds_improved",
        "seeds_worsened",
        "bootstrap_95CI_lower_absolute",
        "bootstrap_95CI_upper_absolute",
        "wilcoxon_p_value",
        "sign_test_p_value",
    ]
    print(paired_table[display_columns].to_string(index=False))

    print("\nSAFETY-GATE AUDIT")
    print(json.dumps(json_safe(gate_summary), indent=2))

    report = {
        "experiment": "Step 10G statistical robustness and safety-gate audit",
        "source_directory": str(INPUT_DIR),
        "number_of_completed_seeds": len(completed_seeds),
        "seeds": completed_seeds,
        "models": [BASELINE_MODEL, CANDIDATE_MODEL, GATED_MODEL],
        "metrics": METRIC_LABELS,
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "paired_results": paired_table.to_dict(orient="records"),
        "gate_audit": gate_summary,
        "interpretation_limitations": [
            "The analysis reuses Step 10F results and does not retrain models.",
            "Seeds reuse the same dataset; they are not independent datasets.",
            "Intervals quantify variability across tested seed-level splits.",
            "Ten seeds provide limited precision for inferential claims.",
            "Bootstrap intervals with ten seed-level observations may be unstable.",
            "Statistical tests are exploratory and should not be overinterpreted.",
            "The gate was evaluated on validation data; held-out results are used only retrospectively for audit.",
            "An accepted update that worsens one held-out metric is not automatically unsafe in every operational sense; the definition is an empirical audit rule.",
            "Synthetic sensor shift does not establish robustness to real industrial drift.",
            "Do not change acceptance criteria based on these results and then reuse the same final results as an unbiased evaluation.",
        ],
        "output_files": {
            "paired_statistical_analysis": str(paired_path),
            "gate_outcome_audit": str(gate_detail_path),
            "checked_input_metrics": str(
                OUTPUT_DIR / "step10G_input_metrics_checked.csv"
            ),
            "plots_directory": str(plot_dir),
        },
    }

    report_path = OUTPUT_DIR / "step10G_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2)

    print("\nSTEP 10G COMPLETE")
    print("Output directory:", OUTPUT_DIR)
    print("Report:", report_path)
    print("Plots:", plot_dir)


if __name__ == "__main__":
    main()
