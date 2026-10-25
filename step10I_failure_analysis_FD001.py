
"""
STEP 10I — Failure Analysis and Evidence Consolidation
Project: Self-Evolving Explainable Rational Industrial Agent

Reads existing Step 10F, 10G and 10H outputs.
Does not retrain models or alter any acceptance criteria.

Outputs:
  step10I_file_inventory.csv
  step10I_seed_level_comparisons.csv
  step10I_gate_decision_analysis.csv
  step10I_evidence_report.json
  step10I_research_report.md

Important:
- Missing evidence is reported as unavailable, not invented.
- Seed runs are repeated splits of the same dataset, not independent datasets.
- Final held-out results are descriptive evaluations.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    r"C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS"
)
OUTPUTS = ROOT / "outputs"

DIRS = {
    "10F": OUTPUTS / "step10F_repeated_seed_robustness",
    "10G": OUTPUTS / "step10G_statistical_robustness",
    "10H": OUTPUTS / "step10H_comparative_safety_gate",
}

OUT_DIR = OUTPUTS / "step10I_failure_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "mean_engine_RMSE",
    "mean_engine_MAE",
    "mean_engine_NASA_score",
]

LOWER_IS_BETTER = True


# ============================================================
# UTILITIES
# ============================================================

def normalized_columns(df):
    """Normalize spaces and punctuation for robust column matching."""
    result = df.copy()
    result.columns = [
        str(c).strip().replace(" ", "_").replace("-", "_")
        for c in result.columns
    ]
    return result


def read_csv_safely(path):
    try:
        return normalized_columns(pd.read_csv(path))
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def read_json_safely(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        warnings.warn(f"Could not parse {path}: {exc}")
        return None


def find_existing_files():
    files = []

    for step, directory in DIRS.items():
        if not directory.exists():
            print(f"WARNING: Step {step} directory not found: {directory}")
            continue

        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
                files.append({
                    "step": step,
                    "path": path,
                    "relative_path": str(path.relative_to(ROOT)),
                    "extension": path.suffix.lower(),
                    "bytes": path.stat().st_size,
                })

    return files


def metric_column_name(df, metric):
    """Return an exact metric column or a recognizable alternative."""
    if metric in df.columns:
        return metric

    aliases = {
        "mean_engine_RMSE": [
            "mean_engine_RMSE",
            "mean_engine_RMSE_mean_across_seeds",
            "RMSE",
            "rmse",
        ],
        "mean_engine_MAE": [
            "mean_engine_MAE",
            "mean_engine_MAE_mean_across_seeds",
            "MAE",
            "mae",
        ],
        "mean_engine_NASA_score": [
            "mean_engine_NASA_score",
            "mean_engine_NASA_score_mean_across_seeds",
            "NASA_score",
            "nasa_score",
        ],
    }

    for candidate in aliases.get(metric, []):
        if candidate in df.columns:
            return candidate

    return None


def metric_improvement(baseline, candidate):
    """Positive percentage means the candidate is better."""
    if (
        pd.isna(baseline)
        or pd.isna(candidate)
        or not np.isfinite(baseline)
        or not np.isfinite(candidate)
        or baseline == 0
    ):
        return np.nan

    return 100.0 * (baseline - candidate) / abs(baseline)


def find_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


# ============================================================
# INVENTORY
# ============================================================

def build_inventory(files):
    rows = []

    for item in files:
        row = {
            "step": item["step"],
            "relative_path": item["relative_path"],
            "extension": item["extension"],
            "bytes": item["bytes"],
            "readable": False,
            "columns_or_json_keys": "",
            "rows": np.nan,
        }

        if item["extension"] == ".csv":
            df = read_csv_safely(item["path"])
            if df is not None:
                row["readable"] = True
                row["rows"] = len(df)
                row["columns_or_json_keys"] = ", ".join(map(str, df.columns))

        elif item["extension"] == ".json":
            data = read_json_safely(item["path"])
            if data is not None:
                row["readable"] = True
                row["columns_or_json_keys"] = ", ".join(
                    map(str, data.keys())
                ) if isinstance(data, dict) else type(data).__name__

        rows.append(row)

    inventory = pd.DataFrame(rows)
    inventory.to_csv(OUT_DIR / "step10I_file_inventory.csv", index=False)
    return inventory


# ============================================================
# IDENTIFY SEED-LEVEL METRICS
# ============================================================

def extract_metric_tables(files):
    """
    Locate final-metric CSVs by their columns rather than assuming
    one exact filename.
    """
    results = []

    for item in files:
        if item["extension"] != ".csv":
            continue

        df = read_csv_safely(item["path"])
        if df is None or df.empty:
            continue

        seed_col = find_column(df, ["seed", "random_seed"])
        model_col = find_column(df, ["model", "model_name", "comparison_model"])

        if seed_col is None or model_col is None:
            continue

        # Avoid mistaking a gate-decision or paired-test file for
        # a final-metric table.
        metric_map = {}
        for metric in METRICS:
            col = metric_column_name(df, metric)
            if col is not None:
                metric_map[metric] = col

        if len(metric_map) < 2:
            continue

        low_name = item["path"].name.lower()

        if any(token in low_name for token in [
            "validation", "paired_results", "wilcoxon", "gate_decision"
        ]):
            continue

        for _, row in df.iterrows():
            record = {
                "step": item["step"],
                "source_file": item["relative_path"],
                "seed": row[seed_col],
                "model": str(row[model_col]).strip(),
            }

            for metric, col in metric_map.items():
                record[metric] = pd.to_numeric(
                    pd.Series([row[col]]), errors="coerce"
                ).iloc[0]

            results.append(record)

    if not results:
        return pd.DataFrame()

    result = pd.DataFrame(results)

    # Remove exact duplicate records that may appear in overlapping exports.
    key_cols = [
        c for c in ["step", "seed", "model"] + METRICS
        if c in result.columns
    ]
    result = result.drop_duplicates(subset=key_cols)

    return result


# ============================================================
# PAIRED COMPARISONS
# ============================================================

def compare_seed_level_models(metrics_df):
    if metrics_df.empty:
        return pd.DataFrame()

    rows = []

    for step, step_df in metrics_df.groupby("step"):
        model_values = step_df["model"].astype(str).unique().tolist()

        baseline_names = [
            m for m in model_values
            if "frozen" in m.lower() or "baseline" in m.lower()
        ]

        candidate_names = [
            m for m in model_values
            if (
                "unconditional" in m.lower()
                or "candidate" in m.lower()
                or "safety_gated" in m.lower()
                or "original_gate" in m.lower()
                or "nasa_prioritized" in m.lower()
            )
            and not (
                "baseline" in m.lower()
                or "frozen" in m.lower()
            )
        ]

        if not baseline_names:
            continue

        baseline_name = baseline_names[0]
        baseline_df = step_df[
            step_df["model"].astype(str) == baseline_name
        ]

        for candidate_name in candidate_names:
            candidate_df = step_df[
                step_df["model"].astype(str) == candidate_name
            ]

            paired = baseline_df.merge(
                candidate_df,
                on="seed",
                suffixes=("_baseline", "_candidate"),
                how="inner",
            )

            if paired.empty:
                continue

            for metric in METRICS:
                base_col = f"{metric}_baseline"
                cand_col = f"{metric}_candidate"

                if base_col not in paired.columns or cand_col not in paired.columns:
                    continue

                for _, pair in paired.iterrows():
                    base_value = pair[base_col]
                    candidate_value = pair[cand_col]

                    rows.append({
                        "step": step,
                        "seed": pair["seed"],
                        "baseline_model": baseline_name,
                        "candidate_model": candidate_name,
                        "metric": metric,
                        "baseline_value": base_value,
                        "candidate_value": candidate_value,
                        "improvement_percent": metric_improvement(
                            base_value, candidate_value
                        ),
                        "candidate_improved": (
                            metric_improvement(base_value, candidate_value) > 0
                            if pd.notna(base_value)
                            and pd.notna(candidate_value)
                            else np.nan
                        ),
                    })

    return pd.DataFrame(rows)


# ============================================================
# GATE DECISION AUDIT
# ============================================================

def extract_gate_tables(files):
    rows = []

    accepted_candidates = [
        "candidate_accepted",
        "original_gate_accepted",
        "nasa_prioritized_gate_accepted",
    ]

    for item in files:
        if item["extension"] != ".csv":
            continue

        df = read_csv_safely(item["path"])
        if df is None or df.empty:
            continue

        seed_col = find_column(df, ["seed", "random_seed"])
        gate_cols = [c for c in accepted_candidates if c in df.columns]

        if seed_col is None or not gate_cols:
            continue

        for _, row in df.iterrows():
            record = {
                "step": item["step"],
                "source_file": item["relative_path"],
                "seed": row[seed_col],
            }

            for col in gate_cols:
                value = row[col]
                if isinstance(value, str):
                    value = value.strip().lower() == "true"
                record[col] = (
                    bool(value) if pd.notna(value) else np.nan
                )

            rows.append(record)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop_duplicates()


# ============================================================
# REPORT GENERATION
# ============================================================

def summarize_paired_comparisons(comparisons):
    summaries = []

    if comparisons.empty:
        return summaries

    valid = comparisons.dropna(subset=["improvement_percent"])

    for (step, candidate, metric), group in valid.groupby(
        ["step", "candidate_model", "metric"]
    ):
        improvements = group["improvement_percent"].to_numpy(dtype=float)

        summaries.append({
            "step": step,
            "candidate_model": candidate,
            "metric": metric,
            "paired_seed_count": int(len(group)),
            "seeds_improved": int((improvements > 0).sum()),
            "seeds_worsened": int((improvements < 0).sum()),
            "mean_improvement_percent": float(np.mean(improvements)),
            "median_improvement_percent": float(np.median(improvements)),
        })

    return summaries


def build_markdown_report(
    inventory,
    metric_df,
    comparisons,
    gate_df,
    report,
):
    lines = [
        "# Step 10I — Failure Analysis and Evidence Consolidation",
        "",
        "## Purpose",
        "",
        "This report consolidates existing Step 10F, 10G and 10H outputs. "
        "It does not retrain models or change the predefined safety-gate rules.",
        "",
        "## Artifact availability",
        "",
        f"- Files discovered: {len(inventory)}",
        f"- Readable files: {int(inventory['readable'].sum()) if not inventory.empty else 0}",
        f"- Seed-level metric records identified: {len(metric_df)}",
        f"- Paired comparison records generated: {len(comparisons)}",
        f"- Gate-decision records identified: {len(gate_df)}",
        "",
    ]

    if inventory.empty:
        lines += [
            "No Step 10F–10H output files were found. Check the configured "
            "directories and run this script again.",
            "",
        ]

    if not comparisons.empty:
        lines += [
            "## Paired held-out comparisons",
            "",
            "Positive improvement percentages mean the candidate metric was "
            "lower than the frozen baseline. Negative values mean worsening.",
            "",
            "| Step | Candidate | Metric | Paired seeds | Improved | Worsened | Mean improvement % | Median improvement % |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]

        for item in summarize_paired_comparisons(comparisons):
            lines.append(
                f"| {item['step']} | {item['candidate_model']} | "
                f"{item['metric']} | {item['paired_seed_count']} | "
                f"{item['seeds_improved']} | {item['seeds_worsened']} | "
                f"{item['mean_improvement_percent']:.3f} | "
                f"{item['median_improvement_percent']:.3f} |"
            )

        lines.append("")
    else:
        lines += [
            "## Paired held-out comparisons",
            "",
            "No compatible paired metric tables were identified. Inspect "
            "`step10I_file_inventory.csv` and verify that the original output "
            "CSVs contain seed, model, and metric columns.",
            "",
        ]

    if not gate_df.empty:
        lines += [
            "## Gate decisions",
            "",
            "Gate decisions are counted from the stored decision files. "
            "A rejected candidate is not automatically a successful decision; "
            "its validation metrics and held-out outcome must also be examined.",
            "",
        ]

        for col in [
            "candidate_accepted",
            "original_gate_accepted",
            "nasa_prioritized_gate_accepted",
        ]:
            if col in gate_df.columns:
                valid = gate_df[col].dropna()
                lines.append(
                    f"- `{col}`: {int(valid.sum())} accepted of "
                    f"{len(valid)} recorded decisions."
                )
        lines.append("")

    lines += [
        "## Research interpretation",
        "",
        "1. The system operationally implements candidate updating, validation "
        "gating, acceptance, and retention of the incumbent model.",
        "2. A validation gate cannot guarantee improvement on unseen engines.",
        "3. RMSE, MAE, and NASA score measure different aspects of prediction "
        "quality and should be reported separately.",
        "4. Repeated seeds reuse the same dataset; they are not independent "
        "industrial fleets.",
        "5. Sensor shifts in these experiments are synthetic. They do not "
        "demonstrate robustness to real industrial concept drift.",
        "6. Feature importance or SHAP results are explanatory associations, "
        "not evidence of causal effects.",
        "7. Any absent, unreadable, or schema-incompatible evidence must be "
        "reported as unavailable, not inferred.",
        "",
        "## Limitations and next research checks",
        "",
        "- Confirm that each held-out engine is excluded from training and "
        "gate selection for its corresponding seed.",
        "- Preserve the original gate thresholds and experiment outputs.",
        "- Report variation across seeds and avoid treating seed runs as "
        "independent datasets.",
        "- Seek external or real fleet validation before making deployment claims.",
        "- Verify the exact feature engineering and preprocessing provenance "
        "used by each earlier experiment.",
        "",
        "## Reproducibility",
        "",
        "The JSON report and CSV tables generated alongside this Markdown "
        "document contain the discovered source files and extracted comparisons.",
        "",
    ]

    (OUT_DIR / "step10I_research_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    print("=" * 72)
    print("STEP 10I — FAILURE ANALYSIS AND EVIDENCE CONSOLIDATION")
    print("=" * 72)
    print("Project:", ROOT)
    print("Output:", OUT_DIR)

    files = find_existing_files()
    inventory = build_inventory(files)

    print(f"\nDiscovered {len(files)} CSV/JSON files.")

    metrics_df = extract_metric_tables(files)
    metrics_df.to_csv(
        OUT_DIR / "step10I_seed_level_comparisons.csv",
        index=False,
    )

    comparisons = compare_seed_level_models(metrics_df)
    comparisons.to_csv(
        OUT_DIR / "step10I_paired_metric_comparisons.csv",
        index=False,
    )

    gate_df = extract_gate_tables(files)
    gate_df.to_csv(
        OUT_DIR / "step10I_gate_decision_analysis.csv",
        index=False,
    )

    summary = summarize_paired_comparisons(comparisons)

    report = {
        "step": "10I",
        "title": "Failure Analysis and Evidence Consolidation",
        "project_root": str(ROOT),
        "output_directory": str(OUT_DIR),
        "source_directories": {
            key: str(path) for key, path in DIRS.items()
        },
        "files_discovered": len(files),
        "readable_files": (
            int(inventory["readable"].sum())
            if not inventory.empty else 0
        ),
        "seed_metric_records": len(metrics_df),
        "paired_comparison_records": len(comparisons),
        "gate_decision_records": len(gate_df),
        "paired_comparison_summary": summary,
        "interpretation_rules": {
            "positive_improvement_percent": "Candidate metric lower than baseline",
            "negative_improvement_percent": "Candidate metric higher than baseline",
            "lower_is_better": LOWER_IS_BETTER,
        },
        "limitations": [
            "This script consolidates available artifacts; it does not retrain models.",
            "Repeated seeds use the same underlying dataset.",
            "Synthetic shifts are not real-world drift evidence.",
            "Metric comparisons do not establish causality.",
            "A validation gate cannot guarantee held-out improvement.",
            "Schema-incompatible files may require manual inspection.",
        ],
        "outputs": [
            "step10I_file_inventory.csv",
            "step10I_seed_level_comparisons.csv",
            "step10I_paired_metric_comparisons.csv",
            "step10I_gate_decision_analysis.csv",
            "step10I_evidence_report.json",
            "step10I_research_report.md",
        ],
    }

    with open(
        OUT_DIR / "step10I_evidence_report.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2, default=str)

    build_markdown_report(
        inventory, metrics_df, comparisons, gate_df, report
    )

    print("\nFiles discovered:", len(files))
    print("Readable files:", report["readable_files"])
    print("Seed metric records:", len(metrics_df))
    print("Paired comparison records:", len(comparisons))
    print("Gate-decision records:", len(gate_df))

    if not comparisons.empty:
        print("\nPaired held-out comparison summary:")
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
    else:
        print(
            "\nNo compatible paired metric comparisons were extracted. "
            "Check step10I_file_inventory.csv before drawing conclusions."
        )

    print("\nSTEP 10I COMPLETE")
    print("Output directory:", OUT_DIR)
    print("Research report:", OUT_DIR / "step10I_research_report.md")
    print("Evidence JSON:", OUT_DIR / "step10I_evidence_report.json")


if __name__ == "__main__":
    main()
