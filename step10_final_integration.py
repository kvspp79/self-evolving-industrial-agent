
# ================================================================
# STEP 10: FINAL INTEGRATION, GRAPHS, AND RESEARCH REPORT
#
# Project:
# A Self-Evolving Explainable Rational Agent for Adaptive
# Industrial Decision Intelligence Using Statistical Trend Learning
#
# This script integrates existing outputs from Steps 1-9.
# It does not retrain models or fabricate missing results.
# ================================================================

import os
import re
import sys
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
FINAL_DIR = os.path.join(OUTPUT_DIR, "step10_final")

os.makedirs(FINAL_DIR, exist_ok=True)


# ------------------------------------------------
# 2. FILE DISCOVERY
# ------------------------------------------------

def normalize_name(value):
    """Ignore spaces, underscores, punctuation, and case."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_output_file(prefix, extension=None):
    """
    Find an output by normalized filename prefix.
    This supports filenames containing spaces or underscores.
    """
    if not os.path.isdir(OUTPUT_DIR):
        return None

    wanted = normalize_name(prefix)
    matches = []

    for filename in os.listdir(OUTPUT_DIR):
        full_path = os.path.join(OUTPUT_DIR, filename)

        if not os.path.isfile(full_path):
            continue

        if extension is not None:
            if not filename.lower().endswith(extension.lower()):
                continue

        normalized_filename = normalize_name(
            os.path.splitext(filename)[0]
        )

        if normalized_filename.startswith(wanted):
            matches.append(full_path)

    if not matches:
        return None

    # Prefer the closest normalized filename match.
    matches.sort(
        key=lambda path: (
            len(normalize_name(os.path.basename(path))),
            os.path.basename(path).lower()
        )
    )

    if len(matches) > 1:
        print(
            "Multiple matches for '{0}'; using '{1}'".format(
                prefix, os.path.basename(matches[0])
            )
        )

    return matches[0]


def load_csv(prefix):
    path = find_output_file(prefix, ".csv")

    if path is None:
        print("Optional CSV not found: {0}".format(prefix))
        return None, None

    try:
        frame = pd.read_csv(path)
        print(
            "Loaded CSV: {0} ({1} rows)".format(
                os.path.basename(path), len(frame)
            )
        )
        return frame, path

    except Exception as error:
        print(
            "Could not read {0}: {1}".format(path, error)
        )
        return None, path


def load_json(prefix):
    path = find_output_file(prefix, ".json")

    if path is None:
        return None, None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file), path

    except Exception as error:
        print(
            "Could not read JSON {0}: {1}".format(path, error)
        )
        return None, path


# ------------------------------------------------
# 3. GENERIC COLUMN DETECTION
# ------------------------------------------------

def find_column(frame, candidate_names):
    normalized_columns = {
        normalize_name(column): column
        for column in frame.columns
    }

    for candidate in candidate_names:
        key = normalize_name(candidate)

        if key in normalized_columns:
            return normalized_columns[key]

    return None


def numeric_columns(frame):
    result = []

    for column in frame.columns:
        converted = pd.to_numeric(
            frame[column], errors="coerce"
        )

        if converted.notna().sum() > 0:
            result.append(column)

    return result


def label_column(frame, numeric):
    """Find a useful category/name column for graph labels."""
    preferred = [
        "policy",
        "model",
        "feature",
        "sensor",
        "feature_name",
        "action",
        "machine_type",
        "type",
        "metric",
        "method",
        "name"
    ]

    normalized_numeric = set(numeric)

    for candidate in preferred:
        column = find_column(frame, [candidate])

        if column is not None and column not in normalized_numeric:
            return column

    for column in frame.columns:
        if column not in normalized_numeric:
            return column

    return None


# ------------------------------------------------
# 4. SAFE GRAPH GENERATION
# ------------------------------------------------

def save_bar_chart(
    frame,
    label_col,
    value_col,
    title,
    y_label,
    filename,
    top_n=20
):
    """
    Create a simple bar chart from an existing results table.
    Invalid or missing numeric values are excluded.
    """
    if frame is None:
        return None

    if label_col not in frame.columns:
        return None

    if value_col not in frame.columns:
        return None

    data = frame[[label_col, value_col]].copy()

    data[value_col] = pd.to_numeric(
        data[value_col], errors="coerce"
    )

    data = data.dropna(subset=[label_col, value_col])

    if data.empty:
        print("Skipped empty graph: {0}".format(filename))
        return None

    data[label_col] = data[label_col].astype(str)

    if len(data) > top_n:
        data = data.reindex(
            data[value_col].abs().sort_values(
                ascending=False
            ).index
        ).head(top_n)

    data = data.sort_values(value_col, ascending=True)

    height = max(5, min(10, 0.35 * len(data) + 2))

    plt.figure(figsize=(10, height))
    plt.barh(data[label_col], data[value_col])
    plt.title(title)
    plt.xlabel(y_label)
    plt.ylabel(label_col)
    plt.tight_layout()

    path = os.path.join(FINAL_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

    print("Saved graph: {0}".format(path))
    return path


def choose_numeric_metric(frame, preferred_terms):
    """
    Choose a numeric column using semantic preferences.
    The choice is recorded in the report for transparency.
    """
    candidates = numeric_columns(frame)

    if not candidates:
        return None

    for term in preferred_terms:
        normalized_term = normalize_name(term)

        for column in candidates:
            if normalized_term in normalize_name(column):
                return column

    # Fall back to the first numeric column.
    return candidates[0]


def make_table_graph(
    frame,
    title,
    filename,
    preferred_terms,
    y_label=None
):
    if frame is None or frame.empty:
        return None

    numbers = numeric_columns(frame)

    if not numbers:
        print("No numeric columns for graph: {0}".format(title))
        return None

    label = label_column(frame, numbers)

    if label is None:
        print("No category column for graph: {0}".format(title))
        return None

    metric = choose_numeric_metric(frame, preferred_terms)

    if metric is None:
        return None

    print(
        "Graph metric for '{0}': {1}".format(title, metric)
    )

    return save_bar_chart(
        frame,
        label,
        metric,
        title,
        y_label or metric,
        filename
    )


# ------------------------------------------------
# 5. DATASET AND MODEL OUTPUT INVENTORY
# ------------------------------------------------

def list_files(folder):
    if not os.path.isdir(folder):
        return []

    return sorted(
        [
            filename
            for filename in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, filename))
        ],
        key=str.lower
    )


def collect_inventory():
    outputs = list_files(OUTPUT_DIR)
    models = list_files(MODEL_DIR)

    return {
        "output_files": outputs,
        "model_files": models,
        "output_file_count": len(outputs),
        "model_file_count": len(models)
    }


# ------------------------------------------------
# 6. LOAD EXISTING RESULTS
# ------------------------------------------------

def load_all_results():
    result_prefixes = [
        "step02_model_metrics",
        "step03_calibration_metrics",
        "step04_shap_feature_importance",
        "step05_two_feature_comparison",
        "step06_policy_cost_comparison",
        "step06_action_summary",
        "step07_sensor_associations",
        "step07_failure_risk_by_type",
        "step07_sensor_condition_risk",
        "step08_policy_cost_comparison",
        "step08_policy_selection_counts",
        "step09_fixed_policy_baselines",
        "step09_robustness_summary",
        "step09_online_episode_summary"
    ]

    results = {}

    for prefix in result_prefixes:
        frame, path = load_csv(prefix)

        results[prefix] = {
            "data": frame,
            "path": path
        }

    reports = {}

    for step in range(1, 10):
        prefix = "step{0:02d}_report".format(step)
        report, path = load_json(prefix)

        reports[prefix] = {
            "data": report,
            "path": path
        }

    return results, reports


# ------------------------------------------------
# 7. FINAL COMPARISON GRAPHS
# ------------------------------------------------

def create_final_graphs(results):
    graph_paths = []
    graph_choices = []

    specifications = [
        (
            "step02_model_metrics",
            "Model Performance Comparison",
            "step10_model_performance.png",
            ["roc_auc", "f1", "average_precision", "accuracy"],
            "Metric value"
        ),
        (
            "step03_calibration_metrics",
            "Probability Calibration Comparison",
            "step10_calibration_comparison.png",
            ["brier", "log_loss", "ece"],
            "Metric value"
        ),
        (
            "step04_shap_feature_importance",
            "SHAP Feature Importance",
            "step10_shap_importance.png",
            ["mean_abs_shap", "mean_absolute_shap", "importance"],
            "Importance"
        ),
        (
            "step05_two_feature_comparison",
            "Six-Feature Versus Two-Feature Experiment",
            "step10_feature_experiment.png",
            ["roc_auc", "f1", "average_precision", "accuracy"],
            "Metric value"
        ),
        (
            "step06_policy_cost_comparison",
            "Step 6 Policy Cost Comparison",
            "step10_step06_policy_cost.png",
            ["mean_cost_per_row", "mean_cost", "total_cost"],
            "Cost"
        ),
        (
            "step07_sensor_associations",
            "Sensor and Failure-Risk Associations",
            "step10_sensor_associations.png",
            ["spearman", "correlation", "association", "rho"],
            "Association value"
        ),
        (
            "step09_fixed_policy_baselines",
            "Step 9 Fixed Policy Cost Comparison",
            "step10_fixed_policy_cost.png",
            ["mean_cost_per_row", "total_cost"],
            "Simulated cost"
        ),
        (
            "step09_robustness_summary",
            "Step 9 Policy Robustness Summary",
            "step10_robustness.png",
            ["mean_cost_per_row", "mean_total_cost"],
            "Mean simulated cost"
        )
    ]

    for (
        key,
        title,
        filename,
        preferred_terms,
        y_label
    ) in specifications:

        frame = results[key]["data"]

        if frame is None or frame.empty:
            print(
                "Skipping graph because data is unavailable: {0}".format(
                    key
                )
            )
            continue

        path = make_table_graph(
            frame,
            title,
            filename,
            preferred_terms,
            y_label
        )

        if path is not None:
            graph_paths.append(path)

            graph_choices.append({
                "source_key": key,
                "source_file": (
                    os.path.basename(results[key]["path"])
                    if results[key]["path"]
                    else None
                ),
                "graph_file": os.path.basename(path),
                "metric_column": choose_numeric_metric(
                    frame, preferred_terms
                )
            })

    return graph_paths, graph_choices


# ------------------------------------------------
# 8. CREATE HUMAN-READABLE REPORT
# ------------------------------------------------

def safe_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

    if isinstance(value, float) and not np.isfinite(value):
        return None

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def frame_summary(frame):
    if frame is None:
        return {
            "available": False
        }

    return {
        "available": True,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns]
    }


def build_report(results, reports, inventory, graph_choices):
    report = {
        "project_title": (
            "A Self-Evolving Explainable Rational Agent for Adaptive "
            "Industrial Decision Intelligence Using Statistical "
            "Trend Learning"
        ),
        "step": 10,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "project_root": ROOT_DIR,
        "inventory": inventory,
        "integrated_tables": {},
        "previous_step_reports_available": {},
        "graphs_generated": graph_choices,
        "interpretation_and_limitations": [
            "The report integrates existing experiment outputs; "
            "it does not retrain the models.",
            "The AI4I 2020 dataset does not provide true chronological "
            "histories for individual machines.",
            "Statistical associations between operating conditions "
            "and failure are not proof of causation.",
            "Cost results use simulated assumptions, not measured "
            "industrial operating expenses.",
            "The online selector chooses among predefined policies. "
            "This does not establish unrestricted autonomous learning.",
            "Lower observed cost alone does not establish statistical "
            "significance or real-world savings.",
            "Graph metrics are selected from existing table columns. "
            "Interpret each graph using its source table and metric."
        ]
    }

    for key, item in results.items():
        frame = item["data"]

        report["integrated_tables"][key] = {
            "source_file": (
                os.path.basename(item["path"])
                if item["path"]
                else None
            ),
            "summary": frame_summary(frame)
        }

        if frame is not None and not frame.empty:
            # Include a small preview to keep the report compact.
            preview = frame.head(5).copy()
            preview = preview.replace(
                [np.inf, -np.inf], np.nan
            )
            preview = preview.astype(object).where(
                pd.notna(preview), None
            )

            report["integrated_tables"][key]["preview"] = [
                {
                    str(column): safe_value(value)
                    for column, value in row.items()
                }
                for row in preview.to_dict(orient="records")
            ]

    for key, item in reports.items():
        report["previous_step_reports_available"][key] = (
            item["data"] is not None
        )

    return report


def write_text_report(report):
    path = os.path.join(
        FINAL_DIR,
        "step10_integrated_research_report.txt"
    )

    lines = []

    lines.append("=" * 76)
    lines.append("FINAL INTEGRATED RESEARCH REPORT")
    lines.append("=" * 76)
    lines.append("")
    lines.append(report["project_title"])
    lines.append("")
    lines.append("Generated: " + report["generated_at"])
    lines.append("Python version: " + report["python_version"])
    lines.append("")

    lines.append("1. PROJECT OUTPUT INVENTORY")
    lines.append("-" * 76)
    lines.append(
        "Output files found: {0}".format(
            report["inventory"]["output_file_count"]
        )
    )
    lines.append(
        "Model files found: {0}".format(
            report["inventory"]["model_file_count"]
        )
    )

    lines.append("")
    lines.append("Model files:")

    for filename in report["inventory"]["model_files"]:
        lines.append("  - " + filename)

    lines.append("")
    lines.append("2. EXPERIMENT OUTPUT AVAILABILITY")
    lines.append("-" * 76)

    for key, item in report["integrated_tables"].items():
        summary = item["summary"]

        if summary["available"]:
            lines.append(
                "{0}: available; {1} rows".format(
                    key, summary["rows"]
                )
            )
        else:
            lines.append("{0}: unavailable".format(key))

    lines.append("")
    lines.append("3. PREVIOUS STEP REPORTS")

    for key, available in (
        report["previous_step_reports_available"].items()
    ):
        lines.append(
            "{0}: {1}".format(
                key,
                "available" if available else "not found"
            )
        )

    lines.append("")
    lines.append("4. FINAL GRAPHS")
    lines.append("-" * 76)

    if report["graphs_generated"]:
        for item in report["graphs_generated"]:
            lines.append(
                "{0} | source: {1} | metric: {2}".format(
                    item["graph_file"],
                    item["source_file"],
                    item["metric_column"]
                )
            )
    else:
        lines.append(
            "No graphs were generated. Check whether the expected "
            "CSV files contain numeric columns."
        )

    lines.append("")
    lines.append("5. SCIENTIFIC INTERPRETATION")
    lines.append("-" * 76)
    lines.append(
        "The project combines failure prediction, probability "
        "calibration, SHAP-based explanation, statistical "
        "condition-risk analysis, decision policies, and simulated "
        "feedback-driven policy selection."
    )
    lines.append(
        "The strength of each conclusion depends on the corresponding "
        "experiment, evaluation split, metric definition, and source "
        "data. The existence of an output file alone is not proof of "
        "model quality or agent effectiveness."
    )

    lines.append("")
    lines.append("6. LIMITATIONS")
    lines.append("-" * 76)

    for limitation in report["interpretation_and_limitations"]:
        lines.append("- " + limitation)

    lines.append("")
    lines.append("7. OUTPUT LOCATION")
    lines.append(FINAL_DIR)
    lines.append("")
    lines.append("=" * 76)
    lines.append("END OF REPORT")
    lines.append("=" * 76)

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return path


# ------------------------------------------------
# 9. MAIN
# ------------------------------------------------

def main():
    print("=" * 76)
    print("STEP 10: FINAL RESEARCH INTEGRATION")
    print("=" * 76)

    print("Python executable: {0}".format(sys.executable))
    print("Python version: {0}".format(sys.version.split()[0]))
    print("Project root: {0}".format(ROOT_DIR))
    print("Final output folder: {0}".format(FINAL_DIR))

    if not os.path.isdir(OUTPUT_DIR):
        raise FileNotFoundError(
            "Output directory not found: {0}".format(OUTPUT_DIR)
        )

    inventory = collect_inventory()

    print(
        "\nExisting output files: {0}".format(
            inventory["output_file_count"]
        )
    )
    print(
        "Existing model files: {0}".format(
            inventory["model_file_count"]
        )
    )

    results, reports = load_all_results()

    graph_paths, graph_choices = create_final_graphs(results)

    report = build_report(
        results,
        reports,
        inventory,
        graph_choices
    )

    json_path = os.path.join(
        FINAL_DIR,
        "step10_integrated_report.json"
    )

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

    text_path = write_text_report(report)

    print("\n" + "=" * 76)
    print("STEP 10 INTEGRATION COMPLETED")
    print("=" * 76)
    print("Generated graphs: {0}".format(len(graph_paths)))
    print("JSON report: {0}".format(json_path))
    print("Text report: {0}".format(text_path))
    print("Final folder: {0}".format(FINAL_DIR))

    print("\nFiles in the final integration folder:")

    for filename in sorted(os.listdir(FINAL_DIR)):
        print("  - " + filename)

    print(
        "\nIMPORTANT: Review the source CSVs and report before "
        "using any result in a paper, presentation, or thesis."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nSTEP 10 FAILED")
        print("Error type: {0}".format(type(error).__name__))
        print("Error message: {0}".format(error))
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
