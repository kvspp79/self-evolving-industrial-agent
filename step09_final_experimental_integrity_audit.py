
from __future__ import annotations

import csv
import json
import math
import hashlib
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# STEP 9: FINAL EXPERIMENTAL INTEGRITY AND METRICS AUDIT
# Dataset: NASA C-MAPSS FD001
#
# READ-ONLY:
# - Does not retrain models.
# - Does not modify predictions or policies.
# - Does not select thresholds.
# - Does not use test labels for tuning.
#
# This script audits stored artifacts and reported metrics.
# It does NOT independently reproduce every model fit.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
AUDIT_DIR = OUTPUT_ROOT / "step09"

AUDIT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now(timezone.utc).isoformat()

findings: list[dict] = []
inventory: list[dict] = []
metric_checks: list[dict] = []
json_reports: dict[str, dict] = {}


def add_finding(
    severity: str,
    category: str,
    artifact: str,
    message: str,
) -> None:
    findings.append(
        {
            "severity": severity,
            "category": category,
            "artifact": str(artifact),
            "message": message,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, Path):
        return str(value)

    return value


def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            result = json.load(handle)

        if not isinstance(result, dict):
            add_finding(
                "WARNING",
                "JSON_STRUCTURE",
                path,
                "JSON is valid, but its root is not an object.",
            )
            return None

        json_reports[str(path)] = result
        return result

    except Exception as exc:
        add_finding(
            "ERROR",
            "JSON_READ",
            path,
            f"Could not parse JSON: {exc}",
        )
        return None


def read_csv(path: Path):
    try:
        frame = pd.read_csv(path, low_memory=False)

        if frame.empty:
            add_finding(
                "WARNING",
                "EMPTY_CSV",
                path,
                "CSV has headers or content but contains zero data rows.",
            )

        duplicate_columns = frame.columns[
            frame.columns.duplicated()
        ].tolist()

        if duplicate_columns:
            add_finding(
                "WARNING",
                "DUPLICATE_COLUMNS",
                path,
                f"Duplicate column names: {duplicate_columns}",
            )

        return frame

    except Exception as exc:
        add_finding(
            "ERROR",
            "CSV_READ",
            path,
            f"Could not read CSV: {exc}",
        )
        return None


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(OUTPUT_ROOT))
    except ValueError:
        return str(path)


# ------------------------------------------------------------
# 1. Check required artifacts
# ------------------------------------------------------------

required_files = [
    "step04/validation_predictions_FD001.csv",
    "step04/test_predictions_FD001.csv",
    "step04/model_comparison_FD001.csv",
    "step04/step04_FD001_report.json",
    "step05/shap_feature_importance_FD001.csv",
    "step05/step05_FD001_report.json",
    "step07_2/step07_2_FD001_report.json",
    "step08/step08_FD001_report.json",
    "step08/step08_frozen_test_policy_results.csv",
    "step08_2/step08_2_FD001_report.json",
    "step08_2/sensor_distribution_shift_FD001.csv",
    "step08_3/step08_3_FD001_report.json",
    "step08_3/explainable_adaptation_audit_FD001.csv",
    "step08_4/step08_4_FD001_report.json",
    "step08_4/validation_metrics_FD001.csv",
    "step08_5/step08_5_FD001_report.json",
    "step08_5/calibration_parameters_FD001.json",
    "step08_5/validation_calibration_metrics_FD001.csv",
    "step08_6/step08_6_FD001_report.json",
    "step08_6/step08_6_validation_metrics_FD001.csv",
    "step08_7/step08_7_FD001_report.json",
    "step08_7/step08_7_engine_bootstrap_FD001.csv",
    "step08_7/step08_7_final_cycle_regression_metrics_FD001.csv",
    "step08_8/step08_8_FD001_evidence_audit.md",
    "step08_8/step08_8_artifact_inventory_FD001.csv",
]

print("=" * 72)
print("NASA C-MAPSS FD001 — STEP 9 INTEGRITY AUDIT")
print("=" * 72)
print("Project:", PROJECT_ROOT)
print("UTC timestamp:", TIMESTAMP)
print()

missing_count = 0

for relative in required_files:
    path = OUTPUT_ROOT / Path(relative)
    exists = path.is_file()

    if exists:
        print("[PRESENT]", relative)
    else:
        missing_count += 1
        print("[MISSING]", relative)
        add_finding(
            "WARNING",
            "REQUIRED_ARTIFACT",
            path,
            "Expected artifact was not found. Confirm whether it was "
            "generated, renamed, or intentionally omitted.",
        )

print()
print("Required artifacts missing:", missing_count)


# ------------------------------------------------------------
# 2. Inventory all existing artifacts through Step 8.8
# ------------------------------------------------------------

print("\nScanning output files...")

for path in sorted(OUTPUT_ROOT.rglob("*")):
    if not path.is_file():
        continue

    if AUDIT_DIR in path.parents:
        continue

    suffix = path.suffix.lower()

    try:
        stat = path.stat()
        size_bytes = stat.st_size
        digest = sha256_file(path) if suffix in {
            ".json", ".csv", ".md", ".py"
        } else ""
        file_status = "present"
    except Exception as exc:
        size_bytes = None
        digest = ""
        file_status = "stat_or_hash_error"
        add_finding(
            "WARNING",
            "FILE_ACCESS",
            path,
            f"Could not inspect file completely: {exc}",
        )

    inventory.append(
        {
            "relative_path": relative_path(path),
            "file_type": suffix or "(no extension)",
            "size_bytes": size_bytes,
            "status": file_status,
            "sha256_if_text_or_data": digest,
        }
    )

print("Files inventoried:", len(inventory))


# ------------------------------------------------------------
# 3. Parse all JSON reports and check basic numeric validity
# ------------------------------------------------------------

print("\nChecking JSON files...")

json_paths = sorted(OUTPUT_ROOT.rglob("*.json"))

for path in json_paths:
    if AUDIT_DIR in path.parents:
        continue

    read_json(path)

print("JSON files parsed:", len(json_reports))


def inspect_numeric_values(value, path_label):
    if isinstance(value, dict):
        for key, item in value.items():
            inspect_numeric_values(item, f"{path_label}.{key}")

    elif isinstance(value, list):
        for index, item in enumerate(value):
            inspect_numeric_values(item, f"{path_label}[{index}]")

    elif isinstance(value, float):
        if not math.isfinite(value):
            add_finding(
                "WARNING",
                "NONFINITE_JSON_VALUE",
                path_label,
                "A numeric JSON value is NaN or infinite.",
            )


for path_label, report in json_reports.items():
    inspect_numeric_values(report, path_label)


# ------------------------------------------------------------
# 4. Read all CSVs and perform structural checks
# ------------------------------------------------------------

print("\nChecking CSV files...")

csv_frames: dict[str, pd.DataFrame] = {}

csv_paths = sorted(OUTPUT_ROOT.rglob("*.csv"))

for path in csv_paths:
    if AUDIT_DIR in path.parents:
        continue

    frame = read_csv(path)

    if frame is None:
        continue

    csv_frames[str(path)] = frame

    # Check infinite values in numeric columns.
    numeric = frame.select_dtypes(include="number")

    if not numeric.empty:
        try:
            infinite_count = int(
                numeric.map(lambda x: math.isinf(x)).sum().sum()
            )
        except Exception:
            infinite_count = 0

        if infinite_count:
            add_finding(
                "WARNING",
                "INFINITE_CSV_VALUE",
                path,
                f"Found {infinite_count} infinite numeric cells.",
            )

print("CSV files successfully read:", len(csv_frames))


# ------------------------------------------------------------
# 5. Record key metric tables and their evaluation units
# ------------------------------------------------------------

metric_targets = [
    (
        "step04/validation_predictions_FD001.csv",
        "All validation cycle rows; predictions and actual RUL.",
    ),
    (
        "step04/test_predictions_FD001.csv",
        "Final-cycle test engine predictions, as reported in Step 4.",
    ),
    (
        "step08_5/validation_calibration_metrics_FD001.csv",
        "Calibration evaluation across validation rows.",
    ),
    (
        "step08_6/step08_6_validation_metrics_FD001.csv",
        "Step 8.6 policy evaluation; inspect the report for exact unit.",
    ),
    (
        "step08_7/step08_7_final_cycle_regression_metrics_FD001.csv",
        "Final-cycle validation engine regression metrics.",
    ),
    (
        "step08_7/step08_7_engine_bootstrap_FD001.csv",
        "Engine-level bootstrap comparisons; inspect difference direction.",
    ),
]

for relative, description in metric_targets:
    path = OUTPUT_ROOT / Path(relative)
    frame = csv_frames.get(str(path))

    if frame is None:
        metric_checks.append(
            {
                "artifact": relative,
                "evaluation_unit": description,
                "status": "unavailable",
                "rows": None,
                "columns": "",
            }
        )
        continue

    metric_checks.append(
        {
            "artifact": relative,
            "evaluation_unit": description,
            "status": "readable",
            "rows": len(frame),
            "columns": " | ".join(map(str, frame.columns)),
        }
    )

    if len(frame) == 0:
        add_finding(
            "WARNING",
            "METRIC_TABLE_EMPTY",
            path,
            "Metric table is empty.",
        )


# ------------------------------------------------------------
# 6. Check prediction-table structure without assuming that
#    every file uses the same column names.
# ------------------------------------------------------------

prediction_paths = [
    OUTPUT_ROOT / "step04" / "validation_predictions_FD001.csv",
    OUTPUT_ROOT / "step04" / "test_predictions_FD001.csv",
]

for path in prediction_paths:
    frame = csv_frames.get(str(path))

    if frame is None:
        continue

    normalized = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    has_actual = any(
        name in normalized
        for name in [
            "actual_rul",
            "rul",
            "true_rul",
            "actual",
        ]
    )

    has_engine_id = any(
        name in normalized
        for name in [
            "unit_id",
            "engine_id",
            "engine",
            "unit",
        ]
    )

    if not has_actual:
        add_finding(
            "WARNING",
            "PREDICTION_SCHEMA",
            path,
            "Could not identify an obvious actual-RUL column. "
            "Review the schema manually.",
        )

    if not has_engine_id:
        add_finding(
            "WARNING",
            "PREDICTION_SCHEMA",
            path,
            "Could not identify an obvious engine identifier column.",
        )

    print(
        "[PREDICTION TABLE]",
        path.name,
        "rows =", len(frame),
        "columns =", list(frame.columns),
    )


# ------------------------------------------------------------
# 7. Record known methodological cautions
# ------------------------------------------------------------

add_finding(
    "INFO",
    "EVALUATION_UNITS",
    "Step 4 / Step 8.5 / Step 8.7",
    "All-cycle validation metrics and final-cycle engine metrics "
    "are different evaluation units. Do not compare them as though "
    "they measured the same set of observations.",
)

add_finding(
    "INFO",
    "CALIBRATION",
    "Step 8.5",
    "Calibration is post-processing fitted using training-partition "
    "out-of-fold predictions. It is not model retraining. Review "
    "the saved report and metrics before making improvement claims.",
)

add_finding(
    "INFO",
    "POLICY_COSTS",
    "Steps 6–8",
    "Maintenance costs are illustrative assumptions, not validated "
    "industrial cost estimates.",
)

add_finding(
    "WARNING",
    "SMALL_VALIDATION_SAMPLE",
    "Step 8 / Step 8.7",
    "The final-cycle validation policy evidence involves only "
    "20 engines. Zero observed misses does not establish zero "
    "real-world failure risk.",
)

add_finding(
    "INFO",
    "DISTRIBUTION_SHIFT",
    "Step 8.2",
    "Training-to-test sensor distribution differences do not alone "
    "prove temporal concept drift, causal degradation, or policy failure.",
)

add_finding(
    "INFO",
    "SELF_EVOLUTION_CLAIM",
    "Steps 8.1–8.8",
    "Policy selection, calibration, drift detection, and auditing "
    "do not by themselves demonstrate continuous online learning. "
    "A controlled feedback-loop experiment is still needed to support "
    "a strong self-evolution claim.",
)

add_finding(
    "INFO",
    "TEST_SET_BOUNDARY",
    "All steps",
    "Test-set metrics are for final reporting only. Do not use them "
    "to tune models, calibration, thresholds, or claims.",
)


# ------------------------------------------------------------
# 8. Summarize report metadata where available
# ------------------------------------------------------------

report_summaries = []

for path_label, report in json_reports.items():
    report_summaries.append(
        {
            "report_path": path_label,
            "project": report.get("project"),
            "dataset": report.get("dataset"),
            "subset": report.get("subset"),
            "step": report.get("step"),
            "method": report.get("method"),
            "run_timestamp_utc": report.get("run_timestamp_utc"),
            "top_level_keys": ", ".join(map(str, report.keys())),
        }
    )


# ------------------------------------------------------------
# 9. Save audit outputs
# ------------------------------------------------------------

inventory_path = AUDIT_DIR / "step09_artifact_inventory_FD001.csv"
findings_path = AUDIT_DIR / "step09_audit_findings_FD001.csv"
metrics_path = AUDIT_DIR / "step09_metric_table_index_FD001.csv"
reports_path = AUDIT_DIR / "step09_report_metadata_FD001.csv"
json_path = AUDIT_DIR / "step09_FD001_integrity_report.json"
markdown_path = AUDIT_DIR / "step09_FD001_integrity_report.md"

pd.DataFrame(inventory).to_csv(
    inventory_path, index=False, encoding="utf-8-sig"
)

pd.DataFrame(findings).to_csv(
    findings_path, index=False, encoding="utf-8-sig"
)

pd.DataFrame(metric_checks).to_csv(
    metrics_path, index=False, encoding="utf-8-sig"
)

pd.DataFrame(report_summaries).to_csv(
    reports_path, index=False, encoding="utf-8-sig"
)


severity_counts = {
    severity: sum(
        1 for item in findings
        if item["severity"] == severity
    )
    for severity in ["ERROR", "WARNING", "INFO"]
}

report = {
    "project": (
        "Self-Evolving Explainable Rational Agent for Adaptive "
        "Industrial Decision Intelligence Using Statistical Trend Learning"
    ),
    "dataset": "NASA C-MAPSS",
    "subset": "FD001",
    "step": "9",
    "method": "Read-only artifact and experimental-integrity audit",
    "run_timestamp_utc": TIMESTAMP,
    "project_root": str(PROJECT_ROOT),
    "audit_output_directory": str(AUDIT_DIR),
    "scope": {
        "retrained_models": False,
        "modified_policy": False,
        "selected_new_thresholds": False,
        "used_test_labels_for_tuning": False,
        "independently_reproduced_every_model_fit": False,
        "verified_stored_artifacts_and_report_structures": True,
    },
    "counts": {
        "required_artifacts": len(required_files),
        "missing_required_artifacts": missing_count,
        "files_inventoried": len(inventory),
        "json_reports_parsed": len(json_reports),
        "csv_files_read": len(csv_frames),
        "audit_findings": len(findings),
        "severity_counts": severity_counts,
    },
    "evaluation_unit_cautions": [
        "All-cycle and final-cycle metrics are not directly interchangeable.",
        "Validation and test results must be reported separately.",
        "Small engine-level validation samples give uncertain safety estimates.",
        "Illustrative maintenance costs are not industrial cost evidence.",
        "Distribution shift is not, by itself, proof of temporal concept drift.",
        "A completed audit does not establish model correctness or deployment safety.",
    ],
    "artifacts": {
        "inventory_csv": str(inventory_path),
        "findings_csv": str(findings_path),
        "metric_table_index_csv": str(metrics_path),
        "report_metadata_csv": str(reports_path),
        "markdown_report": str(markdown_path),
    },
    "findings": findings,
}

with json_path.open("w", encoding="utf-8") as handle:
    json.dump(
        json_safe(report),
        handle,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


# ------------------------------------------------------------
# 10. Human-readable report
# ------------------------------------------------------------

lines = [
    "# Step 9 — Final Experimental Integrity Audit",
    "",
    "**Dataset:** NASA C-MAPSS FD001",
    f"**Generated (UTC):** {TIMESTAMP}",
    "",
    "## Scope",
    "",
    "This is a read-only audit of stored artifacts and report structures. "
    "It does not reproduce every model fit or independently validate "
    "every metric.",
    "",
    "## Summary",
    "",
    f"- Required artifacts: {len(required_files)}",
    f"- Missing required artifacts: {missing_count}",
    f"- Files inventoried: {len(inventory)}",
    f"- JSON reports parsed: {len(json_reports)}",
    f"- CSV files read: {len(csv_frames)}",
    f"- ERROR findings: {severity_counts['ERROR']}",
    f"- WARNING findings: {severity_counts['WARNING']}",
    f"- INFO findings: {severity_counts['INFO']}",
    "",
    "## Evaluation boundaries",
    "",
    "- Keep all-cycle validation metrics separate from final-cycle engine metrics.",
    "- Treat calibration as post-processing, not model retraining.",
    "- Treat simulated maintenance costs as illustrative assumptions.",
    "- A validation sample of 20 engines does not establish real-world safety.",
    "- Train/test sensor shift alone does not prove temporal concept drift.",
    "- Do not tune using test labels or test-set results.",
    "- Existing steps do not yet demonstrate continuous online self-evolution.",
    "",
    "## Findings",
    "",
]

for item in findings:
    lines.append(
        f"- **{item['severity']} — {item['category']}** "
        f"(`{item['artifact']}`): {item['message']}"
    )

lines.extend(
    [
        "",
        "## Metric table index",
        "",
        "| Artifact | Status | Rows | Evaluation unit |",
        "|---|---|---:|---|",
    ]
)

for item in metric_checks:
    lines.append(
        f"| `{item['artifact']}` | {item['status']} | "
        f"{item['rows'] if item['rows'] is not None else 'N/A'} | "
        f"{item['evaluation_unit']} |"
    )

lines.extend(
    [
        "",
        "## Interpretation",
        "",
        "A successful script run confirms that this audit completed. "
        "It does not prove that every model, metric, policy, or scientific "
        "claim is correct. Review warnings and compare key metrics against "
        "their original reports before writing final conclusions.",
        "",
    ]
)

markdown_path.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\n" + "=" * 72)
print("STEP 9 AUDIT COMPLETED")
print("=" * 72)
print("Required artifacts missing:", missing_count)
print("Files inventoried:", len(inventory))
print("JSON reports parsed:", len(json_reports))
print("CSV files read:", len(csv_frames))
print("Findings:", severity_counts)
print()
print("Output directory:", AUDIT_DIR)
print("Inventory:", inventory_path)
print("Findings:", findings_path)
print("Metric index:", metrics_path)
print("JSON report:", json_path)
print("Markdown report:", markdown_path)
print()
print(
    "IMPORTANT: Read the findings. A completed audit is not proof "
    "that all model results are correct."
)
