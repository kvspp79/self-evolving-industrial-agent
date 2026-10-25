
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# STEP 8.8
# Evidence Synthesis and Research-Integrity Audit
# NASA C-MAPSS FD001
#
# This script:
# 1. Collects existing step reports and key CSV outputs.
# 2. Corrects the interpretation labels in the Step 8.7
#    paired RMSE bootstrap table.
# 3. Produces a consolidated JSON report and Markdown summary.
#
# It does NOT:
# - train or retrain models;
# - read or tune against test labels;
# - change the frozen maintenance policy;
# - deploy any model or policy.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
AUDIT_DIR = OUTPUT_ROOT / "step08_8"

BOOTSTRAP_INPUT = (
    OUTPUT_ROOT
    / "step08_7"
    / "step08_7_engine_bootstrap_FD001.csv"
)

BOOTSTRAP_CORRECTED = (
    AUDIT_DIR / "step08_8_corrected_bootstrap_interpretations_FD001.csv"
)

REPORT_JSON = AUDIT_DIR / "step08_8_FD001_evidence_audit.json"
REPORT_MD = AUDIT_DIR / "step08_8_FD001_evidence_audit.md"
INVENTORY_CSV = AUDIT_DIR / "step08_8_artifact_inventory_FD001.csv"

# Existing steps to inspect. Missing optional directories/files
# are recorded as missing rather than treated as fatal errors.
STEP_DIRS = [
    "step02",
    "step03",
    "step04",
    "step05",
    "step06",
    "step07",
    "step07_1",
    "step07_2",
    "step08",
    "step08_1",
    "step08_2",
    "step08_3",
    "step08_4",
    "step08_5",
    "step08_6",
    "step08_7",
]

# Selected CSVs for a compact inventory. Other CSVs are inventoried
# automatically from the step directories.
KEY_CSV_NAMES = [
    "model_comparison_FD001.csv",
    "step08_rul_regression_metrics.csv",
    "step08_validation_policy_selection.csv",
    "step08_frozen_test_policy_results.csv",
    "validation_calibration_metrics_FD001.csv",
    "step08_6_validation_metrics_FD001.csv",
    "step08_7_final_cycle_regression_metrics_FD001.csv",
    "step08_7_selected_validation_candidates_FD001.csv",
    "step08_7_frozen_threshold_reference_FD001.csv",
]


def json_safe(value: Any) -> Any:
    """Convert common pandas/numpy values into JSON-safe Python values."""
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, float):
        if pd.isna(value):
            return None
        return value

    if pd.isna(value):
        return None

    if isinstance(value, (str, int, bool)):
        return value

    return str(value)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column using case-insensitive exact matching."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found

    return None


def corrected_bootstrap_interpretation(
    comparison: Any,
    observed_difference: Any,
    ci_lower: Any,
    ci_upper: Any,
) -> str:
    """
    Assumes the comparison is A_minus_B and the reported difference is:

        RMSE(A) - RMSE(B)

    Negative difference favors A; positive difference favors B.
    A confidence interval containing zero is inconclusive.
    """
    label = str(comparison).strip()
    match = re.match(r"^(.+?)_minus_(.+)$", label)

    if not match:
        return "UNRESOLVED: comparison name is not A_minus_B"

    model_a = match.group(1)
    model_b = match.group(2)

    try:
        lower = float(ci_lower)
        upper = float(ci_upper)
        observed = float(observed_difference)
    except (TypeError, ValueError):
        return "UNRESOLVED: missing or nonnumeric difference/interval"

    if pd.isna(lower) or pd.isna(upper) or pd.isna(observed):
        return "UNRESOLVED: missing or nonnumeric difference/interval"

    if lower > upper:
        return "UNRESOLVED: confidence interval bounds are reversed"

    if upper < 0:
        return f"95% bootstrap CI favors {model_a} (A has lower RMSE)"

    if lower > 0:
        return f"95% bootstrap CI favors {model_b} (B has lower RMSE)"

    return "95% bootstrap CI includes zero; difference is inconclusive"


def load_json_reports() -> dict[str, Any]:
    """Load JSON reports from known output step directories."""
    reports: dict[str, Any] = {}
    for step_name in STEP_DIRS:
        folder = OUTPUT_ROOT / step_name
        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8-sig") as file:
                    reports[str(path.relative_to(OUTPUT_ROOT))] = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                reports[str(path.relative_to(OUTPUT_ROOT))] = {
                    "read_error": str(exc)
                }

    return reports


def inventory_artifacts() -> pd.DataFrame:
    rows = []

    for step_name in STEP_DIRS:
        folder = OUTPUT_ROOT / step_name

        if not folder.exists():
            rows.append({
                "step": step_name,
                "relative_path": str(folder.relative_to(OUTPUT_ROOT)),
                "file_type": None,
                "size_bytes": None,
                "status": "directory_missing",
            })
            continue

        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue

            rows.append({
                "step": step_name,
                "relative_path": str(path.relative_to(OUTPUT_ROOT)),
                "file_type": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "status": "present",
            })

    return pd.DataFrame(
        rows,
        columns=[
            "step",
            "relative_path",
            "file_type",
            "size_bytes",
            "status",
        ],
    )


def correct_bootstrap_file() -> dict[str, Any]:
    if not BOOTSTRAP_INPUT.exists():
        return {
            "status": "not_corrected",
            "reason": "Step 8.7 bootstrap CSV was not found.",
            "expected_input": str(BOOTSTRAP_INPUT),
        }

    df = pd.read_csv(BOOTSTRAP_INPUT)

    comparison_col = find_column(
        df, ["comparison", "contrast", "model_comparison"]
    )
    observed_col = find_column(
        df, ["observed_RMSE_difference", "observed_rmse_difference"]
    )
    lower_col = find_column(
        df, ["bootstrap_CI_lower_95", "ci_lower_95", "lower_95"]
    )
    upper_col = find_column(
        df, ["bootstrap_CI_upper_95", "ci_upper_95", "upper_95"]
    )

    required = {
        "comparison": comparison_col,
        "observed RMSE difference": observed_col,
        "lower CI": lower_col,
        "upper CI": upper_col,
    }

    missing = [name for name, col in required.items() if col is None]

    if missing:
        return {
            "status": "not_corrected",
            "reason": "Required columns are missing.",
            "missing_columns": missing,
            "actual_columns": list(df.columns),
            "expected_input": str(BOOTSTRAP_INPUT),
        }

    original_interpretation_col = find_column(
        df, ["interpretation", "conclusion"]
    )

    if original_interpretation_col is None:
        original_interpretation_col = "original_interpretation"
        df[original_interpretation_col] = ""

    df["interpretation_before_audit"] = df[original_interpretation_col]

    df["interpretation_corrected"] = df.apply(
        lambda row: corrected_bootstrap_interpretation(
            row[comparison_col],
            row[observed_col],
            row[lower_col],
            row[upper_col],
        ),
        axis=1,
    )

    # Keep the original interpretation untouched in its original column.
    # The corrected result is stored in a separate, explicit column.
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(BOOTSTRAP_CORRECTED, index=False)

    return {
        "status": "corrected_copy_created",
        "input": str(BOOTSTRAP_INPUT),
        "output": str(BOOTSTRAP_CORRECTED),
        "rows": int(len(df)),
        "comparison_column": comparison_col,
        "observed_difference_column": observed_col,
        "ci_lower_column": lower_col,
        "ci_upper_column": upper_col,
        "original_interpretation_column": original_interpretation_col,
        "corrected_interpretation_column": "interpretation_corrected",
        "assumption": (
            "Each comparison is A_minus_B and the observed difference "
            "is RMSE(A) - RMSE(B). A negative interval favors A; a positive "
            "interval favors B; an interval containing zero is inconclusive."
        ),
    }


def extract_selected_findings(reports: dict[str, Any]) -> list[str]:
    findings = []

    # Step 8 report: policy-selection/test comparison.
    step8 = next(
        (
            report for name, report in reports.items()
            if name.startswith("step08/")
            and isinstance(report, dict)
            and "adaptive_policy" in report
        ),
        None,
    )

    if step8:
        policy = step8.get("adaptive_policy", {})
        findings.append(
            "Step 8 selected model="
            f"{policy.get('selected_model', 'not reported')}, "
            "threshold="
            f"{policy.get('selected_threshold_cycles', 'not reported')} "
            "cycles, using "
            f"{policy.get('selection_data', 'unspecified')}."
        )

        test_comparison = step8.get("standard_scenario_test_comparison", {})
        if test_comparison:
            findings.append(
                "Step 8 test policy results are descriptive frozen-policy "
                "results only; test results must not be used to retune the "
                "policy."
            )

    # Step 8.5 calibration parameters.
    calibration_json = (
        OUTPUT_ROOT / "step08_5" / "calibration_parameters_FD001.json"
    )
    if calibration_json.exists():
        try:
            with calibration_json.open("r", encoding="utf-8-sig") as file:
                calibration = json.load(file)

            findings.append(
                "Step 8.5 fitted linear calibration using training-partition "
                "out-of-fold predictions only. Validation labels were "
                "evaluation-only; no test labels were used."
            )

            for key in ["baseline_calibrator", "temporal_calibrator"]:
                if key in calibration:
                    findings.append(
                        f"{key}: intercept="
                        f"{calibration[key].get('intercept', 'not reported')}, "
                        f"slope={calibration[key].get('slope', 'not reported')}."
                    )
        except (OSError, json.JSONDecodeError):
            findings.append(
                "Step 8.5 calibration JSON exists but could not be parsed."
            )

    # Step 8.7 regression results.
    metrics_path = (
        OUTPUT_ROOT
        / "step08_7"
        / "step08_7_final_cycle_regression_metrics_FD001.csv"
    )

    if metrics_path.exists():
        try:
            metrics = pd.read_csv(metrics_path)
            findings.append(
                "Step 8.7 final-cycle regression metrics were read from "
                f"{metrics_path.name}; they refer to 20 validation engines, "
                "not all 4,070 validation cycle rows."
            )
        except Exception as exc:
            findings.append(
                f"Step 8.7 final-cycle metrics could not be read: {exc}"
            )

    return findings


def write_markdown_report(
    reports: dict[str, Any],
    bootstrap_result: dict[str, Any],
    inventory: pd.DataFrame,
    findings: list[str],
) -> None:
    lines = [
        "# Step 8.8 — Evidence Synthesis and Research-Integrity Audit",
        "",
        "**Dataset:** NASA C-MAPSS FD001",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Purpose",
        "",
        "This audit consolidates existing outputs and checks interpretation "
        "and evaluation boundaries. It does not retrain models, modify a "
        "maintenance policy, or use test labels for selection.",
        "",
        "## Main findings",
        "",
    ]

    if findings:
        lines.extend(f"- {item}" for item in findings)
    else:
        lines.append(
            "- No selected findings could be extracted automatically. "
            "Inspect the source reports listed in the artifact inventory."
        )

    lines.extend([
        "",
        "## Bootstrap interpretation audit",
        "",
        f"- Status: **{bootstrap_result.get('status', 'unknown')}**",
        f"- Input: `{bootstrap_result.get('input', BOOTSTRAP_INPUT)}`",
        f"- Corrected copy: `{bootstrap_result.get('output', 'not created')}`",
        "",
        "For a comparison A_minus_B defined as RMSE(A) - RMSE(B):",
        "- A 95% interval entirely below zero favors A.",
        "- A 95% interval entirely above zero favors B.",
        "- An interval containing zero is inconclusive at this interval level.",
        "- These labels describe the interval; they do not prove practical "
        "importance or broad generalization.",
        "",
        "## Interpretation safeguards",
        "",
        "1. All-cycle metrics and final-cycle engine metrics are different "
        "evaluation units and must not be compared as if they were the same.",
        "2. Calibration is post-processing, not model retraining.",
        "3. Maintenance costs and cost-optimal thresholds depend on illustrative "
        "assumptions; they are not validated industrial cost estimates.",
        "4. A validation set of 20 engines gives uncertain safety estimates. "
        "Zero observed misses does not establish a zero real-world failure rate.",
        "5. Distribution shift between train and test sensor rows does not, "
        "by itself, establish temporal concept drift or causal degradation.",
        "6. The current experiments do not establish continuous online "
        "self-evolution or safe autonomous deployment.",
        "7. Test-set results are for final reporting only and must not be used "
        "to tune models, calibration, thresholds, or claims.",
        "",
        "## Artifact inventory",
        "",
        f"- Files inventoried: {len(inventory)}",
        "- See `step08_8_artifact_inventory_FD001.csv` for paths and presence.",
        "",
        "## Limitations",
        "",
        "- Only C-MAPSS FD001 is covered by this audit.",
        "- Earlier step reports are summarized as recorded; this script does "
        "not independently reproduce every model fit or metric.",
        "- A successful audit script run does not itself validate model "
        "correctness, safety, or deployment readiness.",
        "",
    ])

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 72)
    print("NASA C-MAPSS FD001 — STEP 8.8")
    print("EVIDENCE SYNTHESIS AND RESEARCH-INTEGRITY AUDIT")
    print("=" * 72)

    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Outputs directory not found: {OUTPUT_ROOT}"
        )

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading existing JSON reports...")
    reports = load_json_reports()
    print(f"   JSON reports read: {len(reports)}")

    print("\n2. Auditing Step 8.7 bootstrap interpretations...")
    bootstrap_result = correct_bootstrap_file()
    print(f"   Status: {bootstrap_result.get('status')}")

    if bootstrap_result.get("status") != "corrected_copy_created":
        print("   Note:", bootstrap_result.get("reason", "No details"))

    print("\n3. Building artifact inventory...")
    inventory = inventory_artifacts()
    inventory.to_csv(INVENTORY_CSV, index=False)
    print(f"   Inventory rows: {len(inventory)}")

    print("\n4. Extracting evidence and limitations...")
    findings = extract_selected_findings(reports)

    report = {
        "project": (
            "Self-Evolving Explainable Rational Agent for Adaptive "
            "Industrial Decision Intelligence Using Statistical Trend Learning"
        ),
        "dataset": "NASA C-MAPSS",
        "subset": "FD001",
        "step": "8.8",
        "method": "Evidence synthesis and research-integrity audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "actions_performed": [
            "Read existing JSON reports from known step output directories",
            "Inventoried existing step artifacts",
            "Created a separate corrected copy of bootstrap interpretation labels",
            "Generated a Markdown evidence audit",
        ],
        "actions_not_performed": [
            "No model retraining",
            "No test-label-driven tuning",
            "No maintenance-policy modification",
            "No deployment",
        ],
        "bootstrap_interpretation_audit": bootstrap_result,
        "extracted_findings": findings,
        "report_files": {
            "markdown_report": str(REPORT_MD),
            "json_report": str(REPORT_JSON),
            "artifact_inventory": str(INVENTORY_CSV),
            "corrected_bootstrap_copy": (
                str(BOOTSTRAP_CORRECTED)
                if bootstrap_result.get("status") == "corrected_copy_created"
                else None
            ),
        },
        "loaded_json_report_paths": list(reports.keys()),
        "limitations": [
            "Only FD001 is audited.",
            "This is a synthesis of existing outputs, not a full independent rerun.",
            "Only 20 engines are available in the reported validation partition.",
            "Illustrative maintenance costs are not measured industrial costs.",
            "Calibration results may not generalize to other subsets or datasets.",
            "The workflow does not prove continuous online self-evolution.",
            "Test results must not be used to retune the system.",
        ],
    }

    with REPORT_JSON.open("w", encoding="utf-8") as file:
        json.dump(json_safe(report), file, indent=2, ensure_ascii=False)

    write_markdown_report(
        reports=reports,
        bootstrap_result=bootstrap_result,
        inventory=inventory,
        findings=findings,
    )

    print("\nSTEP 8.8 COMPLETED")
    print(f"Output directory: {AUDIT_DIR}")
    print(f"Markdown report: {REPORT_MD}")
    print(f"JSON report: {REPORT_JSON}")
    print(f"Artifact inventory: {INVENTORY_CSV}")

    if bootstrap_result.get("status") == "corrected_copy_created":
        print(f"Corrected bootstrap copy: {BOOTSTRAP_CORRECTED}")

    print("\nNo model was retrained.")
    print("No test labels were used for tuning.")
    print("No maintenance policy was modified or deployed.")


if __name__ == "__main__":
    main()
