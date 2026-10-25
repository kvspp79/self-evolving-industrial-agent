
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ============================================================
# STEP 10J — Final Research Evidence Synthesis and Audit
# Audits existing outputs only; does not retrain or modify data.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS"
)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUTS_DIR / "step10J_final_evidence_synthesis"
TABLES_DIR = REPORT_DIR / "evidence_tables"
STAGE_PREFIXES = ("step10F", "step10G", "step10H", "step10I")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_stage_directories() -> list[Path]:
    if not OUTPUTS_DIR.exists():
        return []

    return sorted(
        [
            p for p in OUTPUTS_DIR.iterdir()
            if p.is_dir()
            and any(
                p.name.lower().startswith(prefix.lower())
                for prefix in STAGE_PREFIXES
            )
        ],
        key=lambda p: p.name.lower(),
    )


def safe_json_read(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def missing_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """Detect NA values and empty or whitespace-only strings."""
    mask = frame.isna().copy()

    for col in frame.columns:
        blank_strings = (
            frame[col]
            .astype("string")
            .str.strip()
            .eq("")
            .fillna(False)
        )
        mask[col] = mask[col] | blank_strings

    return mask


def contextual_missing_cells(
    path: Path, frame: pd.DataFrame
) -> tuple[int, int]:
    """
    Return:
      genuine_missing_cells,
      applicable_cells

    Only documented non-applicable fields in the two Step 10I
    tables are excluded. Source files are never modified.
    """
    mask = missing_mask(frame)
    excluded = pd.DataFrame(
        False, index=frame.index, columns=frame.columns
    )
    name = path.name.lower()

    # --------------------------------------------------------
    # Step 10I file inventory:
    # "rows" is a CSV row count. It is not applicable to JSON
    # and other non-CSV files.
    # --------------------------------------------------------
    if (
        name == "step10i_file_inventory.csv"
        and {"extension", "rows"}.issubset(frame.columns)
    ):
        ext = (
            frame["extension"]
            .astype("string")
            .str.lower()
            .str.strip()
        )
        not_csv = ext.ne(".csv").fillna(False)

        excluded.loc[not_csv, "rows"] = True
        mask.loc[not_csv, "rows"] = False

    # --------------------------------------------------------
    # Step 10I gate decision analysis:
    # different experiment families use different gate columns.
    # --------------------------------------------------------
    if (
        name == "step10i_gate_decision_analysis.csv"
        and "step" in frame.columns
    ):
        step = (
            frame["step"]
            .astype("string")
            .str.upper()
            .str.strip()
        )

        # Step 10H uses the two explicit gate columns instead
        # of candidate_accepted.
        if "candidate_accepted" in frame.columns:
            not_10h = step.eq("10H").fillna(False)
            excluded.loc[not_10h, "candidate_accepted"] = True
            mask.loc[not_10h, "candidate_accepted"] = False

        # Steps 10F/10G do not define the Step 10H gate columns.
        not_10fg = step.isin(["10F", "10G"]).fillna(False)

        for col in (
            "original_gate_accepted",
            "nasa_prioritized_gate_accepted",
        ):
            if col in frame.columns:
                excluded.loc[not_10fg, col] = True
                mask.loc[not_10fg, col] = False

    applicable_cells = int(
        frame.size - excluded.to_numpy().sum()
    )
    genuine_missing = int(mask.to_numpy().sum())

    return genuine_missing, applicable_cells


def csv_read_profile(path: Path) -> dict:
    profile = {
        "path": str(path),
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "status": "ERROR",
        "rows": None,
        "columns_count": None,
        "columns": "",
        "duplicate_rows": None,
        "missing_cells": None,
        "missing_percent": None,
        "duplicate_column_names": "",
        "read_error": "",
        "sha256": "",
        "size_bytes": None,
    }

    try:
        profile["size_bytes"] = path.stat().st_size
        profile["sha256"] = sha256_file(path)

        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            raw_header = next(csv.reader(handle), [])

        seen = set()
        duplicates = []

        for col in raw_header:
            if col in seen:
                duplicates.append(col)
            seen.add(col)

        profile["duplicate_column_names"] = "; ".join(
            sorted(set(duplicates))
        )

        frame = pd.read_csv(path, low_memory=False)

        profile["status"] = "OK"
        profile["rows"] = int(len(frame))
        profile["columns_count"] = int(len(frame.columns))
        profile["columns"] = " | ".join(
            map(str, frame.columns)
        )
        profile["duplicate_rows"] = int(
            frame.duplicated().sum()
        )

        missing_count, applicable_cells = (
            contextual_missing_cells(path, frame)
        )

        profile["missing_cells"] = missing_count
        profile["missing_percent"] = (
            round(100 * missing_count / applicable_cells, 4)
            if applicable_cells
            else 0.0
        )

    except Exception as exc:
        profile["read_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return profile


def classify_csv(path: Path, columns: list[str]) -> str:
    name = path.name.lower()
    colset = {
        str(c).strip().lower() for c in columns
    }

    if "gate" in name or any(
        "gate" in c for c in colset
    ):
        return "safety_gate"

    if any(
        term in name
        for term in ("paired", "wilcoxon", "bootstrap")
    ):
        return "statistical_comparison"

    if "failure" in name or "evidence" in name:
        return "evidence_or_failure_analysis"

    if "summary" in name or "across_seed" in name:
        return "aggregate_summary"

    if any(
        term in name
        for term in ("metric", "result", "comparison")
    ):
        return "metrics_or_comparison"

    if "engine" in name:
        return "engine_level_results"

    return "other_csv"


def json_top_level_summary(value) -> str:
    if isinstance(value, dict):
        return "; ".join(
            f"{k}={type(v).__name__}"
            for k, v in list(value.items())[:30]
        )

    if isinstance(value, list):
        return f"list_length={len(value)}"

    return f"top_level_type={type(value).__name__}"


# ============================================================
# Locate project and source artifacts.
# ============================================================

if not PROJECT_ROOT.exists():
    raise FileNotFoundError(
        f"Project directory does not exist: {PROJECT_ROOT}"
    )

stage_dirs = find_stage_directories()

if not stage_dirs:
    raise FileNotFoundError(
        f"No Step 10F–10I output directories found under "
        f"{OUTPUTS_DIR}"
    )

csv_files = []
json_files = []
other_files = []

for stage_dir in stage_dirs:
    for path in stage_dir.rglob("*"):
        if not path.is_file():
            continue

        try:
            path.relative_to(REPORT_DIR)
            continue
        except ValueError:
            pass

        if path.suffix.lower() == ".csv":
            csv_files.append(path)
        elif path.suffix.lower() == ".json":
            json_files.append(path)
        else:
            other_files.append(path)

csv_files = sorted(set(csv_files))
json_files = sorted(set(json_files))
other_files = sorted(set(other_files))

print("=" * 76)
print("STEP 10J — FINAL RESEARCH EVIDENCE SYNTHESIS")
print(f"Project root: {PROJECT_ROOT}")
print(f"Output directory: {REPORT_DIR}")
print(
    f"CSV: {len(csv_files)} | "
    f"JSON: {len(json_files)} | "
    f"Other: {len(other_files)}"
)


# ============================================================
# CSV audit.
# ============================================================

csv_profiles = []
csv_frames = {}
issues = []

for path in csv_files:
    profile = csv_read_profile(path)
    csv_profiles.append(profile)

    if profile["status"] != "OK":
        issues.append({
            "severity": "ERROR",
            "file": profile["relative_path"],
            "issue": profile["read_error"],
        })
        continue

    frame = pd.read_csv(path, low_memory=False)
    csv_frames[path] = frame

    if profile["duplicate_rows"]:
        issues.append({
            "severity": "INFO",
            "file": profile["relative_path"],
            "issue": (
                f"{profile['duplicate_rows']} fully duplicated "
                "rows; review context."
            ),
        })

    if profile["missing_cells"]:
        issues.append({
            "severity": "WARNING",
            "file": profile["relative_path"],
            "issue": (
                f"{profile['missing_cells']} genuinely missing "
                f"applicable cells "
                f"({profile['missing_percent']}%)."
            ),
        })

    if profile["duplicate_column_names"]:
        issues.append({
            "severity": "WARNING",
            "file": profile["relative_path"],
            "issue": (
                "Duplicate raw CSV column names: "
                f"{profile['duplicate_column_names']}"
            ),
        })

csv_inventory_path = REPORT_DIR / "step10J_csv_inventory.csv"
pd.DataFrame(csv_profiles).to_csv(
    csv_inventory_path, index=False
)


# ============================================================
# JSON audit.
# ============================================================

json_profiles = []

for path in json_files:
    value, error = safe_json_read(path)

    profile = {
        "relative_path": str(
            path.relative_to(PROJECT_ROOT)
        ),
        "status": "OK" if error is None else "ERROR",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "top_level_summary": (
            json_top_level_summary(value)
            if error is None
            else ""
        ),
        "read_error": error or "",
    }

    json_profiles.append(profile)

    if error:
        issues.append({
            "severity": "ERROR",
            "file": profile["relative_path"],
            "issue": error,
        })

json_inventory_path = REPORT_DIR / "step10J_json_inventory.csv"
pd.DataFrame(json_profiles).to_csv(
    json_inventory_path, index=False
)


# ============================================================
# Schema and role audit.
# ============================================================

schema_records = []

for path, frame in csv_frames.items():
    schema_records.append({
        "stage_directory": path.parent.name,
        "file_name": path.name,
        "relative_path": str(
            path.relative_to(PROJECT_ROOT)
        ),
        "role": classify_csv(path, list(frame.columns)),
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "columns": " | ".join(map(str, frame.columns)),
        "has_seed_column": any(
            str(c).strip().lower()
            in {"seed", "random_seed", "seed_id"}
            for c in frame.columns
        ),
        "has_model_column": any(
            "model" in str(c).strip().lower()
            or "strategy" in str(c).strip().lower()
            for c in frame.columns
        ),
        "has_metric_column": any(
            "metric" in str(c).strip().lower()
            for c in frame.columns
        ),
    })

schema_audit_path = REPORT_DIR / "step10J_schema_audit.csv"
pd.DataFrame(schema_records).to_csv(
    schema_audit_path, index=False
)


# ============================================================
# Export copies of useful existing evidence tables.
# ============================================================

exported_tables = []

export_roles = {
    "safety_gate",
    "statistical_comparison",
    "evidence_or_failure_analysis",
    "aggregate_summary",
    "metrics_or_comparison",
    "engine_level_results",
}

for path, frame in csv_frames.items():
    role = classify_csv(path, list(frame.columns))

    if role not in export_roles:
        continue

    destination = TABLES_DIR / (
        f"{path.parent.name}__{path.stem}.csv"
    ).replace(" ", "_")

    frame.to_csv(destination, index=False)

    exported_tables.append({
        "source": str(path.relative_to(PROJECT_ROOT)),
        "export": str(
            destination.relative_to(PROJECT_ROOT)
        ),
        "role": role,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
    })


# ============================================================
# Key evidence index.
# ============================================================

interesting_terms = (
    "mean_engine_rmse",
    "mean_engine_mae",
    "mean_engine_nasa_score",
    "seeds_improved",
    "seeds_worsened",
    "bootstrap_95ci",
    "wilcoxon",
    "p_value",
    "candidate_accepted",
    "acceptance_rate",
)

key_evidence = []

for path, frame in csv_frames.items():
    matched = [
        str(c)
        for c in frame.columns
        if any(
            term in str(c).strip().lower()
            for term in interesting_terms
        )
    ]

    if matched:
        key_evidence.append({
            "source": str(path.relative_to(PROJECT_ROOT)),
            "row_count": len(frame),
            "matched_columns": " | ".join(matched),
            "all_columns": " | ".join(
                map(str, frame.columns)
            ),
        })

key_evidence_path = (
    REPORT_DIR / "step10J_key_evidence_index.csv"
)
pd.DataFrame(key_evidence).to_csv(
    key_evidence_path, index=False
)


# ============================================================
# Issues output.
# ============================================================

issues_path = REPORT_DIR / "step10J_issues.csv"

issues_df = pd.DataFrame(
    issues, columns=["severity", "file", "issue"]
)

issues_df.to_csv(issues_path, index=False)


# ============================================================
# Summary JSON and Markdown report.
# ============================================================

csv_ok = sum(
    p["status"] == "OK" for p in csv_profiles
)
csv_error = len(csv_profiles) - csv_ok

json_ok = sum(
    p["status"] == "OK" for p in json_profiles
)
json_error = len(json_profiles) - json_ok

severity_counts = (
    issues_df["severity"].value_counts().to_dict()
    if not issues_df.empty
    else {}
)

report = {
    "step": "10J",
    "title": (
        "Final Research Evidence Synthesis and "
        "Reproducibility Audit"
    ),
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "project_root": str(PROJECT_ROOT),
    "stage_directories_found": [
        str(p) for p in stage_dirs
    ],
    "files": {
        "csv_discovered": len(csv_files),
        "csv_readable": csv_ok,
        "csv_errors": csv_error,
        "json_discovered": len(json_files),
        "json_readable": json_ok,
        "json_errors": json_error,
        "other_files": len(other_files),
        "evidence_tables_exported": len(exported_tables),
    },
    "issues_by_severity": severity_counts,
    "outputs": {
        "csv_inventory": str(csv_inventory_path),
        "json_inventory": str(json_inventory_path),
        "schema_audit": str(schema_audit_path),
        "key_evidence_index": str(key_evidence_path),
        "issues": str(issues_path),
        "evidence_tables_directory": str(TABLES_DIR),
    },
    "important_interpretation_limits": [
        "This audit checks files and schemas; it does not "
        "independently reproduce experiments.",
        "Readable files are not proof that experiments are "
        "methodologically valid.",
        "Repeated seeds are not independent industrial datasets.",
        "Synthetic sensor shifts do not establish real-world "
        "concept drift.",
        "Feature importance/SHAP does not establish causation.",
        "Simulated maintenance costs are not real operational costs.",
        "NASA PHM score is asymmetric and should be reported "
        "separately from RMSE and MAE.",
        "Do not use held-out results to revise acceptance criteria "
        "or select a model after evaluation.",
        "Statistical inference must respect the experimental unit "
        "and dependence structure.",
    ],
    "evidence_table_manifest": exported_tables,
}

json_report_path = REPORT_DIR / "step10J_report.json"

with json_report_path.open("w", encoding="utf-8") as handle:
    json.dump(
        report, handle, indent=2, ensure_ascii=False
    )

markdown_path = (
    REPORT_DIR / "step10J_research_synthesis.md"
)

lines = [
    "# Step 10J — Final Research Evidence Synthesis",
    "",
    f"Audit generated (UTC): {report['created_utc']}",
    "",
    "## Purpose",
    "",
    "This step inventories Step 10F–10I artifacts, checks "
    "CSV/JSON readability, records schemas and basic integrity "
    "indicators, and exports copies of evidence tables. It does "
    "not retrain models or modify source experiment outputs.",
    "",
    "## Audit coverage",
    "",
    f"- Stage directories: {len(stage_dirs)}",
    f"- CSV discovered/readable/errors: "
    f"{len(csv_files)}/{csv_ok}/{csv_error}",
    f"- JSON discovered/readable/errors: "
    f"{len(json_files)}/{json_ok}/{json_error}",
    f"- Other files: {len(other_files)}",
    f"- Evidence table copies: {len(exported_tables)}",
    "",
    "## Integrity observations",
    "",
    f"- Errors: {severity_counts.get('ERROR', 0)}",
    f"- Warnings: {severity_counts.get('WARNING', 0)}",
    f"- Informational observations: "
    f"{severity_counts.get('INFO', 0)}",
    "",
    "See `step10J_issues.csv` for file-specific details. "
    "Contextually non-applicable fields are excluded only for "
    "the two documented Step 10I schemas; original source CSVs "
    "are not changed.",
    "",
    "## Interpretation limits",
    "",
    *[
        f"- {item}"
        for item in report["important_interpretation_limits"]
    ],
    "",
    "This audit does not independently reproduce the underlying "
    "experiments. Verify statistical calculations from seed-level "
    "source data before publication.",
]

markdown_path.write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)


# ============================================================
# Final console summary.
# ============================================================

print("\nSTEP 10J COMPLETE")
print(f"CSV readable/errors: {csv_ok}/{csv_error}")
print(f"JSON readable/errors: {json_ok}/{json_error}")
print(f"Issues: {severity_counts}")
print(f"Issues CSV: {issues_path}")
print(f"Report JSON: {json_report_path}")
print(f"Synthesis Markdown: {markdown_path}")
