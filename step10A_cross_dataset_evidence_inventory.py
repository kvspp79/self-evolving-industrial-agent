
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# STEP 10A — CROSS-DATASET EVIDENCE INVENTORY
# Read-only inventory of existing AI4I and NASA outputs.
# No model training, tuning, or policy modification.
# ============================================================

NASA_ROOT = Path(
    r"C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent_CMAPSS"
)

AI4I_ROOT = Path(
    r"C:\Users\kvspp\Downloads\Self_Evolving_Industrial_Agent"
)

OUTPUT_DIR = NASA_ROOT / "outputs" / "step10A"

INTERESTING_TERMS = (
    "metric", "comparison", "prediction", "importance",
    "policy", "decision", "report", "audit", "calibration",
    "bootstrap", "drift", "performance", "evaluation",
    "trend", "explain", "shap", "cost", "threshold",
    "summary", "result"
)


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def flatten_json(
    value: Any,
    prefix: str = ""
) -> list[dict[str, str]]:
    """Extract scalar values from JSON without interpreting them."""
    rows: list[dict[str, str]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_json(child, child_prefix))

    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            rows.extend(flatten_json(child, child_prefix))

    else:
        if value is None or isinstance(value, (str, int, float, bool)):
            rows.append({
                "key_path": prefix,
                "value_type": type(value).__name__,
                "value": str(value)
            })

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text(
            "No records found.\n",
            encoding="utf-8"
        )
        return

    # Union of keys preserves records with different fields.
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    projects = [
        ("AI4I_2020", AI4I_ROOT),
        ("NASA_CMAPSS_FD001", NASA_ROOT),
    ]

    print("=" * 72)
    print("STEP 10A — CROSS-DATASET EVIDENCE INVENTORY")
    print("=" * 72)
    print("UTC:", datetime.now(timezone.utc).isoformat())

    for name, root in projects:
        if not root.is_dir():
            print(f"[ERROR] {name} project folder not found: {root}")
            print("Check the path at the top of this script.")
            return 2
        print(f"[OK] {name}: {root}")

    # Scan output folders only. Do not scan the input datasets or models.
    inventory: list[dict] = []
    json_values: list[dict] = []
    json_status: list[dict] = []
    csv_status: list[dict] = []
    candidate_files: list[dict] = []
    errors: list[dict] = []

    for project_name, root in projects:
        outputs = root / "outputs"

        if not outputs.is_dir():
            errors.append({
                "project": project_name,
                "path": str(outputs),
                "error": "Outputs directory not found"
            })
            continue

        for path in sorted(outputs.rglob("*")):
            if not path.is_file():
                continue

            # Avoid counting this script's own generated inventory
            # if the script is rerun.
            if path.resolve().is_relative_to(OUTPUT_DIR.resolve()):
                continue

            suffix = path.suffix.lower()
            relative = safe_relative(path, root)
            try:
                stat = path.stat()
                size = stat.st_size
                modified = datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat()
            except OSError as exc:
                errors.append({
                    "project": project_name,
                    "path": str(path),
                    "error": str(exc)
                })
                continue

            filename_lower = path.name.lower()
            interesting = any(
                term in filename_lower for term in INTERESTING_TERMS
            )

            inventory.append({
                "project": project_name,
                "relative_path": relative,
                "extension": suffix,
                "size_bytes": size,
                "modified_utc": modified,
                "filename_matches_research_terms": interesting,
            })

            if interesting:
                candidate_files.append({
                    "project": project_name,
                    "relative_path": relative,
                    "filename": path.name,
                    "extension": suffix,
                    "size_bytes": size,
                })

            if suffix == ".json":
                try:
                    with path.open("r", encoding="utf-8-sig") as file:
                        data = json.load(file)

                    json_status.append({
                        "project": project_name,
                        "relative_path": relative,
                        "status": "parsed",
                        "top_level_type": type(data).__name__,
                    })

                    for record in flatten_json(data):
                        json_values.append({
                            "project": project_name,
                            "relative_path": relative,
                            **record,
                        })

                except Exception as exc:
                    json_status.append({
                        "project": project_name,
                        "relative_path": relative,
                        "status": "ERROR",
                        "top_level_type": "",
                        "error": str(exc),
                    })

            elif suffix == ".csv":
                try:
                    # Read only the header and row count. This avoids
                    # loading large prediction tables into memory.
                    with path.open(
                        "r", newline="", encoding="utf-8-sig"
                    ) as file:
                        reader = csv.reader(file)
                        header = next(reader, [])
                        row_count = sum(1 for _ in reader)

                    csv_status.append({
                        "project": project_name,
                        "relative_path": relative,
                        "status": "readable",
                        "data_rows": row_count,
                        "column_count": len(header),
                        "columns": " | ".join(header),
                    })

                except Exception as exc:
                    csv_status.append({
                        "project": project_name,
                        "relative_path": relative,
                        "status": "ERROR",
                        "data_rows": "",
                        "column_count": "",
                        "columns": "",
                        "error": str(exc),
                    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(OUTPUT_DIR / "step10A_file_inventory.csv", inventory)
    write_csv(OUTPUT_DIR / "step10A_research_candidate_files.csv",
              candidate_files)
    write_csv(OUTPUT_DIR / "step10A_json_parse_status.csv", json_status)
    write_csv(OUTPUT_DIR / "step10A_json_scalar_index.csv", json_values)
    write_csv(OUTPUT_DIR / "step10A_csv_read_status.csv", csv_status)
    write_csv(OUTPUT_DIR / "step10A_scan_errors.csv", errors)

    parsed_json = sum(
        row.get("status") == "parsed" for row in json_status
    )
    failed_json = sum(
        row.get("status") == "ERROR" for row in json_status
    )
    readable_csv = sum(
        row.get("status") == "readable" for row in csv_status
    )
    failed_csv = sum(
        row.get("status") == "ERROR" for row in csv_status
    )

    summary = {
        "step": "10A",
        "title": "Cross-Dataset Evidence Inventory",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Inventory existing artifacts for paper preparation; "
            "no model fitting or policy selection."
        ),
        "projects": {
            name: str(root) for name, root in projects
        },
        "files_inventoried": len(inventory),
        "research_candidate_files": len(candidate_files),
        "json_parsed": parsed_json,
        "json_parse_errors": failed_json,
        "csv_readable": readable_csv,
        "csv_read_errors": failed_csv,
        "scan_errors": len(errors),
        "outputs": str(OUTPUT_DIR),
        "limitations": [
            "File readability does not validate scientific correctness.",
            "JSON scalar values are indexed, not automatically interpreted "
            "as comparable metrics.",
            "Metric comparability must be reviewed using each report's "
            "evaluation unit, dataset, and task.",
            "Test-set results must not be used for model or policy tuning.",
        ],
    }

    with (OUTPUT_DIR / "step10A_inventory_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("STEP 10A COMPLETED")
    print("=" * 72)
    print("Files inventoried:", len(inventory))
    print("Research candidate files:", len(candidate_files))
    print("JSON parsed:", parsed_json)
    print("JSON errors:", failed_json)
    print("CSV readable:", readable_csv)
    print("CSV errors:", failed_csv)
    print("Scan errors:", len(errors))
    print("Output directory:", OUTPUT_DIR)

    if failed_json or failed_csv or errors:
        print("\nReview step10A_scan_errors.csv and status CSVs.")
        return 1

    print("\nNext: review the candidate file list and metric definitions.")
    print("This inventory is not a scientific validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
