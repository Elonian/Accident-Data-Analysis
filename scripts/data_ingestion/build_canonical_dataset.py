"""Build a canonical CSV dataset from a configured source CSV."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory
from utils.time_utils import parse_crash_date

CANONICAL_COLUMNS = [
    "CRASH DATE",
    "CRASH TIME",
    "BOROUGH",
    "ZIP CODE",
    "LATITUDE",
    "LONGITUDE",
    "LOCATION",
    "ON STREET NAME",
    "CROSS STREET NAME",
    "OFF STREET NAME",
    "NUMBER OF PERSONS INJURED",
    "NUMBER OF PERSONS KILLED",
    "NUMBER OF PEDESTRIANS INJURED",
    "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST INJURED",
    "NUMBER OF CYCLIST KILLED",
    "NUMBER OF MOTORIST INJURED",
    "NUMBER OF MOTORIST KILLED",
    "CONTRIBUTING FACTOR VEHICLE 1",
    "CONTRIBUTING FACTOR VEHICLE 2",
    "CONTRIBUTING FACTOR VEHICLE 3",
    "CONTRIBUTING FACTOR VEHICLE 4",
    "CONTRIBUTING FACTOR VEHICLE 5",
    "COLLISION_ID",
    "VEHICLE TYPE CODE 1",
    "VEHICLE TYPE CODE 2",
    "VEHICLE TYPE CODE 3",
    "VEHICLE TYPE CODE 4",
    "VEHICLE TYPE CODE 5",
]

COLUMN_ALIASES = {
    "crash_date": "CRASH DATE",
    "crash_time": "CRASH TIME",
    "borough": "BOROUGH",
    "zip_code": "ZIP CODE",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "location": "LOCATION",
    "on_street_name": "ON STREET NAME",
    "cross_street_name": "CROSS STREET NAME",
    "off_street_name": "OFF STREET NAME",
    "number_of_persons_injured": "NUMBER OF PERSONS INJURED",
    "number_of_persons_killed": "NUMBER OF PERSONS KILLED",
    "number_of_pedestrians_injured": "NUMBER OF PEDESTRIANS INJURED",
    "number_of_pedestrians_killed": "NUMBER OF PEDESTRIANS KILLED",
    "number_of_cyclist_injured": "NUMBER OF CYCLIST INJURED",
    "number_of_cyclist_killed": "NUMBER OF CYCLIST KILLED",
    "number_of_motorist_injured": "NUMBER OF MOTORIST INJURED",
    "number_of_motorist_killed": "NUMBER OF MOTORIST KILLED",
    "contributing_factor_vehicle_1": "CONTRIBUTING FACTOR VEHICLE 1",
    "contributing_factor_vehicle_2": "CONTRIBUTING FACTOR VEHICLE 2",
    "contributing_factor_vehicle_3": "CONTRIBUTING FACTOR VEHICLE 3",
    "contributing_factor_vehicle_4": "CONTRIBUTING FACTOR VEHICLE 4",
    "contributing_factor_vehicle_5": "CONTRIBUTING FACTOR VEHICLE 5",
    "collision_id": "COLLISION_ID",
    "vehicle_type_code1": "VEHICLE TYPE CODE 1",
    "vehicle_type_code2": "VEHICLE TYPE CODE 2",
    "vehicle_type_code_3": "VEHICLE TYPE CODE 3",
    "vehicle_type_code_4": "VEHICLE TYPE CODE 4",
    "vehicle_type_code_5": "VEHICLE TYPE CODE 5",
}


def normalize_row(raw_row: Dict[str, str]) -> Dict[str, str]:
    """Normalize one raw row to canonical columns.

    Args:
        raw_row: Input row dictionary.

    Returns:
        Dict[str, str]: Canonicalized row.

    Raises:
        None
    """
    normalized = {column: "" for column in CANONICAL_COLUMNS}
    for key, value in raw_row.items():
        cleaned_key = (key or "").strip()
        if cleaned_key in normalized:
            normalized[cleaned_key] = value
            continue

        alias = COLUMN_ALIASES.get(cleaned_key.lower())
        if alias:
            normalized[alias] = value
    return normalized


def build_canonical_dataset(source_csv: Path, target_csv: Path) -> Dict[str, str]:
    """Build canonical CSV using streaming read/write.

    Args:
        source_csv: Input source CSV path.
        target_csv: Output canonical CSV path.

    Returns:
        Dict[str, str]: Build summary metrics.

    Raises:
        FileNotFoundError: If source CSV does not exist.
    """
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    ensure_directory(target_csv.parent)
    row_count = 0
    min_date = None
    max_date = None

    with source_csv.open("r", encoding="utf-8-sig", newline="") as src_handle, target_csv.open(
        "w", encoding="utf-8", newline=""
    ) as dst_handle:
        reader = csv.DictReader(src_handle)
        writer = csv.DictWriter(dst_handle, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()

        for raw_row in reader:
            normalized = normalize_row(raw_row)
            writer.writerow(normalized)
            row_count += 1

            parsed_date = parse_crash_date(normalized.get("CRASH DATE", ""))
            if parsed_date is not None:
                if min_date is None or parsed_date < min_date:
                    min_date = parsed_date
                if max_date is None or parsed_date > max_date:
                    max_date = parsed_date

    return {
        "row_count": str(row_count),
        "min_date": str(min_date) if min_date else "N/A",
        "max_date": str(max_date) if max_date else "N/A",
    }


def main() -> None:
    """Run canonical dataset build stage.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If stage fails.
    """
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)
    logger = get_logger("data_ingestion.canonical")

    source_csv = Path(config["source_csv"]).resolve()
    target_csv = Path(config["canonical_csv"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()

    log_step(logger, "INGEST_START", f"Dataset: {config.get('dataset_name', 'unknown')}")
    summary = build_canonical_dataset(source_csv, target_csv)

    write_markdown(
        reports_dir / "canonical_dataset_report.md",
        "Canonical Dataset Build Report",
        [
            f"Dataset: {config.get('dataset_name', 'unknown')}",
            f"Source file: {source_csv.name}",
            f"Rows written: {summary['row_count']}",
            f"Date range: {summary['min_date']} to {summary['max_date']}",
            f"Canonical columns: {len(CANONICAL_COLUMNS)}",
        ],
    )
    log_step(logger, "INGEST_DONE", f"Rows written: {summary['row_count']}")


if __name__ == "__main__":
    main()
