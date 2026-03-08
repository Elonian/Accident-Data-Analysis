"""Run streaming data quality checks for canonical dataset."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_csv_rows, write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory
from utils.plot_utils import save_bar_plot
from utils.time_utils import parse_crash_date


def run_quality_checks(canonical_csv: Path) -> Dict[str, object]:
    """Run quality checks and return summary details.

    Args:
        canonical_csv: Canonical dataset CSV path.

    Returns:
        Dict[str, object]: Quality summary and missingness records.

    Raises:
        FileNotFoundError: If canonical CSV is missing.
    """
    if not canonical_csv.exists():
        raise FileNotFoundError(f"Canonical CSV not found: {canonical_csv}")

    row_count = 0
    invalid_date_count = 0
    missing_counts: Counter[str] = Counter()
    seen_collision_ids: set[str] = set()
    duplicate_collision_rows = 0
    min_date = None
    max_date = None
    columns: List[str] = []

    with canonical_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])

        for row in reader:
            row_count += 1

            for column in columns:
                if not (row.get(column) or "").strip():
                    missing_counts[column] += 1

            collision_id = (row.get("COLLISION_ID") or "").strip()
            if collision_id:
                if collision_id in seen_collision_ids:
                    duplicate_collision_rows += 1
                else:
                    seen_collision_ids.add(collision_id)

            parsed_date = parse_crash_date(row.get("CRASH DATE", ""))
            if parsed_date is None:
                invalid_date_count += 1
            else:
                if min_date is None or parsed_date < min_date:
                    min_date = parsed_date
                if max_date is None or parsed_date > max_date:
                    max_date = parsed_date

    missing_records: List[Dict[str, str]] = []
    for column in columns:
        missing = missing_counts[column]
        percent = (missing * 100.0 / row_count) if row_count else 0.0
        missing_records.append(
            {
                "column": column,
                "missing_count": str(missing),
                "missing_percent": f"{percent:.2f}",
            }
        )

    missing_records.sort(key=lambda item: float(item["missing_percent"]), reverse=True)

    return {
        "row_count": row_count,
        "invalid_date_count": invalid_date_count,
        "duplicate_collision_rows": duplicate_collision_rows,
        "min_date": str(min_date) if min_date else "N/A",
        "max_date": str(max_date) if max_date else "N/A",
        "missing_records": missing_records,
    }


def write_quality_outputs(summary: Dict[str, object], config: Dict[str, str]) -> None:
    """Write quality tables, report, and charts.

    Args:
        summary: Quality summary dictionary.
        config: Runtime config dictionary.

    Returns:
        None

    Raises:
        None
    """
    tables_dir = Path(config["tables_dir"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()
    visual_dir = Path(config["visualizations_dir"]).resolve() / "core"

    ensure_directory(tables_dir)
    ensure_directory(reports_dir)
    ensure_directory(visual_dir)

    missing_records = summary["missing_records"]
    assert isinstance(missing_records, list)

    write_csv_rows(
        tables_dir / "missingness_summary.csv",
        ["column", "missing_count", "missing_percent"],
        missing_records,
    )

    top_missing = missing_records[:12]
    plot_status = "Created top-missing-columns bar chart."
    try:
        save_bar_plot(
            [item["column"] for item in top_missing],
            [float(item["missing_percent"]) for item in top_missing],
            "Top Missing Columns",
            "Column",
            "Missing Percent",
            visual_dir / "top_missing_columns.png",
        )
    except ImportError:
        plot_status = "Skipped quality plot because matplotlib is not installed."

    report_lines = [
        f"Dataset: {config.get('dataset_name', 'unknown')}",
        f"Row count: {summary['row_count']}",
        f"Date range: {summary['min_date']} to {summary['max_date']}",
        f"Invalid crash-date rows: {summary['invalid_date_count']}",
        f"Duplicate collision-ID rows: {summary['duplicate_collision_rows']}",
        "",
        "Top 10 missing columns:",
    ]

    for item in missing_records[:10]:
        report_lines.append(
            f"- {item['column']}: {item['missing_count']} rows ({item['missing_percent']}%)"
        )

    report_lines.append("")
    report_lines.append(plot_status)
    write_markdown(reports_dir / "data_quality_report.md", "Data Quality Report", report_lines)


def main() -> None:
    """Run data quality stage.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If stage fails.
    """
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)
    logger = get_logger("data_quality.checks")

    canonical_csv = Path(config["canonical_csv"]).resolve()
    log_step(logger, "QUALITY_START", f"Dataset: {config.get('dataset_name', 'unknown')}")

    summary = run_quality_checks(canonical_csv)
    write_quality_outputs(summary, config)

    log_step(logger, "QUALITY_DONE", f"Rows checked: {summary['row_count']}")


if __name__ == "__main__":
    main()
