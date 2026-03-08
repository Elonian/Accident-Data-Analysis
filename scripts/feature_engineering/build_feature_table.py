"""Create engineered feature table from canonical CSV."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory
from utils.time_utils import combine_date_time, is_weekend, pandemic_phase


ADDED_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR",
    "IS_WEEKEND",
    "PANDEMIC_PHASE",
    "ANY_INJURY",
    "FATAL_COLLISION",
    "SEVERITY_SCORE",
]


def to_int(value: str) -> int:
    """Convert text number to integer safely.

    Args:
        value: Input text value.

    Returns:
        int: Parsed integer or zero.

    Raises:
        None
    """
    text = (value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def enrich_row(row: Dict[str, str]) -> Dict[str, str]:
    """Add engineered fields to one row.

    Args:
        row: Input canonical row.

    Returns:
        Dict[str, str]: Enriched row.

    Raises:
        None
    """
    output = dict(row)
    crash_dt = combine_date_time(row.get("CRASH DATE", ""), row.get("CRASH TIME", ""))

    if crash_dt is None:
        output["YEAR"] = ""
        output["MONTH"] = ""
        output["DAY_OF_WEEK"] = ""
        output["HOUR"] = ""
        output["IS_WEEKEND"] = ""
        output["PANDEMIC_PHASE"] = ""
    else:
        output["YEAR"] = str(crash_dt.year)
        output["MONTH"] = f"{crash_dt.year}-{crash_dt.month:02d}"
        output["DAY_OF_WEEK"] = crash_dt.strftime("%A")
        output["HOUR"] = str(crash_dt.hour)
        output["IS_WEEKEND"] = str(is_weekend(crash_dt))
        output["PANDEMIC_PHASE"] = pandemic_phase(crash_dt)

    injured = to_int(row.get("NUMBER OF PERSONS INJURED", "0"))
    killed = to_int(row.get("NUMBER OF PERSONS KILLED", "0"))

    output["ANY_INJURY"] = "1" if injured > 0 else "0"
    output["FATAL_COLLISION"] = "1" if killed > 0 else "0"
    output["SEVERITY_SCORE"] = str(injured + (5 * killed))
    return output


def build_feature_table(canonical_csv: Path, feature_csv: Path) -> Dict[str, object]:
    """Build feature table in a streaming pass.

    Args:
        canonical_csv: Canonical input CSV path.
        feature_csv: Feature output CSV path.

    Returns:
        Dict[str, object]: Summary with counts and phase totals.

    Raises:
        FileNotFoundError: If canonical input is missing.
    """
    if not canonical_csv.exists():
        raise FileNotFoundError(f"Canonical CSV not found: {canonical_csv}")

    ensure_directory(feature_csv.parent)

    row_count = 0
    phase_counter: Counter[str] = Counter()

    with canonical_csv.open("r", encoding="utf-8", newline="") as src_handle, feature_csv.open(
        "w", encoding="utf-8", newline=""
    ) as dst_handle:
        reader = csv.DictReader(src_handle)
        if not reader.fieldnames:
            raise ValueError("Canonical CSV has no header.")

        fieldnames = list(reader.fieldnames) + ADDED_COLUMNS
        writer = csv.DictWriter(dst_handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            enriched = enrich_row(row)
            writer.writerow(enriched)
            row_count += 1

            phase = enriched.get("PANDEMIC_PHASE", "")
            if phase:
                phase_counter[phase] += 1

    return {
        "row_count": row_count,
        "phase_counter": dict(phase_counter),
    }


def main() -> None:
    """Run feature engineering stage.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If stage fails.
    """
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)
    logger = get_logger("feature_engineering.build")

    canonical_csv = Path(config["canonical_csv"]).resolve()
    feature_csv = Path(config["feature_csv"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()

    log_step(logger, "FEATURE_START", f"Dataset: {config.get('dataset_name', 'unknown')}")
    summary = build_feature_table(canonical_csv, feature_csv)

    phase_counter = summary["phase_counter"]
    assert isinstance(phase_counter, dict)

    lines = [
        f"Dataset: {config.get('dataset_name', 'unknown')}",
        f"Rows processed: {summary['row_count']}",
        "",
        "Pandemic phase distribution:",
    ]
    for phase, count in sorted(phase_counter.items()):
        lines.append(f"- {phase}: {count}")

    write_markdown(reports_dir / "feature_engineering_report.md", "Feature Engineering Report", lines)
    log_step(logger, "FEATURE_DONE", f"Rows processed: {summary['row_count']}")


if __name__ == "__main__":
    main()
