"""Generate core analysis tables, charts, and heatmaps."""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_csv_rows, write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory
from utils.plot_utils import save_bar_plot, save_heatmap, save_line_plot
from utils.stats_utils import safe_divide

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


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


def aggregate_feature_data(feature_csv: Path) -> Dict[str, object]:
    """Aggregate feature data into reusable counters.

    Args:
        feature_csv: Feature table path.

    Returns:
        Dict[str, object]: Aggregation dictionary.

    Raises:
        FileNotFoundError: If feature CSV is missing.
    """
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    monthly_collisions: Counter[str] = Counter()
    monthly_fatal: Counter[str] = Counter()
    hourly_collisions: Counter[str] = Counter()
    weekday_collisions: Counter[str] = Counter()
    vehicle_counts: Counter[str] = Counter()
    factor_counts: Counter[str] = Counter()
    borough_counts: Counter[str] = Counter()
    day_hour_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    row_count = 0

    with feature_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1

            month = (row.get("MONTH") or "").strip()
            hour = (row.get("HOUR") or "").strip()
            weekday = (row.get("DAY_OF_WEEK") or "").strip()
            borough = (row.get("BOROUGH") or "").strip() or "UNKNOWN"
            vehicle = (row.get("VEHICLE TYPE CODE 1") or "").strip() or "UNKNOWN"
            factor = (row.get("CONTRIBUTING FACTOR VEHICLE 1") or "").strip() or "UNKNOWN"

            fatal = to_int(row.get("FATAL_COLLISION", "0"))

            if month:
                monthly_collisions[month] += 1
                monthly_fatal[month] += fatal
            if hour:
                hourly_collisions[hour] += 1
            if weekday:
                weekday_collisions[weekday] += 1
                if hour:
                    day_hour_counts[weekday][hour] += 1

            borough_counts[borough] += 1
            vehicle_counts[vehicle] += 1
            factor_counts[factor] += 1

    return {
        "row_count": row_count,
        "monthly_collisions": monthly_collisions,
        "monthly_fatal": monthly_fatal,
        "hourly_collisions": hourly_collisions,
        "weekday_collisions": weekday_collisions,
        "vehicle_counts": vehicle_counts,
        "factor_counts": factor_counts,
        "borough_counts": borough_counts,
        "day_hour_counts": day_hour_counts,
    }


def write_core_tables(aggregates: Dict[str, object], tables_dir: Path) -> None:
    """Write aggregated result tables.

    Args:
        aggregates: Aggregation dictionary.
        tables_dir: Output table directory.

    Returns:
        None

    Raises:
        None
    """
    ensure_directory(tables_dir)

    monthly_collisions: Counter = aggregates["monthly_collisions"]  # type: ignore[assignment]
    monthly_fatal: Counter = aggregates["monthly_fatal"]  # type: ignore[assignment]
    hourly_collisions: Counter = aggregates["hourly_collisions"]  # type: ignore[assignment]
    vehicle_counts: Counter = aggregates["vehicle_counts"]  # type: ignore[assignment]
    factor_counts: Counter = aggregates["factor_counts"]  # type: ignore[assignment]
    borough_counts: Counter = aggregates["borough_counts"]  # type: ignore[assignment]

    monthly_rows: List[Dict[str, str]] = []
    for month in sorted(monthly_collisions):
        collisions = monthly_collisions[month]
        fatals = monthly_fatal[month]
        monthly_rows.append(
            {
                "month": month,
                "collisions": str(collisions),
                "fatal_collisions": str(fatals),
                "fatal_rate_percent": f"{100.0 * safe_divide(fatals, collisions):.4f}",
            }
        )

    hourly_rows = [
        {"hour": hour, "collisions": str(hourly_collisions[hour])}
        for hour in sorted(hourly_collisions, key=lambda x: int(x))
    ]

    top_vehicle_rows = [
        {"vehicle_type": name, "collisions": str(count)}
        for name, count in vehicle_counts.most_common(20)
    ]
    top_factor_rows = [
        {"factor": name, "collisions": str(count)} for name, count in factor_counts.most_common(20)
    ]
    borough_rows = [
        {"borough": name, "collisions": str(count)} for name, count in borough_counts.most_common()
    ]

    write_csv_rows(tables_dir / "monthly_kpi.csv", ["month", "collisions", "fatal_collisions", "fatal_rate_percent"], monthly_rows)
    write_csv_rows(tables_dir / "hourly_kpi.csv", ["hour", "collisions"], hourly_rows)
    write_csv_rows(tables_dir / "top_vehicle_types.csv", ["vehicle_type", "collisions"], top_vehicle_rows)
    write_csv_rows(tables_dir / "top_factors.csv", ["factor", "collisions"], top_factor_rows)
    write_csv_rows(tables_dir / "borough_collisions.csv", ["borough", "collisions"], borough_rows)


def create_core_plots(aggregates: Dict[str, object], visual_root: Path) -> List[str]:
    """Create plots and return status messages.

    Args:
        aggregates: Aggregation dictionary.
        visual_root: Dataset visualization root path.

    Returns:
        List[str]: Plot status messages.

    Raises:
        None
    """
    messages: List[str] = []
    ensure_directory(visual_root / "core")
    ensure_directory(visual_root / "heatmaps")

    monthly_collisions: Counter = aggregates["monthly_collisions"]  # type: ignore[assignment]
    monthly_fatal: Counter = aggregates["monthly_fatal"]  # type: ignore[assignment]
    hourly_collisions: Counter = aggregates["hourly_collisions"]  # type: ignore[assignment]
    weekday_collisions: Counter = aggregates["weekday_collisions"]  # type: ignore[assignment]
    vehicle_counts: Counter = aggregates["vehicle_counts"]  # type: ignore[assignment]
    factor_counts: Counter = aggregates["factor_counts"]  # type: ignore[assignment]
    borough_counts: Counter = aggregates["borough_counts"]  # type: ignore[assignment]
    day_hour_counts: Dict[str, Counter] = aggregates["day_hour_counts"]  # type: ignore[assignment]

    months = sorted(monthly_collisions)
    monthly_values = [float(monthly_collisions[m]) for m in months]
    monthly_fatal_rate = [100.0 * safe_divide(monthly_fatal[m], monthly_collisions[m]) for m in months]

    hours = sorted(hourly_collisions, key=lambda x: int(x))
    hour_values = [float(hourly_collisions[h]) for h in hours]

    weekday_labels = DAYS
    weekday_values = [float(weekday_collisions[d]) for d in weekday_labels]

    top_vehicle = vehicle_counts.most_common(12)
    top_factors = factor_counts.most_common(12)
    top_boroughs = borough_counts.most_common()

    heatmap_matrix: List[List[float]] = []
    hour_labels = [str(i) for i in range(24)]
    for day in weekday_labels:
        row = [float(day_hour_counts.get(day, Counter()).get(hour, 0)) for hour in hour_labels]
        heatmap_matrix.append(row)

    try:
        save_line_plot(
            months,
            monthly_values,
            "Monthly Collision Trend",
            "Month",
            "Collisions",
            visual_root / "core" / "monthly_collision_trend.png",
        )
        messages.append("Created monthly collision trend line chart.")

        save_line_plot(
            months,
            monthly_fatal_rate,
            "Monthly Fatal Collision Rate",
            "Month",
            "Fatal Rate (%)",
            visual_root / "core" / "monthly_fatal_rate_trend.png",
        )
        messages.append("Created monthly fatal-rate line chart.")

        save_bar_plot(
            hours,
            hour_values,
            "Hourly Collision Count",
            "Hour",
            "Collisions",
            visual_root / "core" / "hourly_collision_count.png",
        )
        messages.append("Created hourly collision bar chart.")

        save_bar_plot(
            weekday_labels,
            weekday_values,
            "Weekday Collision Count",
            "Weekday",
            "Collisions",
            visual_root / "core" / "weekday_collision_count.png",
        )
        messages.append("Created weekday collision bar chart.")

        save_bar_plot(
            [name for name, _ in top_vehicle],
            [float(value) for _, value in top_vehicle],
            "Top Vehicle Types",
            "Vehicle Type",
            "Collisions",
            visual_root / "core" / "top_vehicle_types.png",
        )
        messages.append("Created top-vehicle-types bar chart.")

        save_bar_plot(
            [name for name, _ in top_factors],
            [float(value) for _, value in top_factors],
            "Top Contributing Factors",
            "Factor",
            "Collisions",
            visual_root / "core" / "top_contributing_factors.png",
        )
        messages.append("Created top-factors bar chart.")

        save_bar_plot(
            [name for name, _ in top_boroughs],
            [float(value) for _, value in top_boroughs],
            "Borough Collision Count",
            "Borough",
            "Collisions",
            visual_root / "core" / "borough_collision_count.png",
        )
        messages.append("Created borough collision bar chart.")

        save_heatmap(
            heatmap_matrix,
            hour_labels,
            weekday_labels,
            "Collision Heatmap: Weekday vs Hour",
            "Hour",
            "Weekday",
            visual_root / "heatmaps" / "weekday_hour_collision_heatmap.png",
        )
        messages.append("Created weekday-hour collision heatmap.")
    except ImportError:
        messages.append("Skipped plots because matplotlib is not installed.")

    return messages


def main() -> None:
    """Run exploratory analysis stage.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If stage fails.
    """
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)
    logger = get_logger("exploratory_analysis.core")

    feature_csv = Path(config["feature_csv"]).resolve()
    tables_dir = Path(config["tables_dir"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()
    visual_root = Path(config["visualizations_dir"]).resolve()

    log_step(logger, "EDA_START", f"Dataset: {config.get('dataset_name', 'unknown')}")
    aggregates = aggregate_feature_data(feature_csv)

    write_core_tables(aggregates, tables_dir)
    plot_messages = create_core_plots(aggregates, visual_root)

    lines = [
        f"Dataset: {config.get('dataset_name', 'unknown')}",
        f"Rows analyzed: {aggregates['row_count']}",
        "",
        "Generated core visual outputs:",
        *[f"- {item}" for item in plot_messages],
    ]
    write_markdown(reports_dir / "exploratory_analysis_report.md", "Exploratory Analysis Report", lines)

    log_step(logger, "EDA_DONE", f"Rows analyzed: {aggregates['row_count']}")


if __name__ == "__main__":
    main()
