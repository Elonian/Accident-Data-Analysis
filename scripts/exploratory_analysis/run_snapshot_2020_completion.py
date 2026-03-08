"""Generate missing 2020 snapshot analysis outputs and visuals.

This script focuses on the 2020 subset requirements before advanced modeling.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_csv_rows, write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory
from utils.plot_utils import save_bar_plot, save_heatmap, save_line_plot, save_scatter_plot
from utils.stats_utils import safe_divide
from utils.time_utils import combine_date_time, parse_crash_date


def to_int(value: str) -> int:
    """Convert text value to integer safely.

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


def read_rows(feature_csv: Path) -> List[Dict[str, str]]:
    """Read feature CSV rows into memory.

    Args:
        feature_csv: Feature CSV path.

    Returns:
        List[Dict[str, str]]: Feature rows.

    Raises:
        FileNotFoundError: If feature file does not exist.
    """
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_csv}")

    with feature_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_phase_summary(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build detailed COVID phase segmentation summary.

    Args:
        rows: Feature rows.

    Returns:
        List[Dict[str, str]]: Phase summary table.

    Raises:
        None
    """
    phase_collision: Counter[str] = Counter()
    phase_injury_coll: Counter[str] = Counter()
    phase_fatal_coll: Counter[str] = Counter()
    phase_person_injured: Counter[str] = Counter()
    phase_person_killed: Counter[str] = Counter()

    for row in rows:
        phase = (row.get("PANDEMIC_PHASE") or "UNKNOWN").strip() or "UNKNOWN"
        phase_collision[phase] += 1

        injury_coll = to_int(row.get("ANY_INJURY", "0"))
        fatal_coll = to_int(row.get("FATAL_COLLISION", "0"))
        phase_injury_coll[phase] += injury_coll
        phase_fatal_coll[phase] += fatal_coll

        phase_person_injured[phase] += to_int(row.get("NUMBER OF PERSONS INJURED", "0"))
        phase_person_killed[phase] += to_int(row.get("NUMBER OF PERSONS KILLED", "0"))

    phase_order = [
        "PRE_PAUSE",
        "PAUSE",
        "REOPEN_PHASE_1",
        "REOPEN_PHASE_2",
        "REOPEN_PHASE_3",
        "REOPEN_PHASE_4_PLUS",
        "UNKNOWN",
    ]

    output_rows: List[Dict[str, str]] = []
    for phase in phase_order:
        collisions = phase_collision.get(phase, 0)
        if collisions == 0:
            continue

        output_rows.append(
            {
                "phase": phase,
                "collisions": str(collisions),
                "injury_collisions": str(phase_injury_coll[phase]),
                "fatal_collisions": str(phase_fatal_coll[phase]),
                "persons_injured": str(phase_person_injured[phase]),
                "persons_killed": str(phase_person_killed[phase]),
                "fatal_collision_rate_percent": f"{100.0 * safe_divide(phase_fatal_coll[phase], collisions):.4f}",
                "persons_killed_per_10k_collisions": f"{10000.0 * safe_divide(phase_person_killed[phase], collisions):.4f}",
                "persons_injured_per_1k_collisions": f"{1000.0 * safe_divide(phase_person_injured[phase], collisions):.4f}",
            }
        )
    return output_rows


def build_weekly_kpi(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build weekly KPI table for 2020 subset.

    Args:
        rows: Feature rows.

    Returns:
        List[Dict[str, str]]: Weekly KPI rows.

    Raises:
        None
    """
    week_collision: Counter[str] = Counter()
    week_injury: Counter[str] = Counter()
    week_fatal: Counter[str] = Counter()

    for row in rows:
        crash_date = parse_crash_date(row.get("CRASH DATE", ""))
        if crash_date is None:
            continue

        iso_year, iso_week, _ = crash_date.isocalendar()
        week_key = f"{iso_year}-W{iso_week:02d}"

        week_collision[week_key] += 1
        week_injury[week_key] += to_int(row.get("ANY_INJURY", "0"))
        week_fatal[week_key] += to_int(row.get("FATAL_COLLISION", "0"))

    output: List[Dict[str, str]] = []
    for week in sorted(week_collision):
        collisions = week_collision[week]
        output.append(
            {
                "iso_week": week,
                "collisions": str(collisions),
                "injury_collisions": str(week_injury[week]),
                "fatal_collisions": str(week_fatal[week]),
                "injury_rate_percent": f"{100.0 * safe_divide(week_injury[week], collisions):.4f}",
                "fatal_rate_percent": f"{100.0 * safe_divide(week_fatal[week], collisions):.4f}",
            }
        )
    return output


def build_hourly_severity(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build frequency vs severity table by hour.

    Args:
        rows: Feature rows.

    Returns:
        List[Dict[str, str]]: Hourly severity summary rows.

    Raises:
        None
    """
    hour_collision: Counter[str] = Counter()
    hour_fatal: Counter[str] = Counter()
    hour_severity: Counter[str] = Counter()

    for row in rows:
        hour = (row.get("HOUR") or "").strip()
        if not hour:
            continue

        hour_collision[hour] += 1
        hour_fatal[hour] += to_int(row.get("FATAL_COLLISION", "0"))
        hour_severity[hour] += to_int(row.get("SEVERITY_SCORE", "0"))

    output = []
    for hour in sorted(hour_collision, key=lambda x: int(x)):
        collisions = hour_collision[hour]
        output.append(
            {
                "hour": hour,
                "collisions": str(collisions),
                "fatal_collisions": str(hour_fatal[hour]),
                "fatal_rate_percent": f"{100.0 * safe_divide(hour_fatal[hour], collisions):.4f}",
                "avg_severity_score": f"{safe_divide(hour_severity[hour], collisions):.4f}",
            }
        )
    return output


def build_weekday_factor(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build weekend vs weekday factor comparison table.

    Args:
        rows: Feature rows.

    Returns:
        List[Dict[str, str]]: Factor comparison rows.

    Raises:
        None
    """
    weekend_count: Counter[str] = Counter()
    weekday_count: Counter[str] = Counter()
    weekend_total = 0
    weekday_total = 0

    for row in rows:
        factor = (row.get("CONTRIBUTING FACTOR VEHICLE 1") or "UNKNOWN").strip() or "UNKNOWN"
        is_weekend_value = to_int(row.get("IS_WEEKEND", "0"))

        if is_weekend_value == 1:
            weekend_count[factor] += 1
            weekend_total += 1
        else:
            weekday_count[factor] += 1
            weekday_total += 1

    top_factors = [factor for factor, _ in (weekend_count + weekday_count).most_common(15)]

    output: List[Dict[str, str]] = []
    for factor in top_factors:
        wend = weekend_count[factor]
        wday = weekday_count[factor]
        output.append(
            {
                "factor": factor,
                "weekend_count": str(wend),
                "weekday_count": str(wday),
                "weekend_share_percent": f"{100.0 * safe_divide(wend, weekend_total):.4f}",
                "weekday_share_percent": f"{100.0 * safe_divide(wday, weekday_total):.4f}",
            }
        )
    return output


def build_vehicle_severity(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build vehicle severity summary table.

    Args:
        rows: Feature rows.

    Returns:
        List[Dict[str, str]]: Vehicle severity rows.

    Raises:
        None
    """
    vehicle_collision: Counter[str] = Counter()
    vehicle_injury: Counter[str] = Counter()
    vehicle_fatal: Counter[str] = Counter()

    for row in rows:
        vehicle = (row.get("VEHICLE TYPE CODE 1") or "UNKNOWN").strip() or "UNKNOWN"
        vehicle_collision[vehicle] += 1
        vehicle_injury[vehicle] += to_int(row.get("NUMBER OF PERSONS INJURED", "0"))
        vehicle_fatal[vehicle] += to_int(row.get("NUMBER OF PERSONS KILLED", "0"))

    output: List[Dict[str, str]] = []
    for vehicle, collisions in vehicle_collision.most_common(20):
        output.append(
            {
                "vehicle_type": vehicle,
                "collisions": str(collisions),
                "persons_injured": str(vehicle_injury[vehicle]),
                "persons_killed": str(vehicle_fatal[vehicle]),
                "persons_injured_per_1k_collisions": f"{1000.0 * safe_divide(vehicle_injury[vehicle], collisions):.4f}",
                "persons_killed_per_10k_collisions": f"{10000.0 * safe_divide(vehicle_fatal[vehicle], collisions):.4f}",
            }
        )
    return output


def build_lethality_matrix(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build 2020 lethality matrix table for VRU harm by vehicle.

    Args:
        rows: Feature rows.

    Returns:
        List[Dict[str, str]]: Lethality matrix rows.

    Raises:
        None
    """
    vehicle_collision: Counter[str] = Counter()
    ped_harm: Counter[str] = Counter()
    cyc_harm: Counter[str] = Counter()

    for row in rows:
        vehicle = (row.get("VEHICLE TYPE CODE 1") or "UNKNOWN").strip() or "UNKNOWN"
        vehicle_collision[vehicle] += 1

        ped_inj = to_int(row.get("NUMBER OF PEDESTRIANS INJURED", "0"))
        ped_kill = to_int(row.get("NUMBER OF PEDESTRIANS KILLED", "0"))
        cyc_inj = to_int(row.get("NUMBER OF CYCLIST INJURED", "0"))
        cyc_kill = to_int(row.get("NUMBER OF CYCLIST KILLED", "0"))

        ped_harm[vehicle] += ped_inj + (5 * ped_kill)
        cyc_harm[vehicle] += cyc_inj + (5 * cyc_kill)

    output: List[Dict[str, str]] = []
    for vehicle, collisions in vehicle_collision.most_common(12):
        output.append(
            {
                "vehicle_type": vehicle,
                "collisions": str(collisions),
                "pedestrian_harm_per_1k_collisions": f"{1000.0 * safe_divide(ped_harm[vehicle], collisions):.4f}",
                "cyclist_harm_per_1k_collisions": f"{1000.0 * safe_divide(cyc_harm[vehicle], collisions):.4f}",
            }
        )
    return output


def build_hotspot_table(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[float], List[float], List[float]]:
    """Build hotspot table from coordinate grid aggregation.

    Args:
        rows: Feature rows.

    Returns:
        Tuple[List[Dict[str, str]], List[float], List[float], List[float]]:
            hotspot rows, longitude sample, latitude sample, collision intensity sample.

    Raises:
        None
    """
    grid_collision: Counter[str] = Counter()
    grid_severity: Counter[str] = Counter()
    grid_lat: Dict[str, float] = {}
    grid_lon: Dict[str, float] = {}

    for row in rows:
        lat_text = (row.get("LATITUDE") or "").strip()
        lon_text = (row.get("LONGITUDE") or "").strip()
        if not lat_text or not lon_text:
            continue

        try:
            lat = float(lat_text)
            lon = float(lon_text)
        except ValueError:
            continue

        key = f"{round(lat, 3)}_{round(lon, 3)}"
        grid_collision[key] += 1
        grid_severity[key] += to_int(row.get("SEVERITY_SCORE", "0"))
        grid_lat[key] = round(lat, 3)
        grid_lon[key] = round(lon, 3)

    output: List[Dict[str, str]] = []
    x_values: List[float] = []
    y_values: List[float] = []
    intensity: List[float] = []

    for key, collisions in grid_collision.most_common(100):
        severity = grid_severity[key]
        output.append(
            {
                "grid_key": key,
                "latitude_center": str(grid_lat[key]),
                "longitude_center": str(grid_lon[key]),
                "collisions": str(collisions),
                "avg_severity_score": f"{safe_divide(severity, collisions):.4f}",
            }
        )
        x_values.append(grid_lon[key])
        y_values.append(grid_lat[key])
        intensity.append(float(collisions))

    return output, x_values, y_values, intensity


def build_dst_spring_summary(rows: List[Dict[str, str]]) -> List[str]:
    """Build spring DST 7-day pre/post summary lines.

    Args:
        rows: Feature rows.

    Returns:
        List[str]: Spring DST summary lines.

    Raises:
        None
    """
    pre_start = date(2020, 3, 1)
    pre_end = date(2020, 3, 7)
    post_start = date(2020, 3, 9)
    post_end = date(2020, 3, 15)

    stats = {
        "pre": {"collisions": 0, "fatal_collisions": 0, "persons_killed": 0, "persons_injured": 0, "early_morning_collisions": 0},
        "post": {"collisions": 0, "fatal_collisions": 0, "persons_killed": 0, "persons_injured": 0, "early_morning_collisions": 0},
    }

    for row in rows:
        crash_date = parse_crash_date(row.get("CRASH DATE", ""))
        if crash_date is None:
            continue

        bucket = None
        if pre_start <= crash_date <= pre_end:
            bucket = "pre"
        elif post_start <= crash_date <= post_end:
            bucket = "post"
        if bucket is None:
            continue

        stats[bucket]["collisions"] += 1
        stats[bucket]["fatal_collisions"] += to_int(row.get("FATAL_COLLISION", "0"))
        stats[bucket]["persons_killed"] += to_int(row.get("NUMBER OF PERSONS KILLED", "0"))
        stats[bucket]["persons_injured"] += to_int(row.get("NUMBER OF PERSONS INJURED", "0"))

        hour_text = (row.get("HOUR") or "").strip()
        if hour_text:
            hour = int(float(hour_text))
            if 4 <= hour <= 8:
                stats[bucket]["early_morning_collisions"] += 1

    lines = [
        "Spring DST 7-day window comparison:",
        f"- Pre window: {pre_start} to {pre_end}",
        f"- Post window: {post_start} to {post_end}",
        "",
    ]

    for bucket in ["pre", "post"]:
        item = stats[bucket]
        collisions = item["collisions"]
        lines.append(f"{bucket.upper()}:")
        lines.append(f"- collisions: {collisions}")
        lines.append(f"- fatal collisions: {item['fatal_collisions']}")
        lines.append(f"- persons killed per 10k collisions: {10000.0 * safe_divide(item['persons_killed'], collisions):.4f}")
        lines.append(f"- persons injured per 1k collisions: {1000.0 * safe_divide(item['persons_injured'], collisions):.4f}")
        lines.append(f"- early morning collision share (%): {100.0 * safe_divide(item['early_morning_collisions'], collisions):.4f}")
        lines.append("")

    return lines


def build_covid_policy_signal_report(phase_rows: List[Dict[str, str]]) -> List[str]:
    """Build COVID lockdown and reopening signal summary lines.

    Args:
        phase_rows: Detailed phase summary rows.

    Returns:
        List[str]: Report lines.

    Raises:
        None
    """
    lookup = {row["phase"]: row for row in phase_rows}

    def _line(phase: str) -> str:
        row = lookup.get(phase)
        if not row:
            return f"- {phase}: no rows"
        return (
            f"- {phase}: collisions={row['collisions']}, "
            f"fatal_rate%={row['fatal_collision_rate_percent']}, "
            f"killed_per_10k={row['persons_killed_per_10k_collisions']}, "
            f"injured_per_1k={row['persons_injured_per_1k_collisions']}"
        )

    return [
        "COVID policy signal summary for 2020 subset:",
        "",
        "Timeline anchors used:",
        "- PAUSE effective: 2020-03-22",
        "- Reopening phase 1: 2020-06-08",
        "- Reopening phase 2: 2020-06-22",
        "- Reopening phase 3: 2020-07-06",
        "- Reopening phase 4: 2020-07-20",
        "",
        "Phase metrics:",
        _line("PRE_PAUSE"),
        _line("PAUSE"),
        _line("REOPEN_PHASE_1"),
        _line("REOPEN_PHASE_2"),
        _line("REOPEN_PHASE_3"),
        _line("REOPEN_PHASE_4_PLUS"),
    ]


def write_outputs(rows: List[Dict[str, str]], config: Dict[str, str]) -> None:
    """Write all required 2020 completion outputs.

    Args:
        rows: Feature rows.
        config: Runtime config dictionary.

    Returns:
        None

    Raises:
        None
    """
    tables_dir = Path(config["tables_dir"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()
    visual_root = Path(config["visualizations_dir"]).resolve()

    ensure_directory(tables_dir)
    ensure_directory(reports_dir)
    ensure_directory(visual_root / "core")
    ensure_directory(visual_root / "heatmaps")
    ensure_directory(visual_root / "clusters")

    phase_rows = build_phase_summary(rows)
    weekly_rows = build_weekly_kpi(rows)
    hourly_rows = build_hourly_severity(rows)
    factor_rows = build_weekday_factor(rows)
    vehicle_rows = build_vehicle_severity(rows)
    lethality_rows = build_lethality_matrix(rows)
    hotspot_rows, hotspot_x, hotspot_y, hotspot_intensity = build_hotspot_table(rows)

    write_csv_rows(
        tables_dir / "phase_segmentation_summary_2020.csv",
        [
            "phase",
            "collisions",
            "injury_collisions",
            "fatal_collisions",
            "persons_injured",
            "persons_killed",
            "fatal_collision_rate_percent",
            "persons_killed_per_10k_collisions",
            "persons_injured_per_1k_collisions",
        ],
        phase_rows,
    )
    write_csv_rows(
        tables_dir / "weekly_kpi_2020.csv",
        ["iso_week", "collisions", "injury_collisions", "fatal_collisions", "injury_rate_percent", "fatal_rate_percent"],
        weekly_rows,
    )
    write_csv_rows(
        tables_dir / "hourly_frequency_severity_2020.csv",
        ["hour", "collisions", "fatal_collisions", "fatal_rate_percent", "avg_severity_score"],
        hourly_rows,
    )
    write_csv_rows(
        tables_dir / "weekend_weekday_factors_2020.csv",
        ["factor", "weekend_count", "weekday_count", "weekend_share_percent", "weekday_share_percent"],
        factor_rows,
    )
    write_csv_rows(
        tables_dir / "vehicle_severity_summary_2020.csv",
        [
            "vehicle_type",
            "collisions",
            "persons_injured",
            "persons_killed",
            "persons_injured_per_1k_collisions",
            "persons_killed_per_10k_collisions",
        ],
        vehicle_rows,
    )
    write_csv_rows(
        tables_dir / "lethality_matrix_2020.csv",
        ["vehicle_type", "collisions", "pedestrian_harm_per_1k_collisions", "cyclist_harm_per_1k_collisions"],
        lethality_rows,
    )
    write_csv_rows(
        tables_dir / "hotspot_summary_2020.csv",
        ["grid_key", "latitude_center", "longitude_center", "collisions", "avg_severity_score"],
        hotspot_rows,
    )

    # Visualizations
    save_bar_plot(
        [row["phase"] for row in phase_rows],
        [float(row["collisions"]) for row in phase_rows],
        "2020 Collisions by COVID Phase",
        "Phase",
        "Collisions",
        visual_root / "core" / "phase_collision_count_2020.png",
    )
    save_bar_plot(
        [row["phase"] for row in phase_rows],
        [float(row["fatal_collision_rate_percent"]) for row in phase_rows],
        "2020 Fatal Collision Rate by COVID Phase",
        "Phase",
        "Fatal Rate (%)",
        visual_root / "core" / "phase_fatal_rate_2020.png",
    )

    save_line_plot(
        [row["iso_week"] for row in weekly_rows],
        [float(row["collisions"]) for row in weekly_rows],
        "2020 Weekly Collision Trend",
        "ISO Week",
        "Collisions",
        visual_root / "core" / "weekly_collision_trend_2020.png",
    )

    save_line_plot(
        [row["hour"] for row in hourly_rows],
        [float(row["fatal_rate_percent"]) for row in hourly_rows],
        "2020 Hourly Fatal Rate",
        "Hour",
        "Fatal Rate (%)",
        visual_root / "core" / "hourly_fatal_rate_2020.png",
    )

    # Dual-axis hourly chart (frequency vs severity).
    try:
        import matplotlib.pyplot as plt  # type: ignore

        hour_labels = [row["hour"] for row in hourly_rows]
        collisions = [float(row["collisions"]) for row in hourly_rows]
        fatal_rates = [float(row["fatal_rate_percent"]) for row in hourly_rows]

        figure, axis_left = plt.subplots(figsize=(10, 5))
        axis_left.bar(hour_labels, collisions, color="steelblue", alpha=0.75)
        axis_left.set_xlabel("Hour")
        axis_left.set_ylabel("Collisions", color="steelblue")
        axis_left.tick_params(axis="y", labelcolor="steelblue")

        axis_right = axis_left.twinx()
        axis_right.plot(hour_labels, fatal_rates, color="darkred", marker="o", linewidth=1.8, markersize=3)
        axis_right.set_ylabel("Fatal Rate (%)", color="darkred")
        axis_right.tick_params(axis="y", labelcolor="darkred")

        plt.title("2020 Frequency vs Severity by Hour")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(visual_root / "core" / "hourly_frequency_vs_fatal_rate_dual_axis_2020.png")
        plt.close(figure)
    except Exception:
        pass

    top_factor_rows = factor_rows[:12]
    save_bar_plot(
        [row["factor"] for row in top_factor_rows],
        [float(row["weekend_share_percent"]) for row in top_factor_rows],
        "Weekend Factor Share (Top Factors)",
        "Factor",
        "Weekend Share (%)",
        visual_root / "core" / "weekend_factor_share_2020.png",
    )

    # Grouped weekend vs weekday factor chart.
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore

        labels = [row["factor"] for row in top_factor_rows]
        weekend_values = [float(row["weekend_share_percent"]) for row in top_factor_rows]
        weekday_values = [float(row["weekday_share_percent"]) for row in top_factor_rows]
        positions = np.arange(len(labels))
        width = 0.4

        figure, axis = plt.subplots(figsize=(12, 5))
        axis.bar(positions - width / 2, weekend_values, width, label="Weekend", color="teal")
        axis.bar(positions + width / 2, weekday_values, width, label="Weekday", color="orange")
        axis.set_title("2020 Weekend vs Weekday Factor Share")
        axis.set_xlabel("Factor")
        axis.set_ylabel("Share (%)")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.legend()
        plt.tight_layout()
        plt.savefig(visual_root / "core" / "weekend_vs_weekday_factor_grouped_2020.png")
        plt.close(figure)
    except Exception:
        pass

    save_bar_plot(
        [row["vehicle_type"] for row in vehicle_rows[:12]],
        [float(row["persons_killed_per_10k_collisions"]) for row in vehicle_rows[:12]],
        "Vehicle Killed per 10k Collisions (Top Vehicle Types)",
        "Vehicle Type",
        "Killed per 10k",
        visual_root / "core" / "vehicle_killed_per_10k_2020.png",
    )

    # Lethality heatmap: rows = [pedestrian, cyclist], columns = vehicle types.
    vehicle_labels = [row["vehicle_type"] for row in lethality_rows]
    matrix = [
        [float(row["pedestrian_harm_per_1k_collisions"]) for row in lethality_rows],
        [float(row["cyclist_harm_per_1k_collisions"]) for row in lethality_rows],
    ]
    save_heatmap(
        matrix,
        vehicle_labels,
        ["Pedestrian Harm", "Cyclist Harm"],
        "2020 Lethality Matrix Heatmap",
        "Vehicle Type",
        "VRU Harm Type",
        visual_root / "heatmaps" / "lethality_matrix_2020_heatmap.png",
    )

    # Phase-hour heatmap
    phase_order = [row["phase"] for row in phase_rows]
    phase_hour_counter: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        phase = (row.get("PANDEMIC_PHASE") or "UNKNOWN").strip() or "UNKNOWN"
        hour = (row.get("HOUR") or "").strip()
        if phase and hour:
            phase_hour_counter[phase][hour] += 1

    hour_labels = [str(i) for i in range(24)]
    phase_hour_matrix = [
        [float(phase_hour_counter[phase].get(hour, 0)) for hour in hour_labels]
        for phase in phase_order
    ]
    save_heatmap(
        phase_hour_matrix,
        hour_labels,
        phase_order,
        "2020 Phase vs Hour Collision Heatmap",
        "Hour",
        "COVID Phase",
        visual_root / "heatmaps" / "phase_hour_collision_heatmap_2020.png",
    )

    # Hotspot scatter sample
    save_scatter_plot(
        hotspot_x,
        hotspot_y,
        "2020 Hotspot Grid Centers (Top 100)",
        "Longitude",
        "Latitude",
        visual_root / "clusters" / "hotspot_grid_centers_2020.png",
    )

    # Reports
    write_markdown(
        reports_dir / "phase_segmentation_report_2020.md",
        "2020 Lockdown Phase Segmentation Report",
        [
            "This report segments Jan-Aug 2020 collisions into pre-PAUSE, PAUSE, and reopening phases.",
            "See phase_segmentation_summary_2020.csv for detailed metrics.",
        ],
    )

    spring_lines = build_dst_spring_summary(rows)
    write_markdown(reports_dir / "dst_spring_check_2020.md", "2020 Spring DST Check", spring_lines)

    # DST pre/post summary chart.
    try:
        import matplotlib.pyplot as plt  # type: ignore

        pre_collisions = 0
        post_collisions = 0
        pre_killed_rate = 0.0
        post_killed_rate = 0.0
        for line in spring_lines:
            if line.startswith("- collisions: ") and pre_collisions == 0:
                pre_collisions = int(line.split(":", 1)[1].strip())
            elif line.startswith("- collisions: "):
                post_collisions = int(line.split(":", 1)[1].strip())
            if line.startswith("- persons killed per 10k collisions: ") and pre_killed_rate == 0.0:
                pre_killed_rate = float(line.split(":", 1)[1].strip())
            elif line.startswith("- persons killed per 10k collisions: "):
                post_killed_rate = float(line.split(":", 1)[1].strip())

        figure, axis_left = plt.subplots(figsize=(7, 4))
        labels = ["Pre DST", "Post DST"]
        axis_left.bar(labels, [pre_collisions, post_collisions], color=["#4c78a8", "#f58518"], alpha=0.8)
        axis_left.set_ylabel("Collisions", color="#4c78a8")
        axis_left.tick_params(axis="y", labelcolor="#4c78a8")

        axis_right = axis_left.twinx()
        axis_right.plot(labels, [pre_killed_rate, post_killed_rate], color="darkred", marker="o", linewidth=2.0)
        axis_right.set_ylabel("Persons Killed per 10k", color="darkred")
        axis_right.tick_params(axis="y", labelcolor="darkred")

        plt.title("2020 Spring DST Pre vs Post")
        plt.tight_layout()
        plt.savefig(visual_root / "core" / "dst_spring_pre_post_2020.png")
        plt.close(figure)
    except Exception:
        pass

    write_markdown(
        reports_dir / "dst_fall_limitation_2020.md",
        "2020 Fall DST Limitation",
        [
            "Fall-back DST analysis cannot be executed on this file window.",
            "Reason: NYC Accidents 2020 subset ends on 2020-08-29 and does not include November 2020 records.",
        ],
    )

    policy_lines = build_covid_policy_signal_report(phase_rows)
    write_markdown(reports_dir / "covid_policy_signal_2020.md", "2020 COVID Lockdown Impact Summary", policy_lines)


def main() -> None:
    """Run missing 2020 output completion workflow.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If workflow fails.
    """
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)

    dataset_name = config.get("dataset_name", "")
    if dataset_name != "snapshot_2020":
        raise RuntimeError("This script is intended for snapshot_2020 config only.")

    logger = get_logger("exploratory_analysis.snapshot_2020_completion")
    feature_csv = Path(config["feature_csv"]).resolve()

    log_step(logger, "SNAPSHOT_COMPLETION_START", f"Input: {feature_csv}")
    rows = read_rows(feature_csv)
    write_outputs(rows, config)
    log_step(logger, "SNAPSHOT_COMPLETION_DONE", f"Rows processed: {len(rows)}")


if __name__ == "__main__":
    main()
