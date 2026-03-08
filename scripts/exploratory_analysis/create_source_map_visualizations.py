"""Create NYC map-style visualizations directly from source collision CSV."""

from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from array import array
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_csv_rows, write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory


def to_float(value: str) -> float | None:
    """Convert text to float safely.

    Args:
        value: Input text value.

    Returns:
        float | None: Parsed float or None.

    Raises:
        None
    """
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: str) -> int:
    """Convert text to integer safely.

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


def parse_hour(value: str) -> int | None:
    """Parse crash-time text to hour.

    Args:
        value: Crash-time text.

    Returns:
        int | None: Hour in [0, 23] or None.

    Raises:
        None
    """
    text = (value or "").strip()
    if not text:
        return None

    token = text.split(":", 1)[0]
    if not token:
        return None
    try:
        hour = int(token)
    except ValueError:
        return None

    if 0 <= hour <= 23:
        return hour
    return None


def is_valid_nyc_coordinate(latitude: float, longitude: float) -> bool:
    """Return True when coordinate is within practical NYC envelope.

    Args:
        latitude: Latitude value.
        longitude: Longitude value.

    Returns:
        bool: True when coordinate is valid for map plotting.

    Raises:
        None
    """
    if math.isclose(latitude, 0.0, abs_tol=1e-9) or math.isclose(longitude, 0.0, abs_tol=1e-9):
        return False
    if not (40.35 <= latitude <= 41.05):
        return False
    if not (-74.35 <= longitude <= -73.55):
        return False
    return True


def reservoir_append(sample: List[float], value: float, seen: int, keep: int, rng: random.Random) -> None:
    """Reservoir-sample one numeric value.

    Args:
        sample: Reservoir sample list.
        value: Candidate value.
        seen: Number of seen elements (1-based).
        keep: Max sample size.
        rng: Random generator.

    Returns:
        None

    Raises:
        None
    """
    if keep <= 0:
        return

    if len(sample) < keep:
        sample.append(value)
        return

    index = rng.randint(0, seen - 1)
    if index < keep:
        sample[index] = value


def clean_window_from_median(lons: Iterable[float], lats: Iterable[float], offset: float) -> Tuple[float, float, float, float]:
    """Build x/y plotting window around coordinate medians.

    Args:
        lons: Longitude values.
        lats: Latitude values.
        offset: Half-window offset in degrees.

    Returns:
        Tuple[float, float, float, float]: x_min, x_max, y_min, y_max.

    Raises:
        ValueError: If no coordinate values are provided.
    """
    lon_values = list(lons)
    lat_values = list(lats)
    if not lon_values or not lat_values:
        raise ValueError("Cannot build map window without coordinates.")

    med_lon = statistics.median(lon_values)
    med_lat = statistics.median(lat_values)

    return med_lon - offset, med_lon + offset, med_lat - offset, med_lat + offset


def plot_road_imprint(
    longitudes: array,
    latitudes: array,
    output_path: Path,
    title: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Plot Rich-Haar-style road-imprint scatter map.

    Args:
        longitudes: Longitude values.
        latitudes: Latitude values.
        output_path: Output image path.
        title: Plot title.
        x_min: Minimum x-axis value.
        x_max: Maximum x-axis value.
        y_min: Minimum y-axis value.
        y_max: Maximum y-axis value.

    Returns:
        None

    Raises:
        ImportError: If matplotlib is unavailable.
    """
    import matplotlib.pyplot as plt  # type: ignore

    ensure_directory(output_path.parent)
    n_points = len(longitudes)
    if n_points >= 1_000_000:
        alpha, size = 0.020, 0.05
    elif n_points >= 200_000:
        alpha, size = 0.025, 0.06
    else:
        alpha, size = 0.035, 0.10

    figure, axis = plt.subplots(figsize=(9, 9))
    axis.scatter(longitudes, latitudes, c="black", s=size, alpha=alpha, linewidths=0)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(title)
    axis.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_density_hexbin(
    longitudes: array,
    latitudes: array,
    output_path: Path,
    title: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Plot collision-density hexbin map.

    Args:
        longitudes: Longitude values.
        latitudes: Latitude values.
        output_path: Output image path.
        title: Plot title.
        x_min: Minimum x-axis value.
        x_max: Maximum x-axis value.
        y_min: Minimum y-axis value.
        y_max: Maximum y-axis value.

    Returns:
        None

    Raises:
        ImportError: If matplotlib is unavailable.
    """
    import matplotlib.pyplot as plt  # type: ignore

    ensure_directory(output_path.parent)
    n_points = len(longitudes)
    gridsize = 250 if n_points >= 1_000_000 else 170 if n_points >= 200_000 else 120

    figure, axis = plt.subplots(figsize=(9, 9))
    hexmap = axis.hexbin(
        longitudes,
        latitudes,
        gridsize=gridsize,
        mincnt=1,
        bins="log",
        cmap="magma",
        extent=[x_min, x_max, y_min, y_max],
    )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(title)
    colorbar = figure.colorbar(hexmap, ax=axis)
    colorbar.set_label("log10(collision density)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_hotspot_overlay(
    longitudes: array,
    latitudes: array,
    hotspot_lons: array,
    hotspot_lats: array,
    output_path: Path,
    title: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    injured_threshold: int,
) -> None:
    """Plot base collision map with high-injury hotspot overlay.

    Args:
        longitudes: All collision longitudes.
        latitudes: All collision latitudes.
        hotspot_lons: Hotspot longitudes.
        hotspot_lats: Hotspot latitudes.
        output_path: Output image path.
        title: Plot title.
        x_min: Minimum x-axis value.
        x_max: Maximum x-axis value.
        y_min: Minimum y-axis value.
        y_max: Maximum y-axis value.
        injured_threshold: Hotspot injury threshold.

    Returns:
        None

    Raises:
        ImportError: If matplotlib is unavailable.
    """
    import matplotlib.pyplot as plt  # type: ignore

    ensure_directory(output_path.parent)

    figure, axis = plt.subplots(figsize=(9, 9))
    n_points = len(longitudes)
    if n_points >= 1_000_000:
        base_alpha, base_size = 0.016, 0.04
    elif n_points >= 200_000:
        base_alpha, base_size = 0.018, 0.05
    else:
        base_alpha, base_size = 0.025, 0.08

    axis.scatter(longitudes, latitudes, c="#2d2d2d", s=base_size, alpha=base_alpha, linewidths=0, label="All collisions")
    if hotspot_lons and hotspot_lats:
        axis.scatter(
            hotspot_lons,
            hotspot_lats,
            c="#e31a1c",
            s=1.1,
            alpha=0.32,
            linewidths=0,
            label=f"Injury >= {injured_threshold}",
        )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(title)
    axis.legend(loc="lower left", fontsize=8, framealpha=0.85)
    axis.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_hourly_hexbin_triptych(
    hour_to_lons: Dict[int, array],
    hour_to_lats: Dict[int, array],
    output_path: Path,
    title: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    focus_hours: List[int],
) -> None:
    """Plot three hour-specific hexbin maps in one figure.

    Args:
        hour_to_lons: Hour-indexed longitude arrays.
        hour_to_lats: Hour-indexed latitude arrays.
        output_path: Output image path.
        title: Figure title.
        x_min: Minimum x-axis value.
        x_max: Maximum x-axis value.
        y_min: Minimum y-axis value.
        y_max: Maximum y-axis value.
        focus_hours: Hours to visualize.

    Returns:
        None

    Raises:
        ImportError: If matplotlib is unavailable.
    """
    import matplotlib.pyplot as plt  # type: ignore

    ensure_directory(output_path.parent)
    figure, axes = plt.subplots(1, len(focus_hours), figsize=(16, 5), sharex=True, sharey=True)
    if len(focus_hours) == 1:
        axes = [axes]  # type: ignore[assignment]

    for axis, hour in zip(axes, focus_hours):
        lons = hour_to_lons.get(hour, array("f"))
        lats = hour_to_lats.get(hour, array("f"))
        if len(lons) > 0:
            hexmap = axis.hexbin(
                lons,
                lats,
                gridsize=95,
                bins="log",
                mincnt=1,
                cmap="viridis",
                extent=[x_min, x_max, y_min, y_max],
            )
            figure.colorbar(hexmap, ax=axis, fraction=0.045, pad=0.02)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{hour:02d}:00 - {hour:02d}:59")
        axis.set_xlabel("Longitude")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Latitude")
    figure.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(figure)


def stream_source_coordinates(
    source_csv: Path,
    injured_threshold: int,
    focus_hours: List[int],
    median_sample_size: int,
    random_seed: int,
) -> Dict[str, object]:
    """Stream source CSV and collect plotting payload and QA stats.

    Args:
        source_csv: Source CSV path.
        injured_threshold: Threshold for hotspot collisions.
        focus_hours: Hour list for hour-specific maps.
        median_sample_size: Reservoir size for median estimation.
        random_seed: Random seed.

    Returns:
        Dict[str, object]: Payload with coordinate arrays and QA counters.

    Raises:
        FileNotFoundError: If source CSV does not exist.
    """
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    rng = random.Random(random_seed)

    all_lons = array("f")
    all_lats = array("f")
    hotspot_lons = array("f")
    hotspot_lats = array("f")

    hour_to_lons: Dict[int, array] = {hour: array("f") for hour in focus_hours}
    hour_to_lats: Dict[int, array] = {hour: array("f") for hour in focus_hours}
    hour_counter: Counter[int] = Counter()

    lon_median_sample: List[float] = []
    lat_median_sample: List[float] = []

    total_rows = 0
    valid_rows = 0
    missing_coord_rows = 0
    invalid_coord_rows = 0
    out_of_bounds_rows = 0

    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            lat = to_float(row.get("LATITUDE", ""))
            lon = to_float(row.get("LONGITUDE", ""))

            if lat is None or lon is None:
                missing_coord_rows += 1
                continue
            if not math.isfinite(lat) or not math.isfinite(lon):
                invalid_coord_rows += 1
                continue
            if not is_valid_nyc_coordinate(lat, lon):
                out_of_bounds_rows += 1
                continue

            valid_rows += 1
            all_lons.append(float(lon))
            all_lats.append(float(lat))

            reservoir_append(lon_median_sample, float(lon), valid_rows, median_sample_size, rng)
            reservoir_append(lat_median_sample, float(lat), valid_rows, median_sample_size, rng)

            injuries = to_int(row.get("NUMBER OF PERSONS INJURED", "0"))
            if injuries >= injured_threshold:
                hotspot_lons.append(float(lon))
                hotspot_lats.append(float(lat))

            hour = parse_hour(row.get("CRASH TIME", ""))
            if hour is not None:
                hour_counter[hour] += 1
                if hour in hour_to_lons:
                    hour_to_lons[hour].append(float(lon))
                    hour_to_lats[hour].append(float(lat))

    return {
        "all_lons": all_lons,
        "all_lats": all_lats,
        "hotspot_lons": hotspot_lons,
        "hotspot_lats": hotspot_lats,
        "hour_to_lons": hour_to_lons,
        "hour_to_lats": hour_to_lats,
        "hour_counter": hour_counter,
        "lon_median_sample": lon_median_sample,
        "lat_median_sample": lat_median_sample,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "missing_coord_rows": missing_coord_rows,
        "invalid_coord_rows": invalid_coord_rows,
        "out_of_bounds_rows": out_of_bounds_rows,
    }


def write_quality_tables(
    payload: Dict[str, object],
    tables_dir: Path,
    dataset_label: str,
) -> None:
    """Write map-data quality tables.

    Args:
        payload: Map payload dictionary.
        tables_dir: Output table directory.
        dataset_label: Dataset label for filenames.

    Returns:
        None

    Raises:
        None
    """
    ensure_directory(tables_dir)

    quality_rows = [
        {"metric": "total_rows", "value": str(payload["total_rows"])},
        {"metric": "valid_coordinate_rows", "value": str(payload["valid_rows"])},
        {"metric": "missing_coordinate_rows", "value": str(payload["missing_coord_rows"])},
        {"metric": "invalid_coordinate_rows", "value": str(payload["invalid_coord_rows"])},
        {"metric": "out_of_bounds_rows", "value": str(payload["out_of_bounds_rows"])},
    ]

    write_csv_rows(
        tables_dir / f"map_quality_summary_{dataset_label}.csv",
        ["metric", "value"],
        quality_rows,
    )

    hour_counter: Counter[int] = payload["hour_counter"]  # type: ignore[assignment]
    hour_rows: List[Dict[str, str]] = []
    for hour in range(24):
        hour_rows.append({"hour": str(hour), "collisions": str(hour_counter[hour])})
    write_csv_rows(
        tables_dir / f"map_hourly_distribution_{dataset_label}.csv",
        ["hour", "collisions"],
        hour_rows,
    )


def write_map_report(
    payload: Dict[str, object],
    reports_dir: Path,
    dataset_label: str,
    source_csv: Path,
    focus_hours: List[int],
) -> None:
    """Write map generation report.

    Args:
        payload: Map payload dictionary.
        reports_dir: Output report directory.
        dataset_label: Dataset label.
        source_csv: Source CSV path.
        focus_hours: Focus hour list.

    Returns:
        None

    Raises:
        None
    """
    ensure_directory(reports_dir)

    total_rows = int(payload["total_rows"])
    valid_rows = int(payload["valid_rows"])

    lines = [
        f"Dataset label: {dataset_label}",
        f"Source CSV: {source_csv}",
        f"Rows scanned: {total_rows}",
        f"Valid NYC coordinates: {valid_rows}",
        f"Coordinate retention (%): {100.0 * safe_divide(valid_rows, total_rows):.4f}",
        f"Missing coordinate rows: {payload['missing_coord_rows']}",
        f"Invalid coordinate rows: {payload['invalid_coord_rows']}",
        f"Out-of-bounds rows: {payload['out_of_bounds_rows']}",
        "",
        "Generated map outputs:",
        f"- road_imprint_scatter_{dataset_label}.png",
        f"- collision_density_hexbin_{dataset_label}.png",
        f"- injury_hotspot_overlay_{dataset_label}.png",
        f"- hourly_hexbin_triptych_{dataset_label}.png",
        "",
        f"Focus hours used: {', '.join(str(hour) for hour in focus_hours)}",
    ]
    write_markdown(
        reports_dir / f"map_visualization_report_{dataset_label}.md",
        "Source Map Visualization Report",
        lines,
    )


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two values.

    Args:
        numerator: Numerator value.
        denominator: Denominator value.

    Returns:
        float: Division result or zero.

    Raises:
        None
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    """Run source-CSV map visualization generation.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If processing fails.
    """
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)
    logger = get_logger("exploratory_analysis.source_maps")

    dataset_name = str(config.get("dataset_name", "dataset")).strip() or "dataset"
    source_csv = Path(config["source_csv"]).resolve()
    tables_dir = Path(config["tables_dir"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()
    visual_root = Path(config["visualizations_dir"]).resolve() / "maps"

    injured_threshold = int(config.get("map_hotspot_injured_threshold", "3"))
    random_seed = int(config.get("random_seed", "143"))
    median_sample_size = int(config.get("map_median_sample_size", "250000"))
    focus_hours_text = str(config.get("map_focus_hours", "2,8,17"))
    focus_hours = [int(item.strip()) for item in focus_hours_text.split(",") if item.strip()]
    focus_hours = [hour for hour in focus_hours if 0 <= hour <= 23]
    if not focus_hours:
        focus_hours = [2, 8, 17]

    log_step(logger, "MAP_VIS_START", f"Dataset: {dataset_name}")
    payload = stream_source_coordinates(
        source_csv=source_csv,
        injured_threshold=injured_threshold,
        focus_hours=focus_hours,
        median_sample_size=median_sample_size,
        random_seed=random_seed,
    )

    lon_median_sample: List[float] = payload["lon_median_sample"]  # type: ignore[assignment]
    lat_median_sample: List[float] = payload["lat_median_sample"]  # type: ignore[assignment]
    x_min, x_max, y_min, y_max = clean_window_from_median(lon_median_sample, lat_median_sample, offset=0.32)

    ensure_directory(visual_root)
    plot_road_imprint(
        longitudes=payload["all_lons"],  # type: ignore[arg-type]
        latitudes=payload["all_lats"],  # type: ignore[arg-type]
        output_path=visual_root / f"road_imprint_scatter_{dataset_name}.png",
        title=f"NYC Collision Road Imprint ({dataset_name})",
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    plot_density_hexbin(
        longitudes=payload["all_lons"],  # type: ignore[arg-type]
        latitudes=payload["all_lats"],  # type: ignore[arg-type]
        output_path=visual_root / f"collision_density_hexbin_{dataset_name}.png",
        title=f"NYC Collision Density Hexbin ({dataset_name})",
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    plot_hotspot_overlay(
        longitudes=payload["all_lons"],  # type: ignore[arg-type]
        latitudes=payload["all_lats"],  # type: ignore[arg-type]
        hotspot_lons=payload["hotspot_lons"],  # type: ignore[arg-type]
        hotspot_lats=payload["hotspot_lats"],  # type: ignore[arg-type]
        output_path=visual_root / f"injury_hotspot_overlay_{dataset_name}.png",
        title=f"NYC Collision Hotspots (injury >= {injured_threshold}) ({dataset_name})",
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        injured_threshold=injured_threshold,
    )
    plot_hourly_hexbin_triptych(
        hour_to_lons=payload["hour_to_lons"],  # type: ignore[arg-type]
        hour_to_lats=payload["hour_to_lats"],  # type: ignore[arg-type]
        output_path=visual_root / f"hourly_hexbin_triptych_{dataset_name}.png",
        title=f"NYC Hourly Collision Density ({dataset_name})",
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        focus_hours=focus_hours,
    )

    write_quality_tables(payload, tables_dir, dataset_name)
    write_map_report(payload, reports_dir, dataset_name, source_csv, focus_hours)

    log_step(logger, "MAP_VIS_DONE", f"Valid coordinates: {payload['valid_rows']}")


if __name__ == "__main__":
    main()
