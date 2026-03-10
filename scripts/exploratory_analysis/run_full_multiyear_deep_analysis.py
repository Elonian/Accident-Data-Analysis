"""Generate deeper multi-year analysis outputs for the official full dataset."""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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
from utils.plot_utils import save_bar_plot, save_heatmap, save_line_plot
from utils.stats_utils import safe_divide
from utils.time_utils import combine_date_time, parse_crash_date

ERA_ORDER = ["PRE_2020", "PANDEMIC_2020_2021", "POST_2022_PLUS", "UNKNOWN"]
RUSH_HOURS = {7, 8, 16, 17, 18}


def to_int(value: str) -> int:
    """Convert text to int safely.

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


def era_from_date(crash_day: date | None) -> str:
    """Map crash date into project-era label.

    Args:
        crash_day: Crash date.

    Returns:
        str: Era label.

    Raises:
        None
    """
    if crash_day is None:
        return "UNKNOWN"
    if crash_day < date(2020, 1, 1):
        return "PRE_2020"
    if crash_day <= date(2021, 12, 31):
        return "PANDEMIC_2020_2021"
    return "POST_2022_PLUS"


def normalize_vehicle_type(raw_vehicle: str) -> str:
    """Normalize high-cardinality vehicle text.

    Args:
        raw_vehicle: Raw vehicle text.

    Returns:
        str: Normalized vehicle class.

    Raises:
        None
    """
    text = (raw_vehicle or "").strip().upper()
    if not text or text == "UNKNOWN":
        return "UNKNOWN"
    if any(token in text for token in ["SEDAN", "4 DR", "4DR", "PASSENGER", "CONV", "COUPE"]):
        return "SEDAN_CAR"
    if any(token in text for token in ["SUV", "SPORT UTILITY", "STATION WAGON", "JEEP"]):
        return "SUV"
    if any(token in text for token in ["TAXI", "LIVERY", "UBER", "LYFT", "FHV"]):
        return "FOR_HIRE"
    if any(token in text for token in ["PICK", "TRUCK", "TRACTOR", "DUMP", "TOW", "SEMI"]):
        return "TRUCK"
    if "VAN" in text:
        return "VAN"
    if "BUS" in text:
        return "BUS"
    if any(token in text for token in ["MOTORCYCLE", "MOTORBIKE", "MOPED", "SCOOTER"]):
        return "MOTORCYCLE_SCOOTER"
    if any(token in text for token in ["BICYCLE", "BIKE", "E-BIKE"]):
        return "BICYCLE"
    return text[:24]


def normalize_factor_category(raw_factor: str) -> str:
    """Normalize contributing factor text into policy-relevant categories.

    Args:
        raw_factor: Raw factor text.

    Returns:
        str: Factor category label.

    Raises:
        None
    """
    text = (raw_factor or "").strip().upper()
    if not text or text == "UNSPECIFIED":
        return "UNSPECIFIED"
    if "UNSAFE SPEED" in text:
        return "UNSAFE_SPEED"
    if "ALCOHOL" in text or "DRUG" in text or "DRUNK" in text:
        return "ALCOHOL_OR_DRUG"
    if "INATTENTION" in text or "DISTRACTION" in text or "CELL PHONE" in text:
        return "INATTENTION_DISTRACTION"
    if "FOLLOWING TOO CLOSELY" in text:
        return "FOLLOWING_TOO_CLOSELY"
    if "FAILURE TO YIELD" in text:
        return "FAILURE_TO_YIELD"
    return "OTHER"


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return nth weekday date for a given year/month.

    Args:
        year: Year value.
        month: Month value.
        weekday: Python weekday number (Monday=0).
        n: Occurrence number (1-based).

    Returns:
        date: Computed calendar date.

    Raises:
        ValueError: If input occurrence is invalid.
    """
    if n <= 0:
        raise ValueError("Occurrence must be positive.")

    first_day = date(year, month, 1)
    shift = (weekday - first_day.weekday()) % 7
    day_number = 1 + shift + (n - 1) * 7
    return date(year, month, day_number)


def us_dst_dates_for_year(year: int) -> Dict[str, date]:
    """Return U.S. DST spring/fall transition dates for year.

    Args:
        year: Year value.

    Returns:
        Dict[str, date]: Mapping with spring and fall transition dates.

    Raises:
        None
    """
    spring = nth_weekday_of_month(year, 3, weekday=6, n=2)  # second Sunday in March
    fall = nth_weekday_of_month(year, 11, weekday=6, n=1)  # first Sunday in November
    return {"spring": spring, "fall": fall}


@dataclass
class Aggregates:
    """Container for streaming multi-year aggregations."""

    rows: int
    years_seen: set[int]
    year_collisions: Counter[int]
    year_injury_collisions: Counter[int]
    year_fatal_collisions: Counter[int]
    year_person_injured: Counter[int]
    year_person_killed: Counter[int]
    year_unsafe_speed: Counter[int]
    era_collisions: Counter[str]
    era_fatal_collisions: Counter[str]
    era_person_killed: Counter[str]
    rush_collisions: Counter[str]
    rush_fatal_collisions: Counter[str]
    rush_person_killed: Counter[str]
    weekend_factor: Counter[str]
    weekday_factor: Counter[str]
    vehicle_collisions: Counter[str]
    vehicle_person_injured: Counter[str]
    vehicle_person_killed: Counter[str]
    vehicle_ped_harm: Counter[str]
    vehicle_cyc_harm: Counter[str]
    era_hour_collision: Dict[str, Counter[int]]
    dst_collision: Counter[str]
    dst_fatal_collisions: Counter[str]
    dst_person_killed: Counter[str]
    dst_early_morning: Counter[str]
    dst_evening_ped_harm: Counter[str]


def initialize_aggregates() -> Aggregates:
    """Initialize empty aggregate container.

    Args:
        None

    Returns:
        Aggregates: Empty aggregation structure.

    Raises:
        None
    """
    return Aggregates(
        rows=0,
        years_seen=set(),
        year_collisions=Counter(),
        year_injury_collisions=Counter(),
        year_fatal_collisions=Counter(),
        year_person_injured=Counter(),
        year_person_killed=Counter(),
        year_unsafe_speed=Counter(),
        era_collisions=Counter(),
        era_fatal_collisions=Counter(),
        era_person_killed=Counter(),
        rush_collisions=Counter(),
        rush_fatal_collisions=Counter(),
        rush_person_killed=Counter(),
        weekend_factor=Counter(),
        weekday_factor=Counter(),
        vehicle_collisions=Counter(),
        vehicle_person_injured=Counter(),
        vehicle_person_killed=Counter(),
        vehicle_ped_harm=Counter(),
        vehicle_cyc_harm=Counter(),
        era_hour_collision=defaultdict(Counter),
        dst_collision=Counter(),
        dst_fatal_collisions=Counter(),
        dst_person_killed=Counter(),
        dst_early_morning=Counter(),
        dst_evening_ped_harm=Counter(),
    )


def update_dst_windows(
    agg: Aggregates,
    crash_day: date | None,
    hour: int | None,
    fatal_collision: int,
    persons_killed: int,
    ped_harm_value: int,
) -> None:
    """Update DST event-window metrics.

    Args:
        agg: Aggregate object.
        crash_day: Crash date.
        hour: Crash hour.
        fatal_collision: Fatal collision binary value.
        persons_killed: Persons killed value.
        ped_harm_value: Pedestrian harm weighted score.

    Returns:
        None

    Raises:
        None
    """
    if crash_day is None:
        return

    dst_dates = us_dst_dates_for_year(crash_day.year)
    spring = dst_dates["spring"]
    fall = dst_dates["fall"]

    if spring - date.resolution * 7 <= crash_day <= spring - date.resolution:
        window = "spring_pre"
    elif spring + date.resolution <= crash_day <= spring + date.resolution * 7:
        window = "spring_post"
    elif fall - date.resolution * 7 <= crash_day <= fall - date.resolution:
        window = "fall_pre"
    elif fall + date.resolution <= crash_day <= fall + date.resolution * 7:
        window = "fall_post"
    else:
        return

    agg.dst_collision[window] += 1
    agg.dst_fatal_collisions[window] += fatal_collision
    agg.dst_person_killed[window] += persons_killed

    if hour is not None and 4 <= hour <= 8:
        agg.dst_early_morning[window] += 1
    if hour is not None and 17 <= hour <= 20:
        agg.dst_evening_ped_harm[window] += ped_harm_value


def aggregate_feature_data(feature_csv: Path) -> Aggregates:
    """Aggregate full multi-year feature data in streaming mode.

    Args:
        feature_csv: Feature dataset path.

    Returns:
        Aggregates: Aggregated multi-year metrics.

    Raises:
        FileNotFoundError: If feature dataset does not exist.
    """
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    agg = initialize_aggregates()

    with feature_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            agg.rows += 1

            crash_day = parse_crash_date(row.get("CRASH DATE", ""))
            crash_dt = combine_date_time(row.get("CRASH DATE", ""), row.get("CRASH TIME", ""))
            year = crash_day.year if crash_day is not None else None
            hour = crash_dt.hour if crash_dt is not None else None

            any_injury = to_int(row.get("ANY_INJURY", "0"))
            fatal_collision = to_int(row.get("FATAL_COLLISION", "0"))
            persons_injured = to_int(row.get("NUMBER OF PERSONS INJURED", "0"))
            persons_killed = to_int(row.get("NUMBER OF PERSONS KILLED", "0"))
            ped_harm = to_int(row.get("NUMBER OF PEDESTRIANS INJURED", "0")) + (5 * to_int(row.get("NUMBER OF PEDESTRIANS KILLED", "0")))
            cyc_harm = to_int(row.get("NUMBER OF CYCLIST INJURED", "0")) + (5 * to_int(row.get("NUMBER OF CYCLIST KILLED", "0")))

            era = era_from_date(crash_day)
            factor_category = normalize_factor_category(row.get("CONTRIBUTING FACTOR VEHICLE 1", ""))
            vehicle_type = normalize_vehicle_type(row.get("VEHICLE TYPE CODE 1", ""))
            is_weekend = to_int(row.get("IS_WEEKEND", "0"))
            rush_bucket = "RUSH_HOUR" if hour in RUSH_HOURS else "NON_RUSH"

            agg.era_collisions[era] += 1
            agg.era_fatal_collisions[era] += fatal_collision
            agg.era_person_killed[era] += persons_killed

            agg.rush_collisions[rush_bucket] += 1
            agg.rush_fatal_collisions[rush_bucket] += fatal_collision
            agg.rush_person_killed[rush_bucket] += persons_killed

            if is_weekend == 1:
                agg.weekend_factor[factor_category] += 1
            else:
                agg.weekday_factor[factor_category] += 1

            agg.vehicle_collisions[vehicle_type] += 1
            agg.vehicle_person_injured[vehicle_type] += persons_injured
            agg.vehicle_person_killed[vehicle_type] += persons_killed
            agg.vehicle_ped_harm[vehicle_type] += ped_harm
            agg.vehicle_cyc_harm[vehicle_type] += cyc_harm

            if hour is not None:
                agg.era_hour_collision[era][hour] += 1

            if year is not None:
                agg.years_seen.add(year)
                agg.year_collisions[year] += 1
                agg.year_injury_collisions[year] += any_injury
                agg.year_fatal_collisions[year] += fatal_collision
                agg.year_person_injured[year] += persons_injured
                agg.year_person_killed[year] += persons_killed
                if factor_category == "UNSAFE_SPEED":
                    agg.year_unsafe_speed[year] += 1

            update_dst_windows(
                agg=agg,
                crash_day=crash_day,
                hour=hour,
                fatal_collision=fatal_collision,
                persons_killed=persons_killed,
                ped_harm_value=ped_harm,
            )

    return agg


def build_year_kpi_rows(agg: Aggregates) -> List[Dict[str, str]]:
    """Build yearly KPI rows.

    Args:
        agg: Aggregation object.

    Returns:
        List[Dict[str, str]]: Year KPI rows.

    Raises:
        None
    """
    output: List[Dict[str, str]] = []
    for year in sorted(agg.year_collisions):
        collisions = agg.year_collisions[year]
        output.append(
            {
                "year": str(year),
                "collisions": str(collisions),
                "injury_collisions": str(agg.year_injury_collisions[year]),
                "fatal_collisions": str(agg.year_fatal_collisions[year]),
                "fatal_rate_percent": f"{100.0 * safe_divide(agg.year_fatal_collisions[year], collisions):.4f}",
                "persons_injured_per_1k_collisions": f"{1000.0 * safe_divide(agg.year_person_injured[year], collisions):.4f}",
                "persons_killed_per_10k_collisions": f"{10000.0 * safe_divide(agg.year_person_killed[year], collisions):.4f}",
                "unsafe_speed_share_percent": f"{100.0 * safe_divide(agg.year_unsafe_speed[year], collisions):.4f}",
            }
        )
    return output


def build_era_kpi_rows(agg: Aggregates) -> List[Dict[str, str]]:
    """Build era-level KPI rows.

    Args:
        agg: Aggregation object.

    Returns:
        List[Dict[str, str]]: Era KPI rows.

    Raises:
        None
    """
    output: List[Dict[str, str]] = []
    for era in ERA_ORDER:
        collisions = agg.era_collisions[era]
        if collisions == 0:
            continue
        output.append(
            {
                "era": era,
                "collisions": str(collisions),
                "fatal_collisions": str(agg.era_fatal_collisions[era]),
                "fatal_rate_percent": f"{100.0 * safe_divide(agg.era_fatal_collisions[era], collisions):.4f}",
                "persons_killed_per_10k_collisions": f"{10000.0 * safe_divide(agg.era_person_killed[era], collisions):.4f}",
            }
        )
    return output


def build_rush_rows(agg: Aggregates) -> List[Dict[str, str]]:
    """Build rush-hour vs non-rush summary rows.

    Args:
        agg: Aggregation object.

    Returns:
        List[Dict[str, str]]: Rush bucket rows.

    Raises:
        None
    """
    output: List[Dict[str, str]] = []
    for bucket in ["RUSH_HOUR", "NON_RUSH"]:
        collisions = agg.rush_collisions[bucket]
        output.append(
            {
                "bucket": bucket,
                "collisions": str(collisions),
                "fatal_collisions": str(agg.rush_fatal_collisions[bucket]),
                "fatal_rate_percent": f"{100.0 * safe_divide(agg.rush_fatal_collisions[bucket], collisions):.4f}",
                "persons_killed_per_10k_collisions": f"{10000.0 * safe_divide(agg.rush_person_killed[bucket], collisions):.4f}",
            }
        )
    return output


def build_weekend_weekday_factor_rows(agg: Aggregates) -> List[Dict[str, str]]:
    """Build weekend vs weekday factor-share rows.

    Args:
        agg: Aggregation object.

    Returns:
        List[Dict[str, str]]: Factor share rows.

    Raises:
        None
    """
    total_weekend = sum(agg.weekend_factor.values())
    total_weekday = sum(agg.weekday_factor.values())
    merged = agg.weekend_factor + agg.weekday_factor

    output: List[Dict[str, str]] = []
    for factor, _ in merged.most_common(8):
        weekend_count = agg.weekend_factor[factor]
        weekday_count = agg.weekday_factor[factor]
        output.append(
            {
                "factor_category": factor,
                "weekend_count": str(weekend_count),
                "weekday_count": str(weekday_count),
                "weekend_share_percent": f"{100.0 * safe_divide(weekend_count, total_weekend):.4f}",
                "weekday_share_percent": f"{100.0 * safe_divide(weekday_count, total_weekday):.4f}",
            }
        )
    return output


def build_vehicle_rows(agg: Aggregates) -> List[Dict[str, str]]:
    """Build vehicle severity rows.

    Args:
        agg: Aggregation object.

    Returns:
        List[Dict[str, str]]: Vehicle severity rows.

    Raises:
        None
    """
    output: List[Dict[str, str]] = []
    for vehicle, collisions in agg.vehicle_collisions.most_common(18):
        output.append(
            {
                "vehicle_type": vehicle,
                "collisions": str(collisions),
                "persons_injured": str(agg.vehicle_person_injured[vehicle]),
                "persons_killed": str(agg.vehicle_person_killed[vehicle]),
                "persons_injured_per_1k_collisions": f"{1000.0 * safe_divide(agg.vehicle_person_injured[vehicle], collisions):.4f}",
                "persons_killed_per_10k_collisions": f"{10000.0 * safe_divide(agg.vehicle_person_killed[vehicle], collisions):.4f}",
            }
        )
    return output


def build_lethality_rows(agg: Aggregates) -> List[Dict[str, str]]:
    """Build lethality matrix rows by vehicle class.

    Args:
        agg: Aggregation object.

    Returns:
        List[Dict[str, str]]: Lethality rows.

    Raises:
        None
    """
    output: List[Dict[str, str]] = []
    for vehicle, collisions in agg.vehicle_collisions.most_common(12):
        output.append(
            {
                "vehicle_type": vehicle,
                "collisions": str(collisions),
                "pedestrian_harm_per_1k_collisions": f"{1000.0 * safe_divide(agg.vehicle_ped_harm[vehicle], collisions):.4f}",
                "cyclist_harm_per_1k_collisions": f"{1000.0 * safe_divide(agg.vehicle_cyc_harm[vehicle], collisions):.4f}",
            }
        )
    return output


def build_dst_rows(agg: Aggregates) -> List[Dict[str, str]]:
    """Build DST window comparison rows.

    Args:
        agg: Aggregation object.

    Returns:
        List[Dict[str, str]]: DST window rows.

    Raises:
        None
    """
    output: List[Dict[str, str]] = []
    for window in ["spring_pre", "spring_post", "fall_pre", "fall_post"]:
        collisions = agg.dst_collision[window]
        output.append(
            {
                "window": window,
                "collisions": str(collisions),
                "fatal_collisions": str(agg.dst_fatal_collisions[window]),
                "fatal_rate_percent": f"{100.0 * safe_divide(agg.dst_fatal_collisions[window], collisions):.4f}",
                "persons_killed_per_10k_collisions": f"{10000.0 * safe_divide(agg.dst_person_killed[window], collisions):.4f}",
                "early_morning_collision_share_percent": f"{100.0 * safe_divide(agg.dst_early_morning[window], collisions):.4f}",
                "evening_ped_harm_per_1k_collisions": f"{1000.0 * safe_divide(agg.dst_evening_ped_harm[window], collisions):.4f}",
            }
        )
    return output


def build_era_hour_rows(agg: Aggregates) -> List[Dict[str, str]]:
    """Build era-hour collision rows for heatmap and export.

    Args:
        agg: Aggregation object.

    Returns:
        List[Dict[str, str]]: Era-hour rows.

    Raises:
        None
    """
    output: List[Dict[str, str]] = []
    for era in ERA_ORDER:
        for hour in range(24):
            output.append(
                {
                    "era": era,
                    "hour": str(hour),
                    "collisions": str(agg.era_hour_collision[era][hour]),
                }
            )
    return output


def build_dst_did_panel_rows(feature_csv: Path) -> List[Dict[str, str]]:
    """Build compact aggregated DST DiD panel (season/year/pre-post/treated-control).

    Args:
        feature_csv: Feature dataset path.

    Returns:
        List[Dict[str, str]]: Aggregated panel rows.

    Raises:
        FileNotFoundError: If feature dataset does not exist.
    """
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    # Hour-bin assumptions for DiD setup.
    treated_by_season = {
        "spring": set(range(4, 9)),
        "fall": set(range(17, 21)),
    }
    control_hours = set(range(10, 15))

    panel_counts: Dict[Tuple[str, int, str, str], int] = Counter()
    panel_fatals: Dict[Tuple[str, int, str, str], int] = Counter()

    with feature_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            crash_day = parse_crash_date(row.get("CRASH DATE", ""))
            crash_dt = combine_date_time(row.get("CRASH DATE", ""), row.get("CRASH TIME", ""))
            if crash_day is None or crash_dt is None:
                continue

            hour = crash_dt.hour
            year = crash_day.year
            dst_dates = us_dst_dates_for_year(year)
            season = ""
            window_side = ""

            spring = dst_dates["spring"]
            fall = dst_dates["fall"]
            if spring - date.resolution * 7 <= crash_day <= spring - date.resolution:
                season = "spring"
                window_side = "pre"
            elif spring + date.resolution <= crash_day <= spring + date.resolution * 7:
                season = "spring"
                window_side = "post"
            elif fall - date.resolution * 7 <= crash_day <= fall - date.resolution:
                season = "fall"
                window_side = "pre"
            elif fall + date.resolution <= crash_day <= fall + date.resolution * 7:
                season = "fall"
                window_side = "post"
            else:
                continue

            if hour in treated_by_season[season]:
                hour_group = "treated_hours"
            elif hour in control_hours:
                hour_group = "control_hours"
            else:
                continue

            key = (season, year, window_side, hour_group)
            panel_counts[key] += 1
            panel_fatals[key] += to_int(row.get("FATAL_COLLISION", "0"))

    complete_years_by_season: Dict[str, List[int]] = {"spring": [], "fall": []}
    for season in ["spring", "fall"]:
        years = sorted({key[1] for key in panel_counts if key[0] == season})
        for year in years:
            required_keys = [
                (season, year, "pre", "treated_hours"),
                (season, year, "post", "treated_hours"),
                (season, year, "pre", "control_hours"),
                (season, year, "post", "control_hours"),
            ]
            if any(panel_counts[key] <= 0 for key in required_keys):
                continue
            complete_years_by_season[season].append(year)

    output: List[Dict[str, str]] = []
    for season in ["spring", "fall"]:
        for year in complete_years_by_season[season]:
            for window_side in ["pre", "post"]:
                for hour_group in ["treated_hours", "control_hours"]:
                    key = (season, year, window_side, hour_group)
                    collisions = panel_counts[key]
                    fatal_collisions = panel_fatals[key]
                    post = 1 if window_side == "post" else 0
                    treated = 1 if hour_group == "treated_hours" else 0
                    output.append(
                        {
                            "season": season,
                            "year": str(year),
                            "window_side": window_side,
                            "post": str(post),
                            "treated": str(treated),
                            "hour_group": hour_group,
                            "collisions": str(collisions),
                            "fatal_collisions": str(fatal_collisions),
                            "fatal_rate_percent": f"{100.0 * safe_divide(fatal_collisions, collisions):.6f}",
                        }
                    )

    return output


def run_dst_did_logistic(panel_rows: List[Dict[str, str]], season: str) -> Dict[str, str]:
    """Fit minimal DiD-style logistic interaction for one season.

    Args:
        panel_rows: Aggregated panel rows.
        season: Season label ("spring" or "fall").

    Returns:
        Dict[str, str]: Logistic interaction outputs.

    Raises:
        None
    """
    season_rows = [row for row in panel_rows if row.get("season") == season]
    if not season_rows:
        return {"coef_post_treated": "0.000000", "odds_ratio_post_treated": "1.000000"}

    x_rows: List[List[float]] = []
    y_rows: List[int] = []

    for row in season_rows:
        post = int(float(row.get("post", "0") or 0))
        treated = int(float(row.get("treated", "0") or 0))
        interaction = post * treated
        collisions = int(float(row.get("collisions", "0") or 0))
        fatals = int(float(row.get("fatal_collisions", "0") or 0))
        non_fatals = max(0, collisions - fatals)

        x_feature = [float(post), float(treated), float(interaction)]
        for _ in range(fatals):
            x_rows.append(x_feature)
            y_rows.append(1)
        for _ in range(non_fatals):
            x_rows.append(x_feature)
            y_rows.append(0)

    if len(set(y_rows)) < 2:
        return {"coef_post_treated": "0.000000", "odds_ratio_post_treated": "1.000000"}

    try:
        import numpy as np  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except ImportError:
        return {"coef_post_treated": "0.000000", "odds_ratio_post_treated": "1.000000"}

    x = np.array(x_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    model = LogisticRegression(max_iter=400, class_weight="balanced")
    model.fit(x, y)

    coef = float(model.coef_[0][2])
    odds_ratio = float(math.exp(coef))
    return {
        "coef_post_treated": f"{coef:.6f}",
        "odds_ratio_post_treated": f"{odds_ratio:.6f}",
    }


def build_dst_did_results(panel_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build per-season DiD-style result rows using 2x2 rates + logistic interaction.

    Args:
        panel_rows: Aggregated panel rows.

    Returns:
        List[Dict[str, str]]: Season-level DiD outputs.

    Raises:
        None
    """
    output: List[Dict[str, str]] = []
    for season in ["spring", "fall"]:
        season_rows = [row for row in panel_rows if row.get("season") == season]
        if not season_rows:
            continue

        totals: Dict[Tuple[int, int], Dict[str, int]] = {
            (1, 0): {"collisions": 0, "fatals": 0},  # treated pre
            (1, 1): {"collisions": 0, "fatals": 0},  # treated post
            (0, 0): {"collisions": 0, "fatals": 0},  # control pre
            (0, 1): {"collisions": 0, "fatals": 0},  # control post
        }

        for row in season_rows:
            treated = int(float(row.get("treated", "0") or 0))
            post = int(float(row.get("post", "0") or 0))
            collisions = int(float(row.get("collisions", "0") or 0))
            fatals = int(float(row.get("fatal_collisions", "0") or 0))
            totals[(treated, post)]["collisions"] += collisions
            totals[(treated, post)]["fatals"] += fatals

        treated_pre_rate = 100.0 * safe_divide(totals[(1, 0)]["fatals"], totals[(1, 0)]["collisions"])
        treated_post_rate = 100.0 * safe_divide(totals[(1, 1)]["fatals"], totals[(1, 1)]["collisions"])
        control_pre_rate = 100.0 * safe_divide(totals[(0, 0)]["fatals"], totals[(0, 0)]["collisions"])
        control_post_rate = 100.0 * safe_divide(totals[(0, 1)]["fatals"], totals[(0, 1)]["collisions"])
        did_effect_pp = (treated_post_rate - treated_pre_rate) - (control_post_rate - control_pre_rate)

        logistic_result = run_dst_did_logistic(panel_rows, season)
        n_observations = sum(int(float(row.get("collisions", "0") or 0)) for row in season_rows)

        output.append(
            {
                "season": season,
                "n_observations": str(n_observations),
                "treated_pre_fatal_rate_percent": f"{treated_pre_rate:.6f}",
                "treated_post_fatal_rate_percent": f"{treated_post_rate:.6f}",
                "control_pre_fatal_rate_percent": f"{control_pre_rate:.6f}",
                "control_post_fatal_rate_percent": f"{control_post_rate:.6f}",
                "did_effect_pp": f"{did_effect_pp:.6f}",
                "coef_post_treated": logistic_result["coef_post_treated"],
                "odds_ratio_post_treated": logistic_result["odds_ratio_post_treated"],
            }
        )

    return output


def save_dst_did_figure(results_rows: List[Dict[str, str]], visual_root: Path) -> None:
    """Save simple season-level DST DiD effect chart.

    Args:
        results_rows: Season-level DiD results.
        visual_root: Visualization root directory.

    Returns:
        None

    Raises:
        None
    """
    if not results_rows:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return

    labels = [row["season"] for row in results_rows]
    values = [float(row["did_effect_pp"]) for row in results_rows]
    colors = ["#1f77b4" if val >= 0 else "#d62728" for val in values]

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, values, color=colors, alpha=0.85)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_title("DST DiD Effect by Season (Fatal Rate pp)")
    axis.set_xlabel("Season")
    axis.set_ylabel("DiD Effect (percentage points)")
    plt.tight_layout()
    plt.savefig(visual_root / "core" / "dst_did_effect_multiyear.png", dpi=180)
    plt.close(figure)


def write_dst_did_report(results_rows: List[Dict[str, str]], reports_dir: Path) -> None:
    """Write dedicated DST DiD regression report.

    Args:
        results_rows: Season-level DiD results.
        reports_dir: Reports directory path.

    Returns:
        None

    Raises:
        None
    """
    lines = [
        "Scope: full multi-year DST DiD-style extension using compact aggregated panel.",
        "Design:",
        "- Separate season models: spring and fall.",
        "- Outcome: fatal collision binary signal aggregated to 2x2 panel rates.",
        "- Treated hours: spring 04-08, fall 17-20.",
        "- Control hours: 10-14 (both seasons).",
        "- Windows: 7-day pre and 7-day post around DST date, transition day excluded.",
        "",
        "Results by season:",
    ]

    for row in results_rows:
        lines.append(f"- Season: {row['season']}")
        lines.append(f"  n_observations: {row['n_observations']}")
        lines.append(f"  treated_pre_fatal_rate_percent: {row['treated_pre_fatal_rate_percent']}")
        lines.append(f"  treated_post_fatal_rate_percent: {row['treated_post_fatal_rate_percent']}")
        lines.append(f"  control_pre_fatal_rate_percent: {row['control_pre_fatal_rate_percent']}")
        lines.append(f"  control_post_fatal_rate_percent: {row['control_post_fatal_rate_percent']}")
        lines.append(f"  did_effect_pp: {row['did_effect_pp']}")
        lines.append(f"  coef_post_treated: {row['coef_post_treated']}")
        lines.append(f"  odds_ratio_post_treated: {row['odds_ratio_post_treated']}")

    lines.append("")
    lines.append("Output artifacts:")
    lines.append("- tables/dst_did_panel_multiyear.csv")
    lines.append("- tables/dst_did_regression_results_multiyear.csv")
    lines.append("- core/dst_did_effect_multiyear.png")

    write_markdown(
        reports_dir / "dst_did_regression_report.md",
        "DST DiD Regression Report",
        lines,
    )


def write_outputs(agg: Aggregates, config: Dict[str, str]) -> None:
    """Write deep-analysis tables, visuals, and report.

    Args:
        agg: Aggregation object.
        config: Runtime config dictionary.

    Returns:
        None

    Raises:
        None
    """
    tables_dir = Path(config["tables_dir"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()
    visual_root = Path(config["visualizations_dir"]).resolve()
    feature_csv = Path(config["feature_csv"]).resolve()

    ensure_directory(tables_dir)
    ensure_directory(reports_dir)
    ensure_directory(visual_root / "core")
    ensure_directory(visual_root / "heatmaps")

    year_rows = build_year_kpi_rows(agg)
    era_rows = build_era_kpi_rows(agg)
    rush_rows = build_rush_rows(agg)
    factor_rows = build_weekend_weekday_factor_rows(agg)
    vehicle_rows = build_vehicle_rows(agg)
    lethality_rows = build_lethality_rows(agg)
    dst_rows = build_dst_rows(agg)
    dst_did_panel_rows = build_dst_did_panel_rows(feature_csv)
    dst_did_results_rows = build_dst_did_results(dst_did_panel_rows)
    era_hour_rows = build_era_hour_rows(agg)

    write_csv_rows(
        tables_dir / "yearly_kpi_multiyear.csv",
        [
            "year",
            "collisions",
            "injury_collisions",
            "fatal_collisions",
            "fatal_rate_percent",
            "persons_injured_per_1k_collisions",
            "persons_killed_per_10k_collisions",
            "unsafe_speed_share_percent",
        ],
        year_rows,
    )
    write_csv_rows(
        tables_dir / "era_kpi_multiyear.csv",
        ["era", "collisions", "fatal_collisions", "fatal_rate_percent", "persons_killed_per_10k_collisions"],
        era_rows,
    )
    write_csv_rows(
        tables_dir / "rush_vs_nonrush_multiyear.csv",
        ["bucket", "collisions", "fatal_collisions", "fatal_rate_percent", "persons_killed_per_10k_collisions"],
        rush_rows,
    )
    write_csv_rows(
        tables_dir / "weekend_weekday_focus_factors_multiyear.csv",
        ["factor_category", "weekend_count", "weekday_count", "weekend_share_percent", "weekday_share_percent"],
        factor_rows,
    )
    write_csv_rows(
        tables_dir / "vehicle_severity_multiyear.csv",
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
        tables_dir / "lethality_matrix_multiyear.csv",
        ["vehicle_type", "collisions", "pedestrian_harm_per_1k_collisions", "cyclist_harm_per_1k_collisions"],
        lethality_rows,
    )
    write_csv_rows(
        tables_dir / "dst_event_windows_multiyear.csv",
        [
            "window",
            "collisions",
            "fatal_collisions",
            "fatal_rate_percent",
            "persons_killed_per_10k_collisions",
            "early_morning_collision_share_percent",
            "evening_ped_harm_per_1k_collisions",
        ],
        dst_rows,
    )
    write_csv_rows(
        tables_dir / "era_hour_collision_matrix_multiyear.csv",
        ["era", "hour", "collisions"],
        era_hour_rows,
    )
    write_csv_rows(
        tables_dir / "dst_did_panel_multiyear.csv",
        [
            "season",
            "year",
            "window_side",
            "post",
            "treated",
            "hour_group",
            "collisions",
            "fatal_collisions",
            "fatal_rate_percent",
        ],
        dst_did_panel_rows,
    )
    write_csv_rows(
        tables_dir / "dst_did_regression_results_multiyear.csv",
        [
            "season",
            "n_observations",
            "treated_pre_fatal_rate_percent",
            "treated_post_fatal_rate_percent",
            "control_pre_fatal_rate_percent",
            "control_post_fatal_rate_percent",
            "did_effect_pp",
            "coef_post_treated",
            "odds_ratio_post_treated",
        ],
        dst_did_results_rows,
    )

    years = [row["year"] for row in year_rows]
    yearly_collisions = [float(row["collisions"]) for row in year_rows]
    yearly_fatal_rate = [float(row["fatal_rate_percent"]) for row in year_rows]
    yearly_unsafe_speed = [float(row["unsafe_speed_share_percent"]) for row in year_rows]

    save_line_plot(
        years,
        yearly_collisions,
        "Multi-year Collision Trend (Official Full Data)",
        "Year",
        "Collisions",
        visual_root / "core" / "yearly_collision_trend_multiyear.png",
    )
    save_line_plot(
        years,
        yearly_fatal_rate,
        "Multi-year Fatal Collision Rate (%)",
        "Year",
        "Fatal Rate (%)",
        visual_root / "core" / "yearly_fatal_rate_multiyear.png",
    )
    save_line_plot(
        years,
        yearly_unsafe_speed,
        "Unsafe Speed Share by Year (%)",
        "Year",
        "Share (%)",
        visual_root / "core" / "unsafe_speed_share_yearly_multiyear.png",
    )

    save_bar_plot(
        [row["bucket"] for row in rush_rows],
        [float(row["fatal_rate_percent"]) for row in rush_rows],
        "Fatal Collision Rate: Rush vs Non-Rush",
        "Bucket",
        "Fatal Rate (%)",
        visual_root / "core" / "rush_nonrush_fatal_rate_multiyear.png",
    )
    save_bar_plot(
        [row["vehicle_type"] for row in vehicle_rows[:12]],
        [float(row["persons_killed_per_10k_collisions"]) for row in vehicle_rows[:12]],
        "Vehicle Severity: Persons Killed per 10k Collisions",
        "Vehicle Type",
        "Killed per 10k",
        visual_root / "core" / "vehicle_killed_per_10k_multiyear.png",
    )

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore

        # Grouped factor shares by day type.
        labels = [row["factor_category"] for row in factor_rows]
        weekend_values = [float(row["weekend_share_percent"]) for row in factor_rows]
        weekday_values = [float(row["weekday_share_percent"]) for row in factor_rows]
        positions = np.arange(len(labels))

        figure, axis = plt.subplots(figsize=(11, 5))
        width = 0.4
        axis.bar(positions - width / 2, weekend_values, width=width, label="Weekend", color="#1f77b4")
        axis.bar(positions + width / 2, weekday_values, width=width, label="Weekday", color="#ff7f0e")
        axis.set_title("Weekend vs Weekday Factor Share (Multi-year)")
        axis.set_xlabel("Factor Category")
        axis.set_ylabel("Share (%)")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.legend()
        plt.tight_layout()
        plt.savefig(visual_root / "core" / "factor_weekend_vs_weekday_multiyear.png", dpi=180)
        plt.close(figure)

        # DST comparison chart.
        dst_labels = [row["window"] for row in dst_rows]
        dst_fatal = [float(row["fatal_rate_percent"]) for row in dst_rows]
        dst_early = [float(row["early_morning_collision_share_percent"]) for row in dst_rows]
        positions = np.arange(len(dst_labels))

        figure, axis_left = plt.subplots(figsize=(10, 5))
        axis_left.bar(positions, dst_fatal, width=0.6, color="#6a3d9a", alpha=0.8)
        axis_left.set_xticks(positions)
        axis_left.set_xticklabels(dst_labels, rotation=20, ha="right")
        axis_left.set_ylabel("Fatal Rate (%)", color="#6a3d9a")
        axis_left.tick_params(axis="y", labelcolor="#6a3d9a")
        axis_left.set_xlabel("DST Window")

        axis_right = axis_left.twinx()
        axis_right.plot(positions, dst_early, color="#e31a1c", marker="o", linewidth=2.0)
        axis_right.set_ylabel("Early Morning Collision Share (%)", color="#e31a1c")
        axis_right.tick_params(axis="y", labelcolor="#e31a1c")
        plt.title("DST Window Signal: Fatal Rate vs Early-Morning Share")
        plt.tight_layout()
        plt.savefig(visual_root / "core" / "dst_window_comparison_multiyear.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass

    hour_labels = [str(hour) for hour in range(24)]
    heatmap_matrix: List[List[float]] = []
    for era in ERA_ORDER:
        heatmap_matrix.append([float(agg.era_hour_collision[era][hour]) for hour in range(24)])
    save_heatmap(
        heatmap_matrix,
        hour_labels,
        ERA_ORDER,
        "Era vs Hour Collision Heatmap (Multi-year)",
        "Hour",
        "Era",
        visual_root / "heatmaps" / "era_hour_collision_heatmap_multiyear.png",
    )

    lethality_vehicle_labels = [row["vehicle_type"] for row in lethality_rows]
    lethality_matrix = [
        [float(row["pedestrian_harm_per_1k_collisions"]) for row in lethality_rows],
        [float(row["cyclist_harm_per_1k_collisions"]) for row in lethality_rows],
    ]
    save_heatmap(
        lethality_matrix,
        lethality_vehicle_labels,
        ["Pedestrian Harm", "Cyclist Harm"],
        "Lethality Matrix (Multi-year)",
        "Vehicle Type",
        "VRU Harm Type",
        visual_root / "heatmaps" / "lethality_matrix_multiyear_heatmap.png",
    )
    save_dst_did_figure(dst_did_results_rows, visual_root)
    write_dst_did_report(dst_did_results_rows, reports_dir)

    report_lines = [
        f"Dataset: {config.get('dataset_name', 'unknown')}",
        f"Rows processed: {agg.rows}",
        f"Years covered: {min(agg.years_seen) if agg.years_seen else 'N/A'} to {max(agg.years_seen) if agg.years_seen else 'N/A'}",
        "",
        "Deep outputs generated:",
        "- yearly_kpi_multiyear.csv",
        "- era_kpi_multiyear.csv",
        "- rush_vs_nonrush_multiyear.csv",
        "- weekend_weekday_focus_factors_multiyear.csv",
        "- vehicle_severity_multiyear.csv",
        "- lethality_matrix_multiyear.csv",
        "- dst_event_windows_multiyear.csv",
        "- era_hour_collision_matrix_multiyear.csv",
        "",
        "Key interpretation pointers:",
        "- Use yearly_kpi_multiyear.csv to compare trend breaks over the full horizon.",
        "- Use unsafe_speed_share_yearly_multiyear.png for pandemic-aftermath signal checks.",
        "- Use rush_vs_nonrush_multiyear.csv and rush_nonrush_fatal_rate_multiyear.png for frequency-vs-severity framing.",
        "- Use dst_event_windows_multiyear.csv for spring/fall transition directional evidence.",
    ]
    write_markdown(
        reports_dir / "deep_multiyear_analysis_report.md",
        "Deep Multi-year Analysis Report",
        report_lines,
    )


def main() -> None:
    """Run deep multi-year exploratory analysis stage.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If stage fails.
    """
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)
    logger = get_logger("exploratory_analysis.deep_multiyear")

    feature_csv = Path(config["feature_csv"]).resolve()
    log_step(logger, "DEEP_EDA_START", f"Dataset: {config.get('dataset_name', 'unknown')}")

    aggregates = aggregate_feature_data(feature_csv)
    write_outputs(aggregates, config)

    log_step(logger, "DEEP_EDA_DONE", f"Rows processed: {aggregates.rows}")


if __name__ == "__main__":
    main()
