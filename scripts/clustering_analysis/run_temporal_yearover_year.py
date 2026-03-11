"""Temporal clustering analysis across years — how collision patterns shift over time.

What this script answers
------------------------
The ST-DBSCAN script answers:
    "Which intersections are dangerous at which time-of-day,
     aggregated across ALL 13 years?"

This script answers:
    "How has the PATTERN of collisions changed year-by-year?
     Which hours/days/boroughs became more or less dangerous over time?
     Did the COVID lockdown create a structural break that persisted?"

Approach: K-Means on yearly temporal profiles
---------------------------------------------
For every (year, borough) combination the dataset contains, we build a
24-bin hourly collision profile and a 7-bin day-of-week profile.  Each
profile is L2-normalised so that volume differences are removed — we cluster
on *shape*, i.e. when collisions happen, not how many.

K-Means (k=4 by default) groups (year, borough) combinations that share a
similar temporal rhythm.  Typical findings:
    - Pre-COVID years (2012–2019) share one cluster: strong bimodal AM/PM
      rush peaks, weekday-heavy.
    - PAUSE 2020 forms its own cluster: flat daytime curve, weekend surge.
    - Reopen 2020–2022 forms a third cluster: PM peak recovers first.
    - Post-2022 forms a fourth: elevated overnight and weekend collisions.

Additionally, for every year the script computes:
    - Per-borough hourly and dow profiles
    - YoY change in fatal rate and injury rate
    - Seasonal month-of-year profiles (monthly collision volume by year)

All raw counts and normalised profiles are written to CSV so downstream
analysis is straightforward.

Outputs
-------
tables/
    yoy_yearly_borough_profiles.csv    — raw + normalised 24h profile per (year, borough)
    yoy_cluster_assignments.csv        — (year, borough) → cluster_id
    yoy_cluster_centroids.csv          — centroid profile per cluster
    yoy_month_of_year_matrix.csv       — collisions per (year, month)
    yoy_severity_by_year.csv           — injuries, fatalities, rates per year
reports/
    yoy_temporal_report.md
visualizations/<dataset>/clusters/
    yoy_hourly_heatmap.png             — heatmap: hour vs year (per borough)
    yoy_dow_heatmap.png                — heatmap: day-of-week vs year
    yoy_profile_cluster_lines.png      — overlaid normalised profiles, coloured by cluster
    yoy_monthly_trend.png              — monthly collision volume by year (line chart)
    yoy_severity_trend.png             — fatal rate and injury rate over years
    yoy_elbow.png                      — inertia vs k (if elbow enabled)
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_csv_rows, write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory


# ── borough colour palette (consistent with ST-DBSCAN script) ─────────────────
_BOROUGH_COLOUR: Dict[str, str] = {
    "Manhattan":     "#E53935",
    "Brooklyn":      "#1E88E5",
    "Queens":        "#43A047",
    "Bronx":         "#FB8C00",
    "Staten Island": "#8E24AA",
    "Unknown":       "#757575",
}

_BOROUGH_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

# Cluster colours for the K-Means groups
_CLUSTER_COLOURS = ["#0D47A1", "#1B5E20", "#B71C1C", "#4A148C",
                    "#E65100", "#006064", "#37474F", "#880E4F"]


# ─── helpers ──────────────────────────────────────────────────────────────────

def to_float(value: str) -> Optional[float]:
    """Convert text to float or None."""
    t = (value or "").strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def to_int(value: str) -> int:
    """Convert text to int or 0."""
    t = (value or "").strip()
    if not t:
        return 0
    try:
        return int(float(t))
    except ValueError:
        return 0


def normalise_l2(vector: List[float]) -> List[float]:
    """L2-normalise a vector; return zeros if norm is zero.

    Args:
        vector: Input numeric list.

    Returns:
        List[float]: Unit-length vector or zero vector.

    Raises:
        None
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    return [v / norm for v in vector]


def borough_from_name(raw: str) -> str:
    """Normalise borough name from raw CSV value.

    Args:
        raw: Raw borough string from the dataset.

    Returns:
        str: Canonical borough name or 'Unknown'.

    Raises:
        None
    """
    mapping = {
        "MANHATTAN":     "Manhattan",
        "BROOKLYN":      "Brooklyn",
        "QUEENS":        "Queens",
        "BRONX":         "Bronx",
        "STATEN ISLAND": "Staten Island",
    }
    return mapping.get((raw or "").strip().upper(), "Unknown")


# ── streaming data accumulation ───────────────────────────────────────────────

def stream_feature_csv(
    feature_csv: Path,
) -> Tuple[
    Dict[Tuple[int, str], List[float]],   # (year, borough) → hourly counts [24]
    Dict[Tuple[int, str], List[float]],   # (year, borough) → dow counts [7]
    Dict[Tuple[int, int], int],           # (year, month) → collision count
    Dict[int, Dict[str, int]],            # year → {collisions, injuries, fatalities}
]:
    """Single streaming pass over feature CSV — accumulate all yearly profiles.

    Args:
        feature_csv: Path to the feature-engineered CSV.

    Returns:
        Tuple of four dictionaries:
            hourly_profiles, dow_profiles, month_matrix, severity_by_year.

    Raises:
        FileNotFoundError: If feature_csv does not exist.
    """
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    dow_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }

    hourly_profiles: Dict[Tuple[int, str], List[float]] = defaultdict(lambda: [0.0] * 24)
    dow_profiles:    Dict[Tuple[int, str], List[float]] = defaultdict(lambda: [0.0] * 7)
    month_matrix:    Dict[Tuple[int, int], int]         = defaultdict(int)
    severity_by_year: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"collisions": 0, "injuries": 0, "fatalities": 0}
    )

    with feature_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            year_str = (row.get("YEAR") or "").strip()
            if not year_str:
                continue
            year = to_int(year_str)
            if year < 2010 or year > 2030:
                continue

            # Borough — prefer the BOROUGH column, fall back to GPS-free
            borough = borough_from_name(row.get("BOROUGH") or "")

            hour = to_int(row.get("HOUR") or "")
            if not (0 <= hour <= 23):
                continue

            dow_name = (row.get("DAY_OF_WEEK") or "").strip()
            dow = dow_map.get(dow_name, -1)
            if dow == -1:
                continue

            # Month from "YYYY-MM" MONTH column
            month_raw = (row.get("MONTH") or "").strip()
            month = -1
            if len(month_raw) >= 7 and "-" in month_raw:
                try:
                    month = int(month_raw.split("-")[1])
                except (ValueError, IndexError):
                    pass

            key = (year, borough)
            hourly_profiles[key][hour] += 1.0
            dow_profiles[key][dow]     += 1.0

            if 1 <= month <= 12:
                month_matrix[(year, month)] += 1

            sev = severity_by_year[year]
            sev["collisions"]  += 1
            sev["injuries"]    += to_int(row.get("NUMBER OF PERSONS INJURED") or "0")
            sev["fatalities"]  += to_int(row.get("NUMBER OF PERSONS KILLED")  or "0")

    return hourly_profiles, dow_profiles, month_matrix, severity_by_year


# ── K-Means on profiles ───────────────────────────────────────────────────────

def run_kmeans(
    keys: List[Tuple[int, str]],
    norm_matrix: "object",
    n_clusters: int,
    n_init: int,
    random_seed: int,
) -> Tuple[List[int], float]:
    """Fit K-Means and return labels and inertia.

    Args:
        keys: Ordered list of (year, borough) keys.
        norm_matrix: (N, D) numpy array of normalised profiles.
        n_clusters: k.
        n_init: Number of initialisations.
        random_seed: Random seed.

    Returns:
        Tuple[List[int], float]: Labels and inertia.

    Raises:
        ImportError: If scikit-learn unavailable.
    """
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except ImportError as e:
        raise ImportError("scikit-learn is required.") from e

    model = KMeans(n_clusters=n_clusters, n_init=n_init,
                   max_iter=500, random_state=random_seed)
    model.fit(norm_matrix)
    return list(map(int, model.labels_)), float(model.inertia_)


def elbow_search(
    norm_matrix: "object",
    k_max: int,
    n_init: int,
    random_seed: int,
) -> List[Tuple[int, float]]:
    """Compute inertia for k = 2 … k_max.

    Args:
        norm_matrix: (N, D) feature matrix.
        k_max: Maximum k to test.
        n_init: K-Means initialisations per k.
        random_seed: Random seed.

    Returns:
        List[Tuple[int, float]]: (k, inertia) pairs.

    Raises:
        ImportError: If scikit-learn unavailable.
    """
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except ImportError as e:
        raise ImportError("scikit-learn is required.") from e

    results = []
    n = len(norm_matrix)
    for k in range(2, min(k_max + 1, n)):
        m = KMeans(n_clusters=k, n_init=n_init, max_iter=300, random_state=random_seed)
        m.fit(norm_matrix)
        results.append((k, float(m.inertia_)))
    return results


# ── visualisations ────────────────────────────────────────────────────────────

def save_hourly_heatmap(
    hourly_profiles: Dict[Tuple[int, str], List[float]],
    years: List[int],
    output_path: Path,
) -> None:
    """Heatmap of normalised hourly collision density: year (rows) × hour (cols).

    One subplot per borough.  Colour intensity = fraction of collisions at
    that hour in that year (normalised so each year sums to 1).

    Args:
        hourly_profiles: Raw hourly counts per (year, borough).
        years: Sorted list of years to include.
        output_path: Output PNG path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as e:
        raise ImportError("matplotlib and numpy required.") from e

    ensure_directory(output_path.parent)
    boroughs = _BOROUGH_ORDER
    n_cols = 3
    n_rows = math.ceil(len(boroughs) / n_cols)
    hour_labels = [f"{h:02d}:00" for h in range(24)]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5.5, n_rows * 3.5),
                             squeeze=False)
    fig.patch.set_facecolor("#F9FAFB")

    for idx, borough in enumerate(boroughs):
        r_idx, c_idx = divmod(idx, n_cols)
        ax = axes[r_idx][c_idx]

        grid = np.zeros((len(years), 24), dtype=float)
        for y_idx, year in enumerate(years):
            counts = hourly_profiles.get((year, borough), [0.0] * 24)
            total  = sum(counts) or 1.0
            grid[y_idx] = [c / total for c in counts]

        base_col = _BOROUGH_COLOUR.get(borough, "#757575")
        import matplotlib.colors as mcolors  # type: ignore
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"cmap_{borough}", ["#FFFFFF", base_col], N=256)

        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Fraction of crashes")

        ax.set_title(f"{borough}", fontsize=10, color=base_col, fontweight="bold")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)],
                           fontsize=6.5, rotation=35, ha="right")
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels([str(y) for y in years], fontsize=7)
        ax.set_xlabel("Hour of Day", fontsize=8)
        ax.set_ylabel("Year", fontsize=8)

    # Hide unused panels
    for extra in range(len(boroughs), n_rows * n_cols):
        r_idx, c_idx = divmod(extra, n_cols)
        axes[r_idx][c_idx].set_visible(False)

    fig.suptitle(
        "Year-over-Year Hourly Collision Profile by Borough\n"
        "Each row = one year.  Each column = one hour.  "
        "Colour intensity = fraction of that year's crashes at that hour.\n"
        "Brighter = more crashes at that hour relative to that year's total.",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def save_dow_heatmap(
    dow_profiles: Dict[Tuple[int, str], List[float]],
    years: List[int],
    output_path: Path,
) -> None:
    """Heatmap of normalised day-of-week collision density: year × day.

    Args:
        dow_profiles: Raw dow counts per (year, borough).
        years: Sorted list of years.
        output_path: Output PNG path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        import matplotlib.colors as mcolors  # type: ignore
    except ImportError as e:
        raise ImportError("matplotlib and numpy required.") from e

    ensure_directory(output_path.parent)
    boroughs  = _BOROUGH_ORDER
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    n_cols = 3
    n_rows = math.ceil(len(boroughs) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.8, n_rows * 3.5),
                             squeeze=False)
    fig.patch.set_facecolor("#F9FAFB")

    for idx, borough in enumerate(boroughs):
        r_idx, c_idx = divmod(idx, n_cols)
        ax = axes[r_idx][c_idx]

        grid = np.zeros((len(years), 7), dtype=float)
        for y_idx, year in enumerate(years):
            counts = dow_profiles.get((year, borough), [0.0] * 7)
            total  = sum(counts) or 1.0
            grid[y_idx] = [c / total for c in counts]

        base_col = _BOROUGH_COLOUR.get(borough, "#757575")
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"cmap_dow_{borough}", ["#FFFFFF", base_col], N=256)

        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Fraction of crashes")

        ax.set_title(f"{borough}", fontsize=10, color=base_col, fontweight="bold")
        ax.set_xticks(range(7))
        ax.set_xticklabels(dow_names, fontsize=8)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels([str(y) for y in years], fontsize=7)
        ax.set_xlabel("Day of Week", fontsize=8)
        ax.set_ylabel("Year", fontsize=8)

    for extra in range(len(boroughs), n_rows * n_cols):
        r_idx, c_idx = divmod(extra, n_cols)
        axes[r_idx][c_idx].set_visible(False)

    fig.suptitle(
        "Year-over-Year Day-of-Week Collision Profile by Borough\n"
        "Each row = one year.  Each column = one day.  "
        "Colour intensity = fraction of that year's crashes on that day.\n"
        "Look for the 2020 row: weekend fraction rose sharply during PAUSE.",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def save_profile_cluster_lines(
    keys: List[Tuple[int, str]],
    hourly_profiles: Dict[Tuple[int, str], List[float]],
    labels: List[int],
    n_clusters: int,
    output_path: Path,
) -> None:
    """Overlay normalised hourly profiles, coloured by K-Means cluster.

    Each line = one (year, borough) combination.  Lines in the same cluster
    share colour.  This makes it easy to see which years share a temporal
    rhythm and which are outliers.

    Args:
        keys: (year, borough) keys, same order as labels.
        hourly_profiles: Raw hourly counts per key.
        labels: K-Means cluster label per key.
        n_clusters: Total number of clusters.
        output_path: Output PNG path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.patches as mpatches  # type: ignore
    except ImportError as e:
        raise ImportError("matplotlib required.") from e

    ensure_directory(output_path.parent)
    x = list(range(24))
    x_labels = [f"{h:02d}:00" for h in range(24)]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F3F4F6")

    # One subplot per cluster
    cluster_profiles: Dict[int, List[List[float]]] = defaultdict(list)
    cluster_meta:     Dict[int, List[str]]          = defaultdict(list)

    for (year, borough), lbl in zip(keys, labels):
        counts = hourly_profiles.get((year, borough), [0.0] * 24)
        total  = sum(counts) or 1.0
        normed = [c / total for c in counts]
        cluster_profiles[lbl].append(normed)
        cluster_meta[lbl].append(f"{year} {borough}")

    for lbl in range(n_clusters):
        colour = _CLUSTER_COLOURS[lbl % len(_CLUSTER_COLOURS)]
        profiles = cluster_profiles[lbl]
        for i, profile in enumerate(profiles):
            ax.plot(x, profile, color=colour, linewidth=1.0, alpha=0.45)
        # Cluster mean
        if profiles:
            mean_profile = [
                sum(p[h] for p in profiles) / len(profiles) for h in range(24)
            ]
            ax.plot(x, mean_profile, color=colour, linewidth=3.0, alpha=0.95,
                    label=f"Cluster {lbl}  ({len(profiles)} year-borough pairs)")

    # Rush bands
    for start, end, txt in [(7, 10, "AM Rush"), (15, 19, "PM Rush")]:
        ax.axvspan(start, end, alpha=0.07, color="#F97316")
        ax.text((start+end)/2, ax.get_ylim()[1]*0.98 if ax.get_ylim()[1] > 0 else 0.1,
                txt, ha="center", va="top", fontsize=8, color="#EA580C", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Hour of Day", fontsize=10)
    ax.set_ylabel("Fraction of that year-borough's crashes at this hour", fontsize=9)
    ax.set_title(
        "Year-over-Year K-Means Temporal Clusters — Normalised Hourly Profile\n"
        "Each thin line = one (year, borough) combination.  "
        "Thick line = cluster centroid (mean profile).\n"
        "Clusters with similar temporal rhythms share a colour.  "
        "Shaded bands = typical NYC rush hours.",
        fontsize=9, pad=8,
    )
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.grid(axis="x", linewidth=0.3, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              facecolor="white", edgecolor="#D1D5DB")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def save_monthly_trend(
    month_matrix: Dict[Tuple[int, int], int],
    years: List[int],
    output_path: Path,
) -> None:
    """Line chart: monthly collision count for each year (one line per year).

    Args:
        month_matrix: {(year, month): collision_count}.
        years: Sorted list of years.
        output_path: Output PNG path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.cm as cm  # type: ignore
    except ImportError as e:
        raise ImportError("matplotlib required.") from e

    ensure_directory(output_path.parent)
    months     = list(range(1, 13))
    month_abbr = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F3F4F6")

    cmap   = cm.get_cmap("tab20", len(years))
    covid_marked = False

    for y_idx, year in enumerate(years):
        counts = [month_matrix.get((year, m), 0) for m in months]
        style  = "--" if year == 2020 else "-"
        lw     = 2.5 if year == 2020 else 1.5
        ax.plot(months, counts, marker="o", markersize=3,
                linewidth=lw, linestyle=style,
                color=cmap(y_idx), label=str(year))

    # COVID annotation
    ax.axvspan(3.22, 6.07, alpha=0.08, color="#EF4444",
               label="_nolegend_")
    ax.text(4.6, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 100,
            "NYC\nPAUSE", ha="center", va="top",
            fontsize=8, color="#EF4444", alpha=0.9)

    ax.set_xticks(months)
    ax.set_xticklabels(month_abbr, fontsize=9)
    ax.set_xlabel("Month of Year", fontsize=10)
    ax.set_ylabel("Collision count", fontsize=10)
    ax.set_title(
        "Monthly Collision Volume by Year — All Boroughs Combined\n"
        "Each line = one calendar year.  Dashed line = 2020.  "
        "Red band = NYC PAUSE period (Mar–Jun 2020).\n"
        "Seasonal pattern: typically peaks in summer, dips in February.",
        fontsize=9, pad=8,
    )
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=7.5, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              facecolor="white", edgecolor="#D1D5DB",
              title="Year", title_fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def save_severity_trend(
    severity_by_year: Dict[int, Dict[str, int]],
    years: List[int],
    output_path: Path,
) -> None:
    """Dual-axis line chart: fatal rate and injury rate over years.

    Args:
        severity_by_year: {year: {collisions, injuries, fatalities}}.
        years: Sorted list of years.
        output_path: Output PNG path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as e:
        raise ImportError("matplotlib required.") from e

    ensure_directory(output_path.parent)
    fatal_rates  = []
    injury_rates = []
    collision_counts = []

    for year in years:
        sev = severity_by_year.get(year, {"collisions":1,"injuries":0,"fatalities":0})
        n   = sev["collisions"] or 1
        fatal_rates.append(10000 * sev["fatalities"] / n)
        injury_rates.append(100  * sev["injuries"]   / n)
        collision_counts.append(sev["collisions"])

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax1.set_facecolor("#F3F4F6")

    ax2 = ax1.twinx()

    ax1.bar(years, collision_counts, color="#CBD5E1", alpha=0.5,
            width=0.6, label="Total collisions (left)")
    ax1.set_ylabel("Total collisions per year", fontsize=9, color="#64748B")
    ax1.tick_params(axis="y", labelcolor="#64748B")

    ax2.plot(years, fatal_rates, color="#DC2626", linewidth=2.5,
             marker="o", markersize=5, label="Fatal rate (per 10,000 crashes)")
    ax2.plot(years, injury_rates, color="#EA580C", linewidth=2.0,
             marker="s", markersize=4, linestyle="--",
             label="Injury rate (% of crashes)")
    ax2.set_ylabel("Rate", fontsize=9, color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")

    # COVID band
    ax1.axvspan(2019.5, 2021.5, alpha=0.07, color="#EF4444")
    ax1.text(2020, max(collision_counts) * 0.92,
             "COVID\n2020", ha="center", fontsize=8,
             color="#EF4444", alpha=0.9)

    ax1.set_xlabel("Year", fontsize=10)
    ax1.set_xticks(years)
    ax1.set_xticklabels([str(y) for y in years], fontsize=8, rotation=30)
    ax1.set_title(
        "Year-over-Year Severity Trends\n"
        "Bars = total collision volume.  "
        "Red line = fatalities per 10,000 crashes.  "
        "Orange dashed = injury rate (%).\n"
        "Note: fatal rate rose during COVID despite fewer total crashes — "
        "empty roads encouraged higher speeds.",
        fontsize=9, pad=8,
    )
    ax1.grid(axis="y", linewidth=0.5, alpha=0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=8, loc="upper right",
               facecolor="white", edgecolor="#D1D5DB")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def save_elbow_plot(
    inertia_curve: List[Tuple[int, float]],
    output_path: Path,
) -> None:
    """Inertia vs k elbow chart.

    Args:
        inertia_curve: List of (k, inertia) pairs.
        output_path: Output PNG path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as e:
        raise ImportError("matplotlib required.") from e

    ensure_directory(output_path.parent)
    ks       = [p[0] for p in inertia_curve]
    inertias = [p[1] for p in inertia_curve]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F3F4F6")
    ax.plot(ks, inertias, marker="o", linewidth=2, color="#1D4ED8")
    ax.set_title(
        "K-Means Elbow Curve — Year-over-Year Temporal Profiles\n"
        "Choose k at the 'elbow' where adding more clusters gives diminishing returns.",
        fontsize=9,
    )
    ax.set_xlabel("Number of clusters (k)", fontsize=10)
    ax.set_ylabel("Inertia", fontsize=10)
    ax.grid(linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── table builders ────────────────────────────────────────────────────────────

def build_profile_rows(
    keys: List[Tuple[int, str]],
    hourly_profiles: Dict[Tuple[int, str], List[float]],
    dow_profiles: Dict[Tuple[int, str], List[float]],
    labels: List[int],
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Build per (year, borough) profile CSV rows.

    Args:
        keys: Ordered (year, borough) keys.
        hourly_profiles: Raw hourly counts.
        dow_profiles: Raw dow counts.
        labels: K-Means cluster label per key.

    Returns:
        Tuple[List[str], List[Dict[str, str]]]: fieldnames and rows.

    Raises:
        None
    """
    hour_raw_cols  = [f"raw_h{h:02d}" for h in range(24)]
    hour_norm_cols = [f"norm_h{h:02d}" for h in range(24)]
    dow_raw_cols   = ["raw_Mon","raw_Tue","raw_Wed","raw_Thu","raw_Fri","raw_Sat","raw_Sun"]
    dow_norm_cols  = ["norm_Mon","norm_Tue","norm_Wed","norm_Thu","norm_Fri","norm_Sat","norm_Sun"]

    fieldnames = (
        ["year", "borough", "total_collisions", "cluster_id"]
        + hour_raw_cols + hour_norm_cols
        + dow_raw_cols  + dow_norm_cols
    )

    rows: List[Dict[str, str]] = []
    for (year, borough), lbl in zip(keys, labels):
        h_raw  = hourly_profiles.get((year, borough), [0.0]*24)
        d_raw  = dow_profiles.get((year, borough), [0.0]*7)
        h_norm = normalise_l2(h_raw)
        d_norm = normalise_l2(d_raw)
        record: Dict[str, str] = {
            "year":              str(year),
            "borough":           borough,
            "total_collisions":  str(int(sum(h_raw))),
            "cluster_id":        str(lbl),
        }
        for col, val in zip(hour_raw_cols,  h_raw):  record[col] = str(int(val))
        for col, val in zip(hour_norm_cols, h_norm): record[col] = f"{val:.6f}"
        for col, val in zip(dow_raw_cols,   d_raw):  record[col] = str(int(val))
        for col, val in zip(dow_norm_cols,  d_norm): record[col] = f"{val:.6f}"
        rows.append(record)

    return fieldnames, rows


def build_severity_rows(
    severity_by_year: Dict[int, Dict[str, int]],
    years: List[int],
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Build severity-by-year CSV rows.

    Args:
        severity_by_year: {year: {collisions, injuries, fatalities}}.
        years: Sorted list of years.

    Returns:
        Tuple[List[str], List[Dict[str, str]]]: fieldnames and rows.

    Raises:
        None
    """
    fieldnames = ["year", "collisions", "injuries", "fatalities",
                  "injury_rate_pct", "fatal_rate_per_10k"]
    rows: List[Dict[str, str]] = []
    for year in years:
        sev = severity_by_year.get(year, {"collisions":0,"injuries":0,"fatalities":0})
        n   = sev["collisions"] or 1
        rows.append({
            "year":                str(year),
            "collisions":          str(sev["collisions"]),
            "injuries":            str(sev["injuries"]),
            "fatalities":          str(sev["fatalities"]),
            "injury_rate_pct":     f"{100*sev['injuries']/n:.2f}",
            "fatal_rate_per_10k":  f"{10000*sev['fatalities']/n:.2f}",
        })
    return fieldnames, rows


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run year-over-year temporal analysis and write all outputs."""
    config_path = parse_config_path(sys.argv[1:])
    config      = load_config(config_path)
    logger      = get_logger("clustering.yoy")

    feature_csv = Path(config["feature_csv"]).resolve()
    tables_dir  = Path(config["tables_dir"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()
    visual_root = Path(config["visualizations_dir"]).resolve()

    random_seed  = int(config.get("random_seed",          "143"))
    n_clusters   = int(config.get("yoy_n_clusters",       "4"))
    n_init       = int(config.get("yoy_n_init",           "25"))
    elbow_max_k  = int(config.get("yoy_elbow_max_k",      "0"))
    profile_type = config.get("yoy_profile_type", "hour").lower()  # "hour" or "dow"

    log_step(logger, "YOY_START",
             f"Dataset: {config.get('dataset_name','unknown')}  "
             f"k={n_clusters}  profile={profile_type}")

    # ── stream data ────────────────────────────────────────────────────────────
    log_step(logger, "YOY_STREAM", "Streaming feature CSV (single pass) …")
    hourly_profiles, dow_profiles, month_matrix, severity_by_year = \
        stream_feature_csv(feature_csv)

    years = sorted({year for year, _ in hourly_profiles.keys()
                    if 2012 <= year <= 2030})
    log_step(logger, "YOY_STREAM_DONE",
             f"Years: {years[0]}–{years[-1]}  "
             f"(year,borough) keys: {len(hourly_profiles)}")

    # ── build profile matrix ───────────────────────────────────────────────────
    keys = sorted(hourly_profiles.keys())
    if profile_type == "dow":
        raw_profiles = {k: dow_profiles.get(k, [0.0]*7) for k in keys}
    else:
        raw_profiles = {k: hourly_profiles.get(k, [0.0]*24) for k in keys}

    try:
        import numpy as np  # type: ignore
        norm_matrix = np.array(
            [normalise_l2(raw_profiles[k]) for k in keys], dtype=float)
    except ImportError:
        raise RuntimeError("numpy is required for year-over-year analysis.")

    # ── elbow search ──────────────────────────────────────────────────────────
    inertia_curve: List[Tuple[int, float]] = []
    if elbow_max_k > 1:
        log_step(logger, "YOY_ELBOW", f"Elbow search up to k={elbow_max_k} …")
        try:
            inertia_curve = elbow_search(norm_matrix, elbow_max_k, 10, random_seed)
        except ImportError:
            log_step(logger, "YOY_ELBOW_SKIP", "scikit-learn unavailable — skipping.")

    # ── K-Means ────────────────────────────────────────────────────────────────
    log_step(logger, "YOY_KMEANS",
             f"Fitting K-Means k={n_clusters} on {len(keys)} (year,borough) profiles …")
    try:
        labels, inertia = run_kmeans(keys, norm_matrix, n_clusters, n_init, random_seed)
    except ImportError as e:
        raise RuntimeError(str(e)) from e

    from collections import Counter as _Counter
    lc = _Counter(labels)
    log_step(logger, "YOY_KMEANS_DONE",
             f"Cluster sizes: {dict(sorted(lc.items()))}")

    # ── tables ─────────────────────────────────────────────────────────────────
    ensure_directory(tables_dir)
    ensure_directory(reports_dir)
    ensure_directory(visual_root / "clusters")

    profile_fieldnames, profile_rows = build_profile_rows(
        keys, hourly_profiles, dow_profiles, labels)
    write_csv_rows(
        tables_dir / "yoy_yearly_borough_profiles.csv",
        profile_fieldnames, profile_rows,
    )

    assignment_rows = [
        {"year": str(k[0]), "borough": k[1], "cluster_id": str(l)}
        for k, l in zip(keys, labels)
    ]
    write_csv_rows(
        tables_dir / "yoy_cluster_assignments.csv",
        ["year", "borough", "cluster_id"],
        assignment_rows,
    )

    # Centroids
    centroid_rows: List[Dict[str, str]] = []
    n_bins = norm_matrix.shape[1]
    for c in range(n_clusters):
        mask = [i for i, l in enumerate(labels) if l == c]
        if not mask:
            continue
        centroid = [float(norm_matrix[mask, b].mean()) for b in range(n_bins)]
        row: Dict[str, str] = {"cluster_id": str(c), "size": str(len(mask))}
        for b, v in enumerate(centroid):
            row[f"bin_{b:02d}"] = f"{v:.6f}"
        centroid_rows.append(row)
    centroid_fieldnames = ["cluster_id","size"] + [f"bin_{b:02d}" for b in range(n_bins)]
    write_csv_rows(
        tables_dir / "yoy_cluster_centroids.csv",
        centroid_fieldnames, centroid_rows,
    )

    # Month matrix
    all_months = sorted(month_matrix.keys())
    month_rows = [
        {"year": str(y), "month": str(m), "collision_count": str(month_matrix[(y,m)])}
        for y, m in all_months
    ]
    write_csv_rows(
        tables_dir / "yoy_month_of_year_matrix.csv",
        ["year", "month", "collision_count"], month_rows,
    )

    # Severity
    sev_fieldnames, sev_rows = build_severity_rows(severity_by_year, years)
    write_csv_rows(
        tables_dir / "yoy_severity_by_year.csv",
        sev_fieldnames, sev_rows,
    )

    # ── visualisations ─────────────────────────────────────────────────────────
    plot_lines: List[str] = []
    try:
        save_hourly_heatmap(
            hourly_profiles, years,
            visual_root / "clusters" / "yoy_hourly_heatmap.png")
        plot_lines.append("yoy_hourly_heatmap.png")

        save_dow_heatmap(
            dow_profiles, years,
            visual_root / "clusters" / "yoy_dow_heatmap.png")
        plot_lines.append("yoy_dow_heatmap.png")

        save_profile_cluster_lines(
            keys, hourly_profiles, labels, n_clusters,
            visual_root / "clusters" / "yoy_profile_cluster_lines.png")
        plot_lines.append("yoy_profile_cluster_lines.png")

        save_monthly_trend(
            month_matrix, years,
            visual_root / "clusters" / "yoy_monthly_trend.png")
        plot_lines.append("yoy_monthly_trend.png")

        save_severity_trend(
            severity_by_year, years,
            visual_root / "clusters" / "yoy_severity_trend.png")
        plot_lines.append("yoy_severity_trend.png")

        if inertia_curve:
            save_elbow_plot(
                inertia_curve,
                visual_root / "clusters" / "yoy_elbow.png")
            plot_lines.append("yoy_elbow.png")

    except ImportError:
        plot_lines.append("Skipped — matplotlib not installed.")

    # ── report ─────────────────────────────────────────────────────────────────
    report_lines = [
        f"Dataset: {config.get('dataset_name','unknown')}",
        f"Years covered: {years[0]}–{years[-1]}",
        f"(year, borough) combinations: {len(keys)}",
        f"Profile type used for clustering: {profile_type}",
        f"K-Means k: {n_clusters}  |  Inertia: {inertia:.4f}",
        "",
        "Cluster membership:",
    ]
    for c in range(n_clusters):
        members = [(k[0], k[1]) for k, l in zip(keys, labels) if l == c]
        report_lines.append(f"  Cluster {c} ({len(members)} members):")
        for year, borough in sorted(members):
            report_lines.append(f"    {year}  {borough}")
    report_lines += [
        "",
        f"Tables written: {len(profile_rows)} profile rows, "
        f"{len(sev_rows)} severity rows, "
        f"{len(month_rows)} month rows",
        "",
        "Outputs:",
    ] + [f"- {l}" for l in plot_lines]

    write_markdown(
        reports_dir / "yoy_temporal_report.md",
        "Year-over-Year Temporal Analysis Report",
        report_lines,
    )
    log_step(logger, "YOY_DONE",
             f"k={n_clusters}  years={years[0]}-{years[-1]}  → {tables_dir}")


if __name__ == "__main__":
    main()
