"""Run ST-DBSCAN spatio-temporal clustering on NYC collision records.

What is a cluster?
------------------
Each cluster CN is a specific *geographic hotspot* (a street intersection or
short road segment) that has a consistently dangerous *time-of-day window*.
For example:

    C0  → Atlantic Ave & Flatbush, Brooklyn  — peak 17:20 (5:20 PM)
    C7  → Queens Blvd, Queens               — peak 08:45 (8:45 AM)

Clusters are NOT boroughs.  A borough (Brooklyn, Queens, …) can contain
dozens of clusters.  Each cluster is coloured by its borough on the map so
geography is immediately apparent, and labelled with its exact peak time
(e.g. "08:45" not "AM Rush").

Temporal scope
--------------
The analysis covers the FULL dataset (13+ years, 2012–2026).  Every record is
given a cyclic *hour-of-day* coordinate (0.0 – 23.99).  This means a crash at
17:20 on any Tuesday in any year is considered a temporal neighbour of another
crash at 17:05 on any Thursday.  The question answered is:
    "Which intersections are systematically dangerous at which hours of the
     day, aggregated across ALL years in the dataset?"

This is distinct from a time-series analysis (which would show change over
years).  See run_temporal_yearover_year.py for that.

Algorithm
---------
Two crashes are neighbours when BOTH hold simultaneously:
    spatial_dist(p, q)  ≤  eps1   (metres — same intersection / block)
    cyclic_hour_dist(p, q)  ≤  eps2   (hours — same time window)

Distance is implemented as max(spatial/eps1, hour/eps2) — the Chebyshev unit
ball is exactly the ST-DBSCAN neighbourhood.

Outputs
-------
tables/
    stdbscan_assignments_sample.csv
    stdbscan_cluster_summary.csv
reports/
    stdbscan_clustering_report.md
visualizations/<dataset>/clusters/
    stdbscan_spatial_scatter.png    — NYC map, colour = borough, size ∝ cluster size
    stdbscan_temporal_density.png   — per-cluster day×hour heatmap (exact HH:MM labels)
    stdbscan_cluster_profiles.png   — normalised hourly profile per cluster
    stdbscan_cluster_size_bar.png   — cluster sizes, coloured by borough
"""

from __future__ import annotations

import csv
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_csv_rows, write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory


# ── NYC bounding box ──────────────────────────────────────────────────────────
_NYC_LAT_MIN = 40.40
_NYC_LAT_MAX = 40.95
_NYC_LON_MIN = -74.30
_NYC_LON_MAX = -73.65

# ── Borough definitions + colours ─────────────────────────────────────────────
# Each borough has: (name, lat_lo, lat_hi, lon_lo, lon_hi, hex_colour)
_BOROUGH_DEFS: List[Tuple[str, float, float, float, float, str]] = [
    ("Manhattan",     40.700, 40.882, -74.020, -73.907, "#E53935"),  # red
    ("Brooklyn",      40.570, 40.740, -74.042, -73.833, "#1E88E5"),  # blue
    ("Queens",        40.541, 40.800, -73.962, -73.700, "#43A047"),  # green
    ("Bronx",         40.785, 40.915, -73.933, -73.765, "#FB8C00"),  # orange
    ("Staten Island", 40.477, 40.651, -74.259, -74.034, "#8E24AA"),  # purple
    ("Unknown",       -90.0,  90.0,  -180.0,  180.0,   "#757575"),  # grey fallback
]

_BOROUGH_COLOUR: Dict[str, str] = {b[0]: b[5] for b in _BOROUGH_DEFS}
_BOROUGH_CENTRE: Dict[str, Tuple[float, float]] = {
    "Manhattan":     (40.783, -73.966),
    "Brooklyn":      (40.650, -73.950),
    "Queens":        (40.700, -73.820),
    "Bronx":         (40.845, -73.865),
    "Staten Island": (40.579, -74.152),
}


def borough_for(lat: float, lon: float) -> str:
    """Return borough name for a coordinate (first bbox match, Unknown otherwise).

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        str: Borough name.

    Raises:
        None
    """
    for name, lat_lo, lat_hi, lon_lo, lon_hi, _ in _BOROUGH_DEFS[:-1]:
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            return name
    return "Unknown"


def hour_frac_to_hhmm(hour_frac: float) -> str:
    """Convert fractional hour (e.g. 14.4) to 'HH:MM' string (e.g. '14:24').

    Args:
        hour_frac: Hour as a decimal, e.g. 14.4 means 14 hours 24 minutes.

    Returns:
        str: Time string in HH:MM format.

    Raises:
        None
    """
    h = int(hour_frac) % 24
    m = int(round((hour_frac - int(hour_frac)) * 60)) % 60
    return f"{h:02d}:{m:02d}"


# ─── helpers ──────────────────────────────────────────────────────────────────

def to_float(value: str) -> Optional[float]:
    """Convert text to float, returning None on failure."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: str) -> int:
    """Convert text to int, returning zero on failure."""
    text = (value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def cyclic_hour_distance(h1: float, h2: float) -> float:
    """Shortest cyclic distance between two hour-of-day values (max = 12).

    Args:
        h1: Hour in [0, 24).
        h2: Hour in [0, 24).

    Returns:
        float: Distance in hours, range [0, 12].

    Raises:
        None
    """
    diff = abs(h1 - h2) % 24.0
    return min(diff, 24.0 - diff)


# ── data loading ──────────────────────────────────────────────────────────────

def iter_valid_records(
    feature_csv: Path,
) -> Iterator[Tuple[str, float, float, float, int]]:
    """Yield validated records inside the NYC bbox with a parseable crash time.

    Args:
        feature_csv: Path to the feature-engineered CSV.

    Yields:
        Tuple: (collision_id, lat, lon, hour_frac, day_of_week_0based)

    Raises:
        FileNotFoundError: If feature_csv does not exist.
    """
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    dow_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }

    with feature_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lat = to_float(row.get("LATITUDE", ""))
            lon = to_float(row.get("LONGITUDE", ""))
            if lat is None or lon is None:
                continue
            if not (_NYC_LAT_MIN <= lat <= _NYC_LAT_MAX and
                    _NYC_LON_MIN <= lon <= _NYC_LON_MAX):
                continue

            hour_raw = to_int(row.get("HOUR", ""))
            if not (0 <= hour_raw <= 23):
                continue

            # Sub-hour precision from CRASH TIME
            hour_frac = float(hour_raw)
            crash_time = (row.get("CRASH TIME") or "").strip()
            if crash_time and ":" in crash_time:
                parts = crash_time.split(":")
                if len(parts) >= 2:
                    try:
                        hour_frac = float(parts[0]) + float(parts[1]) / 60.0
                    except ValueError:
                        pass

            dow_name = (row.get("DAY_OF_WEEK") or "").strip()
            dow = dow_map.get(dow_name, -1)
            if dow == -1:
                continue

            collision_id = (row.get("COLLISION_ID") or "").strip()
            yield collision_id, lat, lon, hour_frac, dow


def reservoir_sample(
    feature_csv: Path,
    sample_size: int,
    random_seed: int,
) -> List[Tuple[str, float, float, float, int]]:
    """Uniform random sample via Vitter's reservoir algorithm.

    Args:
        feature_csv: Feature CSV path.
        sample_size: Target sample size.
        random_seed: Reproducibility seed.

    Returns:
        List of (collision_id, lat, lon, hour_frac, dow) tuples.

    Raises:
        ValueError: If sample_size < 1.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")
    rng = random.Random(random_seed)
    reservoir: List[Tuple[str, float, float, float, int]] = []
    n_seen = 0
    for record in iter_valid_records(feature_csv):
        n_seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(record)
        else:
            idx = rng.randint(0, n_seen - 1)
            if idx < sample_size:
                reservoir[idx] = record
    return reservoir


# ── feature matrices ──────────────────────────────────────────────────────────

def make_scaled_feature_matrix(
    records: List[Tuple[str, float, float, float, int]],
    cos_lat_ref: float,
    metres_per_degree: float,
    eps1_m: float,
    eps2_h: float,
) -> "object":
    """(N,3) scaled matrix for Chebyshev DBSCAN.  d_chebyshev≤1 ↔ ST-DBSCAN neighbour.

    Args:
        records: Sampled record tuples.
        cos_lat_ref: cos(lat_reference).
        metres_per_degree: ~111194.
        eps1_m: Spatial epsilon metres.
        eps2_h: Temporal epsilon hours.

    Returns:
        numpy.ndarray shape (N,3).

    Raises:
        ImportError: If numpy unavailable.
    """
    import numpy as np  # type: ignore
    rows = []
    for _, lat, lon, hour_frac, _ in records:
        rows.append([
            lon * cos_lat_ref * metres_per_degree / eps1_m,
            lat * metres_per_degree / eps1_m,
            hour_frac / eps2_h,
        ])
    return np.array(rows, dtype=float)


def make_pairwise_distance_matrix(
    records: List[Tuple[str, float, float, float, int]],
    cos_lat_ref: float,
    metres_per_degree: float,
    eps1_m: float,
    eps2_h: float,
) -> "object":
    """Exact (N,N) ST-DBSCAN distance with correct cyclic-hour arithmetic.

    d(p,q) = max(spatial_m/eps1, cyclic_hour/eps2).

    Args:
        records: Sampled record tuples.
        cos_lat_ref: cos(lat_reference).
        metres_per_degree: ~111194.
        eps1_m: Spatial epsilon metres.
        eps2_h: Temporal epsilon hours.

    Returns:
        numpy.ndarray shape (N,N).

    Raises:
        ImportError: If numpy unavailable.
    """
    import numpy as np  # type: ignore
    lats  = np.array([r[1] for r in records], dtype=float)
    lons  = np.array([r[2] for r in records], dtype=float)
    hours = np.array([r[3] for r in records], dtype=float)
    x_m = lons * cos_lat_ref * metres_per_degree
    y_m = lats * metres_per_degree
    spatial_m = np.sqrt((x_m[:,None]-x_m[None,:])**2 + (y_m[:,None]-y_m[None,:])**2)
    raw_diff  = np.abs(hours[:,None] - hours[None,:])
    cyclic_h  = np.minimum(raw_diff, 24.0 - raw_diff)
    return np.maximum(spatial_m / eps1_m, cyclic_h / eps2_h)


# ── clustering ────────────────────────────────────────────────────────────────

def fit_stdbscan(
    records: List[Tuple[str, float, float, float, int]],
    cos_lat_ref: float,
    metres_per_degree: float,
    eps1_m: float,
    eps2_h: float,
    min_samples: int,
    precompute_threshold: int,
    random_seed: int,
    logger: object,
) -> List[int]:
    """Run ST-DBSCAN; return per-record labels (−1 = noise).

    Args:
        records: Sampled record tuples.
        cos_lat_ref: cos(lat_reference).
        metres_per_degree: ~111194.
        eps1_m: Spatial epsilon metres.
        eps2_h: Temporal epsilon hours.
        min_samples: Core-point threshold.
        precompute_threshold: Use exact matrix below this N.
        random_seed: Passed to sklearn.
        logger: Logger instance.

    Returns:
        List[int]: Cluster labels.

    Raises:
        ImportError: If scikit-learn unavailable.
    """
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
    except ImportError as e:
        raise ImportError("scikit-learn is required for ST-DBSCAN.") from e

    n = len(records)
    if n <= precompute_threshold:
        log_step(logger, "STDBSCAN_MATRIX",
                 f"Exact pairwise matrix {n:,}×{n:,} (cyclic-hour correct) …")
        D = make_pairwise_distance_matrix(
            records, cos_lat_ref, metres_per_degree, eps1_m, eps2_h)
        model = DBSCAN(eps=1.0, min_samples=min_samples,
                       metric="precomputed", n_jobs=-1)
        model.fit(D)
    else:
        log_step(logger, "STDBSCAN_BALLTREE",
                 f"BallTree Chebyshev, N={n:,} …")
        X = make_scaled_feature_matrix(
            records, cos_lat_ref, metres_per_degree, eps1_m, eps2_h)
        model = DBSCAN(eps=1.0, min_samples=min_samples,
                       metric="chebyshev", algorithm="ball_tree", n_jobs=-1)
        model.fit(X)
    return list(map(int, model.labels_))


# ── summary ───────────────────────────────────────────────────────────────────

_DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_cluster_summary(
    records: List[Tuple[str, float, float, float, int]],
    labels: List[int],
    top_n: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Compute per-cluster stats.  peak_time is exact HH:MM, not a vague label.

    Args:
        records: Sampled record tuples.
        labels: Cluster labels.
        top_n: Top clusters to include.

    Returns:
        Tuple: (assignment_rows, summary_rows).

    Raises:
        None
    """
    c_lat:   Dict[int, List[float]] = defaultdict(list)
    c_lon:   Dict[int, List[float]] = defaultdict(list)
    c_hours: Dict[int, List[float]] = defaultdict(list)
    c_dows:  Dict[int, List[int]]   = defaultdict(list)

    for (cid, lat, lon, hf, dow), lbl in zip(records, labels):
        if lbl == -1:
            continue
        c_lat[lbl].append(lat)
        c_lon[lbl].append(lon)
        c_hours[lbl].append(hf)
        c_dows[lbl].append(dow)

    label_counts: Counter[int] = Counter(labels)

    assignment_rows = [
        {
            "COLLISION_ID": cid,
            "LATITUDE":     f"{lat:.6f}",
            "LONGITUDE":    f"{lon:.6f}",
            "HOUR":         f"{hf:.2f}",
            "HOUR_HHMM":    hour_frac_to_hhmm(hf),
            "DAY_OF_WEEK":  _DOW_NAMES[dow] if 0 <= dow <= 6 else "?",
            "CLUSTER_ID":   str(lbl),
        }
        for (cid, lat, lon, hf, dow), lbl in zip(records, labels)
    ]

    top_non_noise = [(l, c) for l, c in label_counts.most_common() if l != -1][:top_n]
    summary_rows: List[Dict[str, str]] = []

    for lbl, cnt in top_non_noise:
        hours   = c_hours[lbl]
        dows    = c_dows[lbl]
        mean_lat = sum(c_lat[lbl]) / len(c_lat[lbl])
        mean_lon = sum(c_lon[lbl]) / len(c_lon[lbl])
        mean_h   = sum(hours) / len(hours)

        # Peak hour = most-common integer hour bin
        peak_h_bin = Counter(int(h) % 24 for h in hours).most_common(1)[0][0]
        # Mean minute within that peak hour
        peak_mins = [
            int((h - int(h)) * 60) for h in hours if int(h) % 24 == peak_h_bin
        ]
        peak_min = int(sum(peak_mins) / len(peak_mins)) if peak_mins else 0
        peak_time_str = f"{peak_h_bin:02d}:{peak_min:02d}"

        peak_dow_idx = Counter(dows).most_common(1)[0][0] if dows else -1
        borough = borough_for(mean_lat, mean_lon)

        summary_rows.append({
            "cluster_id":       str(lbl),
            "collision_count":  str(cnt),
            "center_latitude":  f"{mean_lat:.6f}",
            "center_longitude": f"{mean_lon:.6f}",
            "mean_hour_decimal":f"{mean_h:.2f}",
            "mean_hour_hhmm":   hour_frac_to_hhmm(mean_h),
            "peak_hour_hhmm":   peak_time_str,
            "peak_day_of_week": _DOW_NAMES[peak_dow_idx] if 0 <= peak_dow_idx <= 6 else "?",
            "borough":          borough,
        })

    return assignment_rows, summary_rows


# ── visualisations ────────────────────────────────────────────────────────────

def save_spatial_scatter(
    records: List[Tuple[str, float, float, float, int]],
    labels: List[int],
    summary_rows: List[Dict[str, str]],
    output_path: Path,
    eps1_m: float,
    eps2_h: float,
) -> None:
    """NYC map: each cluster dot is coloured by borough.  Noise is faint grey.

    Every non-noise point is coloured by its cluster's borough so geography is
    the primary visual grouping.  Cluster centroids are annotated with their
    peak time (HH:MM) and cluster ID.

    Args:
        records: Sampled record tuples.
        labels: Cluster labels.
        summary_rows: Output of build_cluster_summary.
        output_path: Output PNG path.
        eps1_m: eps1 for subtitle.
        eps2_h: eps2 for subtitle.

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

    # cluster → borough colour
    cluster_borough: Dict[int, str] = {}
    for row in summary_rows:
        cluster_borough[int(row["cluster_id"])] = row["borough"]

    # Bucket points by borough
    borough_lons: Dict[str, List[float]] = defaultdict(list)
    borough_lats: Dict[str, List[float]] = defaultdict(list)
    noise_lons: List[float] = []
    noise_lats: List[float] = []

    for (_, lat, lon, _, _), lbl in zip(records, labels):
        if lbl == -1:
            noise_lons.append(lon)
            noise_lats.append(lat)
        else:
            b = cluster_borough.get(lbl, "Unknown")
            borough_lons[b].append(lon)
            borough_lats[b].append(lat)

    fig, ax = plt.subplots(figsize=(11, 10))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    if noise_lons:
        ax.scatter(noise_lons, noise_lats, s=1.0, alpha=0.08,
                   color="#9CA3AF", linewidths=0, rasterized=True)

    borough_order = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "Unknown"]
    for b in borough_order:
        lons_b = borough_lons.get(b, [])
        lats_b = borough_lats.get(b, [])
        if not lons_b:
            continue
        ax.scatter(lons_b, lats_b, s=4, alpha=0.75,
                   color=_BOROUGH_COLOUR[b],
                   linewidths=0, rasterized=True, label=b)

    # Annotate top-15 cluster centroids with peak time
    for row in summary_rows[:15]:
        lat_c = float(row["center_latitude"])
        lon_c = float(row["center_longitude"])
        ax.annotate(
            f"C{row['cluster_id']}\n{row['peak_hour_hhmm']}",
            xy=(lon_c, lat_c),
            fontsize=5.5,
            color="white",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="#00000099", ec="none"),
        )

    ax.set_xlim(_NYC_LON_MIN, _NYC_LON_MAX)
    ax.set_ylim(_NYC_LAT_MIN, _NYC_LAT_MAX)

    # Borough ghost labels
    for name, (blat, blon) in _BOROUGH_CENTRE.items():
        ax.text(blon, blat, name, color=_BOROUGH_COLOUR[name],
                alpha=0.25, fontsize=11, ha="center", va="center",
                style="italic", fontweight="bold")

    ax.set_title(
        "ST-DBSCAN — Spatio-Temporal Collision Clusters\n"
        f"Colour = borough  |  eps₁={eps1_m:.0f} m (spatial)  "
        f"eps₂={eps2_h:.1f} h (cyclic hour-of-day)  |  "
        f"Full dataset 2012–2026, {len(records):,} sampled",
        color="white", fontsize=10, pad=10,
    )
    ax.set_xlabel("Longitude", color="#9CA3AF", fontsize=9)
    ax.set_ylabel("Latitude",  color="#9CA3AF", fontsize=9)
    ax.tick_params(colors="#9CA3AF", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#374151")

    handles = [
        mpatches.Patch(color=_BOROUGH_COLOUR[b], label=b)
        for b in borough_order if b in borough_lons
    ]
    handles.append(mpatches.Patch(color="#9CA3AF", alpha=0.4, label="Noise"))
    ax.legend(handles=handles, loc="lower left", fontsize=9,
              facecolor="#1F2937", edgecolor="#374151", labelcolor="white",
              title="Borough", title_fontsize=9)

    n_noise = sum(1 for l in labels if l == -1)
    n_total = len(labels)
    ax.text(0.99, 0.01,
            f"Noise: {n_noise:,}/{n_total:,} ({100*n_noise/n_total:.0f}%)\n"
            f"Clusters cover the FULL 2012–2026 period",
            transform=ax.transAxes, color="#9CA3AF", fontsize=7.5,
            ha="right", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_temporal_density_heatmap(
    records: List[Tuple[str, float, float, float, int]],
    labels: List[int],
    summary_rows: List[Dict[str, str]],
    output_path: Path,
    top_n_clusters: int = 8,
) -> None:
    """Day × hour heatmap per cluster.  Axes show exact HH:MM labels.

    Title of each panel shows: cluster ID, borough, peak time, and crash count.

    Args:
        records: Sampled record tuples.
        labels: Cluster labels.
        summary_rows: Output of build_cluster_summary.
        output_path: Output PNG path.
        top_n_clusters: Number of clusters to display.

    Returns:
        None

    Raises:
        ImportError: If matplotlib unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.colors as mcolors  # type: ignore
    except ImportError as e:
        raise ImportError("matplotlib required.") from e

    ensure_directory(output_path.parent)

    top_rows = summary_rows[:top_n_clusters]
    if not top_rows:
        return

    label_counts: Counter[int] = Counter(lbl for lbl in labels if lbl != -1)
    top_labels = [int(r["cluster_id"]) for r in top_rows]
    n_show = len(top_labels)
    n_cols  = min(4, n_show)
    n_rows  = math.ceil(n_show / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.6, n_rows * 3.2),
                             squeeze=False)
    fig.patch.set_facecolor("#F9FAFB")

    hour_labels = [f"{h:02d}:00" for h in range(24)]

    for ax_idx, (lbl, row) in enumerate(zip(top_labels, top_rows)):
        grid = [[0] * 24 for _ in range(7)]
        for (_, _, _, hf, dow), cluster_lbl in zip(records, labels):
            if cluster_lbl == lbl:
                grid[dow][int(hf) % 24] += 1

        r_idx, c_idx = divmod(ax_idx, n_cols)
        ax = axes[r_idx][c_idx]

        borough = row["borough"]
        base_col = _BOROUGH_COLOUR.get(borough, "#757575")
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"cm{lbl}", ["#F9FAFB", base_col], N=256)

        im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("crash count", fontsize=6)
        cbar.ax.tick_params(labelsize=6)

        # Vertical dashed line at peak hour
        peak_h = int(row["peak_hour_hhmm"].split(":")[0])
        ax.axvline(x=peak_h, color=base_col, linewidth=1.8,
                   linestyle="--", alpha=0.9)

        for sp in ax.spines.values():
            sp.set_linewidth(2.0)
            sp.set_edgecolor(base_col)

        ax.set_title(
            f"C{lbl} · {borough}\n"
            f"Peak: {row['peak_hour_hhmm']}  Mean: {row['mean_hour_hhmm']}  "
            f"n={label_counts[lbl]:,}",
            fontsize=8, pad=4, color="#111827",
        )
        ax.set_xticks(range(0, 24, 4))
        ax.set_xticklabels(
            [f"{h:02d}:00" for h in range(0, 24, 4)],
            fontsize=6, rotation=30, ha="right",
        )
        ax.set_yticks(range(7))
        ax.set_yticklabels(_DOW_NAMES, fontsize=7)

    for extra in range(n_show, n_rows * n_cols):
        r_idx, c_idx = divmod(extra, n_cols)
        axes[r_idx][c_idx].set_visible(False)

    fig.suptitle(
        "ST-DBSCAN — Collision Density by Day-of-Week × Hour-of-Day  (top clusters)\n"
        "Each cell = how many collisions from that cluster fell in that day/hour combination "
        "across the FULL 2012–2026 dataset.\n"
        "Colour = borough.  Dashed line = peak hour.  Cluster titles show exact peak and mean time (HH:MM).",
        fontsize=9, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def save_cluster_profiles_plot(
    records: List[Tuple[str, float, float, float, int]],
    labels: List[int],
    summary_rows: List[Dict[str, str]],
    output_path: Path,
    top_n: int = 12,
) -> None:
    """Normalised hourly profile lines, one per cluster, coloured by borough.

    X-axis is exact HH:MM.  Each line shows what fraction of that cluster's
    crashes occur at each hour.  Clusters from the same borough share colour.

    Args:
        records: Sampled record tuples.
        labels: Cluster labels.
        summary_rows: Output of build_cluster_summary.
        output_path: Output PNG path.
        top_n: How many clusters to overlay.

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

    top_labels = [int(r["cluster_id"]) for r in summary_rows[:top_n]]
    if not top_labels:
        return

    hour_profiles: Dict[int, List[int]] = {l: [0]*24 for l in top_labels}
    label_set = set(top_labels)
    for (_, _, _, hf, _), lbl in zip(records, labels):
        if lbl in label_set:
            hour_profiles[lbl][int(hf) % 24] += 1

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F3F4F6")

    x = list(range(24))
    x_labels = [f"{h:02d}:00" for h in range(24)]

    for row in summary_rows[:top_n]:
        lbl     = int(row["cluster_id"])
        counts  = hour_profiles[lbl]
        total   = sum(counts)
        if total == 0:
            continue
        normed  = [c / total for c in counts]
        borough = row["borough"]
        colour  = _BOROUGH_COLOUR.get(borough, "#757575")
        peak_str = row["peak_hour_hhmm"]

        ax.plot(x, normed, color=colour, linewidth=1.8, alpha=0.82,
                label=f"C{lbl} · {borough} peak {peak_str} (n={row['collision_count']})")
        # Mark peak
        peak_h = int(peak_str.split(":")[0])
        ax.scatter([peak_h], [normed[peak_h]], color=colour, s=55, zorder=6,
                   edgecolors="white", linewidths=0.8)

    # Rush-hour bands
    for start, end, txt in [(7, 10, "AM Rush\n07:00–10:00"),
                            (15, 19, "PM Rush\n15:00–19:00")]:
        ax.axvspan(start, end, alpha=0.08, color="#F97316")
        ax.text((start+end)/2, ax.get_ylim()[1]*0.98, txt,
                ha="center", va="top", fontsize=7.5, color="#EA580C", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Hour of Day", fontsize=10)
    ax.set_ylabel("Fraction of cluster's crashes at this hour", fontsize=10)
    ax.set_title(
        "ST-DBSCAN — Normalised Hourly Crash Profile per Cluster  (full 2012–2026 dataset)\n"
        "Each line = one cluster.  Height = fraction of that cluster's crashes at each hour.\n"
        "Colour = borough.  Dots mark exact peak hour.  Shaded bands = typical NYC rush hours.",
        fontsize=9, pad=8,
    )
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.grid(axis="x", linewidth=0.3, alpha=0.25)

    # De-duplicate legend entries by borough
    seen_boroughs: set = set()
    handles, leg_labels = [], []
    for line, label_text in zip(ax.lines, [
        f"C{r['cluster_id']} · {r['borough']} peak {r['peak_hour_hhmm']} (n={r['collision_count']})"
        for r in summary_rows[:top_n]
    ]):
        handles.append(line)
        leg_labels.append(label_text)

    ax.legend(handles, leg_labels,
              fontsize=7, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              facecolor="white", edgecolor="#D1D5DB",
              title="Cluster ID · Borough · Peak time", title_fontsize=7.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def save_cluster_size_bar(
    summary_rows: List[Dict[str, str]],
    label_counts: Counter,
    n_noise: int,
    n_total: int,
    output_path: Path,
) -> None:
    """Horizontal bar chart: cluster sizes, coloured by borough, log scale.

    Each bar is annotated with borough, peak time (HH:MM), and crash count.

    Args:
        summary_rows: Cluster summaries.
        label_counts: Full label counter.
        n_noise: Number of noise points.
        n_total: Total sample size.
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
    if not summary_rows:
        return

    rows   = sorted(summary_rows, key=lambda r: int(r["collision_count"]), reverse=True)
    labels = [f"C{r['cluster_id']}" for r in rows]
    counts = [int(r["collision_count"]) for r in rows]
    colours= [_BOROUGH_COLOUR.get(r["borough"], "#757575") for r in rows]
    annots = [f"  {r['borough']}  peak {r['peak_hour_hhmm']}" for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(4, len(rows) * 0.45)))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F3F4F6")

    bars = ax.barh(list(range(len(rows))), counts,
                   color=colours, edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()

    for bar, count, annot in zip(bars, counts, annots):
        ax.text(count * 1.06,
                bar.get_y() + bar.get_height() / 2,
                f" {count:,}{annot}",
                va="center", ha="left", fontsize=7.5, color="#1F2937")

    ax.set_xlabel("Collision count in sample  (log scale)", fontsize=9)
    ax.set_title(
        "ST-DBSCAN — Cluster Sizes  (full 2012–2026 dataset, colour = borough)\n"
        f"Total sample: {n_total:,}  |  "
        f"Noise: {n_noise:,} ({100*n_noise/n_total:.0f}%)  |  "
        f"Clustered: {n_total-n_noise:,} ({100*(n_total-n_noise)/n_total:.0f}%)\n"
        "Bar annotations: borough and exact peak hour (HH:MM)",
        fontsize=9, pad=8,
    )

    handles = [
        mpatches.Patch(color=_BOROUGH_COLOUR[b], label=b)
        for b in ["Manhattan","Brooklyn","Queens","Bronx","Staten Island","Unknown"]
        if any(r["borough"] == b for r in rows)
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              facecolor="white", edgecolor="#D1D5DB", title="Borough")
    ax.grid(axis="x", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run ST-DBSCAN and write all outputs."""
    config_path = parse_config_path(sys.argv[1:])
    config      = load_config(config_path)
    logger      = get_logger("clustering.stdbscan")

    feature_csv  = Path(config["feature_csv"]).resolve()
    tables_dir   = Path(config["tables_dir"]).resolve()
    reports_dir  = Path(config["reports_dir"]).resolve()
    visual_root  = Path(config["visualizations_dir"]).resolve()

    random_seed           = int(config.get("random_seed",                "143"))
    sample_limit          = int(config.get("stdbscan_sample_limit",      "250000"))
    eps1_metres           = float(config.get("stdbscan_eps1_metres",     "150.0"))
    eps2_hours            = float(config.get("stdbscan_eps2_hours",      "1.5"))
    min_samples           = int(config.get("stdbscan_min_samples",       "5"))
    top_n                 = int(config.get("stdbscan_top_n",             "20"))
    latitude_reference    = float(config.get("cluster_latitude_reference","40.7128"))
    precompute_threshold  = int(config.get("stdbscan_precompute_threshold","30000"))

    metres_per_degree = 111_194.0
    cos_lat_ref       = math.cos(math.radians(latitude_reference))

    log_step(logger, "STDBSCAN_START",
             f"Dataset: {config.get('dataset_name','unknown')}  "
             f"(full multi-year — hour-of-day aggregated across ALL years)")
    log_step(logger, "STDBSCAN_PARAMS",
             f"eps1={eps1_metres}m  eps2={eps2_hours}h (cyclic)  "
             f"min_samples={min_samples}  n={sample_limit:,}")

    log_step(logger, "STDBSCAN_SAMPLE", "Reservoir-sampling (NYC bbox only) …")
    records = reservoir_sample(feature_csv, sample_limit, random_seed)
    n_records = len(records)
    if n_records == 0:
        raise RuntimeError("No valid records found.")
    log_step(logger, "STDBSCAN_SAMPLE_DONE", f"Sampled: {n_records:,}")

    log_step(logger, "STDBSCAN_FIT", "Running ST-DBSCAN …")
    labels = fit_stdbscan(
        records, cos_lat_ref, metres_per_degree,
        eps1_metres, eps2_hours, min_samples,
        precompute_threshold, random_seed, logger,
    )

    lc = Counter(labels)
    n_clusters = sum(1 for l in lc if l != -1)
    n_noise    = lc.get(-1, 0)
    noise_pct  = 100.0 * n_noise / n_records
    log_step(logger, "STDBSCAN_FIT_DONE",
             f"Clusters: {n_clusters}  Noise: {n_noise:,} ({noise_pct:.1f}%)")

    assignment_rows, summary_rows = build_cluster_summary(records, labels, top_n)

    ensure_directory(tables_dir)
    ensure_directory(reports_dir)
    ensure_directory(visual_root / "clusters")

    write_csv_rows(
        tables_dir / "stdbscan_assignments_sample.csv",
        ["COLLISION_ID","LATITUDE","LONGITUDE","HOUR","HOUR_HHMM",
         "DAY_OF_WEEK","CLUSTER_ID"],
        assignment_rows,
    )
    write_csv_rows(
        tables_dir / "stdbscan_cluster_summary.csv",
        ["cluster_id","collision_count","center_latitude","center_longitude",
         "mean_hour_decimal","mean_hour_hhmm","peak_hour_hhmm",
         "peak_day_of_week","borough"],
        summary_rows,
    )

    plot_lines: List[str] = []
    try:
        save_spatial_scatter(
            records, labels, summary_rows,
            visual_root / "clusters" / "stdbscan_spatial_scatter.png",
            eps1_metres, eps2_hours,
        )
        plot_lines.append("stdbscan_spatial_scatter.png")

        if n_clusters > 0:
            save_temporal_density_heatmap(
                records, labels, summary_rows,
                visual_root / "clusters" / "stdbscan_temporal_density.png",
                top_n_clusters=min(8, n_clusters),
            )
            plot_lines.append("stdbscan_temporal_density.png")

            save_cluster_profiles_plot(
                records, labels, summary_rows,
                visual_root / "clusters" / "stdbscan_cluster_profiles.png",
                top_n=min(12, n_clusters),
            )
            plot_lines.append("stdbscan_cluster_profiles.png")

            save_cluster_size_bar(
                summary_rows, lc, n_noise, n_records,
                visual_root / "clusters" / "stdbscan_cluster_size_bar.png",
            )
            plot_lines.append("stdbscan_cluster_size_bar.png")

    except ImportError:
        plot_lines.append("Skipped — matplotlib not installed.")

    report_lines = [
        f"Dataset: {config.get('dataset_name','unknown')}",
        f"Temporal scope: FULL dataset 2012–2026 (hour-of-day aggregated across all years)",
        f"Algorithm: ST-DBSCAN",
        f"Neighbourhood: spatial ≤ {eps1_metres} m  AND  cyclic_hour ≤ {eps2_hours} h",
        f"Sample: {n_records:,}  |  Clusters: {n_clusters}  |  "
        f"Noise: {n_noise:,} ({noise_pct:.1f}%)",
        "",
        "Top clusters:",
    ]
    for row in summary_rows[:15]:
        report_lines.append(
            f"  C{row['cluster_id']:>3}: {int(row['collision_count']):>6} crashes  "
            f"{row['borough']:<15}  peak {row['peak_hour_hhmm']}  "
            f"mean {row['mean_hour_hhmm']}  peak_day {row['peak_day_of_week']}"
        )
    report_lines += ["", "Outputs:"] + [f"- {l}" for l in plot_lines]

    write_markdown(
        reports_dir / "stdbscan_clustering_report.md",
        "ST-DBSCAN Temporal Clustering Report",
        report_lines,
    )
    log_step(logger, "STDBSCAN_DONE",
             f"Clusters: {n_clusters}  Noise: {noise_pct:.1f}%  → {tables_dir}")


if __name__ == "__main__":
    main()
