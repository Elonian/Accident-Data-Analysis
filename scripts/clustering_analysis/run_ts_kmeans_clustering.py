"""Run Time Series K-Means clustering on temporal collision profiles.

This script clusters *time-series profiles* rather than individual records.
Each geographic zone (or optionally each borough) is summarised into a
normalised collision-count vector across hours-of-day, days-of-week, or
months-of-year (controlled by the ``ts_kmeans_profile`` config key).

The zones are built by discretising the city into a regular lat/lon grid;
each occupied cell accumulates hourly (or daily / monthly) collision counts
which are then L2-normalised to form a unit-length profile vector.  K-Means
with DTW-inspired Euclidean distance on the profiles partitions zones by
*when* collisions tend to happen, independent of zone volume.

Because the profiles are small vectors (length 24, 7, or 12), the full
``sklearn.cluster.KMeans`` is used rather than the mini-batch variant.

Outputs
-------
tables/
    ts_kmeans_zone_profiles.csv         — per-zone raw + normalised profile
    ts_kmeans_zone_assignments.csv      — zone → cluster mapping
    ts_kmeans_cluster_summary.csv       — per-cluster centroid profile
reports/
    ts_kmeans_clustering_report.md
visualizations/<dataset>/clusters/
    ts_kmeans_centroid_profiles.png     — time-series plot of each cluster's centroid
    ts_kmeans_zone_map.png              — colour-coded zone map
    ts_kmeans_elbow.png                 — inertia vs k (if elbow search enabled)
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_csv_rows, write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory

# ── profile dimension labels ──────────────────────────────────────────────────

_PROFILE_META: Dict[str, List[str]] = {
    "hour": [str(h) for h in range(24)],
    "dow": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "month": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ],
}

# ─── helpers ──────────────────────────────────────────────────────────────────

def to_float(value: str) -> Optional[float]:
    text = (value or "").strip()
    if not text: return None
    try: return float(text)
    except ValueError: return None

def to_int(value: str) -> int:
    text = (value or "").strip()
    if not text: return 0
    try: return int(float(text))
    except ValueError: return 0

def latlon_to_zone(lat: float, lon: float, lat_min: float, lon_min: float, cell_degrees: float) -> Tuple[int, int]:
    row = int((lat - lat_min) / cell_degrees)
    col = int((lon - lon_min) / cell_degrees)
    return row, col

def normalise_l2(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0: return list(vector)
    return [v / norm for v in vector]

# ── data loading ──────────────────────────────────────────────────────────────

def build_zone_profiles(
    feature_csv: Path,
    profile_type: str,
    cell_degrees: float,
    min_collisions_per_zone: int,
    nyc_bounds: Tuple[float, float, float, float],
) -> Tuple[
    Dict[Tuple[int, int], List[float]],
    Dict[Tuple[int, int], Tuple[float, float]],
    Dict[Tuple[int, int], str],
]:
    if profile_type not in _PROFILE_META:
        raise ValueError(f"Unknown profile_type '{profile_type}'")

    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    n_bins = len(_PROFILE_META[profile_type])
    lat_min, lat_max, lon_min, lon_max = nyc_bounds

    zone_raw = defaultdict(lambda: [0.0] * n_bins)
    zone_geo = defaultdict(lambda: [0.0, 0.0, 0.0])
    zone_b_counts = defaultdict(lambda: defaultdict(int))

    with feature_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lat = to_float(row.get("LATITUDE", ""))
            lon = to_float(row.get("LONGITUDE", ""))
            if lat is None or lon is None: continue
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max): continue

            if profile_type == "hour":
                bin_idx = to_int(row.get("HOUR", ""))
            elif profile_type == "dow":
                dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
                bin_idx = dow_map.get((row.get("DAY_OF_WEEK") or "").strip(), -1)
            else: # month
                month_str = (row.get("MONTH") or "").strip()
                try: bin_idx = int(month_str.split("-")[1]) - 1
                except: continue

            if not (0 <= bin_idx < n_bins): continue

            zone_key = latlon_to_zone(lat, lon, lat_min, lon_min, cell_degrees)
            zone_raw[zone_key][bin_idx] += 1.0
            zone_b_counts[zone_key][(row.get("BOROUGH") or "UNKNOWN").strip().title()] += 1
            geo = zone_geo[zone_key]
            geo[0], geo[1], geo[2] = geo[0]+lat, geo[1]+lon, geo[2]+1.0

    zone_profiles, zone_centres, zone_boroughs = {}, {}, {}
    for zone_key, counts in zone_raw.items():
        if sum(counts) < min_collisions_per_zone: continue
        zone_profiles[zone_key] = counts
        geo = zone_geo[zone_key]
        zone_centres[zone_key] = (geo[0] / geo[2], geo[1] / geo[2])
        zone_boroughs[zone_key] = max(zone_b_counts[zone_key], key=zone_b_counts[zone_key].get)

    return zone_profiles, zone_centres, zone_boroughs

# ── K-Means ───────────────────────────────────────────────────────────────────

def run_kmeans_on_profiles(zone_keys, norm_profiles, n_clusters, random_seed, n_init):
    import numpy as np
    from sklearn.cluster import KMeans
    X = np.array([norm_profiles[k] for k in zone_keys], dtype=float)
    model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=n_init, max_iter=400)
    model.fit(X)
    return list(map(int, model.labels_))

def compute_cluster_centroids(zone_keys, norm_profiles, labels, n_clusters, n_bins):
    sums = [[0.0] * n_bins for _ in range(n_clusters)]
    counts = [0] * n_clusters
    for zone_key, label in zip(zone_keys, labels):
        for i, v in enumerate(norm_profiles[zone_key]):
            sums[label][i] += v
        counts[label] += 1
    return [[s / max(1, counts[i]) for s in sums[i]] for i in range(n_clusters)]

# ── visualisations ─────────────────────────────────────────────────────────────

def save_centroid_profiles_plot(
    centroids, bin_labels, profile_type, n_zones_per_cluster, cluster_boroughs, output_path
):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    ensure_directory(output_path.parent)
    
    n = len(centroids)
    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5), squeeze=False)
    cmap = cm.get_cmap("tab10", n)
    
    profile_names = {"hour": "Hour of Day", "dow": "Day of Week", "month": "Month of Year"}
    x_label = profile_names.get(profile_type, profile_type.capitalize())
    
    for idx, (centroid, ax) in enumerate(zip(centroids, axes.flatten())):
        ax.plot(range(len(bin_labels)), centroid, marker="o", markersize=4, color=cmap(idx))
        ax.fill_between(range(len(bin_labels)), centroid, alpha=0.15, color=cmap(idx))
        
        title = f"Cluster {idx} ({n_zones_per_cluster[idx]} zones)\nTop: {cluster_boroughs[idx]}"
        ax.set_title(title, fontsize=8)
        
        # Axis Information
        ax.set_xlabel(x_label, fontsize=7)
        ax.set_ylabel("Norm. Collision Density", fontsize=7)
        
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, fontsize=7, rotation=45 if profile_type=="hour" else 0)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    for i in range(n, n_rows * n_cols): axes.flatten()[i].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

# ── table builders ────────────────────────────────────────────────────────────

def build_cluster_summary_rows(
    centroids, n_zones_per_cluster, total_collisions_per_cluster, cluster_boroughs, bin_labels
) -> Tuple[List[str], List[Dict[str, str]]]:
    centroid_cols = [f"centroid_{bl}" for bl in bin_labels]
    fieldnames = ["cluster_id", "n_zones", "total_collisions", "top_boroughs", "peak_bin"] + centroid_cols

    rows = []
    for idx, (centroid, n_zones, total, boroughs) in enumerate(
        zip(centroids, n_zones_per_cluster, total_collisions_per_cluster, cluster_boroughs)
    ):
        peak_idx = centroid.index(max(centroid)) if centroid else 0
        record = {
            "cluster_id": str(idx),
            "n_zones": str(n_zones),
            "total_collisions": str(total),
            "top_boroughs": boroughs,
            "peak_bin": bin_labels[peak_idx] if bin_labels else str(peak_idx),
        }
        for col_name, val in zip(centroid_cols, centroid):
            record[col_name] = f"{val:.6f}"
        rows.append(record)
    return fieldnames, rows

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config(parse_config_path(sys.argv[1:]))
    logger = get_logger("clustering.ts_kmeans")

    profile_type = config.get("ts_kmeans_profile", "hour").strip().lower()
    n_clusters = int(config.get("ts_kmeans_n_clusters", "8"))
    cell_degrees = float(config.get("ts_kmeans_cell_degrees", "0.01"))
    min_collisions_per_zone = int(config.get("ts_kmeans_min_zone_collisions", "30"))
    nyc_bounds = (40.477, 40.917, -74.260, -73.700)

    log_step(logger, "TS_KMEANS_PROFILE", "Building profiles and borough mapping...")
    zone_profiles, zone_centres, zone_boroughs = build_zone_profiles(
        Path(config["feature_csv"]), profile_type, cell_degrees, min_collisions_per_zone, nyc_bounds
    )
    
    zone_keys = list(zone_profiles.keys())
    norm_profiles = {k: normalise_l2(v) for k, v in zone_profiles.items()}
    labels = run_kmeans_on_profiles(zone_keys, norm_profiles, n_clusters, 143, 20)
    centroids = compute_cluster_centroids(zone_keys, norm_profiles, labels, n_clusters, len(_PROFILE_META[profile_type]))

    # Stats and Borough Summaries
    label_counts = Counter(labels)
    n_zones_per_cluster = [label_counts[i] for i in range(n_clusters)]
    total_collisions_per_cluster = [0] * n_clusters
    cluster_borough_summary = []
    
    for i in range(n_clusters):
        b_list = []
        for j, lbl in enumerate(labels):
            if lbl == i:
                z_key = zone_keys[j]
                b_list.append(zone_boroughs[z_key])
                total_collisions_per_cluster[i] += int(sum(zone_profiles[z_key]))
        
        top_2 = Counter(b_list).most_common(2)
        cluster_borough_summary.append(", ".join([f"{name}" for name, count in top_2]))

    # Write Tables
    tables_dir = Path(config["tables_dir"])
    summary_fieldnames, summary_rows = build_cluster_summary_rows(
        centroids, n_zones_per_cluster, total_collisions_per_cluster, 
        cluster_borough_summary, _PROFILE_META[profile_type]
    )
    write_csv_rows(tables_dir / "ts_kmeans_cluster_summary.csv", summary_fieldnames, summary_rows)

    # Visualise
    visual_root = Path(config["visualizations_dir"])
    save_centroid_profiles_plot(
        centroids, _PROFILE_META[profile_type], profile_type, 
        n_zones_per_cluster, cluster_borough_summary,
        visual_root / "clusters" / "ts_kmeans_centroid_profiles.png"
    )

    log_step(logger, "TS_KMEANS_DONE", "Tables and plots with axis labels generated.")

if __name__ == "__main__":
    main()