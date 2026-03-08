"""Run spatial clustering with MiniBatchKMeans and write colored cluster outputs."""

from __future__ import annotations

import csv
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_csv_rows, write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory
from utils.plot_utils import save_cluster_centroid_bubble_plot, save_cluster_scatter_plot, save_hexbin_plot


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


def iter_valid_coordinates(feature_csv: Path) -> Iterator[Tuple[str, float, float]]:
    """Yield collision ID and valid coordinates from feature CSV.

    Args:
        feature_csv: Feature dataset path.

    Returns:
        Tuple[str, float, float]: Collision ID, latitude, longitude.

    Raises:
        FileNotFoundError: If feature CSV is missing.
    """
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    with feature_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lat = to_float(row.get("LATITUDE", ""))
            lon = to_float(row.get("LONGITUDE", ""))
            if lat is None or lon is None:
                continue
            collision_id = (row.get("COLLISION_ID") or "").strip()
            yield collision_id, lat, lon


def count_valid_points(feature_csv: Path) -> int:
    """Count valid coordinate rows.

    Args:
        feature_csv: Feature dataset path.

    Returns:
        int: Number of rows with valid coordinates.

    Raises:
        None
    """
    return sum(1 for _ in iter_valid_coordinates(feature_csv))


def choose_cluster_count(valid_points: int, configured_count: int) -> int:
    """Choose cluster count based on config and data size.

    Args:
        valid_points: Number of valid coordinate rows.
        configured_count: Configured cluster count (0 means auto).

    Returns:
        int: Final cluster count.

    Raises:
        ValueError: If there are fewer than 2 points.
    """
    if valid_points < 2:
        raise ValueError("Need at least 2 valid coordinate points for clustering.")

    if configured_count > 0:
        return min(configured_count, valid_points)

    auto_count = max(12, min(80, valid_points // 1800))
    return max(2, min(auto_count, valid_points))


def project_for_clustering(latitude: float, longitude: float, cos_latitude_ref: float) -> List[float]:
    """Project lat/lon into a locally scaled 2D space.

    Args:
        latitude: Latitude value.
        longitude: Longitude value.
        cos_latitude_ref: Longitude scale factor.

    Returns:
        List[float]: Scaled [x, y] feature pair.

    Raises:
        None
    """
    return [longitude * cos_latitude_ref, latitude]


def fit_minibatch_kmeans(
    feature_csv: Path,
    cluster_count: int,
    fit_batch_size: int,
    random_seed: int,
    cos_latitude_ref: float,
) -> object:
    """Fit MiniBatchKMeans in streaming batches.

    Args:
        feature_csv: Feature dataset path.
        cluster_count: Number of clusters.
        fit_batch_size: Batch size for partial_fit.
        random_seed: Random seed.
        cos_latitude_ref: Longitude scale factor.

    Returns:
        object: Fitted MiniBatchKMeans model.

    Raises:
        ImportError: If sklearn/numpy is unavailable.
    """
    import numpy as np  # type: ignore
    from sklearn.cluster import MiniBatchKMeans  # type: ignore

    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=random_seed,
        batch_size=max(fit_batch_size, cluster_count),
        n_init=10,
        reassignment_ratio=0.01,
    )

    batch: List[List[float]] = []
    fitted_any = False

    for _, latitude, longitude in iter_valid_coordinates(feature_csv):
        batch.append(project_for_clustering(latitude, longitude, cos_latitude_ref))
        if len(batch) >= max(fit_batch_size, cluster_count):
            model.partial_fit(np.array(batch, dtype=float))
            batch.clear()
            fitted_any = True

    if batch:
        model.partial_fit(np.array(batch, dtype=float))
        fitted_any = True

    if not fitted_any:
        raise ValueError("No valid coordinates were found for model fitting.")

    return model


def update_plot_reservoir(
    longitudes: List[float],
    latitudes: List[float],
    labels: List[int],
    longitude: float,
    latitude: float,
    label: int,
    seen_points: int,
    plot_limit: int,
    rng: random.Random,
) -> None:
    """Keep a fixed-size random sample of plot points.

    Args:
        longitudes: Reservoir longitude list.
        latitudes: Reservoir latitude list.
        labels: Reservoir label list.
        longitude: New point longitude.
        latitude: New point latitude.
        label: New point cluster label.
        seen_points: Number of points seen so far (1-based).
        plot_limit: Maximum number of plot points.
        rng: Random generator.

    Returns:
        None

    Raises:
        None
    """
    if plot_limit <= 0:
        return

    if len(longitudes) < plot_limit:
        longitudes.append(longitude)
        latitudes.append(latitude)
        labels.append(label)
        return

    replacement_index = rng.randint(0, seen_points - 1)
    if replacement_index < plot_limit:
        longitudes[replacement_index] = longitude
        latitudes[replacement_index] = latitude
        labels[replacement_index] = label


def assign_clusters_and_summarize(
    feature_csv: Path,
    model: object,
    predict_batch_size: int,
    assignment_limit: int,
    top_cluster_count: int,
    plot_limit: int,
    random_seed: int,
    cos_latitude_ref: float,
) -> Dict[str, object]:
    """Assign clusters and collect summary outputs.

    Args:
        feature_csv: Feature dataset path.
        model: Fitted clustering model.
        predict_batch_size: Batch size for predict.
        assignment_limit: Max assignment rows to write.
        top_cluster_count: Number of top clusters to report.
        plot_limit: Maximum points for cluster plot.
        random_seed: Random seed for reservoir sampling.
        cos_latitude_ref: Longitude scale factor.

    Returns:
        Dict[str, object]: Assignments, summaries, and plot payload.

    Raises:
        ImportError: If numpy is unavailable.
    """
    import numpy as np  # type: ignore

    cluster_counts: Counter[int] = Counter()
    cluster_lat_sum: Dict[int, float] = defaultdict(float)
    cluster_lon_sum: Dict[int, float] = defaultdict(float)
    assignments: List[Dict[str, str]] = []

    plot_lons: List[float] = []
    plot_lats: List[float] = []
    plot_labels: List[int] = []

    rng = random.Random(random_seed)
    seen_points = 0

    pending_meta: List[Tuple[str, float, float]] = []
    pending_features: List[List[float]] = []

    def flush_pending() -> None:
        """Predict labels for pending rows and update outputs.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        nonlocal seen_points
        if not pending_features:
            return

        feature_matrix = np.array(pending_features, dtype=float)
        predicted = model.predict(feature_matrix).tolist()

        for (collision_id, latitude, longitude), cluster_label in zip(pending_meta, predicted):
            label = int(cluster_label)
            seen_points += 1

            cluster_counts[label] += 1
            cluster_lat_sum[label] += latitude
            cluster_lon_sum[label] += longitude

            if len(assignments) < assignment_limit:
                assignments.append(
                    {
                        "COLLISION_ID": collision_id,
                        "LATITUDE": f"{latitude:.6f}",
                        "LONGITUDE": f"{longitude:.6f}",
                        "CLUSTER_ID": str(label),
                    }
                )

            update_plot_reservoir(
                longitudes=plot_lons,
                latitudes=plot_lats,
                labels=plot_labels,
                longitude=longitude,
                latitude=latitude,
                label=label,
                seen_points=seen_points,
                plot_limit=plot_limit,
                rng=rng,
            )

        pending_meta.clear()
        pending_features.clear()

    for collision_id, latitude, longitude in iter_valid_coordinates(feature_csv):
        pending_meta.append((collision_id, latitude, longitude))
        pending_features.append(project_for_clustering(latitude, longitude, cos_latitude_ref))
        if len(pending_features) >= predict_batch_size:
            flush_pending()

    flush_pending()

    top_clusters = cluster_counts.most_common(top_cluster_count)
    top_cluster_rows: List[Dict[str, str]] = []
    for cluster_id, count in top_clusters:
        center_latitude = cluster_lat_sum[cluster_id] / count
        center_longitude = cluster_lon_sum[cluster_id] / count
        top_cluster_rows.append(
            {
                "cluster_id": str(cluster_id),
                "collision_count": str(count),
                "center_latitude": f"{center_latitude:.6f}",
                "center_longitude": f"{center_longitude:.6f}",
            }
        )

    return {
        "seen_points": seen_points,
        "unique_clusters": len(cluster_counts),
        "assignments": assignments,
        "top_cluster_rows": top_cluster_rows,
        "plot_lons": plot_lons,
        "plot_lats": plot_lats,
        "plot_labels": plot_labels,
    }


def main() -> None:
    """Run clustering stage and write outputs.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If stage fails.
    """
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)
    logger = get_logger("clustering.spatial")

    feature_csv = Path(config["feature_csv"]).resolve()
    tables_dir = Path(config["tables_dir"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()
    visual_root = Path(config["visualizations_dir"]).resolve()

    random_seed = int(config.get("random_seed", "143"))
    assignment_limit = int(config.get("cluster_sample_limit", "120000"))
    plot_limit = int(config.get("cluster_plot_limit", "120000"))
    fit_batch_size = int(config.get("cluster_fit_batch_size", "20000"))
    predict_batch_size = int(config.get("cluster_predict_batch_size", "5000"))
    top_cluster_count = int(config.get("cluster_top_n", "25"))
    configured_cluster_count = int(config.get("n_spatial_clusters", "0"))
    latitude_reference = float(config.get("cluster_latitude_reference", "40.7128"))

    log_step(logger, "CLUSTER_START", f"Dataset: {config.get('dataset_name', 'unknown')}")
    valid_points = count_valid_points(feature_csv)
    cluster_count = choose_cluster_count(valid_points, configured_cluster_count)

    cos_latitude_ref = math.cos(math.radians(latitude_reference))

    model_method = "minibatch_kmeans"
    try:
        model = fit_minibatch_kmeans(
            feature_csv=feature_csv,
            cluster_count=cluster_count,
            fit_batch_size=fit_batch_size,
            random_seed=random_seed,
            cos_latitude_ref=cos_latitude_ref,
        )
    except ImportError:
        raise RuntimeError("scikit-learn and numpy are required for spatial clustering.")

    summary = assign_clusters_and_summarize(
        feature_csv=feature_csv,
        model=model,
        predict_batch_size=predict_batch_size,
        assignment_limit=assignment_limit,
        top_cluster_count=top_cluster_count,
        plot_limit=plot_limit,
        random_seed=random_seed,
        cos_latitude_ref=cos_latitude_ref,
    )

    ensure_directory(tables_dir)
    ensure_directory(reports_dir)
    ensure_directory(visual_root / "clusters")

    write_csv_rows(
        tables_dir / "spatial_cluster_assignments_sample.csv",
        ["COLLISION_ID", "LATITUDE", "LONGITUDE", "CLUSTER_ID"],
        summary["assignments"],
    )
    write_csv_rows(
        tables_dir / "top_spatial_clusters.csv",
        ["cluster_id", "collision_count", "center_latitude", "center_longitude"],
        summary["top_cluster_rows"],
    )

    plot_status = "Created color-coded spatial cluster map."
    plot_all_points = summary["seen_points"] <= plot_limit
    plot_title_suffix = "all points" if plot_all_points else f"sampled points ({plot_limit:,})"
    try:
        save_cluster_scatter_plot(
            summary["plot_lons"],
            summary["plot_lats"],
            summary["plot_labels"],
            f"Spatial Collision Clusters ({plot_title_suffix})",
            "Longitude",
            "Latitude",
            visual_root / "clusters" / "spatial_collision_sample.png",
        )
        save_cluster_scatter_plot(
            summary["plot_lons"],
            summary["plot_lats"],
            summary["plot_labels"],
            f"Spatial Collision Clusters ({plot_title_suffix})",
            "Longitude",
            "Latitude",
            visual_root / "clusters" / "spatial_collision_clustered_colored.png",
        )
        save_hexbin_plot(
            summary["plot_lons"],
            summary["plot_lats"],
            f"Spatial Collision Density ({plot_title_suffix})",
            "Longitude",
            "Latitude",
            visual_root / "clusters" / "spatial_collision_density_hexbin.png",
            gridsize=100,
        )

        top_rows = summary["top_cluster_rows"][:20]
        if top_rows:
            max_count = max(float(row["collision_count"]) for row in top_rows)
            centroid_x = [float(row["center_longitude"]) for row in top_rows]
            centroid_y = [float(row["center_latitude"]) for row in top_rows]
            centroid_sizes = [80.0 + 900.0 * (float(row["collision_count"]) / max_count) for row in top_rows]
            centroid_labels = [f"C{row['cluster_id']}" for row in top_rows]
            save_cluster_centroid_bubble_plot(
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                centroid_sizes=centroid_sizes,
                centroid_labels=centroid_labels,
                title="Top Spatial Cluster Centroids (by collision volume)",
                x_label="Longitude",
                y_label="Latitude",
                output_path=visual_root / "clusters" / "spatial_cluster_centroids_top20.png",
            )
    except ImportError:
        plot_status = "Skipped cluster map because matplotlib is not installed."

    write_markdown(
        reports_dir / "spatial_clustering_report.md",
        "Spatial Clustering Report",
        [
            f"Dataset: {config.get('dataset_name', 'unknown')}",
            f"Model method: {model_method}",
            f"Valid coordinate points: {summary['seen_points']}",
            f"Configured clusters: {cluster_count}",
            f"Unique clusters generated: {summary['unique_clusters']}",
            f"Assignment rows written: {len(summary['assignments'])}",
            f"Plot points used: {len(summary['plot_lats'])}",
            f"Plot coverage: {'all valid points' if plot_all_points else 'reservoir sample'}",
            plot_status,
        ],
    )

    log_step(
        logger,
        "CLUSTER_DONE",
        f"Clusters: {summary['unique_clusters']} | plot_points: {len(summary['plot_lats'])}",
    )


if __name__ == "__main__":
    main()
