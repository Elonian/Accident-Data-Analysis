"""Plot helpers for saving common chart types."""

from pathlib import Path
from typing import Iterable, List

from utils.path_utils import ensure_directory


def _load_pyplot():
    """Load matplotlib pyplot lazily.

    Returns:
        module: Imported matplotlib.pyplot module.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as error:
        raise ImportError("matplotlib is required for plotting.") from error
    return plt


def save_line_plot(
    x_values: Iterable,
    y_values: Iterable[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Save a simple line chart.

    Args:
        x_values: X-axis values.
        y_values: Y-axis values.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        output_path: Output image path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib is not installed.
    """
    plt = _load_pyplot()
    ensure_directory(output_path.parent)

    plt.figure(figsize=(10, 5))
    plt.plot(list(x_values), list(y_values), marker="o", linewidth=1.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_bar_plot(
    labels: Iterable[str],
    values: Iterable[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Save a simple bar chart.

    Args:
        labels: Category labels.
        values: Bar values.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        output_path: Output image path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib is not installed.
    """
    plt = _load_pyplot()
    ensure_directory(output_path.parent)

    plt.figure(figsize=(10, 5))
    plt.bar(list(labels), list(values))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_scatter_plot(
    x_values: List[float],
    y_values: List[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Save a simple scatter plot.

    Args:
        x_values: X-axis numeric values.
        y_values: Y-axis numeric values.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        output_path: Output image path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib is not installed.
    """
    plt = _load_pyplot()
    ensure_directory(output_path.parent)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, s=6, alpha=0.6)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_cluster_scatter_plot(
    x_values: List[float],
    y_values: List[float],
    cluster_labels: List[int],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Save a cluster-colored scatter plot.

    Args:
        x_values: X-axis numeric values (usually longitude).
        y_values: Y-axis numeric values (usually latitude).
        cluster_labels: Cluster ID for each point.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        output_path: Output image path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If point and label lengths do not match.
    """
    if len(x_values) != len(y_values) or len(x_values) != len(cluster_labels):
        raise ValueError("Cluster scatter inputs must have the same length.")

    plt = _load_pyplot()
    ensure_directory(output_path.parent)

    # Show largest clusters first, and keep noise (-1) in gray.
    cluster_counts = {}
    for label in cluster_labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1
    ordered_labels = [item[0] for item in sorted(cluster_counts.items(), key=lambda pair: pair[1], reverse=True)]

    non_noise = [label for label in ordered_labels if label != -1]
    cmap = plt.get_cmap("tab20", max(1, len(non_noise)))
    color_lookup = {label: cmap(index) for index, label in enumerate(non_noise)}
    color_lookup[-1] = (0.55, 0.55, 0.55, 0.35)

    plt.figure(figsize=(9, 7))
    for label in ordered_labels:
        xs = [x for x, c in zip(x_values, cluster_labels) if c == label]
        ys = [y for y, c in zip(y_values, cluster_labels) if c == label]
        if not xs:
            continue
        if label == -1:
            plt.scatter(xs, ys, s=4, alpha=0.25, c=[color_lookup[label]], linewidths=0, label="noise")
        else:
            plt.scatter(xs, ys, s=4, alpha=0.55, c=[color_lookup[label]], linewidths=0)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_hexbin_plot(
    x_values: List[float],
    y_values: List[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    gridsize: int = 90,
) -> None:
    """Save a density hexbin map.

    Args:
        x_values: X-axis numeric values.
        y_values: Y-axis numeric values.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        output_path: Output image path.
        gridsize: Hexbin grid size.

    Returns:
        None

    Raises:
        ImportError: If matplotlib is not installed.
    """
    plt = _load_pyplot()
    ensure_directory(output_path.parent)

    figure, axis = plt.subplots(figsize=(9, 7))
    hexmap = axis.hexbin(x_values, y_values, gridsize=gridsize, mincnt=1, cmap="viridis", bins="log")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    cbar = figure.colorbar(hexmap, ax=axis)
    cbar.set_label("log10(point density)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def save_cluster_centroid_bubble_plot(
    centroid_x: List[float],
    centroid_y: List[float],
    centroid_sizes: List[float],
    centroid_labels: List[str],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Save bubble plot for cluster centroids.

    Args:
        centroid_x: Centroid x-values (longitude).
        centroid_y: Centroid y-values (latitude).
        centroid_sizes: Marker sizes (collision count scale).
        centroid_labels: Label text for markers.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        output_path: Output image path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If input lengths mismatch.
    """
    if not (len(centroid_x) == len(centroid_y) == len(centroid_sizes) == len(centroid_labels)):
        raise ValueError("Centroid bubble inputs must have the same length.")

    plt = _load_pyplot()
    ensure_directory(output_path.parent)

    figure, axis = plt.subplots(figsize=(9, 7))
    scatter = axis.scatter(centroid_x, centroid_y, s=centroid_sizes, c=centroid_sizes, cmap="plasma", alpha=0.7, edgecolors="black", linewidths=0.25)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Relative cluster size")

    for x_value, y_value, label in zip(centroid_x, centroid_y, centroid_labels):
        axis.text(x_value, y_value, label, fontsize=7, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def save_heatmap(
    matrix: List[List[float]],
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Save a simple heatmap image.

    Args:
        matrix: 2D numeric matrix.
        x_labels: Label list for heatmap columns.
        y_labels: Label list for heatmap rows.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        output_path: Output image path.

    Returns:
        None

    Raises:
        ImportError: If matplotlib is not installed.
    """
    plt = _load_pyplot()
    ensure_directory(output_path.parent)

    fig, axis = plt.subplots(figsize=(12, 4))
    heatmap = axis.imshow(matrix, aspect="auto")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_xticks(range(len(x_labels)))
    axis.set_xticklabels(x_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(y_labels)))
    axis.set_yticklabels(y_labels)
    fig.colorbar(heatmap, ax=axis)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
