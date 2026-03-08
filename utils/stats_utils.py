"""Reusable small statistics helpers."""

from statistics import mean
from typing import Dict, Iterable, List


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two values.

    Args:
        numerator: Top value.
        denominator: Bottom value.

    Returns:
        float: Division result, or 0.0 when denominator is zero.

    Raises:
        None
    """
    return numerator / denominator if denominator else 0.0


def percentage(numerator: float, denominator: float) -> float:
    """Return percentage value between 0 and 100.

    Args:
        numerator: Top value.
        denominator: Bottom value.

    Returns:
        float: Percentage result.

    Raises:
        None
    """
    return safe_divide(numerator * 100.0, denominator)


def summarize_numeric(values: Iterable[float]) -> Dict[str, float]:
    """Return simple numeric summary metrics.

    Args:
        values: Numeric values.

    Returns:
        Dict[str, float]: Dictionary with count, min, max, mean.

    Raises:
        ValueError: If input is empty.
    """
    value_list: List[float] = list(values)
    if not value_list:
        raise ValueError("Cannot summarize an empty sequence.")

    return {
        "count": float(len(value_list)),
        "min": float(min(value_list)),
        "max": float(max(value_list)),
        "mean": float(mean(value_list)),
    }
