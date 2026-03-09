"""Build grouped fatality-count dataset and compare Poisson vs NB models."""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_csv_rows, write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory

REQUIRED_FIELDS = [
    "MONTH",
    "HOUR",
    "IS_WEEKEND",
    "NUMBER OF PERSONS KILLED",
    "NUMBER OF PERSONS INJURED",
]


def to_int(value: str) -> Tuple[int, bool]:
    text = (value or "").strip()
    if not text:
        return 0, False
    try:
        return int(float(text)), True
    except ValueError:
        return 0, False


def parse_year_month(value: str) -> Tuple[int, int, str, bool]:
    text = (value or "").strip()
    if len(text) >= 7 and text[4] == "-":
        try:
            year_value = int(text[:4])
            month_value = int(text[5:7])
            if 1 <= month_value <= 12:
                return year_value, month_value, f"{year_value:04d}-{month_value:02d}", True
        except ValueError:
            return 0, 0, "", False
    return 0, 0, "", False


def parse_hour(value: str) -> Tuple[int, bool]:
    parsed, ok = to_int(value)
    if not ok or parsed < 0 or parsed > 23:
        return 0, False
    return parsed, True


def parse_weekend_flag(value: str) -> Tuple[int, bool]:
    text = (value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1, True
    if text in {"0", "false", "no", "n"}:
        return 0, True
    return 0, False


def stable_hash_bucket(text: str, buckets: int = 10) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def build_time_count_dataset(feature_csv: Path) -> Tuple[List[Dict[str, object]], Dict[str, int], int]:
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    grouped = defaultdict(lambda: {"collision_count": 0, "fatality_count": 0, "injury_count": 0})
    missing_or_invalid: Counter[str] = Counter()
    total_rows = 0

    with feature_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1

            year_value, month_value, year_month, month_ok = parse_year_month(row.get("MONTH", ""))
            hour_value, hour_ok = parse_hour(row.get("HOUR", ""))
            weekend_value, weekend_ok = parse_weekend_flag(row.get("IS_WEEKEND", ""))
            killed_value, killed_ok = to_int(row.get("NUMBER OF PERSONS KILLED", ""))
            injured_value, injured_ok = to_int(row.get("NUMBER OF PERSONS INJURED", ""))

            if not month_ok:
                missing_or_invalid["MONTH"] += 1
            if not hour_ok:
                missing_or_invalid["HOUR"] += 1
            if not weekend_ok:
                missing_or_invalid["IS_WEEKEND"] += 1
            if not killed_ok:
                missing_or_invalid["NUMBER OF PERSONS KILLED"] += 1
            if not injured_ok:
                missing_or_invalid["NUMBER OF PERSONS INJURED"] += 1

            if not (month_ok and hour_ok and weekend_ok):
                continue

            key = (year_month, year_value, month_value, hour_value, weekend_value)
            grouped[key]["collision_count"] += 1
            grouped[key]["fatality_count"] += max(0, killed_value)
            grouped[key]["injury_count"] += max(0, injured_value)

    grouped_rows: List[Dict[str, object]] = []
    for (year_month, year_value, month_value, hour_value, weekend_value), agg in sorted(grouped.items()):
        collisions = int(agg["collision_count"])
        fatalities = int(agg["fatality_count"])
        injuries = int(agg["injury_count"])
        grouped_rows.append(
            {
                "year_month": year_month,
                "year": year_value,
                "month": month_value,
                "hour": hour_value,
                "is_weekend": weekend_value,
                "collision_count": collisions,
                "fatality_count": fatalities,
                "injury_count": injuries,
                "fatality_rate_per_1000_collisions": (1000.0 * fatalities / collisions) if collisions > 0 else 0.0,
                "log_collision_count": math.log(collisions) if collisions > 0 else 0.0,
            }
        )

    return grouped_rows, dict(missing_or_invalid), total_rows


def build_missingness_rows(missing_or_invalid: Dict[str, int], total_rows: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for field in REQUIRED_FIELDS:
        count = int(missing_or_invalid.get(field, 0))
        share = (count / total_rows) if total_rows > 0 else 0.0
        rows.append(
            {
                "field": field,
                "missing_or_invalid_count": str(count),
                "missing_or_invalid_share": f"{share:.6f}",
            }
        )
    return rows


def compute_summary_stats(grouped_rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not grouped_rows:
        return {
            "n_groups": 0.0,
            "mean_fatality_count": 0.0,
            "variance_fatality_count": 0.0,
            "zero_fatality_group_share": 0.0,
            "mean_collision_count": 0.0,
            "overdispersion_ratio_var_over_mean": 0.0,
        }

    fatality_counts = [float(row["fatality_count"]) for row in grouped_rows]
    collision_counts = [float(row["collision_count"]) for row in grouped_rows]
    zero_groups = sum(1 for value in fatality_counts if value == 0.0)
    mean_fatality = statistics.mean(fatality_counts)
    variance_fatality = statistics.variance(fatality_counts) if len(fatality_counts) > 1 else 0.0
    return {
        "n_groups": float(len(grouped_rows)),
        "mean_fatality_count": float(mean_fatality),
        "variance_fatality_count": float(variance_fatality),
        "zero_fatality_group_share": float(zero_groups / len(grouped_rows)),
        "mean_collision_count": float(statistics.mean(collision_counts)),
        "overdispersion_ratio_var_over_mean": float(variance_fatality / mean_fatality) if mean_fatality > 0 else 0.0,
    }


def build_fatality_distribution(grouped_rows: Sequence[Dict[str, object]]) -> List[Dict[str, str]]:
    total_groups = max(1, len(grouped_rows))
    frequency: Counter[int] = Counter()
    for row in grouped_rows:
        frequency[int(row["fatality_count"])] += 1
    return [
        {
            "fatality_count": str(fatality_count),
            "group_count": str(group_count),
            "share_of_groups": f"{(group_count / total_groups):.6f}",
        }
        for fatality_count, group_count in sorted(frequency.items())
    ]


def build_hourly_profiles(grouped_rows: Sequence[Dict[str, object]]) -> List[Dict[str, str]]:
    grouped = defaultdict(lambda: {"collision_count": 0, "fatality_count": 0, "group_count": 0})
    for row in grouped_rows:
        key = (int(row["hour"]), int(row["is_weekend"]))
        grouped[key]["collision_count"] += int(row["collision_count"])
        grouped[key]["fatality_count"] += int(row["fatality_count"])
        grouped[key]["group_count"] += 1

    output_rows: List[Dict[str, str]] = []
    for hour_value in range(24):
        for weekend_value in [0, 1]:
            agg = grouped.get((hour_value, weekend_value), {"collision_count": 0, "fatality_count": 0, "group_count": 0})
            collisions = int(agg["collision_count"])
            fatalities = int(agg["fatality_count"])
            output_rows.append(
                {
                    "hour": str(hour_value),
                    "is_weekend": str(weekend_value),
                    "collision_count": str(collisions),
                    "fatality_count": str(fatalities),
                    "group_count": str(int(agg["group_count"])),
                    "fatality_rate_per_1000_collisions": f"{(1000.0 * fatalities / collisions) if collisions > 0 else 0.0:.6f}",
                }
            )
    return output_rows


def split_train_test_rows(grouped_rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    for row in grouped_rows:
        key_text = f"{row['year_month']}|{row['hour']}|{row['is_weekend']}"
        if stable_hash_bucket(key_text, buckets=10) < 8:
            train_rows.append(dict(row))
        else:
            test_rows.append(dict(row))
    if not test_rows and train_rows:
        test_rows.append(train_rows[-1])
        train_rows = train_rows[:-1]
    return train_rows, test_rows


def build_design_matrix(
    rows: Sequence[Dict[str, object]], hour_levels: Sequence[int], year_levels: Sequence[int]
) -> Tuple[List[List[float]], List[str]]:
    base_hour = hour_levels[0]
    base_year = year_levels[0]

    feature_names = ["intercept"]
    feature_names.extend([f"hour_{hour}" for hour in hour_levels if hour != base_hour])
    feature_names.append("is_weekend")
    feature_names.extend([f"hour_{hour}_x_weekend" for hour in hour_levels if hour != base_hour])
    feature_names.extend([f"year_{year}" for year in year_levels if year != base_year])

    matrix_rows: List[List[float]] = []
    for row in rows:
        hour_value = int(row["hour"])
        year_value = int(row["year"])
        weekend_value = int(row["is_weekend"])

        values: List[float] = [1.0]
        values.extend([1.0 if hour_value == hour else 0.0 for hour in hour_levels if hour != base_hour])
        values.append(float(weekend_value))
        values.extend([1.0 if (hour_value == hour and weekend_value == 1) else 0.0 for hour in hour_levels if hour != base_hour])
        values.extend([1.0 if year_value == year else 0.0 for year in year_levels if year != base_year])
        matrix_rows.append(values)

    return matrix_rows, feature_names


def evaluate_predictions(y_true: Sequence[float], y_pred: Sequence[float]) -> Tuple[float, float]:
    if len(y_true) != len(y_pred) or not y_true:
        return 0.0, 0.0
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))
    return float(mae), float(rmse)


def fit_count_models(
    grouped_rows: Sequence[Dict[str, object]],
) -> Tuple[str, List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    if len(grouped_rows) < 50:
        return "insufficient_group_rows", [], [], [], ["Not enough grouped rows for model fitting."]

    train_rows, test_rows = split_train_test_rows(grouped_rows)
    if len(train_rows) < 30 or len(test_rows) < 10:
        return "insufficient_split_rows", [], [], [], [f"Split rows too small (train={len(train_rows)}, test={len(test_rows)})."]

    try:
        import numpy as np  # type: ignore
        from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson  # type: ignore
    except ImportError:
        return "missing_dependency", [], [], [], ["statsmodels and numpy are required for Poisson/NB model fitting."]

    hour_levels = list(range(24))
    year_levels = sorted({int(row["year"]) for row in train_rows})
    x_train_list, feature_names = build_design_matrix(train_rows, hour_levels, year_levels)
    x_test_list, _ = build_design_matrix(test_rows, hour_levels, year_levels)

    x_train = np.array(x_train_list, dtype=float)
    x_test = np.array(x_test_list, dtype=float)
    y_train = np.array([float(row["fatality_count"]) for row in train_rows], dtype=float)
    y_test = np.array([float(row["fatality_count"]) for row in test_rows], dtype=float)
    exposure_train = np.array([max(1.0, float(row["collision_count"])) for row in train_rows], dtype=float)
    exposure_test = np.array([max(1.0, float(row["collision_count"])) for row in test_rows], dtype=float)

    keep_indices = [0]
    for index in range(1, x_train.shape[1]):
        column = x_train[:, index]
        if np.any(column != column[0]):
            keep_indices.append(index)
    x_train = x_train[:, keep_indices]
    x_test = x_test[:, keep_indices]
    kept_feature_names = [feature_names[index] for index in keep_indices]

    try:
        poisson_result = Poisson(y_train, x_train, exposure=exposure_train).fit(disp=0, maxiter=200)
        nb_result = NegativeBinomial(y_train, x_train, exposure=exposure_train, loglike_method="nb2").fit(disp=0, maxiter=200)
    except Exception as error:  # pragma: no cover
        return "fit_failed", [], [], [], [f"Model fit failed: {error}"]

    poisson_pred = [float(max(0.0, value)) for value in poisson_result.predict(x_test, exposure=exposure_test).tolist()]
    nb_pred = [float(max(0.0, value)) for value in nb_result.predict(x_test, exposure=exposure_test).tolist()]
    y_test_list = y_test.tolist()

    poisson_mae, poisson_rmse = evaluate_predictions(y_test_list, poisson_pred)
    nb_mae, nb_rmse = evaluate_predictions(y_test_list, nb_pred)

    comparison_rows = [
        {
            "model": "Poisson",
            "train_groups": str(len(train_rows)),
            "test_groups": str(len(test_rows)),
            "log_likelihood": f"{float(poisson_result.llf):.6f}",
            "aic": f"{float(poisson_result.aic):.6f}",
            "mae_test": f"{poisson_mae:.6f}",
            "rmse_test": f"{poisson_rmse:.6f}",
            "n_parameters": str(len(poisson_result.params)),
        },
        {
            "model": "NegativeBinomial",
            "train_groups": str(len(train_rows)),
            "test_groups": str(len(test_rows)),
            "log_likelihood": f"{float(nb_result.llf):.6f}",
            "aic": f"{float(nb_result.aic):.6f}",
            "mae_test": f"{nb_mae:.6f}",
            "rmse_test": f"{nb_rmse:.6f}",
            "n_parameters": str(len(nb_result.params)),
        },
    ]

    coefficient_rows: List[Dict[str, str]] = []
    for name, coef in zip(kept_feature_names, poisson_result.params.tolist()):
        coefficient_rows.append({"model": "Poisson", "feature": name, "coefficient": f"{float(coef):.6f}"})
    nb_params = nb_result.params.tolist()
    for index, coef in enumerate(nb_params):
        label = kept_feature_names[index] if index < len(kept_feature_names) else "nb_dispersion_term"
        coefficient_rows.append({"model": "NegativeBinomial", "feature": label, "coefficient": f"{float(coef):.6f}"})

    prediction_rows = []
    for row, observed, p_pred, nb_pred_value in zip(test_rows, y_test_list, poisson_pred, nb_pred):
        prediction_rows.append(
            {
                "year_month": str(row["year_month"]),
                "year": str(row["year"]),
                "month": str(row["month"]),
                "hour": str(row["hour"]),
                "is_weekend": str(row["is_weekend"]),
                "collision_count": str(row["collision_count"]),
                "observed_fatality_count": str(int(observed)),
                "predicted_poisson": f"{p_pred:.6f}",
                "predicted_negative_binomial": f"{nb_pred_value:.6f}",
            }
        )

    alpha_value = getattr(nb_result, "alpha", None)
    if alpha_value is None and getattr(nb_result, "lnalpha", None) is not None:
        alpha_value = math.exp(float(nb_result.lnalpha))
    messages = [f"Model fit complete (train={len(train_rows)}, test={len(test_rows)})."]
    if alpha_value is not None:
        messages.append(f"Estimated NB overdispersion alpha: {float(alpha_value):.6f}")

    return "ok", comparison_rows, prediction_rows, coefficient_rows, messages


def hour_series(hourly_rows: Sequence[Dict[str, str]], key: str, is_weekend: int) -> List[float]:
    lookup = {int(row["hour"]): float(row[key]) for row in hourly_rows if int(row["is_weekend"]) == is_weekend}
    return [float(lookup.get(hour, 0.0)) for hour in range(24)]


def create_fatality_count_plots(
    grouped_rows: Sequence[Dict[str, object]],
    hourly_rows: Sequence[Dict[str, str]],
    comparison_rows: Sequence[Dict[str, str]],
    prediction_rows: Sequence[Dict[str, str]],
    visual_root: Path,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return ["Skipped count-model plots because matplotlib is not installed."]

    output_dir = visual_root / "count_models"
    ensure_directory(output_dir)
    hours = list(range(24))
    messages: List[str] = []

    weekly_collision = hour_series(hourly_rows, "collision_count", 0)
    weekend_collision = hour_series(hourly_rows, "collision_count", 1)
    weekly_fatal = hour_series(hourly_rows, "fatality_count", 0)
    weekend_fatal = hour_series(hourly_rows, "fatality_count", 1)
    weekly_rate = hour_series(hourly_rows, "fatality_rate_per_1000_collisions", 0)
    weekend_rate = hour_series(hourly_rows, "fatality_rate_per_1000_collisions", 1)

    plt.figure(figsize=(10, 5))
    plt.plot(hours, weekly_collision, marker="o", label="Weekday")
    plt.plot(hours, weekend_collision, marker="o", label="Weekend")
    plt.title("Hourly Collision Count: Weekday vs Weekend")
    plt.xlabel("Hour of Day")
    plt.ylabel("Collision Count")
    plt.xticks(hours)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hourly_collision_count_weekday_vs_weekend.png")
    plt.close()
    messages.append("Created hourly collision count comparison plot.")

    plt.figure(figsize=(10, 5))
    plt.plot(hours, weekly_fatal, marker="o", label="Weekday")
    plt.plot(hours, weekend_fatal, marker="o", label="Weekend")
    plt.title("Hourly Fatality Count: Weekday vs Weekend")
    plt.xlabel("Hour of Day")
    plt.ylabel("Fatality Count")
    plt.xticks(hours)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hourly_fatality_count_weekday_vs_weekend.png")
    plt.close()
    messages.append("Created hourly fatality count comparison plot.")

    plt.figure(figsize=(10, 5))
    plt.plot(hours, weekly_rate, marker="o", label="Weekday")
    plt.plot(hours, weekend_rate, marker="o", label="Weekend")
    plt.title("Hourly Fatality Rate per 1,000 Collisions: Weekday vs Weekend")
    plt.xlabel("Hour of Day")
    plt.ylabel("Fatality Rate per 1,000 Collisions")
    plt.xticks(hours)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hourly_fatality_rate_weekday_vs_weekend.png")
    plt.close()
    messages.append("Created hourly fatality rate comparison plot.")

    fatality_values = [int(row["fatality_count"]) for row in grouped_rows]
    bins = list(range(0, max(fatality_values) + 2)) if fatality_values and max(fatality_values) <= 25 else 30
    plt.figure(figsize=(8, 5))
    plt.hist(fatality_values, bins=bins, edgecolor="black", alpha=0.8)
    plt.title("Distribution of Grouped Fatality Count")
    plt.xlabel("Fatality Count per Group")
    plt.ylabel("Number of Groups")
    plt.tight_layout()
    plt.savefig(output_dir / "grouped_fatality_count_histogram.png")
    plt.close()
    messages.append("Created grouped fatality count histogram.")

    if comparison_rows:
        figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        models = [row["model"] for row in comparison_rows]
        for axis, metric, title in zip(
            axes,
            ["aic", "mae_test", "rmse_test"],
            ["AIC (Lower Better)", "Test MAE (Lower Better)", "Test RMSE (Lower Better)"],
        ):
            axis.bar(models, [float(row[metric]) for row in comparison_rows])
            axis.set_title(title)
            axis.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.savefig(output_dir / "count_model_comparison_metrics.png")
        plt.close(figure)
        messages.append("Created Poisson vs NB metric comparison plot.")

    if prediction_rows:
        observed = [float(row["observed_fatality_count"]) for row in prediction_rows]
        poisson_pred = [float(row["predicted_poisson"]) for row in prediction_rows]
        nb_pred = [float(row["predicted_negative_binomial"]) for row in prediction_rows]
        upper = max(observed + poisson_pred + nb_pred + [1.0])
        plt.figure(figsize=(7, 6))
        plt.scatter(observed, poisson_pred, s=12, alpha=0.4, label="Poisson")
        plt.scatter(observed, nb_pred, s=12, alpha=0.4, label="NegativeBinomial")
        plt.plot([0, upper], [0, upper], linestyle="--", linewidth=1)
        plt.title("Observed vs Predicted Fatality Count (Test Groups)")
        plt.xlabel("Observed Fatality Count")
        plt.ylabel("Predicted Fatality Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "observed_vs_predicted_fatality_count.png")
        plt.close()
        messages.append("Created observed vs predicted test-group plot.")

    return messages


def write_summary_stats_csv(summary_stats: Dict[str, float], output_csv: Path) -> None:
    rows = [{"metric": key, "value": f"{value:.6f}"} for key, value in summary_stats.items()]
    write_csv_rows(output_csv, ["metric", "value"], rows)


def pick_best_model(comparison_rows: Sequence[Dict[str, str]]) -> str:
    if not comparison_rows:
        return "NA"
    return min(comparison_rows, key=lambda row: float(row["aic"]))["model"]


def write_fatality_modeling_report(
    report_path: Path,
    dataset_name: str,
    total_rows: int,
    grouped_rows: Sequence[Dict[str, object]],
    dropped_rows: int,
    summary_stats: Dict[str, float],
    missing_rows: Sequence[Dict[str, str]],
    model_status: str,
    model_messages: Sequence[str],
    comparison_rows: Sequence[Dict[str, str]],
    plot_messages: Sequence[str],
) -> None:
    lines = [
        f"Dataset: {dataset_name}",
        "Modeling target: grouped fatality_count by year_month x hour x is_weekend.",
        "",
        "Data construction:",
        f"- Feature rows scanned: {total_rows}",
        f"- Grouped rows generated: {len(grouped_rows)}",
        f"- Rows dropped for invalid grouping keys (MONTH/HOUR/IS_WEEKEND): {dropped_rows}",
        "",
        "Descriptive summary:",
        f"- Mean fatality_count: {summary_stats.get('mean_fatality_count', 0.0):.6f}",
        f"- Variance fatality_count: {summary_stats.get('variance_fatality_count', 0.0):.6f}",
        f"- Overdispersion ratio (var/mean): {summary_stats.get('overdispersion_ratio_var_over_mean', 0.0):.6f}",
        f"- Zero-fatality group share: {summary_stats.get('zero_fatality_group_share', 0.0):.6f}",
        f"- Mean collision_count per group: {summary_stats.get('mean_collision_count', 0.0):.6f}",
        "",
        "Required-field missing/invalid share:",
    ]
    for row in missing_rows:
        lines.append(f"- {row['field']}: {row['missing_or_invalid_count']} ({row['missing_or_invalid_share']})")
    lines.extend(["", "Poisson vs Negative Binomial comparison:", f"- Model status: {model_status}"])
    for message in model_messages:
        lines.append(f"- {message}")
    for row in comparison_rows:
        lines.append(
            "- "
            + f"{row['model']}: llf={row['log_likelihood']}, aic={row['aic']}, "
            + f"test_mae={row['mae_test']}, test_rmse={row['rmse_test']}"
        )
    if comparison_rows:
        lines.append(f"- Best model by AIC: {pick_best_model(comparison_rows)}")
    lines.extend(["", "Plot status:"])
    for message in plot_messages:
        lines.append(f"- {message}")
    write_markdown(report_path, "Fatality Count Modeling Report", lines)


def main() -> None:
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)
    logger = get_logger("fatality_count_models.run")

    feature_csv = Path(config["feature_csv"]).resolve()
    tables_dir = Path(config["tables_dir"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()
    visual_root = Path(config["visualizations_dir"]).resolve()

    dataset_name = config.get("dataset_name", "unknown")
    log_step(logger, "FATALITY_COUNT_START", f"Dataset: {dataset_name}")

    grouped_rows, missing_or_invalid, total_rows = build_time_count_dataset(feature_csv)
    dropped_rows = total_rows - sum(int(row["collision_count"]) for row in grouped_rows)
    if not grouped_rows:
        raise RuntimeError("No grouped rows were generated for count modeling.")

    summary_stats = compute_summary_stats(grouped_rows)
    missing_rows = build_missingness_rows(missing_or_invalid, total_rows)
    distribution_rows = build_fatality_distribution(grouped_rows)
    hourly_rows = build_hourly_profiles(grouped_rows)
    model_status, comparison_rows, prediction_rows, coefficient_rows, model_messages = fit_count_models(grouped_rows)

    ensure_directory(tables_dir)
    write_csv_rows(
        tables_dir / "fatality_count_modeling_dataset.csv",
        [
            "year_month",
            "year",
            "month",
            "hour",
            "is_weekend",
            "collision_count",
            "fatality_count",
            "injury_count",
            "fatality_rate_per_1000_collisions",
            "log_collision_count",
        ],
        [
            {
                "year_month": str(row["year_month"]),
                "year": str(row["year"]),
                "month": str(row["month"]),
                "hour": str(row["hour"]),
                "is_weekend": str(row["is_weekend"]),
                "collision_count": str(row["collision_count"]),
                "fatality_count": str(row["fatality_count"]),
                "injury_count": str(row["injury_count"]),
                "fatality_rate_per_1000_collisions": f"{float(row['fatality_rate_per_1000_collisions']):.6f}",
                "log_collision_count": f"{float(row['log_collision_count']):.6f}",
            }
            for row in grouped_rows
        ],
    )
    write_csv_rows(
        tables_dir / "hourly_weekpart_summary.csv",
        ["hour", "is_weekend", "collision_count", "fatality_count", "group_count", "fatality_rate_per_1000_collisions"],
        hourly_rows,
    )
    write_csv_rows(tables_dir / "fatality_count_distribution.csv", ["fatality_count", "group_count", "share_of_groups"], distribution_rows)
    write_summary_stats_csv(summary_stats, tables_dir / "fatality_modeling_summary_stats.csv")
    write_csv_rows(
        tables_dir / "fatality_count_missingness_summary.csv",
        ["field", "missing_or_invalid_count", "missing_or_invalid_share"],
        missing_rows,
    )
    if comparison_rows:
        write_csv_rows(
            tables_dir / "fatality_count_model_comparison.csv",
            ["model", "train_groups", "test_groups", "log_likelihood", "aic", "mae_test", "rmse_test", "n_parameters"],
            comparison_rows,
        )
    if prediction_rows:
        write_csv_rows(
            tables_dir / "fatality_count_model_predictions_sample.csv",
            [
                "year_month",
                "year",
                "month",
                "hour",
                "is_weekend",
                "collision_count",
                "observed_fatality_count",
                "predicted_poisson",
                "predicted_negative_binomial",
            ],
            prediction_rows[:1500],
        )
    if coefficient_rows:
        write_csv_rows(
            tables_dir / "fatality_count_model_coefficients.csv",
            ["model", "feature", "coefficient"],
            coefficient_rows,
        )

    plot_messages = create_fatality_count_plots(grouped_rows, hourly_rows, comparison_rows, prediction_rows, visual_root)
    write_fatality_modeling_report(
        reports_dir / "fatality_count_modeling_report.md",
        dataset_name,
        total_rows,
        grouped_rows,
        dropped_rows,
        summary_stats,
        missing_rows,
        model_status,
        model_messages,
        comparison_rows,
        plot_messages,
    )
    log_step(logger, "FATALITY_COUNT_DONE", f"Grouped rows: {len(grouped_rows)} | Model status: {model_status}")


if __name__ == "__main__":
    main()
