"""Train baseline severity model and write model outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_csv_rows, write_markdown
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory
from utils.plot_utils import save_bar_plot
from utils.stats_utils import safe_divide

PHASES = [
    "PRE_PAUSE",
    "PAUSE",
    "REOPEN_PHASE_1",
    "REOPEN_PHASE_2",
    "REOPEN_PHASE_3",
    "REOPEN_PHASE_4_PLUS",
]


def stable_hash_bucket(text: str, buckets: int = 10) -> int:
    """Hash text into a deterministic bucket index.

    Args:
        text: Input identifier string.
        buckets: Number of buckets.

    Returns:
        int: Bucket index.

    Raises:
        ValueError: If bucket count is invalid.
    """
    if buckets <= 0:
        raise ValueError("Bucket count must be positive.")
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def to_float(value: str) -> float:
    """Convert text to float safely.

    Args:
        value: Input text.

    Returns:
        float: Parsed float or zero.

    Raises:
        None
    """
    text = (value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def build_feature_vector(row: Dict[str, str]) -> List[float]:
    """Build numeric model feature vector from row.

    Args:
        row: Feature row dictionary.

    Returns:
        List[float]: Numeric feature vector.

    Raises:
        None
    """
    hour = to_float(row.get("HOUR", "0"))
    is_weekend = to_float(row.get("IS_WEEKEND", "0"))
    phase = (row.get("PANDEMIC_PHASE", "") or "").strip()

    phase_vector = [1.0 if phase == phase_name else 0.0 for phase_name in PHASES]
    return [hour, is_weekend] + phase_vector


def get_target(row: Dict[str, str]) -> int:
    """Return binary target label for any injury.

    Args:
        row: Feature row dictionary.

    Returns:
        int: 1 when any injury collision, else 0.

    Raises:
        None
    """
    return int(float(row.get("ANY_INJURY", "0") or 0))


def sample_train_test_rows(feature_csv: Path, max_rows: int, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Sample train and test rows from feature CSV.

    Args:
        feature_csv: Feature table path.
        max_rows: Maximum sampled rows.
        seed: Random seed.

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: Train and test rows.

    Raises:
        FileNotFoundError: If feature CSV does not exist.
    """
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    rng = random.Random(seed)
    train_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []
    train_seen = 0
    test_seen = 0
    test_limit = max(1, max_rows // 4)

    with feature_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            collision_id = (row.get("COLLISION_ID") or "").strip() or f"ROW_{len(train_rows)+len(test_rows)}"
            bucket = stable_hash_bucket(collision_id, buckets=10)

            if bucket < 8:
                train_seen += 1
                if len(train_rows) < max_rows:
                    train_rows.append(row)
                else:
                    replacement_index = rng.randint(0, train_seen - 1)
                    if replacement_index < max_rows:
                        train_rows[replacement_index] = row
            else:
                test_seen += 1
                if len(test_rows) < test_limit:
                    test_rows.append(row)
                else:
                    replacement_index = rng.randint(0, test_seen - 1)
                    if replacement_index < test_limit:
                        test_rows[replacement_index] = row

    return train_rows, test_rows


def evaluate_predictions(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute basic classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dict[str, float]: Accuracy, precision, recall, and F1.

    Raises:
        ValueError: If label lengths differ.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Label lengths must match.")

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "accuracy": safe_divide(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def train_with_sklearn(train_rows: List[Dict[str, str]], test_rows: List[Dict[str, str]]) -> Tuple[str, Dict[str, float], List[Dict[str, str]], Dict[str, object]]:
    """Train logistic baseline with sklearn.

    Args:
        train_rows: Training rows.
        test_rows: Testing rows.

    Returns:
        Tuple[str, Dict[str, float], List[Dict[str, str]], Dict[str, object]]: Method, metrics, prediction rows, and curve data.

    Raises:
        ImportError: If sklearn or numpy is not installed.
    """
    import numpy as np  # type: ignore
    from sklearn.calibration import calibration_curve  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import auc, precision_recall_curve, roc_curve  # type: ignore

    x_train = np.array([build_feature_vector(row) for row in train_rows], dtype=float)
    y_train = np.array([get_target(row) for row in train_rows], dtype=int)
    x_test = np.array([build_feature_vector(row) for row in test_rows], dtype=float)
    y_test = np.array([get_target(row) for row in test_rows], dtype=int)

    model = LogisticRegression(max_iter=300, class_weight="balanced")
    model.fit(x_train, y_train)

    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = evaluate_predictions(y_test.tolist(), pred.tolist())

    fpr, tpr, _ = roc_curve(y_test, prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall_curve, precision_curve)

    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)

    prediction_rows: List[Dict[str, str]] = []
    for row, p_score, p_label, actual in zip(test_rows, prob.tolist(), pred.tolist(), y_test.tolist()):
        prediction_rows.append(
            {
                "COLLISION_ID": row.get("COLLISION_ID", ""),
                "CRASH DATE": row.get("CRASH DATE", ""),
                "predicted_probability": f"{float(p_score):.6f}",
                "predicted_label": str(int(p_label)),
                "actual_label": str(int(actual)),
            }
        )

    metrics["roc_auc"] = float(roc_auc)
    metrics["pr_auc"] = float(pr_auc)

    curve_data: Dict[str, object] = {
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "pr_recall": recall_curve.tolist(),
        "pr_precision": precision_curve.tolist(),
        "calibration_mean_pred": mean_pred.tolist(),
        "calibration_frac_pos": frac_pos.tolist(),
    }

    return "sklearn_logistic", metrics, prediction_rows, curve_data


def train_rule_baseline(test_rows: List[Dict[str, str]]) -> Tuple[str, Dict[str, float], List[Dict[str, str]], Dict[str, object]]:
    """Run rule-based fallback model when sklearn is unavailable.

    Args:
        test_rows: Testing rows.

    Returns:
        Tuple[str, Dict[str, float], List[Dict[str, str]], Dict[str, object]]: Method, metrics, prediction rows, and curve data.

    Raises:
        None
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    prediction_rows: List[Dict[str, str]] = []

    for row in test_rows:
        hour = to_float(row.get("HOUR", "0"))
        is_weekend = to_float(row.get("IS_WEEKEND", "0"))
        score = 0.2
        if hour <= 4 or hour >= 20:
            score += 0.2
        if is_weekend >= 1:
            score += 0.15

        score = max(0.01, min(0.99, score))
        label = 1 if score >= 0.5 else 0
        actual = get_target(row)

        y_prob.append(score)
        y_pred.append(label)
        y_true.append(actual)

        prediction_rows.append(
            {
                "COLLISION_ID": row.get("COLLISION_ID", ""),
                "CRASH DATE": row.get("CRASH DATE", ""),
                "predicted_probability": f"{score:.6f}",
                "predicted_label": str(label),
                "actual_label": str(actual),
            }
        )

    metrics = evaluate_predictions(y_true, y_pred)
    curve_data: Dict[str, object] = {
        "score_bins": [str(x / 10.0) for x in range(1, 11)],
        "score_counts": [
            sum(1 for score in y_prob if (x - 1) / 10.0 < score <= x / 10.0) for x in range(1, 11)
        ],
    }

    return "rule_baseline", metrics, prediction_rows, curve_data


def save_model_plots(method: str, curve_data: Dict[str, object], visual_root: Path) -> List[str]:
    """Save model performance plots.

    Args:
        method: Model method name.
        curve_data: Curve data dictionary.
        visual_root: Visualization root directory.

    Returns:
        List[str]: Plot status messages.

    Raises:
        None
    """
    messages: List[str] = []
    models_dir = visual_root / "models"
    ensure_directory(models_dir)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return ["Skipped model plots because matplotlib is not installed."]

    if method == "sklearn_logistic":
        roc_fpr = curve_data.get("roc_fpr", [])
        roc_tpr = curve_data.get("roc_tpr", [])
        pr_recall = curve_data.get("pr_recall", [])
        pr_precision = curve_data.get("pr_precision", [])
        calib_x = curve_data.get("calibration_mean_pred", [])
        calib_y = curve_data.get("calibration_frac_pos", [])

        plt.figure(figsize=(6, 5))
        plt.plot(roc_fpr, roc_tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(models_dir / "roc_curve.png")
        plt.close()
        messages.append("Created ROC curve plot.")

        plt.figure(figsize=(6, 5))
        plt.plot(pr_recall, pr_precision)
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(models_dir / "precision_recall_curve.png")
        plt.close()
        messages.append("Created precision-recall curve plot.")

        plt.figure(figsize=(6, 5))
        plt.plot(calib_x, calib_y, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("Calibration Curve")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Observed Positive Rate")
        plt.tight_layout()
        plt.savefig(models_dir / "calibration_curve.png")
        plt.close()
        messages.append("Created calibration curve plot.")
    else:
        score_bins = curve_data.get("score_bins", [])
        score_counts = curve_data.get("score_counts", [])
        save_bar_plot(
            [str(item) for item in score_bins],
            [float(item) for item in score_counts],
            "Rule Baseline Score Distribution",
            "Score Bin",
            "Count",
            models_dir / "rule_score_distribution.png",
        )
        messages.append("Created rule-baseline score distribution plot.")

    return messages


def main() -> None:
    """Run severity model stage.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If stage fails.
    """
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)
    logger = get_logger("severity_models.baseline")

    feature_csv = Path(config["feature_csv"]).resolve()
    tables_dir = Path(config["tables_dir"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()
    visual_root = Path(config["visualizations_dir"]).resolve()

    max_rows = int(config.get("max_rows_for_model", "250000"))
    seed = int(config.get("random_seed", "143"))

    log_step(logger, "MODEL_START", f"Dataset: {config.get('dataset_name', 'unknown')}")
    train_rows, test_rows = sample_train_test_rows(feature_csv, max_rows=max_rows, seed=seed)

    method = "sklearn_logistic"
    try:
        method, metrics, predictions, curve_data = train_with_sklearn(train_rows, test_rows)
    except ImportError:
        method, metrics, predictions, curve_data = train_rule_baseline(test_rows)

    write_csv_rows(
        tables_dir / "severity_model_predictions_sample.csv",
        ["COLLISION_ID", "CRASH DATE", "predicted_probability", "predicted_label", "actual_label"],
        predictions[:1000],
    )

    metrics_rows = [{"metric": key, "value": f"{value:.6f}"} for key, value in metrics.items()]
    write_csv_rows(tables_dir / "severity_model_metrics.csv", ["metric", "value"], metrics_rows)

    plot_messages = save_model_plots(method, curve_data, visual_root)

    report_lines = [
        f"Dataset: {config.get('dataset_name', 'unknown')}",
        f"Model method: {method}",
        f"Train rows used: {len(train_rows)}",
        f"Test rows used: {len(test_rows)}",
        "",
        "Metrics:",
    ]
    for key, value in metrics.items():
        report_lines.append(f"- {key}: {value:.6f}")
    report_lines.append("")
    report_lines.append("Plot status:")
    for message in plot_messages:
        report_lines.append(f"- {message}")

    write_markdown(reports_dir / "severity_model_report.md", "Severity Model Report", report_lines)
    log_step(logger, "MODEL_DONE", f"Method: {method}")


if __name__ == "__main__":
    main()
