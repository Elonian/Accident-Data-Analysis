"""Run validation checks for one dataset-specific pipeline run."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import load_config, parse_config_path
from utils.io_utils import write_markdown
from utils.log_utils import get_logger, log_step


def build_expected_paths(config: Dict[str, str]) -> List[Path]:
    """Build expected output path list for validation.

    Args:
        config: Runtime config dictionary.

    Returns:
        List[Path]: Expected output paths.

    Raises:
        KeyError: If required config keys are missing.
    """
    tables_dir = Path(config["tables_dir"]).resolve()
    reports_dir = Path(config["reports_dir"]).resolve()
    visual_root = Path(config["visualizations_dir"]).resolve()

    return [
        Path(config["canonical_csv"]).resolve(),
        Path(config["feature_csv"]).resolve(),
        tables_dir / "monthly_kpi.csv",
        tables_dir / "hourly_kpi.csv",
        tables_dir / "severity_model_metrics.csv",
        tables_dir / "top_spatial_clusters.csv",
        reports_dir / "canonical_dataset_report.md",
        reports_dir / "data_quality_report.md",
        reports_dir / "feature_engineering_report.md",
        reports_dir / "exploratory_analysis_report.md",
        reports_dir / "severity_model_report.md",
        reports_dir / "spatial_clustering_report.md",
        visual_root / "core" / "monthly_collision_trend.png",
        visual_root / "heatmaps" / "weekday_hour_collision_heatmap.png",
        visual_root / "clusters" / "spatial_collision_sample.png",
    ]


def run_validation(expected_paths: List[Path]) -> List[str]:
    """Validate output existence and file size.

    Args:
        expected_paths: Paths to validate.

    Returns:
        List[str]: Validation status lines.

    Raises:
        None
    """
    lines: List[str] = []
    for path in expected_paths:
        if path.exists() and path.stat().st_size > 0:
            lines.append(f"PASS: {path}")
        elif path.exists():
            lines.append(f"FAIL: {path} exists but is empty")
        else:
            lines.append(f"FAIL: {path} missing")
    return lines


def main() -> None:
    """Run validation stage for one dataset config.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If any required output is missing.
    """
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)
    logger = get_logger("validation.checks")

    log_step(logger, "VALIDATION_START", f"Dataset: {config.get('dataset_name', 'unknown')}")
    expected_paths = build_expected_paths(config)
    lines = run_validation(expected_paths)

    report_path = Path(config["reports_dir"]).resolve() / "validation_checks_report.md"
    write_markdown(report_path, "Validation Checks Report", lines)

    failed = [line for line in lines if line.startswith("FAIL")]
    if failed:
        raise RuntimeError(f"Validation failed with {len(failed)} failing checks.")

    log_step(logger, "VALIDATION_DONE", f"All {len(lines)} checks passed.")


if __name__ == "__main__":
    main()
