"""Run all pipeline stages sequentially for one dataset config."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_utils import parse_config_path
from utils.log_utils import get_logger, log_step


STAGES: List[str] = [
    "scripts/data_ingestion/build_canonical_dataset.py",
    "scripts/data_quality/run_data_quality_checks.py",
    "scripts/feature_engineering/build_feature_table.py",
    "scripts/exploratory_analysis/create_core_outputs.py",
    "scripts/severity_models/train_baseline_severity_model.py",
    "scripts/clustering_analysis/run_spatial_clustering.py",
    "scripts/validation_tests/run_validation_checks.py",
]


def run_stage(project_root: Path, script_relative: str, config_path: Path) -> None:
    """Run one pipeline stage with a config argument.

    Args:
        project_root: Project root path.
        script_relative: Relative stage script path.
        config_path: Config JSON path.

    Returns:
        None

    Raises:
        RuntimeError: If stage script fails.
    """
    script_path = project_root / script_relative
    if not script_path.exists():
        raise RuntimeError(f"Stage script not found: {script_relative}")

    result = subprocess.run(
        [sys.executable, str(script_path), "--config", str(config_path)],
        cwd=str(project_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Stage failed: {script_relative}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def main() -> None:
    """Run full stage sequence for a config.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If any stage fails.
    """
    config_path = parse_config_path(sys.argv[1:])
    logger = get_logger("setup.run_pipeline")
    log_step(logger, "PIPELINE_CONFIG", str(config_path))

    for stage in STAGES:
        log_step(logger, "RUN_STAGE", stage)
        run_stage(PROJECT_ROOT, stage, config_path)

    log_step(logger, "PIPELINE_DONE", "All stages completed.")


if __name__ == "__main__":
    main()
