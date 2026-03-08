"""Initialize project folders and dataset-specific config files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import write_json
from utils.log_utils import get_logger, log_step
from utils.path_utils import ensure_directory


def build_dataset_config(dataset_name: str, source_file_name: str) -> Dict[str, str]:
    """Build one dataset-specific runtime configuration.

    Args:
        dataset_name: Short config name (for example `dataset_2020`).
        source_file_name: Source CSV file name from the data folder.

    Returns:
        Dict[str, str]: Config dictionary.

    Raises:
        ValueError: If input values are empty.
    """
    if not dataset_name or not source_file_name:
        raise ValueError("dataset_name and source_file_name are required.")

    base_output = PROJECT_ROOT / "outputs" / dataset_name
    base_visual = PROJECT_ROOT / "visualizations" / dataset_name

    return {
        "dataset_name": dataset_name,
        "source_csv": str(PROJECT_ROOT / "data" / source_file_name),
        "canonical_csv": str(base_output / "tables" / "canonical_dataset.csv"),
        "feature_csv": str(base_output / "tables" / "feature_table.csv"),
        "reports_dir": str(base_output / "reports"),
        "tables_dir": str(base_output / "tables"),
        "models_dir": str(base_output / "models"),
        "visualizations_dir": str(base_visual),
        "random_seed": "143",
        "max_rows_for_model": "250000",
        "cluster_sample_limit": "120000",
        "plot_sample_limit": "60000",
    }


def create_project_directories(configs: List[Dict[str, str]]) -> None:
    """Create common and dataset-specific directories.

    Args:
        configs: List of dataset config dictionaries.

    Returns:
        None

    Raises:
        OSError: If directory creation fails.
    """
    ensure_directory(PROJECT_ROOT / "configs")
    ensure_directory(PROJECT_ROOT / "outputs")
    ensure_directory(PROJECT_ROOT / "visualizations")

    for config in configs:
        ensure_directory(Path(config["reports_dir"]))
        ensure_directory(Path(config["tables_dir"]))
        ensure_directory(Path(config["models_dir"]))

        visual_root = Path(config["visualizations_dir"])
        ensure_directory(visual_root / "core")
        ensure_directory(visual_root / "heatmaps")
        ensure_directory(visual_root / "models")
        ensure_directory(visual_root / "clusters")


def write_dataset_configs(configs: List[Dict[str, str]]) -> List[Path]:
    """Write config files to the configs folder.

    Args:
        configs: List of config dictionaries.

    Returns:
        List[Path]: Paths to written config files.

    Raises:
        OSError: If file writing fails.
    """
    written: List[Path] = []
    for config in configs:
        config_path = PROJECT_ROOT / "configs" / f"{config['dataset_name']}.json"
        write_json(config_path, config)
        written.append(config_path)

    # Keep run_config.json pointing to full dataset by default.
    full_config = next(item for item in configs if item["dataset_name"] == "full_multiyear")
    run_config_path = PROJECT_ROOT / "configs" / "run_config.json"
    write_json(run_config_path, full_config)
    written.append(run_config_path)
    return written


def main() -> None:
    """Run project initialization.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If initialization fails.
    """
    logger = get_logger("setup.init_project")
    log_step(logger, "SETUP", "Creating dataset configs and folders.")

    try:
        configs = [
            build_dataset_config("snapshot_2020", "NYC Accidents 2020.csv"),
            build_dataset_config("full_multiyear", "NYC_Collisions_Official_full.csv"),
        ]
        create_project_directories(configs)
        written = write_dataset_configs(configs)
        for path in written:
            log_step(logger, "CONFIG_WRITTEN", str(path))
        log_step(logger, "SETUP_DONE", "Initialization complete.")
    except Exception as error:  # pragma: no cover
        raise RuntimeError(f"Initialization failed: {error}") from error


if __name__ == "__main__":
    main()
