"""Configuration helpers for dataset-specific pipeline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence


def parse_config_path(argv: Sequence[str], default_config: str = "configs/run_config.json") -> Path:
    """Parse config path from CLI arguments.

    Args:
        argv: Raw command-line argument sequence.
        default_config: Default config path to use when not provided.

    Returns:
        Path: Resolved config file path.

    Raises:
        None
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=default_config)
    args, _ = parser.parse_known_args(list(argv))
    return Path(args.config).resolve()


def load_config(config_path: Path) -> Dict[str, str]:
    """Load a JSON config file.

    Args:
        config_path: Config JSON path.

    Returns:
        Dict[str, str]: Loaded config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config payload is not a dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object.")
    return payload
