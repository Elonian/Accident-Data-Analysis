"""I/O helper functions for reading and writing project artifacts."""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from utils.path_utils import ensure_directory


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read CSV rows into a list of dictionaries.

    Args:
        csv_path: Input CSV file path.

    Returns:
        List[Dict[str, str]]: Loaded row dictionaries.

    Raises:
        FileNotFoundError: If input CSV file does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv_rows(csv_path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, str]]) -> None:
    """Write row dictionaries to a CSV file.

    Args:
        csv_path: Output CSV file path.
        fieldnames: Ordered column names.
        rows: Row dictionary iterable.

    Returns:
        None

    Raises:
        ValueError: If fieldnames is empty.
    """
    if not fieldnames:
        raise ValueError("Fieldnames cannot be empty.")

    ensure_directory(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(md_path: Path, title: str, lines: Sequence[str]) -> None:
    """Write a markdown file from title and line content.

    Args:
        md_path: Output markdown path.
        title: Document title.
        lines: Body lines to write.

    Returns:
        None

    Raises:
        ValueError: If title is empty.
    """
    if not title:
        raise ValueError("Markdown title cannot be empty.")

    ensure_directory(md_path.parent)
    content = [f"# {title}", ""] + list(lines)
    md_path.write_text("\n".join(content), encoding="utf-8")


def write_json(json_path: Path, payload: Dict) -> None:
    """Write a dictionary payload to JSON.

    Args:
        json_path: Output JSON file path.
        payload: Dictionary payload.

    Returns:
        None

    Raises:
        TypeError: If payload is not a dictionary.
    """
    if not isinstance(payload, dict):
        raise TypeError("Payload must be a dictionary.")

    ensure_directory(json_path.parent)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
