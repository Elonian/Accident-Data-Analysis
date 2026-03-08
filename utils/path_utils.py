"""Path helper functions used across pipeline modules."""

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Return the absolute project root path.

    Returns:
        Path: Absolute path to the project root.

    Raises:
        RuntimeError: If the current file is not inside the expected project tree.
    """
    root = Path(__file__).resolve().parents[1]
    if not root.exists():
        raise RuntimeError("Project root could not be resolved.")
    return root


def ensure_directory(path: Union[str, Path]) -> Path:
    """Create a directory if missing and return its path.

    Args:
        path: Directory path to create.

    Returns:
        Path: Absolute directory path.

    Raises:
        ValueError: If the provided path is empty.
    """
    if not path:
        raise ValueError("Directory path cannot be empty.")
    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(relative_or_absolute: Union[str, Path]) -> Path:
    """Resolve a path against project root when not absolute.

    Args:
        relative_or_absolute: Relative or absolute input path.

    Returns:
        Path: Absolute resolved path.

    Raises:
        ValueError: If the input value is empty.
    """
    if not relative_or_absolute:
        raise ValueError("Path value cannot be empty.")
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate
    return (get_project_root() / candidate).resolve()
