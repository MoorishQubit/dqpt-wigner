"""Portable labels for data and execution paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def repository_path(path: Path) -> str:
    path = (ROOT / path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
