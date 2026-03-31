"""Small utilities shared across the experiment scripts."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable, Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path_like: str | Path) -> Path:
    """Resolve a path relative to the project root unless already absolute."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path_like: str | Path) -> Path:
    """Create a directory if needed and return its resolved path."""
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path_like: str | Path) -> dict[str, Any]:
    """Load a YAML file into a Python dictionary."""
    path = resolve_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def write_yaml(path_like: str | Path, data: dict[str, Any]) -> Path:
    """Write a YAML file with stable ordering."""
    path = resolve_path(path_like)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return path


def read_jsonl(path_like: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    path = resolve_path(path_like)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path_like: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write rows to JSONL format."""
    path = resolve_path(path_like)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_json(path_like: str | Path, data: dict[str, Any]) -> Path:
    """Write a JSON file with indentation for inspection."""
    path = resolve_path(path_like)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return path
