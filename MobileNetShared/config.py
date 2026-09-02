from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Config file not found: {path.resolve()}")
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def merged_section(
    common_config: dict[str, Any],
    experiment_config: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    return {
        **section(common_config, name),
        **section(experiment_config, name),
    }


def optional_path(value: str | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(str(value))


def extensions(value: list[str] | str) -> frozenset[str]:
    parts = value.split(",") if isinstance(value, str) else value
    return frozenset(part.strip().lower() for part in map(str, parts) if part.strip())
