from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

def load_yaml_config(config_path: Path, required: bool = True) -> dict[str, Any]:
    if not config_path.is_file():
        if required:
            raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return data

def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value

def merged_config_section(
    common_config: dict[str, Any],
    experiment_config: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    return {
        **config_section(common_config, name),
        **config_section(experiment_config, name),
    }

def config_path_value(value: str | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(str(value))

def config_extensions(value: list[str] | str) -> frozenset[str]:
    parts = value.split(",") if isinstance(value, str) else value
    return frozenset(
        str(extension).strip().lower()
        for extension in parts
        if str(extension).strip()
    )

def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}.")

def configured_value(
    env_name: str,
    yaml_section: dict[str, Any],
    yaml_key: str,
    default: object,
) -> object:
    yaml_value = yaml_section.get(yaml_key)
    if yaml_value is not None:
        return yaml_value
    env_value = os.getenv(env_name)
    if env_value is not None:
        return env_value
    return default

def configured_bool(
    env_name: str,
    yaml_section: dict[str, Any],
    yaml_key: str,
    default: bool,
) -> bool:
    yaml_value = yaml_section.get(yaml_key)
    if yaml_value is not None:
        return bool(yaml_value)
    if os.getenv(env_name) is not None:
        return env_bool(env_name, default)
    return default
