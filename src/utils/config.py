"""Configuration loading with explicit path validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject missing or malformed configuration."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    for section in ("project", "paths", "data", "runtime"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing mapping section '{section}' in {path}")
    return config

