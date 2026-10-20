"""Load reproducible experiment settings from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load one experiment configuration from a YAML mapping."""
    config_path = Path(path)

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError("experiment configuration must be a YAML mapping")

    return config