"""OmegaConf merge and validated runtime config loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

from .spec_schema import RuntimeConfig, validate_config_dict


def _load_yaml(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


def merge_yaml_configs(base_path: Path | str, override_paths: Sequence[Path | str] | None = None) -> DictConfig:
    """Merge base config with optional override files, preserving order."""

    merged_stack: list[DictConfig] = [_load_yaml(Path(base_path))]
    for override_path in override_paths or []:
        merged_stack.append(_load_yaml(Path(override_path)))
    return OmegaConf.merge(*merged_stack)


def load_runtime_config(
    base_path: Path | str,
    override_paths: Sequence[Path | str] | None = None,
) -> RuntimeConfig:
    """Load and validate runtime configuration."""

    merged = merge_yaml_configs(base_path=base_path, override_paths=override_paths)
    resolved = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved config is not a dictionary")
    return validate_config_dict(resolved)
