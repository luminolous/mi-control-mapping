"""Config loading.

Hydra composes every experiment; nothing that affects a number lives outside
`configs/`. Two rules matter here. Struct mode is on, so a typo in a key raises
instead of silently creating a new one. And configs are read, never patched:
there is no helper in this module for setting a value, because a value set from
Python would not appear in the config saved next to the run output.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

_CONFIG_DIR_ENV: Final[str] = "MICM_CONFIG_DIR"
_MARKER: Final[str] = "base.yaml"
_HYDRA_VERSION_BASE: Final[str] = "1.3"


def default_config_dir() -> Path:
    """Locate the repository `configs/` directory.

    Checks `MICM_CONFIG_DIR` first, then walks up from this file looking for a
    `configs/base.yaml`. The walk exists so an editable install run from any
    working directory still finds the configs.

    Raises:
        FileNotFoundError: if no candidate directory contains `base.yaml`.
    """
    override = os.environ.get(_CONFIG_DIR_ENV)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not (candidate / _MARKER).is_file():
            raise FileNotFoundError(
                f"{_CONFIG_DIR_ENV}={candidate} does not contain {_MARKER}"
            )
        return candidate

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs"
        if (candidate / _MARKER).is_file():
            return candidate

    raise FileNotFoundError(
        f"could not locate a configs/ directory containing {_MARKER}; "
        f"set {_CONFIG_DIR_ENV} to point at it"
    )


def load_config(
    config_name: str,
    *,
    overrides: tuple[str, ...] = (),
    config_dir: Path | None = None,
) -> DictConfig:
    """Compose a config by name and return it in struct mode.

    Args:
        config_name: name relative to the config directory, without the `.yaml`
            suffix, e.g. `"base"` or `"experiment/smoke"`.
        overrides: Hydra override strings, e.g. `("seed=7",)`. An override that
            names a key not present in the config raises.
        config_dir: config root; defaults to `default_config_dir()`.

    Returns:
        The composed config, resolved lazily, with struct mode enabled.
    """
    root = (config_dir or default_config_dir()).resolve()
    with initialize_config_dir(config_dir=str(root), version_base=_HYDRA_VERSION_BASE):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    OmegaConf.set_struct(cfg, True)
    return cfg


def resolved_container(cfg: DictConfig) -> dict[str, Any]:
    """Fully resolve a config into plain Python containers.

    Raises:
        ValueError: if the config still holds a mandatory missing value (`???`).
    """
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(container, dict):
        raise TypeError(f"config must resolve to a mapping, got {type(container).__name__}")
    return {str(k): v for k, v in container.items()}


def save_config(cfg: DictConfig, path: Path) -> None:
    """Write the fully resolved config next to a run's outputs.

    Resolved rather than raw, because an interpolation that referred to an
    environment variable would otherwise make the saved config unreadable on
    another machine. Creates parent directories.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.create(resolved_container(cfg))
    OmegaConf.save(config=resolved, f=path)
