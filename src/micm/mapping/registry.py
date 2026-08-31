"""Config name to mapping class.

A plain dict, like the decoder registry. No dynamic import, no plugin discovery,
no `eval`.

Keys are **config names**, not class names, so one class can back several
configured variants. S2 with and without entropy scaling are different
conditions in the experiment and need different values in the `mapping` column,
but they are the same twenty lines of code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from micm.mapping.argmax import ArgmaxMapping
from micm.mapping.base import Mapping
from micm.mapping.evidence import EvidenceMapping
from micm.mapping.shared import SharedMapping
from micm.mapping.weighted import WeightedMapping

MAPPINGS: dict[str, type[Any]] = {
    "s1_argmax": ArgmaxMapping,
    "s2_weighted": WeightedMapping,
    "s2_weighted_entropy": WeightedMapping,
    "s3_evidence": EvidenceMapping,
    "s4_shared": SharedMapping,
}


def build_mapping(cfg: DictConfig, *, directions: np.ndarray) -> Mapping:
    """Instantiate a mapping from its config node.

    The node carries `name` and a `params` block holding exactly the constructor
    keywords, so a misspelled key raises rather than sitting unused in the saved
    run config. `directions` is passed in rather than configured: it is permuted
    per seed by the task, so it belongs to the episode, not to the mapping.

    Raises:
        KeyError: on an unknown name or a missing `params` block.
        TypeError: if `params` does not match the constructor.
    """
    node = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(node, dict):
        raise TypeError(f"mapping config must be a mapping, got {type(node).__name__}")

    name = node.get("name")
    if name not in MAPPINGS:
        raise KeyError(f"unknown mapping {name!r}, available: {sorted(MAPPINGS)}")

    params = node.get("params")
    if not isinstance(params, dict):
        raise KeyError(f"mapping {name!r} config has no 'params' block")

    mapping: Mapping = MAPPINGS[str(name)](directions=directions, **params)
    return mapping


def select_mapping(
    cfg: DictConfig,
    name: str,
    *,
    directions: np.ndarray,
    overrides: dict[str, Any] | None = None,
) -> Mapping:
    """Build the mapping a cell names, from the `mappings` block of a run config.

    The grid varies the mapping, so the runner cannot use a single composed
    `mapping` group. Every mapping the experiment needs is composed into
    `cfg.mappings` under its own key, and this picks the one for the cell.

    `overrides` carries the values the grid varies per cell rather than per
    config, such as `alpha` and `intent_mode`. A `None` override is ignored,
    because most cells have no autonomy weight to set. A non-`None` override for
    a parameter the mapping does not have raises: a cell that specifies an alpha
    for argmax is a mistake in the grid, and applying it silently to nothing
    would leave the episode row claiming a condition that never took effect.

    Raises:
        KeyError: if the experiment did not compose that mapping, or an override
            names a parameter it does not have.
    """
    if name not in cfg.mappings:
        raise KeyError(
            f"the experiment config does not compose mapping {name!r}; add it to the "
            f"defaults list as `- /mapping@mappings.{name}: {name}`. "
            f"Composed: {sorted(cfg.mappings)}"
        )

    node = OmegaConf.to_container(cfg.mappings[name], resolve=True, throw_on_missing=True)
    if not isinstance(node, dict):
        raise TypeError(f"mapping config must be a mapping, got {type(node).__name__}")

    params = node.get("params")
    if not isinstance(params, dict):
        raise KeyError(f"mapping {name!r} config has no 'params' block")

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key not in params:
            raise KeyError(
                f"the grid sets {key}={value!r} but mapping {name!r} has no such parameter; "
                "the episode row would record a condition that never took effect"
            )
        params[key] = value

    return build_mapping(OmegaConf.create(node), directions=directions)
