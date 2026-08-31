"""Config name to mapping class.

A plain dict, like the decoder registry. No dynamic import, no plugin discovery,
no `eval`. Adding a mapping means one import and one entry.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from micm.mapping.argmax import ArgmaxMapping
from micm.mapping.base import Mapping

MAPPINGS: dict[str, type[Any]] = {
    ArgmaxMapping.name: ArgmaxMapping,
}


def build_mapping(cfg: DictConfig, *, directions: np.ndarray) -> Mapping:
    """Instantiate a mapping from its config node.

    The node carries `name` and a `params` block holding exactly the constructor
    keywords, so a misspelled key raises rather than sitting unused in the saved
    run config. `directions` is passed in rather than configured: it is permuted
    per seed by the task, so it is a property of the episode, not of the mapping.

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
