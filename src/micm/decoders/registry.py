"""Config name to decoder class.

A plain dict. No dynamic import, no plugin discovery, no `eval`. Adding a
decoder means one import and one entry, and an unknown name fails immediately
with the list of what is available rather than at the first call.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from micm.decoders.base import Decoder
from micm.decoders.fbcsp import FBCSPDecoder
from micm.decoders.riemann import RiemannDecoder

DECODERS: dict[str, type[Any]] = {
    FBCSPDecoder.name: FBCSPDecoder,
    RiemannDecoder.name: RiemannDecoder,
}


def build_decoder(cfg: DictConfig) -> Decoder:
    """Instantiate a decoder from its config node.

    The node carries `name`, a `params` block holding exactly the constructor
    keywords, and any metadata the pipeline reads but the decoder does not, such
    as `expected_kappa`. Keeping the constructor arguments in their own block is
    what lets a stray key raise instead of being ignored: everything under
    `params` is passed through, so an unrecognised one is a `TypeError` rather
    than a value that silently never reaches the decoder.

    Raises:
        KeyError: on an unknown decoder name or a missing `params` block.
        TypeError: if `params` does not match the decoder's constructor.
    """
    node = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(node, dict):
        raise TypeError(f"decoder config must be a mapping, got {type(node).__name__}")

    name = node.get("name")
    if name not in DECODERS:
        raise KeyError(f"unknown decoder {name!r}, available: {sorted(DECODERS)}")

    params = node.get("params")
    if not isinstance(params, dict):
        raise KeyError(f"decoder {name!r} config has no 'params' block")

    decoder: Decoder = DECODERS[str(name)](**params)
    return decoder
