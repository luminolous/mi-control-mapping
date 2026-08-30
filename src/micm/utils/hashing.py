"""Stable hashing of resolved configs.

Every artifact in this project is named by the hash of the config that produced
it. The hash therefore has one hard requirement: two configs that produce
identical numbers must hash identically, and two configs that produce different
numbers must not. Keys that cannot affect a result are removed before hashing,
and that list is written out explicitly rather than inferred, because an
accidentally-included key silently invalidates every cache on the machine.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

from omegaconf import DictConfig, OmegaConf

# Top-level config keys that cannot change any number a run produces. Anything
# not listed here is assumed to matter. Adding a key to this list invalidates
# the assumption that equal hash implies equal results, so add deliberately.
EXCLUDED_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "paths",  # where files are read and written
        "n_workers",  # parallelism, results are seeded per episode
        "device",  # cuda or cpu; decoders must be deterministic on both
        "logging",  # verbosity
        "hydra",  # Hydra's own runtime block
        "progress_every",  # how often the runner logs an ETA
    }
)

_HASH_LENGTH: Final[int] = 8


def _canonical(obj: Any) -> Any:
    """Convert a resolved config value into something json.dumps orders stably.

    Mappings are recursed with sorted keys. Sequences keep their order, because
    order is meaningful in this project (subject lists, lambda grids). Sets are
    not expected in configs and raise rather than being silently ordered.
    """
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        raise TypeError("sets have no stable order and must not appear in a config")
    return obj


def resolved_payload(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Return the hashable payload of a config: resolved, filtered, canonical.

    Exposed separately from `config_hash` so a hash mismatch can be debugged by
    diffing two payloads instead of two opaque digests.
    """
    if isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    else:
        container = dict(cfg)

    if not isinstance(container, dict):
        raise TypeError(f"config must resolve to a mapping, got {type(container).__name__}")

    kept = {k: v for k, v in container.items() if str(k) not in EXCLUDED_TOP_LEVEL_KEYS}
    canonical = _canonical(kept)
    if not isinstance(canonical, dict):  # pragma: no cover - guarded by the check above
        raise TypeError("canonical payload must be a mapping")
    return canonical


def config_hash(cfg: DictConfig | dict[str, Any], *, length: int = _HASH_LENGTH) -> str:
    """Stable short hash of a resolved config.

    Excludes the keys in `EXCLUDED_TOP_LEVEL_KEYS`, which cannot affect results.
    Two loads of the same config file give the same hash within and across
    processes: the digest is blake2b over JSON with sorted keys, not Python's
    salted `hash()`.

    Raises:
        TypeError: if the config contains a value with no stable ordering, or
            does not resolve to a mapping.
        ValueError: if `length` is outside 1..32.
    """
    if not 1 <= length <= 32:
        raise ValueError(f"length must be in 1..32, got {length}")

    payload = resolved_payload(cfg)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()[:length]
