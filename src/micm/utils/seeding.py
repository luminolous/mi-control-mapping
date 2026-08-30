"""Deterministic random number generation.

Two runs with the same config must produce identical outputs. That rules out
the global `np.random.*` functions, module-level generators, and the common
`seed + i` trick for parallel workers, which correlates streams that are
supposed to be independent. Everything here builds `np.random.Generator`
objects from an explicit seed via `SeedSequence`.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np

# Number of 32-bit words drawn from the key digest when deriving a stream.
# Four words (128 bits) is far more entropy than the number of episodes needs,
# and costs nothing.
_KEY_WORDS: Final[int] = 4


def root_generator(seed: int) -> np.random.Generator:
    """Generator for the top of a run. Use `spawn_generators` for parallel work."""
    return np.random.default_rng(np.random.SeedSequence(seed))


def spawn_generators(seed: int, n: int) -> list[np.random.Generator]:
    """Return `n` independent generators derived from `seed`.

    Uses `SeedSequence.spawn`, which guarantees the child streams do not
    overlap. Never derive worker seeds as `seed + i`: adjacent seeds produce
    correlated streams for some bit generators, and the correlation is invisible
    in aggregate statistics.

    Raises:
        ValueError: if `n` is negative.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(n)]


def _key_words(key: tuple[object, ...]) -> list[int]:
    """Hash a key tuple into 32-bit words for use as SeedSequence entropy.

    Uses blake2b rather than `hash()`, which is salted per process and would
    make runs irreproducible across invocations.
    """
    encoded = "\x1f".join(f"{type(part).__name__}:{part!r}" for part in key).encode("utf-8")
    digest = hashlib.blake2b(encoded, digest_size=4 * _KEY_WORDS).digest()
    return [
        int.from_bytes(digest[i * 4 : (i + 1) * 4], "big", signed=False)
        for i in range(_KEY_WORDS)
    ]


def generator_for(seed: int, *key: object) -> np.random.Generator:
    """Generator identified by its content, not by its position in a loop.

    An episode's randomness should depend on which cell it belongs to, so that
    re-running a subset of the grid reproduces exactly the same episodes as the
    full grid did. Passing the cell identity as `key` gives that property:

        rng = generator_for(cfg.seed, subject, mapping, lam, episode_seed)

    The key is hashed by value, so the same key always gives the same stream and
    two different keys effectively never collide.
    """
    return np.random.default_rng(np.random.SeedSequence([seed, *_key_words(key)]))
