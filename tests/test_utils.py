"""Tests for config loading, config hashing, seeding, and logging.

The hashing tests are the ones that matter: the config hash decides which
cached artifact a run reuses, so a hash that is unstable across processes
silently recomputes everything, and a hash that ignores a substantive key
silently returns the wrong file.
"""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from micm.utils import (
    config_hash,
    configure_logging,
    generator_for,
    get_logger,
    load_config,
    resolved_container,
    root_generator,
    spawn_generators,
)
from micm.utils.hashing import EXCLUDED_TOP_LEVEL_KEYS, resolved_payload


def test_base_config_loads_and_resolves() -> None:
    cfg = load_config("base")
    container = resolved_container(cfg)
    assert container["seed"] == 1337
    # Exactly these two. A path nobody reads sends a reviewer looking for the
    # stage that writes it; see docs/decisions.md D54.
    assert set(container["paths"]) == {"data_raw", "artifacts"}


def test_load_config_is_struct_mode() -> None:
    """A typo in a key must raise instead of creating a new key."""
    with pytest.raises(ConfigCompositionException):
        load_config("base", overrides=("sed=7",))


def test_config_hash_stable_across_two_loads() -> None:
    assert config_hash(load_config("base")) == config_hash(load_config("base"))


def test_config_hash_stable_across_processes() -> None:
    """Guards against Python's salted `hash()` sneaking into the digest."""
    code = (
        "from micm.utils import config_hash, load_config;"
        "print(config_hash(load_config('base')))"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert runs[0] == runs[1] == config_hash(load_config("base"))


def test_config_hash_changes_on_substantive_key() -> None:
    assert config_hash(load_config("base")) != config_hash(load_config("base", overrides=("seed=7",)))


@pytest.mark.parametrize("override", ["n_workers=1", "device=cpu", "logging.level=DEBUG"])
def test_config_hash_ignores_excluded_keys(override: str) -> None:
    """Keys that cannot change a number must not invalidate a cache."""
    assert config_hash(load_config("base")) == config_hash(
        load_config("base", overrides=(override,))
    )


def test_excluded_keys_are_absent_from_the_payload() -> None:
    payload = resolved_payload(load_config("base"))
    assert EXCLUDED_TOP_LEVEL_KEYS.isdisjoint(payload)


def test_config_hash_is_order_independent() -> None:
    """Two mappings differing only in key order describe the same experiment."""
    a = OmegaConf.create({"seed": 1, "mapping": {"name": "s1", "v_max": 1.0}})
    b = OmegaConf.create({"mapping": {"v_max": 1.0, "name": "s1"}, "seed": 1})
    assert config_hash(a) == config_hash(b)


def test_config_hash_respects_sequence_order() -> None:
    """Subject and lambda lists are ordered; reordering them is a different run."""
    a = OmegaConf.create({"lambdas": [0.0, 0.5]})
    b = OmegaConf.create({"lambdas": [0.5, 0.0]})
    assert config_hash(a) != config_hash(b)


def test_config_hash_rejects_unordered_containers() -> None:
    with pytest.raises(TypeError):
        config_hash({"subjects": {1, 2, 3}})


@pytest.mark.parametrize("length", [0, 33])
def test_config_hash_rejects_bad_length(length: int) -> None:
    with pytest.raises(ValueError, match=r"length must be in 1\.\.32"):
        config_hash({"seed": 1}, length=length)


def test_root_generator_is_reproducible() -> None:
    assert root_generator(1337).random() == root_generator(1337).random()


def test_spawn_generators_are_reproducible_and_distinct() -> None:
    first = [g.random() for g in spawn_generators(1337, 4)]
    second = [g.random() for g in spawn_generators(1337, 4)]
    assert first == second
    assert len(set(first)) == 4


def test_spawn_generators_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        spawn_generators(1337, -1)


def test_generator_for_is_content_addressed() -> None:
    """The same cell identity must give the same stream regardless of loop order."""
    assert generator_for(1337, 3, "s1_argmax", 0.4).random() == (
        generator_for(1337, 3, "s1_argmax", 0.4).random()
    )
    assert generator_for(1337, 3, "s1_argmax", 0.4).random() != (
        generator_for(1337, 3, "s1_argmax", 0.6).random()
    )


def test_generator_for_distinguishes_types() -> None:
    """`1` and `"1"` name different cells and must not collide."""
    assert generator_for(1337, 1).random() != generator_for(1337, "1").random()


def test_configure_logging_does_not_duplicate_handlers() -> None:
    """The property is that a second call adds nothing, so the count is compared
    against itself. An absolute count would also be counting pytest's own
    capture handlers, which attach to `micm` because it does not propagate."""
    configure_logging(logging.DEBUG)
    before = len(logging.getLogger("micm").handlers)
    configure_logging(logging.INFO)
    assert len(logging.getLogger("micm").handlers) == before
    assert before >= 1


def test_get_logger_nests_under_micm() -> None:
    assert get_logger("micm.env.task").name == "micm.env.task"
    assert get_logger("scratch").name == "micm.scratch"
