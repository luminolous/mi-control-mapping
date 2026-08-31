"""Mapping tests: the interface contract, and S1 itself.

The update-detection test is the one that will matter later. `step` is called at
100 Hz while the posterior changes at 4 Hz, so a mapping that accumulates
evidence per call rather than per decoder update has parameters that mean
nothing. S1 has no state, so the machinery is checked here, before S3 depends on
it.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from micm.env.dynamics import RobotState
from micm.env.task import DEFAULT_DIRECTIONS, direction_table
from micm.mapping.argmax import ArgmaxMapping
from micm.mapping.base import BaseMapping, Mapping, validate_directions, validate_posterior
from micm.mapping.registry import MAPPINGS, build_mapping
from micm.utils import load_config

V_MAX = 0.15
STATE = RobotState(pos=np.zeros(2), vel=np.zeros(2), target_idx=0, dwell_t=0.0, t=0.0)
DT = 0.01


def make_argmax(directions: np.ndarray | None = None) -> ArgmaxMapping:
    return ArgmaxMapping(
        directions=DEFAULT_DIRECTIONS if directions is None else directions, v_max=V_MAX
    )


class CountingMapping(BaseMapping):
    """Records how often the update hook fires, so the 4 Hz contract can be checked."""

    name = "counting"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.updates = 0

    def on_reset(self, rng: np.random.Generator) -> None:
        self.updates = 0

    def on_new_posterior(self, posterior: np.ndarray) -> None:
        self.updates += 1

    def command(self, posterior: np.ndarray, state: RobotState, dt: float) -> np.ndarray:
        return np.zeros(2)


# --- the contract ---


def test_argmax_satisfies_the_protocol() -> None:
    assert isinstance(make_argmax(), Mapping)
    assert isinstance(make_argmax(), BaseMapping)


def test_no_posterior_gives_a_zero_command() -> None:
    """The burst protocol working as intended, not an edge case to paper over."""
    command = make_argmax().step(None, STATE, DT)
    np.testing.assert_array_equal(command, np.zeros(2))


@pytest.mark.parametrize("cls", [0, 1, 2, 3])
def test_argmax_drives_full_speed_along_the_winning_class(cls: int) -> None:
    posterior = np.full(4, 0.1)
    posterior[cls] = 0.7
    command = make_argmax().step(posterior, STATE, DT)
    np.testing.assert_allclose(command, DEFAULT_DIRECTIONS[cls] * V_MAX)


def test_argmax_ignores_everything_but_the_winner() -> None:
    """The information S1 discards is the quantity this project measures the cost of."""
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    barely = np.array([0.28, 0.24, 0.24, 0.24])
    mapping = make_argmax()
    np.testing.assert_allclose(
        mapping.step(peaked, STATE, DT), mapping.step(barely, STATE, DT)
    )


def test_argmax_follows_a_permuted_direction_table() -> None:
    permuted = direction_table(np.array([3, 2, 1, 0]))
    posterior = np.array([0.7, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(
        make_argmax(permuted).step(posterior, STATE, DT), permuted[0] * V_MAX
    )


def test_command_speed_never_exceeds_v_max() -> None:
    posterior = np.array([0.7, 0.1, 0.1, 0.1])
    command = make_argmax().step(posterior, STATE, DT)
    assert float(np.hypot(*command)) == pytest.approx(V_MAX)


# --- decoder-update detection ---


def test_the_update_hook_fires_once_per_posterior_not_once_per_step() -> None:
    """25 calls per posterior at 100 Hz against 4 Hz. Per-call accumulation is meaningless."""
    mapping = CountingMapping(directions=DEFAULT_DIRECTIONS, v_max=V_MAX)
    mapping.reset(np.random.default_rng(0))

    first = np.array([0.7, 0.1, 0.1, 0.1])
    second = np.array([0.1, 0.7, 0.1, 0.1])
    for _ in range(25):
        mapping.step(first, STATE, DT)
    for _ in range(25):
        mapping.step(second, STATE, DT)

    assert mapping.updates == 2


def test_two_equal_posteriors_still_count_as_two_updates() -> None:
    """Detection is by identity, because consecutive posteriors can hold equal values."""
    mapping = CountingMapping(directions=DEFAULT_DIRECTIONS, v_max=V_MAX)
    mapping.reset(np.random.default_rng(0))

    values = [0.7, 0.1, 0.1, 0.1]
    for _ in range(3):
        mapping.step(np.array(values), STATE, DT)
    assert mapping.updates == 3


def test_a_gap_ends_the_current_posterior() -> None:
    """After a gap the same array is a new arrival, not a continuation."""
    mapping = CountingMapping(directions=DEFAULT_DIRECTIONS, v_max=V_MAX)
    mapping.reset(np.random.default_rng(0))

    posterior = np.array([0.7, 0.1, 0.1, 0.1])
    mapping.step(posterior, STATE, DT)
    mapping.step(None, STATE, DT)
    mapping.step(posterior, STATE, DT)
    assert mapping.updates == 2


def test_reset_clears_the_update_state() -> None:
    mapping = CountingMapping(directions=DEFAULT_DIRECTIONS, v_max=V_MAX)
    mapping.reset(np.random.default_rng(0))
    mapping.step(np.array([0.7, 0.1, 0.1, 0.1]), STATE, DT)
    mapping.reset(np.random.default_rng(0))
    assert mapping.updates == 0


# --- validation ---


def test_directions_must_be_unit_vectors() -> None:
    with pytest.raises(ValueError, match="unit vectors"):
        validate_directions(np.array([[2.0, 0.0], [0.0, 1.0]]))


def test_directions_must_be_two_dimensional() -> None:
    with pytest.raises(ValueError, match=r"\(K, 2\)"):
        validate_directions(np.array([[1.0, 0.0, 0.0]]))


def test_v_max_must_be_positive() -> None:
    with pytest.raises(ValueError, match="v_max"):
        ArgmaxMapping(directions=DEFAULT_DIRECTIONS, v_max=0.0)


@pytest.mark.parametrize(
    ("posterior", "match"),
    [
        (np.array([0.5, 0.5, 0.5, 0.5]), "sum to 1"),
        (np.array([1.2, -0.2, 0.0, 0.0]), "negative"),
        (np.array([np.nan, 0.0, 0.0, 1.0]), "NaN"),
        (np.array([0.5, 0.5]), r"\(4,\)"),
    ],
)
def test_an_invalid_posterior_raises(posterior: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_posterior(posterior, 4)


def test_step_validates_the_posterior_it_is_given() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        make_argmax().step(np.full(4, 0.5), STATE, DT)


# --- registry ---


def test_registry_builds_s1_from_its_config() -> None:
    cfg = load_config("experiment/smoke")
    mapping = build_mapping(cfg.mapping, directions=DEFAULT_DIRECTIONS)
    assert isinstance(mapping, Mapping)
    assert mapping.name == "s1_argmax"


def test_the_mapping_speed_limit_follows_the_environment() -> None:
    """One speed limit, in the env config, so a mapping cannot quietly outrun the robot."""
    cfg = load_config("experiment/smoke")
    built = build_mapping(cfg.mapping, directions=DEFAULT_DIRECTIONS)
    assert isinstance(built, BaseMapping)
    assert built.v_max == pytest.approx(float(cfg.env.v_max))


def test_registry_rejects_an_unknown_name() -> None:
    cfg = OmegaConf.create({"name": "s9_magic", "params": {"v_max": 1.0}})
    assert isinstance(cfg, DictConfig)
    with pytest.raises(KeyError, match="unknown mapping"):
        build_mapping(cfg, directions=DEFAULT_DIRECTIONS)


def test_registry_rejects_a_node_without_params() -> None:
    cfg = OmegaConf.create({"name": "s1_argmax"})
    assert isinstance(cfg, DictConfig)
    with pytest.raises(KeyError, match="no 'params' block"):
        build_mapping(cfg, directions=DEFAULT_DIRECTIONS)


def test_registry_rejects_an_unexpected_param() -> None:
    cfg = OmegaConf.create({"name": "s1_argmax", "params": {"v_max": 1.0, "typo": 2}})
    assert isinstance(cfg, DictConfig)
    with pytest.raises(TypeError):
        build_mapping(cfg, directions=DEFAULT_DIRECTIONS)


def test_only_s1_is_registered_so_far() -> None:
    """T10 and T11 add the rest; this fails loudly when they do, which is the point."""
    assert set(MAPPINGS) == {"s1_argmax"}
