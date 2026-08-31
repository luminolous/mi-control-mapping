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
from micm.mapping.evidence import EvidenceMapping
from micm.mapping.registry import MAPPINGS, build_mapping, select_mapping
from micm.mapping.weighted import WeightedMapping, normalised_entropy
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


@pytest.mark.parametrize(
    "name", ["s1_argmax", "s2_weighted", "s2_weighted_entropy", "s3_evidence"]
)
def test_registry_builds_every_configured_mapping(name: str) -> None:
    cfg = load_config("experiment/smoke")
    mapping = select_mapping(cfg, name, directions=DEFAULT_DIRECTIONS)
    assert isinstance(mapping, Mapping)


def test_selecting_a_mapping_the_experiment_did_not_compose_says_how_to_add_it() -> None:
    """The grid varies the mapping, so a single composed group could only hold one."""
    cfg = load_config("experiment/smoke")
    with pytest.raises(KeyError, match="does not compose mapping"):
        select_mapping(cfg, "s4_shared", directions=DEFAULT_DIRECTIONS)


@pytest.mark.parametrize(
    "name", ["s1_argmax", "s2_weighted", "s2_weighted_entropy", "s3_evidence"]
)
def test_the_mapping_speed_limit_follows_the_environment(name: str) -> None:
    """One speed limit, in the env config, so a mapping cannot quietly outrun the robot."""
    cfg = load_config("experiment/smoke")
    built = select_mapping(cfg, name, directions=DEFAULT_DIRECTIONS)
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


def test_the_registry_holds_the_mappings_implemented_so_far() -> None:
    """S4 lands in T11; this fails loudly when it does, which is the point."""
    assert set(MAPPINGS) == {
        "s1_argmax",
        "s2_weighted",
        "s2_weighted_entropy",
        "s3_evidence",
    }


def test_one_class_can_back_several_configured_variants() -> None:
    """Registry keys are config names, so S2 with and without scaling are distinct."""
    assert MAPPINGS["s2_weighted"] is MAPPINGS["s2_weighted_entropy"]


# --- S2: posterior-weighted ---


def make_weighted(entropy_scaling: bool = False) -> WeightedMapping:
    return WeightedMapping(
        directions=DEFAULT_DIRECTIONS, v_max=V_MAX, entropy_scaling=entropy_scaling
    )


def test_a_one_hot_posterior_makes_s2_agree_with_s1() -> None:
    """The two mappings can only differ where there is uncertainty to use."""
    one_hot = np.array([0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        make_weighted().step(one_hot, STATE, DT), make_argmax().step(one_hot, STATE, DT)
    )


def test_opposing_classes_cancel() -> None:
    """Half left and half right is not a confident sideways command."""
    split = np.array([0.5, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(make_weighted().step(split, STATE, DT), np.zeros(2), atol=1e-12)


def test_s2_can_point_at_a_diagonal_that_argmax_cannot() -> None:
    """Where the structural advantage over S1 comes from: eight targets, four directions."""
    diagonal = np.array([0.0, 0.5, 0.0, 0.5])
    command = make_weighted().step(diagonal, STATE, DT)
    assert command[0] > 0.0
    assert command[0] == pytest.approx(command[1])


def test_an_uncertain_posterior_gives_a_slower_command() -> None:
    confident = np.array([0.0, 0.9, 0.05, 0.05])
    unsure = np.array([0.1, 0.4, 0.25, 0.25])
    mapping = make_weighted()
    assert float(np.hypot(*mapping.step(unsure, STATE, DT))) < float(
        np.hypot(*mapping.step(confident, STATE, DT))
    )


def test_entropy_scaling_brings_a_flat_posterior_to_a_standstill() -> None:
    uniform = np.full(4, 0.25)
    scaled = make_weighted(entropy_scaling=True).step(uniform, STATE, DT)
    np.testing.assert_allclose(scaled, np.zeros(2), atol=1e-12)


def test_entropy_scaling_leaves_a_certain_posterior_alone() -> None:
    """Scaling is by 1 - H/log K, and a one-hot posterior has zero entropy."""
    one_hot = np.array([0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        make_weighted(entropy_scaling=True).step(one_hot, STATE, DT),
        make_weighted(entropy_scaling=False).step(one_hot, STATE, DT),
    )


def test_entropy_scaling_only_ever_slows_the_command() -> None:
    posterior = np.array([0.1, 0.5, 0.2, 0.2])
    plain = float(np.hypot(*make_weighted(False).step(posterior, STATE, DT)))
    scaled = float(np.hypot(*make_weighted(True).step(posterior, STATE, DT)))
    assert 0.0 < scaled < plain


def test_normalised_entropy_spans_zero_to_one() -> None:
    assert normalised_entropy(np.array([0.0, 1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert normalised_entropy(np.full(4, 0.25)) == pytest.approx(1.0)


def test_s2_returns_zero_without_a_posterior() -> None:
    np.testing.assert_array_equal(make_weighted().step(None, STATE, DT), np.zeros(2))


# --- S3: evidence accumulation ---


def make_evidence(
    theta: float = -1.0, gamma: float = 0.8, hold_s: float = 0.5
) -> EvidenceMapping:
    mapping = EvidenceMapping(
        directions=DEFAULT_DIRECTIONS,
        v_max=V_MAX,
        gamma=gamma,
        theta=theta,
        hold_s=hold_s,
        epsilon=1e-6,
    )
    mapping.reset(np.random.default_rng(0))
    return mapping


def test_evidence_accumulates_once_per_decoder_update_not_once_per_step() -> None:
    """The test that fails if the accumulation ever moves into the per-step path.

    step runs at 100 Hz against a 4 Hz decoder, so integrating per call would add
    the same posterior twenty-five times and apply the leak twenty-five times,
    leaving gamma and theta meaningless.
    """
    posterior = np.array([0.1, 0.7, 0.1, 0.1])

    once = make_evidence(theta=1e9)
    once.step(posterior, STATE, DT)
    after_one_update = once.evidence.copy()

    held = make_evidence(theta=1e9)
    for _ in range(25):
        held.step(posterior, STATE, DT)

    np.testing.assert_allclose(held.evidence, after_one_update)


def test_two_updates_apply_the_leak_exactly_once_between_them() -> None:
    posterior = np.array([0.1, 0.7, 0.1, 0.1])
    gamma = 0.8
    mapping = make_evidence(theta=1e9, gamma=gamma)

    # One array object per arrival, held across the 25 environment steps it
    # spans. That is the contract the runner honours; building a new array each
    # step would look like 25 separate arrivals.
    first_arrival = np.array(posterior)
    for _ in range(25):
        mapping.step(first_arrival, STATE, DT)
    first = mapping.evidence.copy()

    second_arrival = np.array(posterior)
    for _ in range(25):
        mapping.step(second_arrival, STATE, DT)

    expected = gamma * first + np.log(posterior + 1e-6)
    np.testing.assert_allclose(mapping.evidence, expected, rtol=1e-9)


def test_no_command_is_emitted_before_the_threshold_is_crossed() -> None:
    """Between emissions the command is zero, which is the mapping working."""
    mapping = make_evidence(theta=1e9)
    command = mapping.step(np.array([0.1, 0.7, 0.1, 0.1]), STATE, DT)
    np.testing.assert_array_equal(command, np.zeros(2))
    assert mapping.n_emissions == 0


def test_crossing_the_threshold_emits_the_winning_direction() -> None:
    mapping = make_evidence(theta=-1.0)
    command = mapping.step(np.array([0.02, 0.94, 0.02, 0.02]), STATE, DT)
    np.testing.assert_allclose(command, DEFAULT_DIRECTIONS[1] * V_MAX)
    assert mapping.n_emissions == 1


def test_an_emission_persists_for_the_hold_and_then_stops() -> None:
    hold_s = 0.5
    mapping = make_evidence(theta=-1.0, hold_s=hold_s)
    posterior = np.array([0.02, 0.94, 0.02, 0.02])

    mapping.step(posterior, STATE, DT)
    for _ in range(round(hold_s / DT) - 2):
        assert float(np.hypot(*mapping.step(posterior, STATE, DT))) > 0.0

    for _ in range(3):
        mapping.step(posterior, STATE, DT)
    np.testing.assert_array_equal(mapping.step(posterior, STATE, DT), np.zeros(2))


def test_evidence_resets_at_emission() -> None:
    """Evidence already acted on must not count toward the next decision."""
    mapping = make_evidence(theta=-1.0)
    mapping.step(np.array([0.02, 0.94, 0.02, 0.02]), STATE, DT)
    np.testing.assert_array_equal(mapping.evidence, np.zeros(4))


def test_an_uncertain_stream_never_emits_which_is_the_watchdog_case() -> None:
    """A burst with no emission is legitimate and is recorded, not smoothed over."""
    mapping = make_evidence(theta=-0.5)
    uniform = np.full(4, 0.25)
    for _ in range(200):
        np.testing.assert_array_equal(mapping.step(np.array(uniform), STATE, DT), np.zeros(2))
    assert mapping.n_emissions == 0


def test_reset_clears_the_accumulator_between_episodes() -> None:
    mapping = make_evidence(theta=1e9)
    mapping.step(np.array([0.1, 0.7, 0.1, 0.1]), STATE, DT)
    mapping.reset(np.random.default_rng(0))
    np.testing.assert_array_equal(mapping.evidence, np.zeros(4))
    assert mapping.n_emissions == 0


def test_s3_returns_zero_without_a_posterior() -> None:
    np.testing.assert_array_equal(make_evidence().step(None, STATE, DT), np.zeros(2))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"gamma": 1.5}, "gamma"), ({"hold_s": 0.0}, "hold_s"), ({"epsilon": 0.0}, "epsilon")],
)
def test_s3_rejects_bad_parameters(kwargs: dict[str, float], match: str) -> None:
    settings: dict[str, object] = {
        "directions": DEFAULT_DIRECTIONS,
        "v_max": V_MAX,
        "gamma": 0.8,
        "theta": -1.0,
        "hold_s": 0.5,
        "epsilon": 1e-6,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        EvidenceMapping(**settings)  # type: ignore[arg-type]
