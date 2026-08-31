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
from micm.env.task import DEFAULT_DIRECTIONS, direction_table, target_positions
from micm.mapping.argmax import ArgmaxMapping
from micm.mapping.base import BaseMapping, Mapping, validate_directions, validate_posterior
from micm.mapping.evidence import EvidenceMapping
from micm.mapping.registry import MAPPINGS, build_mapping, select_mapping
from micm.mapping.shared import SharedMapping
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


@pytest.mark.parametrize("name", ["s1_argmax", "s2_weighted", "s3_evidence", "s4_shared"])
def test_registry_builds_every_configured_mapping(name: str) -> None:
    cfg = load_config("experiment/smoke")
    mapping = select_mapping(cfg, name, directions=DEFAULT_DIRECTIONS)
    assert isinstance(mapping, Mapping)


def test_selecting_a_mapping_the_experiment_did_not_compose_says_how_to_add_it() -> None:
    """The grid varies the mapping, so a single composed group could only hold one."""
    cfg = load_config("experiment/smoke")
    with pytest.raises(KeyError, match="does not compose mapping"):
        select_mapping(cfg, "s5_imaginary", directions=DEFAULT_DIRECTIONS)


@pytest.mark.parametrize("name", ["s1_argmax", "s2_weighted", "s3_evidence", "s4_shared"])
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


def test_the_registry_holds_every_mapping_in_the_study() -> None:
    """All four strategies, with S2 in both of its configured variants."""
    assert set(MAPPINGS) == {
        "s1_argmax",
        "s2_weighted",
        "s2_weighted_entropy",
        "s3_evidence",
        "s4_shared",
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


# --- S4: shared control ---

TARGETS = target_positions(8, 1.0)
OBSTACLES = np.array([[0.45, 0.45], [-0.45, -0.45]])


def make_shared(
    alpha: float = 0.5,
    intent_mode: str = "aware",
    alpha_mode: str = "fixed",
    *,
    obstacles: np.ndarray | None = None,
    entropy_scaling: bool = False,
) -> SharedMapping:
    mapping = SharedMapping(
        directions=DEFAULT_DIRECTIONS,
        v_max=V_MAX,
        alpha=alpha,
        alpha_mode=alpha_mode,
        alpha_min=0.0,
        alpha_max=0.9,
        intent_mode=intent_mode,
        entropy_scaling=entropy_scaling,
        k_rep=0.02,
        d0=0.35,
        angular_window_deg=60.0,
    )
    mapping.bind_task(
        targets=TARGETS, obstacles=OBSTACLES if obstacles is None else obstacles
    )
    mapping.reset(np.random.default_rng(0))
    return mapping


def state_at(pos: tuple[float, float], target_idx: int = 0) -> RobotState:
    return RobotState(
        pos=np.array(pos, dtype=np.float64),
        vel=np.zeros(2),
        target_idx=target_idx,
        dwell_t=0.0,
        t=0.0,
    )


@pytest.mark.parametrize(
    "posterior",
    [
        np.array([0.7, 0.1, 0.1, 0.1]),
        np.array([0.1, 0.4, 0.25, 0.25]),
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.0, 1.0, 0.0, 0.0]),
    ],
)
def test_alpha_zero_is_exactly_s2(posterior: np.ndarray) -> None:
    """Bit equality, not closeness. S4 uses S2 itself rather than a copy of it."""
    shared = make_shared(alpha=0.0)
    weighted = WeightedMapping(
        directions=DEFAULT_DIRECTIONS, v_max=V_MAX, entropy_scaling=False
    )
    state = state_at((0.2, -0.1), target_idx=3)

    np.testing.assert_array_equal(
        shared.step(posterior, state, DT), weighted.step(posterior, state, DT)
    )


def test_alpha_zero_matches_s2_without_any_task_geometry() -> None:
    """At alpha 0 the autonomy term is never consulted, so it need not exist."""
    unbound = SharedMapping(
        directions=DEFAULT_DIRECTIONS,
        v_max=V_MAX,
        alpha=0.0,
        alpha_mode="fixed",
        alpha_min=0.0,
        alpha_max=0.9,
        intent_mode="aware",
        entropy_scaling=False,
        k_rep=0.02,
        d0=0.35,
        angular_window_deg=60.0,
    )
    unbound.reset(np.random.default_rng(0))
    posterior = np.array([0.7, 0.1, 0.1, 0.1])
    weighted = WeightedMapping(
        directions=DEFAULT_DIRECTIONS, v_max=V_MAX, entropy_scaling=False
    )
    np.testing.assert_array_equal(
        unbound.step(posterior, state_at((0.0, 0.0)), DT),
        weighted.step(posterior, state_at((0.0, 0.0)), DT),
    )


def test_an_unbound_mapping_raises_rather_than_guessing_the_geometry() -> None:
    mapping = SharedMapping(
        directions=DEFAULT_DIRECTIONS,
        v_max=V_MAX,
        alpha=1.0,
        alpha_mode="fixed",
        alpha_min=0.0,
        alpha_max=0.9,
        intent_mode="blind",
        entropy_scaling=False,
        k_rep=0.02,
        d0=0.35,
        angular_window_deg=60.0,
    )
    mapping.reset(np.random.default_rng(0))
    with pytest.raises(RuntimeError, match="bind_task"):
        mapping.step(np.array([0.7, 0.1, 0.1, 0.1]), state_at((0.0, 0.0)), DT)


def test_intent_blind_at_alpha_one_ignores_the_posterior_entirely() -> None:
    """The alpha = 1 independence check: two decoders, identical behaviour."""
    mapping = make_shared(alpha=1.0, intent_mode="blind", obstacles=np.zeros((0, 2)))
    state = state_at((0.1, 0.1), target_idx=2)

    first = mapping.step(np.array([0.7, 0.1, 0.1, 0.1]), state, DT)
    second = mapping.step(np.array([0.1, 0.1, 0.1, 0.7]), state, DT)
    np.testing.assert_array_equal(first, second)


def test_intent_aware_at_alpha_one_still_reads_the_posterior() -> None:
    """Not a bug: this is the leakage the intent-blind ablation exists to expose.

    Under `aware` the decoder informs the choice of attractive target, so part of
    what looks like shared control succeeding is the decoder working through the
    autonomy term. A test asserting independence here would be asserting the
    wrong thing.
    """
    mapping = make_shared(alpha=1.0, intent_mode="aware", obstacles=np.zeros((0, 2)))
    state = state_at((0.0, 0.0), target_idx=0)

    first = mapping.step(np.array([0.02, 0.94, 0.02, 0.02]), state, DT)
    second = mapping.step(np.array([0.02, 0.02, 0.02, 0.94]), state, DT)
    assert not np.allclose(first, second)


def test_intent_blind_pulls_toward_the_active_target() -> None:
    mapping = make_shared(alpha=1.0, intent_mode="blind", obstacles=np.zeros((0, 2)))
    state = state_at((0.0, 0.0), target_idx=2)

    command = mapping.step(np.array([0.7, 0.1, 0.1, 0.1]), state, DT)
    expected = TARGETS[2] / float(np.hypot(*TARGETS[2])) * V_MAX
    np.testing.assert_allclose(command, expected, atol=1e-9)


def test_intent_aware_picks_the_target_the_posterior_points_at() -> None:
    mapping = make_shared(alpha=1.0, intent_mode="aware", obstacles=np.zeros((0, 2)))
    # Class 1 is +x, and target 0 sits at (1, 0). The active target is elsewhere.
    command = mapping.step(
        np.array([0.02, 0.94, 0.02, 0.02]), state_at((0.0, 0.0), target_idx=4), DT
    )
    np.testing.assert_allclose(command, np.array([V_MAX, 0.0]), atol=1e-9)


def test_intent_aware_withholds_attraction_outside_the_angular_window() -> None:
    """Falling back on the active target would quietly make `aware` behave like `blind`."""
    mapping = make_shared(alpha=1.0, intent_mode="aware", obstacles=np.zeros((0, 2)))
    # Sitting at the centre of the circle of targets with a uniform posterior:
    # the blend has no direction, so the autonomy has no opinion.
    command = mapping.step(np.full(4, 0.25), state_at((0.0, 0.0), target_idx=0), DT)
    np.testing.assert_allclose(command, np.zeros(2), atol=1e-12)


def test_repulsion_pushes_back_on_a_head_on_approach() -> None:
    mapping = make_shared(alpha=1.0, intent_mode="blind", obstacles=np.array([[0.5, 0.0]]))
    # Target 0 is at (1, 0), the obstacle sits directly between.
    near = mapping.autonomy_command(state_at((0.35, 0.0)), np.full(4, 0.25))
    far = mapping.autonomy_command(state_at((0.0, 0.0)), np.full(4, 0.25))

    assert near[0] < far[0]


def test_repulsion_falls_to_nothing_beyond_its_range() -> None:
    mapping = make_shared(alpha=1.0, intent_mode="blind", obstacles=np.array([[0.5, 0.0]]))
    without = make_shared(alpha=1.0, intent_mode="blind", obstacles=np.zeros((0, 2)))
    state = state_at((-0.5, 0.0))  # a full unit from the obstacle, well beyond d0

    np.testing.assert_allclose(
        mapping.autonomy_command(state, np.full(4, 0.25)),
        without.autonomy_command(state, np.full(4, 0.25)),
    )


def test_a_grazing_pass_is_deflected_sideways() -> None:
    """Head-on repulsion opposes progress; a grazing one should mostly steer."""
    mapping = make_shared(alpha=1.0, intent_mode="blind", obstacles=np.array([[0.5, 0.2]]))
    command = mapping.autonomy_command(state_at((0.45, 0.0)), np.full(4, 0.25))
    assert command[1] < 0.0


def test_the_autonomy_command_respects_the_speed_limit() -> None:
    mapping = make_shared(alpha=1.0, intent_mode="blind", obstacles=np.array([[0.01, 0.0]]))
    command = mapping.autonomy_command(state_at((0.011, 0.0)), np.full(4, 0.25))
    assert float(np.hypot(*command)) <= V_MAX + 1e-9


def test_adaptive_alpha_rises_with_uncertainty() -> None:
    """The robot takes over exactly when the user is unsure, which is what H3 tests."""
    mapping = make_shared(alpha_mode="adaptive")
    assert mapping.effective_alpha(np.array([0.0, 1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert mapping.effective_alpha(np.full(4, 0.25)) == pytest.approx(0.9)

    middling = mapping.effective_alpha(np.array([0.1, 0.6, 0.15, 0.15]))
    assert 0.0 < middling < 0.9


def test_adaptive_alpha_at_full_certainty_reduces_to_s2() -> None:
    mapping = make_shared(alpha_mode="adaptive")
    weighted = WeightedMapping(
        directions=DEFAULT_DIRECTIONS, v_max=V_MAX, entropy_scaling=False
    )
    one_hot = np.array([0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        mapping.step(one_hot, state_at((0.2, 0.2), target_idx=1), DT),
        weighted.step(one_hot, state_at((0.2, 0.2), target_idx=1), DT),
    )


def test_a_higher_alpha_moves_the_command_toward_the_autonomy() -> None:
    posterior = np.array([0.7, 0.1, 0.1, 0.1])  # points -x
    state = state_at((0.0, 0.0), target_idx=0)  # active target is +x

    low = make_shared(alpha=0.2, intent_mode="blind", obstacles=np.zeros((0, 2)))
    high = make_shared(alpha=0.8, intent_mode="blind", obstacles=np.zeros((0, 2)))
    assert high.step(posterior, state, DT)[0] > low.step(posterior, state, DT)[0]


def test_s4_returns_zero_without_a_posterior() -> None:
    np.testing.assert_array_equal(make_shared().step(None, state_at((0.0, 0.0)), DT), np.zeros(2))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 1.5}, "alpha"),
        ({"alpha_mode": "linear"}, "alpha_mode"),
        ({"intent_mode": "psychic"}, "intent_mode"),
        ({"alpha_min": 0.9, "alpha_max": 0.1}, "alpha_min"),
        ({"d0": 0.0}, "d0"),
        ({"angular_window_deg": 0.0}, "angular_window_deg"),
    ],
)
def test_s4_rejects_bad_parameters(kwargs: dict[str, object], match: str) -> None:
    settings: dict[str, object] = {
        "directions": DEFAULT_DIRECTIONS,
        "v_max": V_MAX,
        "alpha": 0.5,
        "alpha_mode": "fixed",
        "alpha_min": 0.0,
        "alpha_max": 0.9,
        "intent_mode": "aware",
        "entropy_scaling": False,
        "k_rep": 0.02,
        "d0": 0.35,
        "angular_window_deg": 60.0,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        SharedMapping(**settings)  # type: ignore[arg-type]


def test_the_s4_config_shares_s2s_entropy_setting() -> None:
    """Or the alpha = 0 identity would be against a mapping S2 is not."""
    cfg = load_config("experiment/smoke")
    assert cfg.mappings.s4_shared.params.entropy_scaling == (
        cfg.mappings.s2_weighted.params.entropy_scaling
    )


def test_a_grid_override_for_a_parameter_a_mapping_lacks_raises() -> None:
    """A cell claiming an alpha for argmax would record a condition that never applied."""
    cfg = load_config("experiment/smoke")
    with pytest.raises(KeyError, match="has no such parameter"):
        select_mapping(
            cfg, "s1_argmax", directions=DEFAULT_DIRECTIONS, overrides={"alpha": 0.5}
        )


def test_a_none_override_is_ignored() -> None:
    cfg = load_config("experiment/smoke")
    mapping = select_mapping(
        cfg,
        "s1_argmax",
        directions=DEFAULT_DIRECTIONS,
        overrides={"alpha": None, "intent_mode": None},
    )
    assert isinstance(mapping, Mapping)


def test_a_grid_override_reaches_the_mapping() -> None:
    cfg = load_config("experiment/smoke")
    built = select_mapping(
        cfg, "s4_shared", directions=DEFAULT_DIRECTIONS, overrides={"alpha": 0.25}
    )
    assert isinstance(built, SharedMapping)
    assert built.alpha == pytest.approx(0.25)
