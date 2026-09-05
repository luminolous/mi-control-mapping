"""The experiment matrix: cell counts, replay variants, and parallel execution.

The cell counts are written out literally rather than recomputed from the grid,
for the same reason the episode schema is: a test that recomputes the number
from the config agrees with whatever the config currently says, which is the
opposite of what a contract test is for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from micm.eval.runner import (
    cache_key,
    enumerate_cells,
    plan_chunks,
    resolve_variants,
    run_all,
    variant_for,
    variant_key,
)
from micm.eval.writer import episodes_frame
from micm.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# `agents/05-experiments-and-outputs.md` §1, copied by hand.
#
# `smoke` is the one deviation: the table says two episodes at one mapping, and
# it runs four at four mappings, because the sanity block checks the oracle
# ceiling for every mapping in the grid and a one-mapping grid would check one.
# See docs/decisions.md D46.
EXPECTED_EPISODES = {
    "main": 1080,
    "lambda_sweep": 2160,
    "alpha_sweep": 2160,
    "ablation_window": 540,
    "ablation_latency": 720,
    "ablation_protocol": 360,
    "ablation_errstruct": 540,
    "ablation_intent_blind": 540,
    "smoke": 4,
}

EXPERIMENTS = sorted(EXPECTED_EPISODES)


@pytest.fixture(scope="module", params=EXPERIMENTS)
def experiment(request: pytest.FixtureRequest) -> tuple[str, DictConfig]:
    loaded = load_config(f"experiment/{request.param}")
    assert isinstance(loaded, DictConfig)
    return str(request.param), loaded


# --- the matrix ---


def test_every_experiment_enumerates_the_episode_count_the_spec_states(
    experiment: tuple[str, DictConfig],
) -> None:
    name, cfg = experiment
    assert len(enumerate_cells(cfg)) == EXPECTED_EPISODES[name]


def test_every_cell_of_every_experiment_is_distinct(
    experiment: tuple[str, DictConfig],
) -> None:
    """Two cells with the same key would share a random stream and an episode."""
    cells = enumerate_cells(experiment[1])
    assert len({cell.key() for cell in cells}) == len(cells)


def test_every_experiment_names_itself(experiment: tuple[str, DictConfig]) -> None:
    """The `experiment` column has to identify the run that produced the row."""
    name, cfg = experiment
    assert str(cfg.name) == name
    assert {cell.experiment for cell in enumerate_cells(cfg)} == {name}


def test_the_matrix_is_about_eight_thousand_episodes() -> None:
    """The figure `agents/05` §1 gives, which sets what the runner has to survive."""
    assert 7000 <= sum(EXPECTED_EPISODES.values()) <= 9000


# --- replay variants ---


def test_no_experiment_composes_a_top_level_replay_group(
    experiment: tuple[str, DictConfig],
) -> None:
    """One `replay` block could describe only one condition, and two ablations
    vary it. Everything replay lives under `replay_variants`, keyed so a cell
    cannot reach a condition it does not name. See docs/decisions.md D45."""
    assert "replay" not in experiment[1]
    assert "replay_variants" in experiment[1]


def test_every_grid_combination_has_a_variant(experiment: tuple[str, DictConfig]) -> None:
    cfg = experiment[1]
    variants = resolve_variants(cfg)
    for cell in enumerate_cells(cfg):
        assert variant_for(cell, variants).protocol == cell.protocol


def test_every_composed_variant_is_reachable_from_the_grid(
    experiment: tuple[str, DictConfig],
) -> None:
    """A variant nothing reaches is a cache file nobody will read and a hash
    somebody will keep filling in for no reason."""
    cfg = experiment[1]
    reachable = {cache_key(cell)[2] for cell in enumerate_cells(cfg)}
    assert set(resolve_variants(cfg)) == reachable


def test_a_variant_filed_under_the_wrong_key_is_rejected(
    experiment: tuple[str, DictConfig],
) -> None:
    """Otherwise a grid combination would load another condition's posteriors."""
    broken = OmegaConf.merge(experiment[1], {"replay_variants": {"burst_w2000": {"window_s": 1.5}}})
    assert isinstance(broken, DictConfig)
    with pytest.raises(ValueError, match="belongs under 'burst_w1500'"):
        resolve_variants(broken)


def test_a_grid_reaching_an_undescribed_condition_raises(
    experiment: tuple[str, DictConfig],
) -> None:
    cfg = experiment[1]
    broken = OmegaConf.merge(cfg, {"grid": {"windows_s": [2.5]}})
    assert isinstance(broken, DictConfig)
    with pytest.raises(KeyError, match="burst_w2500"):
        enumerate_cells(broken)


def test_a_composed_variant_keeps_the_protocol_group_it_came_from(
    experiment: tuple[str, DictConfig],
) -> None:
    """The variants are composed from `configs/replay/*.yaml` rather than
    restated, so the gap and the stride cannot drift from the protocol they
    belong to. Only the window is ever overridden."""
    for variant in resolve_variants(experiment[1]).values():
        path = REPO_ROOT / "configs" / "replay" / f"{variant.protocol}.yaml"
        group = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert variant.protocol == group["protocol"]
        assert variant.stride_s == pytest.approx(group["stride_s"])
        assert variant.gap_s == pytest.approx(group["gap_s"])


def test_the_variant_key_is_derived_from_the_condition() -> None:
    assert variant_key("burst", 2.0) == "burst_w2000"
    assert variant_key("stitched", 1.5) == "stitched_w1500"


# --- what varies per cell rather than per config ---


def test_the_window_column_follows_the_cell_not_the_config() -> None:
    """`ablation_window` varies the window, so a `window_s` column read from
    `cfg.replay` would claim one length for every row. See docs/decisions.md D45."""
    cfg = load_config("experiment/ablation_window")
    assert isinstance(cfg, DictConfig)
    assert {cell.window_s for cell in enumerate_cells(cfg)} == {1.0, 2.0, 3.0}


def test_the_protocol_axis_selects_a_different_posterior_file() -> None:
    """Otherwise `ablation_protocol` would run one condition twice."""
    cfg = load_config("experiment/ablation_protocol")
    assert isinstance(cfg, DictConfig)
    keys = {cache_key(cell)[2] for cell in enumerate_cells(cfg)}
    assert keys == {"burst_w2000", "stitched_w2000"}


def test_a_grid_missing_an_axis_raises_rather_than_assuming_one() -> None:
    cfg = load_config("experiment/smoke")
    assert isinstance(cfg, DictConfig)
    stripped = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(stripped, DictConfig)
    del stripped.grid["windows_s"]
    with pytest.raises(KeyError, match="windows_s"):
        enumerate_cells(stripped)


# --- work planning ---


def test_every_cell_is_planned_exactly_once() -> None:
    cfg = load_config("experiment/main")
    assert isinstance(cfg, DictConfig)
    cells = enumerate_cells(cfg)
    planned = [cell for chunk in plan_chunks(cells, 8) for cell in chunk]
    assert sorted(planned, key=lambda cell: cell.key()) == sorted(
        cells, key=lambda cell: cell.key()
    )


def test_a_block_never_spans_two_posterior_files() -> None:
    """A worker loads one file per block. Mixed blocks would load several."""
    cfg = load_config("experiment/ablation_window")
    assert isinstance(cfg, DictConfig)
    for chunk in plan_chunks(enumerate_cells(cfg), 8):
        assert len({cache_key(cell) for cell in chunk}) == 1


def test_there_are_more_blocks_than_workers() -> None:
    """One block per worker would leave seven idle behind the slowest."""
    cfg = load_config("experiment/main")
    assert isinstance(cfg, DictConfig)
    assert len(plan_chunks(enumerate_cells(cfg), 8)) >= 8


def test_planning_for_no_workers_is_refused() -> None:
    cfg = load_config("experiment/smoke")
    assert isinstance(cfg, DictConfig)
    with pytest.raises(ValueError, match="n_workers must be at least 1"):
        plan_chunks(enumerate_cells(cfg), 0)


# --- parallel execution ---


def _with(cfg: DictConfig, **overrides: object) -> DictConfig:
    merged = OmegaConf.merge(cfg, overrides)
    assert isinstance(merged, DictConfig)
    return merged


def test_workers_do_not_change_a_single_number() -> None:
    """The guarantee that lets `n_workers` stay out of the config hash.

    Episode seeds are derived from the content of the cell rather than from its
    position in the queue or the worker that drew it, so the same grid gives the
    same episodes at any worker count. `agents/05` §2 asks for `SeedSequence`
    spawned per worker; this is the stronger property that requirement was
    protecting. See docs/decisions.md D47.
    """
    cfg = load_config("experiment/smoke")
    assert isinstance(cfg, DictConfig)

    serial = episodes_frame(run_all(_with(cfg, n_workers=1)), cfg, "run")
    parallel = episodes_frame(run_all(_with(cfg, n_workers=2)), cfg, "run")

    # wall_time_s is the one column a worker legitimately changes.
    columns = [column for column in serial.columns if column != "wall_time_s"]
    assert serial[columns].equals(parallel[columns])


def test_parallel_results_come_back_in_grid_order() -> None:
    """Completion order is not grid order, and two runs of the same config have
    to produce identical Parquet contents."""
    cfg = load_config("experiment/smoke")
    assert isinstance(cfg, DictConfig)
    results = run_all(_with(cfg, n_workers=2))
    assert [result.cell for result in results] == enumerate_cells(cfg)


def test_one_failed_episode_aborts_the_whole_parallel_run() -> None:
    """A matrix silently missing cells is worse than no matrix at all."""
    cfg = load_config("experiment/smoke")
    assert isinstance(cfg, DictConfig)
    broken = _with(cfg, n_workers=2, grid={"mappings": ["s1_argmax", "s9_nonexistent"]})
    with pytest.raises(KeyError, match="s9_nonexistent"):
        run_all(broken)


# --- the task must be solvable at every condition the grid names ---


@pytest.mark.slow
def test_an_oracle_decoder_reaches_every_target_at_every_latency() -> None:
    """Obstacle placement is relative to the target ring, and a grid axis can move
    the robot onto an obstacle that a different axis value clears.

    This is the check that would have caught D61 before the numbers were run.
    Obstacles sat on three of the four diagonal target bearings; a diagonal
    target is approached in a staircase, so argmax walked into one head on, and
    both cardinal directions its staircase alternates between pushed into it. An
    oracle decoder reached 4 of 8 targets at 0 ms latency and 5 of 8 at 500 ms
    while reaching all 8 at 125 and 250 ms, which reads exactly like a latency
    effect and is not one.

    The two extremes only: they are the ones that failed, and the middle two are
    covered by the sanity block of every run.
    """
    from micm.eval.runner import Cell, run_episode, synthetic_arrays
    from micm.utils.seeding import generator_for

    cfg = load_config("experiment/ablation_latency")
    assert isinstance(cfg, DictConfig)
    arrays = synthetic_arrays(
        generator_for(int(cfg.seed), "sanity", "oracle"),
        n_bursts=int(cfg.synthetic.n_bursts),
        n_windows=int(cfg.synthetic.n_windows),
        n_classes=int(cfg.data.n_classes),
        accuracy=1.0,
        window_s=2.0,
        stride_s=0.25,
        confidence=1.0,
    )

    for latency in (int(min(cfg.grid.latencies_ms)), int(max(cfg.grid.latencies_ms))):
        for mapping in cfg.grid.mappings:
            metrics = run_episode(
                Cell(
                    experiment="ceiling",
                    subject=1,
                    decoder="synthetic",
                    mapping=str(mapping),
                    quality_level=1.0,
                    protocol="burst",
                    window_s=2.0,
                    error_struct="none",
                    intent_mode=None,
                    alpha=None,
                    latency_ms=latency,
                    seed=0,
                ),
                arrays,
                cfg,
            ).metrics
            assert metrics.success_rate == 1.0, (
                f"{mapping} reached {metrics.n_success} of {int(cfg.env.n_targets)} targets "
                f"at {latency} ms with a perfect decoder"
            )
