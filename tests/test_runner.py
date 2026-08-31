"""Runner and writer tests: the trial pool, the output contract, determinism.

The schema test lists the contracted columns literally rather than importing
them from the writer. Importing them would make the test agree with whatever the
code currently does, which is the opposite of what a contract test is for.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from micm.eval.runner import (
    Cell,
    TrialPool,
    enumerate_cells,
    load_arrays,
    run_all,
    sanity_block,
    synthetic_arrays,
)
from micm.eval.writer import episodes_frame, validate_schema, write_run
from micm.utils import load_config

# The columns of agents/05-experiments-and-outputs.md §3, in order, plus
# `quality_level` (docs/decisions.md D35). Written out here on purpose.
CONTRACT_COLUMNS = [
    "experiment",
    "run_id",
    "subject",
    "decoder",
    "mapping",
    "quality_level",
    "lam",
    "alpha",
    "window_s",
    "stride_s",
    "latency_ms",
    "protocol",
    "error_struct",
    "intent_mode",
    "seed",
    "direction_perm",
    "kappa_offline",
    "effective_acc",
    "success_rate",
    "n_success",
    "time_to_target_median",
    "n_timeouts",
    "path_efficiency",
    "direction_reversals",
    "collisions",
    "effective_itr",
    "user_contribution_index",
    "uci_excluded_frac",
    "bursts_without_command",
    "episode_duration_s",
    "wall_time_s",
]

# Not results: one embeds a timestamp, the other measures the machine.
NOT_A_RESULT = ["run_id", "wall_time_s"]


@pytest.fixture
def cfg(tmp_path: Path) -> DictConfig:
    """The smoke config, writing into a temporary artifacts directory."""
    loaded = load_config(
        "experiment/smoke", overrides=(f"paths.artifacts={tmp_path.as_posix()}",)
    )
    assert isinstance(loaded, DictConfig)
    return loaded


def arrays(accuracy: float = 0.75, n_bursts: int = 24, seed: int = 0) -> dict[str, np.ndarray]:
    return synthetic_arrays(
        np.random.default_rng(seed),
        n_bursts=n_bursts,
        n_windows=7,
        n_classes=4,
        accuracy=accuracy,
        window_s=2.0,
        stride_s=0.25,
    )


# --- synthetic posteriors ---


def test_synthetic_posteriors_hit_the_requested_accuracy() -> None:
    data = arrays(accuracy=0.75, n_bursts=200)
    achieved = float((data["posterior"].argmax(axis=1) == data["label"]).mean())
    assert achieved == pytest.approx(0.75, abs=0.03)


def test_synthetic_posteriors_are_valid_distributions() -> None:
    data = arrays()
    np.testing.assert_allclose(data["posterior"].sum(axis=1), 1.0, atol=1e-5)
    assert (data["posterior"] >= 0.0).all()


def test_every_class_appears_so_the_pool_can_always_match() -> None:
    """A missing class would leave the task unable to express an intent."""
    assert set(np.unique(arrays()["label"]).tolist()) == {0, 1, 2, 3}


def test_synthetic_t_rel_marks_window_ends_from_the_window_length() -> None:
    data = arrays()
    first = data["t_rel"][data["burst_id"] == 0]
    assert float(first[0]) == pytest.approx(2.0)
    np.testing.assert_allclose(np.diff(first), 0.25, atol=1e-5)


# --- trial pool ---


def test_the_pool_returns_a_burst_of_the_requested_class() -> None:
    """The link between the dataset and the task. Without it an oracle points nowhere."""
    data = arrays()
    pool = TrialPool(
        data["posterior"], data["label"], data["burst_id"], data["t_rel"],
        np.random.default_rng(0),
    )
    for cls in range(4):
        assert pool.draw(cls).label == cls


def test_the_pool_samples_without_replacement_until_a_class_runs_out() -> None:
    data = arrays(n_bursts=8)
    pool = TrialPool(
        data["posterior"], data["label"], data["burst_id"], data["t_rel"],
        np.random.default_rng(0),
    )
    drawn = [pool.draw(0).burst_id for _ in range(2)]
    assert len(set(drawn)) == 2


def test_the_pool_refills_rather_than_failing_when_a_class_is_exhausted() -> None:
    data = arrays(n_bursts=8)
    pool = TrialPool(
        data["posterior"], data["label"], data["burst_id"], data["t_rel"],
        np.random.default_rng(0),
    )
    for _ in range(6):
        assert pool.draw(0).label == 0


def test_the_pool_raises_when_a_class_is_absent_entirely() -> None:
    """Silently substituting another class would replay the wrong intent."""
    data = arrays()
    keep = data["label"] != 3
    pool = TrialPool(
        data["posterior"][keep], data["label"][keep], data["burst_id"][keep],
        data["t_rel"][keep], np.random.default_rng(0),
    )
    with pytest.raises(KeyError, match="no burst of class 3"):
        pool.draw(3)


def test_the_pool_is_reproducible_from_its_generator() -> None:
    data = arrays()
    draws = []
    for _ in range(2):
        pool = TrialPool(
            data["posterior"], data["label"], data["burst_id"], data["t_rel"],
            np.random.default_rng(7),
        )
        draws.append([pool.draw(cls % 4).burst_id for cls in range(8)])
    assert draws[0] == draws[1]


# --- grid ---


def test_the_grid_enumerates_the_product_of_its_axes(cfg: DictConfig) -> None:
    cells = enumerate_cells(cfg)
    expected = (
        len(cfg.grid.subjects)
        * len(cfg.grid.decoders)
        * len(cfg.grid.mappings)
        * len(cfg.grid.quality_levels)
        * len(cfg.grid.protocols)
        * len(cfg.grid.error_structures)
        * len(cfg.grid.intent_modes)
        * len(cfg.grid.alphas)
        * len(cfg.grid.latencies_ms)
        * int(cfg.grid.seeds)
    )
    assert len(cells) == expected
    assert len({cell.key() for cell in cells}) == len(cells)


def test_a_missing_cache_names_the_file_it_wanted(cfg: DictConfig) -> None:
    """No silent recomputation: the caching script has simply not been run."""
    real = OmegaConf.merge(
        cfg, {"synthetic": {"enabled": False}, "posterior_hash": "deadbeef"}
    )
    assert isinstance(real, DictConfig)
    cell = enumerate_cells(real)[0]
    with pytest.raises(FileNotFoundError, match="no posterior cache at"):
        load_arrays(cell, real)


def test_a_real_run_without_a_posterior_hash_says_so(cfg: DictConfig) -> None:
    real = OmegaConf.merge(cfg, {"synthetic": {"enabled": False}})
    assert isinstance(real, DictConfig)
    with pytest.raises(ValueError, match="posterior_hash is null"):
        load_arrays(enumerate_cells(real)[0], real)


# --- output contract ---


def test_the_episode_schema_matches_the_contract_column_for_column(cfg: DictConfig) -> None:
    frame = episodes_frame(run_all(cfg), cfg, "test-run")
    assert list(frame.columns) == CONTRACT_COLUMNS


def test_a_frame_with_a_renamed_column_is_rejected(cfg: DictConfig) -> None:
    frame = episodes_frame(run_all(cfg), cfg, "test-run").rename(
        columns={"success_rate": "success"}
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_schema(frame)


def test_every_independent_variable_is_its_own_column(cfg: DictConfig) -> None:
    """So the file can be grouped without joining anything."""
    frame = episodes_frame(run_all(cfg), cfg, "test-run")
    for column in ("subject", "decoder", "mapping", "quality_level", "protocol", "seed"):
        assert column in frame.columns
    assert frame["quality_level"].nunique() == len(cfg.grid.quality_levels)


def test_alpha_is_nan_when_the_mapping_has_no_autonomy_weight(cfg: DictConfig) -> None:
    frame = episodes_frame(run_all(cfg), cfg, "test-run")
    assert frame["alpha"].isna().all()


def test_a_run_writes_all_three_files_and_four_summary_blocks(cfg: DictConfig) -> None:
    run = write_run(run_all(cfg), cfg, sanity=sanity_block(cfg), wall_time_s=1.0)

    assert sorted(path.name for path in run.directory.iterdir()) == [
        "config.yaml",
        "episodes.parquet",
        "summary.json",
    ]
    summary = json.loads((run.directory / "summary.json").read_text(encoding="utf-8"))
    assert sorted(summary) == ["meta", "sanity", "scores", "stats"]
    assert sorted(summary["scores"]) == ["by_cell", "by_subject", "overall"]
    assert summary["meta"]["n_episodes"] == len(run.episodes)


def test_no_partial_directory_survives_a_failed_write(cfg: DictConfig) -> None:
    """A crashed run must not leave something a later analysis reads as complete."""
    results = run_all(cfg)
    broken = OmegaConf.merge(cfg, {"replay": {"window_s": "not-a-number"}})
    assert isinstance(broken, DictConfig)

    with pytest.raises(ValueError):
        write_run(results, broken, sanity={"all_passed": True}, wall_time_s=1.0)

    runs = Path(cfg.paths.artifacts) / "runs"
    partials = list(runs.rglob(".*partial")) if runs.exists() else []
    assert partials == []


# --- sanity block ---


def test_the_sanity_block_is_always_written_even_when_a_check_cannot_run(
    cfg: DictConfig,
) -> None:
    """A check that does not apply is stated, never omitted."""
    block = sanity_block(cfg)
    assert block["alpha0_equals_s2"]["applicable"] is False
    assert block["alpha0_equals_s2"]["reason"]
    assert "all_passed" in block


def test_the_oracle_ceiling_and_chance_floor_hold_on_synthetic_data(
    cfg: DictConfig,
) -> None:
    block = sanity_block(cfg)
    assert block["oracle_success"]["value"] >= float(cfg.sanity.oracle_threshold)
    assert block["uniform_posterior_chance"]["value"] <= float(cfg.sanity.chance_threshold)
    assert block["all_passed"]


def test_a_failed_sanity_check_is_still_written(cfg: DictConfig) -> None:
    """Never suppressed. The run is written and the flag records what happened."""
    run = write_run(
        run_all(cfg),
        cfg,
        sanity={"all_passed": False, "made_up": {"applicable": True, "pass": False}},
        wall_time_s=1.0,
    )
    summary = json.loads((run.directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["sanity"]["all_passed"] is False


# --- determinism ---


def test_two_identical_runs_produce_identical_episodes(cfg: DictConfig) -> None:
    """Unseeded randomness or dict ordering would show up here and nowhere else."""
    first = episodes_frame(run_all(cfg), cfg, "run-a").drop(columns=NOT_A_RESULT)
    second = episodes_frame(run_all(cfg), cfg, "run-b").drop(columns=NOT_A_RESULT)
    assert first.equals(second)


def test_a_different_seed_changes_the_episodes(cfg: DictConfig) -> None:
    """Otherwise the determinism test above would pass for the wrong reason."""
    other = OmegaConf.merge(cfg, {"seed": int(cfg.seed) + 1})
    assert isinstance(other, DictConfig)

    first = episodes_frame(run_all(cfg), cfg, "run-a").drop(columns=NOT_A_RESULT)
    second = episodes_frame(run_all(other), other, "run-a").drop(columns=NOT_A_RESULT)
    assert not first.equals(second)


def test_the_smoke_configuration_stays_inside_its_time_budget(cfg: DictConfig) -> None:
    """Five seconds is what makes it usable as a check after every change."""
    started = time.perf_counter()
    write_run(run_all(cfg), cfg, sanity=sanity_block(cfg), wall_time_s=0.0)
    assert time.perf_counter() - started < 5.0


def test_the_smoke_grid_stays_small(cfg: DictConfig) -> None:
    assert len(enumerate_cells(cfg)) <= 4
    assert bool(cfg.synthetic.enabled)


def test_a_cell_key_is_content_addressed() -> None:
    """Re-running a subset of the grid must reproduce the same episodes."""
    fields = {
        "experiment": "x",
        "subject": 1,
        "decoder": "d",
        "mapping": "m",
        "quality_level": 0.5,
        "protocol": "burst",
        "error_struct": "none",
        "intent_mode": "aware",
        "alpha": None,
        "latency_ms": 250,
        "seed": 0,
    }
    assert Cell(**fields).key() == Cell(**fields).key()  # type: ignore[arg-type]
    assert Cell(**{**fields, "seed": 1}).key() != Cell(**fields).key()  # type: ignore[arg-type]
