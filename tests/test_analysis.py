"""Aggregation and statistics.

The aggregation cases are hand-computed: the frames are small enough that the
expected mean and spread are written in the test rather than produced by the
code under test. The statistics cases are built from a known generating model,
so a fitted slope can be checked against the slope that was put in.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from micm.eval.aggregate import bootstrap_ci, estimate, scores_block, subject_means
from micm.eval.stats import (
    NotFittableError,
    apply_transform,
    h2_block,
    h3_block,
    half_count_logit,
    prepare,
    resample_subjects,
    response_diagnostics,
    spread,
    stats_block,
)
from micm.eval.writer import EPISODE_SCHEMA
from micm.utils import load_config
from micm.utils.config import save_config
from micm.utils.seeding import generator_for

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stats_cfg() -> DictConfig:
    loaded = load_config("stats/lmm")
    assert isinstance(loaded, DictConfig)
    return loaded


def tiny_frame() -> pd.DataFrame:
    """Three subjects, two episodes each, values chosen for hand computation.

    Subject means are 0.2, 0.5 and 0.8, so the mean of those is 0.5 and their
    sample standard deviation is 0.3 exactly.
    """
    return pd.DataFrame(
        {
            "subject": [1, 1, 2, 2, 3, 3],
            "mapping": ["s1_argmax"] * 6,
            "success_rate": [0.1, 0.3, 0.4, 0.6, 0.7, 0.9],
        }
    )


# --- aggregation, hand computed ---


def test_subject_means_average_within_a_subject_first() -> None:
    means = subject_means(tiny_frame(), "success_rate")
    assert means.tolist() == pytest.approx([0.2, 0.5, 0.8])


def test_the_reported_mean_is_the_mean_of_subject_means() -> None:
    """Not the mean of episodes. A subject with more episodes must not get more say."""
    found = estimate(
        tiny_frame(),
        "success_rate",
        n_boot=200,
        ci=0.95,
        min_subjects=3,
        rng=np.random.default_rng(0),
    )
    assert found is not None
    assert found.mean == pytest.approx(0.5)
    assert found.sd == pytest.approx(0.3)
    assert found.n == 6
    assert found.n_subjects == 3


def test_an_unbalanced_subject_does_not_dominate_the_mean() -> None:
    """The failure the subject-level mean exists to prevent."""
    frame = pd.DataFrame(
        {
            "subject": [1, 1, 1, 1, 2, 3],
            "mapping": ["s1_argmax"] * 6,
            "success_rate": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        }
    )
    found = estimate(
        frame, "success_rate", n_boot=200, ci=0.95, min_subjects=3, rng=np.random.default_rng(0)
    )
    assert found is not None
    # Episode-level would give 1/3. Subject-level gives 2/3.
    assert found.mean == pytest.approx(2.0 / 3.0)


def test_a_metric_that_is_never_recorded_gets_no_block() -> None:
    """time_to_target_median is NaN for an episode that acquired nothing."""
    frame = tiny_frame().assign(success_rate=np.nan)
    assert (
        estimate(
            frame,
            "success_rate",
            n_boot=200,
            ci=0.95,
            min_subjects=3,
            rng=np.random.default_rng(0),
        )
        is None
    )


def test_too_few_subjects_gives_no_interval_and_says_why() -> None:
    """A zero-width interval from one subject would read as precision."""
    frame = tiny_frame()
    frame = frame[frame["subject"] == 1]
    found = estimate(
        frame, "success_rate", n_boot=200, ci=0.95, min_subjects=3, rng=np.random.default_rng(0)
    )
    assert found is not None
    assert found.ci95 is None
    assert "1 subject(s)" in str(found.ci_note)


# --- the bootstrap itself ---


def test_the_bootstrap_interval_brackets_the_mean() -> None:
    values = [0.2, 0.5, 0.8, 0.4, 0.6]
    low, high = bootstrap_ci(values, n_boot=4000, ci=0.95, rng=np.random.default_rng(0))
    assert low < float(np.mean(values)) < high


def test_the_bootstrap_of_identical_subjects_has_zero_width() -> None:
    """Not a bug: with no between-subject variation there is nothing to resample."""
    low, high = bootstrap_ci([0.5] * 5, n_boot=200, ci=0.95, rng=np.random.default_rng(0))
    assert low == high == pytest.approx(0.5)


def test_a_bootstrap_of_one_value_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two values"):
        bootstrap_ci([0.5], n_boot=200, ci=0.95, rng=np.random.default_rng(0))


def test_the_bootstrap_is_reproducible_from_its_generator() -> None:
    values = [0.2, 0.5, 0.8, 0.4]
    first = bootstrap_ci(values, n_boot=500, ci=0.95, rng=generator_for(7, "a"))
    second = bootstrap_ci(values, n_boot=500, ci=0.95, rng=generator_for(7, "a"))
    assert first == second


def test_a_wider_interval_covers_a_narrower_one() -> None:
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    narrow = bootstrap_ci(values, n_boot=4000, ci=0.5, rng=generator_for(1, "n"))
    wide = bootstrap_ci(values, n_boot=4000, ci=0.99, rng=generator_for(1, "n"))
    assert wide[0] <= narrow[0] and narrow[1] <= wide[1]


# --- the scores block ---


def test_the_scores_block_has_exactly_the_three_contracted_keys() -> None:
    """agents/05 §3 fixes them. What the intervals came from lives in meta."""
    block = scores_block(tiny_frame().assign(**_missing_score_columns()), n_boot=100)
    assert sorted(block) == ["by_cell", "by_subject", "overall"]


def test_every_subject_appears_in_the_breakdown() -> None:
    block = scores_block(tiny_frame().assign(**_missing_score_columns()), n_boot=100)
    assert [entry["subject"] for entry in block["by_subject"]] == [1, 2, 3]


def _missing_score_columns() -> dict[str, float]:
    return {
        "effective_acc": 0.7,
        "path_efficiency": 1.0,
        "time_to_target_median": 5.0,
        "direction_reversals": 2.0,
        "effective_itr": 10.0,
        "user_contribution_index": 0.5,
    }


# --- transforms ---


def test_the_half_count_logit_is_finite_at_both_ends() -> None:
    """A proportion out of eight targets hits 0 and 1 often, and log(0) is not a number."""
    values = half_count_logit(np.array([0.0, 0.5, 1.0]), n_trials=8)
    assert np.all(np.isfinite(values))


def test_the_half_count_logit_is_symmetric_about_a_half() -> None:
    low, mid, high = half_count_logit(np.array([0.0, 0.5, 1.0]), n_trials=8)
    assert mid == pytest.approx(0.0)
    assert low == pytest.approx(-high)


def test_the_half_count_correction_shrinks_as_trials_grow() -> None:
    """More trials means a zero is stronger evidence, so it sits further out."""
    few = half_count_logit(np.array([0.0]), n_trials=4)[0]
    many = half_count_logit(np.array([0.0]), n_trials=64)[0]
    assert many < few


def test_the_logit_is_monotone() -> None:
    values = half_count_logit(np.linspace(0.0, 1.0, 17), n_trials=8)
    assert np.all(np.diff(values) > 0)


def test_a_logit_of_something_unbounded_is_refused() -> None:
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        apply_transform(np.array([0.5, 42.0]), "logit", n_trials=8)


def test_a_log_of_a_non_positive_value_is_refused() -> None:
    with pytest.raises(ValueError, match="needs positive values"):
        apply_transform(np.array([1.0, 0.0]), "log", n_trials=8)


def test_an_unknown_transform_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown transform"):
        apply_transform(np.array([0.5]), "sqrt", n_trials=8)


def test_identity_leaves_the_values_alone() -> None:
    values = np.array([0.0, 3.0, -2.0])
    assert apply_transform(values, "identity", n_trials=8).tolist() == values.tolist()


# --- synthetic runs with a known generating model ---


MAPPING_OFFSET = {"s1_argmax": 0.0, "s2_weighted": 0.06, "s3_evidence": 0.03, "s4_shared": 0.09}
ACC_SLOPE = 0.5


def sweep_frame(
    *, n_subjects: int = 9, seeds: int = 3, noise: float = 0.02, seed: int = 11
) -> pd.DataFrame:
    """A lambda_sweep-shaped frame from a known model.

    success_rate rises with effective_acc at a fixed slope and each mapping adds
    a fixed offset, so the trade ratio the fit reports has a value it can be
    checked against: offset / slope.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(1, n_subjects + 1):
        aptitude = (subject - (n_subjects + 1) / 2) * 0.01
        kappa = 0.35 + 0.04 * subject
        for mapping, offset in MAPPING_OFFSET.items():
            for level in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                accuracy = 0.45 + 0.45 * level
                for seed_index in range(seeds):
                    value = (
                        0.2
                        + ACC_SLOPE * accuracy
                        + offset
                        + aptitude
                        + rng.normal(0.0, noise)
                    )
                    rows.append(
                        {
                            "subject": subject,
                            "mapping": mapping,
                            "decoder": "riemann",
                            "quality_level": level,
                            "alpha": np.nan,
                            "intent_mode": None,
                            "protocol": "burst",
                            "window_s": 2.0,
                            "latency_ms": 250,
                            "error_struct": "none",
                            "seed": seed_index,
                            "kappa_offline": kappa,
                            "effective_acc": accuracy,
                            "success_rate": float(np.clip(value, 0.0, 1.0)),
                            "path_efficiency": 0.6,
                            "time_to_target_median": 40.0 - 10.0 * accuracy,
                            "direction_reversals": 2.0,
                            "effective_itr": 8.0,
                            "user_contribution_index": 0.5,
                        }
                    )
    return pd.DataFrame(rows)


def alpha_frame(*, n_subjects: int = 9, seed: int = 13) -> pd.DataFrame:
    """An alpha_sweep-shaped frame where the kappa effect fades as alpha rises.

    Built so alpha* exists: above alpha 0.6 the decoder contributes nothing and
    the user contribution index falls with alpha, which is exactly the pair of
    facts H3 has to report together.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(1, n_subjects + 1):
        # Real between-subject variation, so the random intercept has something
        # to estimate and the fit is not singular.
        aptitude = (subject - (n_subjects + 1) / 2) * 0.02
        for decoder, kappa in (("eegnet", 0.25), ("fbcsp", 0.53), ("riemann", 0.68)):
            for alpha in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                weight = max(0.0, 1.0 - alpha / 0.6)
                for seed_index in range(3):
                    rows.append(
                        {
                            "subject": subject,
                            "mapping": "s4_shared",
                            "decoder": decoder,
                            "quality_level": 0.0,
                            "alpha": alpha,
                            "intent_mode": "aware",
                            "protocol": "burst",
                            "window_s": 2.0,
                            "latency_ms": 250,
                            "error_struct": "none",
                            "seed": seed_index,
                            "kappa_offline": kappa,
                            "effective_acc": 0.4 + kappa * 0.3,
                            "success_rate": float(
                                np.clip(
                                    0.35 + 0.6 * weight * kappa + 0.3 * alpha
                                    + aptitude
                                    + rng.normal(0.0, 0.02),
                                    0.0,
                                    1.0,
                                )
                            ),
                            "path_efficiency": 0.6,
                            "time_to_target_median": 30.0,
                            "direction_reversals": 2.0,
                            "effective_itr": 8.0,
                            "user_contribution_index": float(
                                np.clip(
                                    0.8 - 0.7 * alpha + aptitude + rng.normal(0.0, 0.02),
                                    -1.0,
                                    1.0,
                                )
                            ),
                        }
                    )
    return pd.DataFrame(rows)


# --- refusals ---


def test_a_saturated_response_is_refused_and_names_the_alternatives(
    stats_cfg: DictConfig,
) -> None:
    """The failure D42b predicted: every mapping reaches every target."""
    frame = sweep_frame(seeds=1).assign(success_rate=1.0)
    with pytest.raises(NotFittableError, match="saturated"):
        stats_block(frame, stats_cfg, n_targets=8, seed=1337)


def test_the_refusal_points_at_a_response_that_still_varies(stats_cfg: DictConfig) -> None:
    frame = sweep_frame(seeds=1).assign(success_rate=1.0)
    with pytest.raises(NotFittableError, match="time_to_target_median"):
        stats_block(frame, stats_cfg, n_targets=8, seed=1337)


def test_too_few_subjects_is_refused_rather_than_fitted(stats_cfg: DictConfig) -> None:
    frame = sweep_frame(n_subjects=2, seeds=1)
    with pytest.raises(NotFittableError, match="random intercept"):
        stats_block(frame, stats_cfg, n_targets=8, seed=1337)


def test_response_diagnostics_expose_a_saturated_metric() -> None:
    frame = sweep_frame(seeds=1).assign(success_rate=1.0)
    rows = {row["column"]: row for row in response_diagnostics(frame, ["success_rate"])}
    assert rows["success_rate"]["sd"] == 0.0
    assert rows["success_rate"]["n_distinct"] == 1


def test_spread_of_a_single_value_is_zero_not_nan() -> None:
    assert spread(pd.Series([0.5])) == 0.0


# --- H2 ---


def test_the_trade_ratio_recovers_the_offset_over_the_slope(stats_cfg: DictConfig) -> None:
    """The number the paper leads with, checked against the model it came from.

    The response is fitted on the logit scale while the frame was built on the
    raw one, so the ratio is not expected to equal offset/slope exactly. The
    ordering is what has to survive, and s4 was given the largest offset.
    """
    cfg = OmegaConf.merge(stats_cfg, {"bootstrap": {"n_boot": 60}})
    assert isinstance(cfg, DictConfig)
    prepared = prepare(sweep_frame(), response="success_rate", transform="logit", n_trials=8)
    block = h2_block(prepared, cfg, seed=1337)

    assert block["applicable"]
    ratios = block["trade_ratio"]
    assert set(ratios) == {"s2_weighted", "s3_evidence", "s4_shared"}
    assert all(entry["estimate"] > 0 for entry in ratios.values())
    assert (
        ratios["s4_shared"]["estimate"]
        > ratios["s2_weighted"]["estimate"]
        > ratios["s3_evidence"]["estimate"]
    )


def test_every_trade_ratio_comes_with_a_bootstrap_interval(stats_cfg: DictConfig) -> None:
    cfg = OmegaConf.merge(stats_cfg, {"bootstrap": {"n_boot": 60}})
    assert isinstance(cfg, DictConfig)
    prepared = prepare(sweep_frame(), response="success_rate", transform="logit", n_trials=8)
    block = h2_block(prepared, cfg, seed=1337)
    for entry in block["trade_ratio"].values():
        assert entry["ci95"] is not None
        assert entry["ci95"][0] <= entry["estimate"] <= entry["ci95"][1]


def test_h2_is_inapplicable_when_the_accuracy_dial_does_not_move(
    stats_cfg: DictConfig,
) -> None:
    """`main` holds quality at level 0, so there is nothing to trade against."""
    frame = sweep_frame(seeds=1)
    frame = frame[frame["quality_level"] == 0.0]
    prepared = prepare(frame, response="success_rate", transform="logit", n_trials=8)
    block = h2_block(prepared, stats_cfg, seed=1337)
    assert not block["applicable"]
    assert "effective_acc" in block["reason"]


def test_a_missing_reference_mapping_is_refused(stats_cfg: DictConfig) -> None:
    """Every offset is measured against it, so silently picking another would
    change what the ratio means without changing its name."""
    frame = sweep_frame(seeds=1)
    frame = frame[frame["mapping"] != "s1_argmax"]
    prepared = prepare(frame, response="success_rate", transform="logit", n_trials=8)
    with pytest.raises(NotFittableError, match="reference mapping"):
        h2_block(prepared, stats_cfg, seed=1337)


# --- the bootstrap replicate ---


def test_a_resampled_subject_is_relabelled_so_it_counts_twice() -> None:
    """Sharing a random intercept between two copies would narrow the interval,
    which is the error this bootstrap exists to avoid."""
    frame = sweep_frame(n_subjects=4, seeds=1)
    replicate = resample_subjects(frame, np.random.default_rng(0))
    assert replicate["subject"].nunique() == 4
    assert len(replicate) == len(frame)


def test_a_replicate_draws_only_subjects_that_exist() -> None:
    frame = sweep_frame(n_subjects=4, seeds=1)
    replicate = resample_subjects(frame, np.random.default_rng(3))
    origins = {label.split("_r")[0] for label in replicate["subject"]}
    assert origins <= {str(value) for value in frame["subject"].unique()}


# --- H3 ---


def test_alpha_star_is_found_and_labelled_a_grid_estimate(stats_cfg: DictConfig) -> None:
    prepared = prepare(alpha_frame(), response="success_rate", transform="logit", n_trials=8)
    block = h3_block(prepared, stats_cfg)
    assert block["applicable"]
    assert block["estimator"] == "grid"
    assert block["alpha_star"] in set(block["grid"])
    assert "not a solved threshold" in block["note"]


def test_alpha_star_lands_where_the_decoder_stops_mattering(stats_cfg: DictConfig) -> None:
    """The frame fades the kappa effect out by alpha 0.6, so alpha* belongs there."""
    prepared = prepare(alpha_frame(), response="success_rate", transform="logit", n_trials=8)
    block = h3_block(prepared, stats_cfg)
    assert block["alpha_star"] is not None
    assert block["alpha_star"] >= 0.6


def test_every_alpha_on_the_grid_is_reported(stats_cfg: DictConfig) -> None:
    """Not only the crossing: a reader has to see the slope shrink to believe it."""
    prepared = prepare(alpha_frame(), response="success_rate", transform="logit", n_trials=8)
    block = h3_block(prepared, stats_cfg)
    assert [entry["alpha"] for entry in block["by_alpha"]] == block["grid"]


def test_the_guard_reports_the_user_contribution_falling_with_alpha(
    stats_cfg: DictConfig,
) -> None:
    """An alpha that equalises the decoders by ignoring the user is not the finding."""
    prepared = prepare(alpha_frame(), response="success_rate", transform="logit", n_trials=8)
    guard = h3_block(prepared, stats_cfg)["guard"]
    slope = next(row for row in guard["coefficients"] if row["term"] == "alpha")
    assert guard["metric"] == "user_contribution_index"
    assert slope["estimate"] < 0.0


def test_h3_is_inapplicable_when_alpha_does_not_move(stats_cfg: DictConfig) -> None:
    prepared = prepare(sweep_frame(seeds=1), response="success_rate", transform="logit", n_trials=8)
    block = h3_block(prepared, stats_cfg)
    assert not block["applicable"]
    assert "alpha_sweep" in block["reason"]


# --- the whole block ---


def test_the_stats_block_carries_every_hypothesis_and_the_diagnostics(
    stats_cfg: DictConfig,
) -> None:
    cfg = OmegaConf.merge(stats_cfg, {"bootstrap": {"n_boot": 40}})
    assert isinstance(cfg, DictConfig)
    block = stats_block(sweep_frame(seeds=2), cfg, n_targets=8, seed=1337)
    assert sorted(block) == [
        "h1",
        "h2",
        "h3",
        "n_trials",
        "notes",
        "response",
        "response_diagnostics",
        "transform",
    ]
    assert block["h1"]["applicable"]
    assert block["h2"]["applicable"]
    assert not block["h3"]["applicable"]


def test_h1_reports_the_interaction_terms_it_is_testing(stats_cfg: DictConfig) -> None:
    """H1 is the interaction. The main effects are not the hypothesis."""
    cfg = OmegaConf.merge(stats_cfg, {"bootstrap": {"n_boot": 40}})
    assert isinstance(cfg, DictConfig)
    block = stats_block(sweep_frame(seeds=2), cfg, n_targets=8, seed=1337)["h1"]
    assert block["interaction_terms"]
    assert all(":" in term for term in block["interaction_terms"])


def test_the_stats_block_is_json_serialisable(stats_cfg: DictConfig) -> None:
    """It goes straight into summary.json, so a numpy scalar in there is a bug."""
    import json

    cfg = OmegaConf.merge(stats_cfg, {"bootstrap": {"n_boot": 40}})
    assert isinstance(cfg, DictConfig)
    block = stats_block(sweep_frame(seeds=2), cfg, n_targets=8, seed=1337)
    json.dumps(block, allow_nan=False)


# --- scripts/04_analyze.py ---


@pytest.fixture
def analyze() -> Any:
    """The analysis script, imported by path because its name is not an identifier."""
    spec = importlib.util.spec_from_file_location(
        "analyze_script", REPO_ROOT / "scripts" / "04_analyze.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_episodes(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Fill in the schema columns the generating model does not produce.

    `load_run` validates the frame against the contract, so a hand-built one has
    to carry every column even where the value is a placeholder. That is the
    validation working: a figure must never see a frame the writer would have
    rejected.
    """
    filled = frame.copy()
    filled["experiment"] = experiment
    filled["run_id"] = "synthetic"
    filled["lam"] = filled["quality_level"]
    filled["stride_s"] = 0.25
    filled["direction_perm"] = "0,1,2,3"
    filled["n_success"] = np.round(filled["success_rate"] * 8).astype(int)
    filled["n_timeouts"] = 8 - filled["n_success"]
    filled["collisions"] = 0
    filled["uci_excluded_frac"] = 0.1
    filled["bursts_without_command"] = 0
    filled["episode_duration_s"] = 300.0
    filled["wall_time_s"] = 12.0
    return filled[list(EPISODE_SCHEMA)].astype(EPISODE_SCHEMA)


def build_run(directory: Path, frame: pd.DataFrame, experiment: str) -> Path:
    """A run directory holding a synthetic episode frame."""
    directory.mkdir(parents=True, exist_ok=True)
    as_episodes(frame, experiment).to_parquet(directory / "episodes.parquet", index=False)
    cfg = load_config(f"experiment/{experiment}")
    save_config(cfg, directory / "config.yaml")
    (directory / "summary.json").write_text(
        json.dumps({"meta": {}, "scores": {}, "stats": {"model": None}, "sanity": {}}),
        encoding="utf-8",
    )
    return directory


def test_a_directory_that_is_not_a_run_names_the_file_it_wanted(
    analyze: Any, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match=r"episodes\.parquet"):
        analyze.load_run(tmp_path)


def test_the_analysis_writes_the_stats_block_into_the_summary(
    analyze: Any, tmp_path: Path
) -> None:
    run = build_run(tmp_path / "run", sweep_frame(seeds=2), "lambda_sweep")
    assert analyze.main([str(run), "bootstrap.n_boot=20"]) == 0

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["stats"]["h1"]["applicable"]
    assert summary["stats"]["h2"]["trade_ratio"]
    assert sorted(summary["scores"]) == ["by_cell", "by_subject", "overall"]


def test_the_analysis_leaves_the_other_summary_blocks_alone(
    analyze: Any, tmp_path: Path
) -> None:
    """It fits models; it does not re-run the episodes, so sanity and meta stand."""
    run = build_run(tmp_path / "run", sweep_frame(seeds=2), "lambda_sweep")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["sanity"] = {"all_passed": True, "marker": "kept"}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    assert analyze.main([str(run), "bootstrap.n_boot=20"]) == 0
    written = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert written["sanity"]["marker"] == "kept"


def test_re_analysing_without_force_is_refused(analyze: Any, tmp_path: Path) -> None:
    """Overwriting a fitted block silently would leave no record that it changed."""
    run = build_run(tmp_path / "run", sweep_frame(seeds=2), "lambda_sweep")
    assert analyze.main([str(run), "bootstrap.n_boot=20"]) == 0
    assert analyze.main([str(run), "bootstrap.n_boot=20"]) == 1
    assert analyze.main(["--force", str(run), "bootstrap.n_boot=20"]) == 0


def test_an_unfittable_run_reports_the_reason_rather_than_crashing(
    analyze: Any, tmp_path: Path
) -> None:
    frame = sweep_frame(seeds=2).assign(success_rate=1.0)
    run = build_run(tmp_path / "run", frame, "lambda_sweep")
    assert analyze.main([str(run), "bootstrap.n_boot=20"]) == 2
    # The summary is left as it was: nothing was fitted, so nothing is recorded.
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["stats"] == {"model": None}


def test_the_summary_is_replaced_rather_than_appended_to(
    analyze: Any, tmp_path: Path
) -> None:
    run = build_run(tmp_path / "run", sweep_frame(seeds=2), "lambda_sweep")
    analyze.write_summary(run, {"meta": {"x": 1}})
    assert json.loads((run / "summary.json").read_text(encoding="utf-8")) == {"meta": {"x": 1}}
    assert not (run / ".summary.json.partial").exists()


def test_an_alpha_sweep_run_reports_alpha_star(analyze: Any, tmp_path: Path) -> None:
    run = build_run(tmp_path / "run", alpha_frame(), "alpha_sweep")
    assert analyze.main([str(run), "bootstrap.n_boot=20"]) == 0
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["stats"]["h3"]["applicable"]
    assert summary["stats"]["h3"]["estimator"] == "grid"
