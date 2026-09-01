"""Figures.

The figures carry no result of their own, so these tests are about the two ways
a figure can lie: by drawing a number it computed itself, and by looking the same
whether the data was there or not. Everything here checks one of those, plus that
all six render from a run that varies almost nothing, which is what the smoke run
is for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from omegaconf import DictConfig

from micm.eval.aggregate import scores_block
from micm.eval.writer import LoadedRun, load_run
from micm.utils import load_config
from micm.utils.config import save_config
from micm.viz import figures as viz
from micm.viz.theme import (
    MAPPING_STYLES,
    UNKNOWN_STYLE,
    apply_theme,
    label_for,
    save_figure,
    style_for,
)
from test_analysis import alpha_frame, as_episodes, sweep_frame


def build_run(
    directory: Path, frame: pd.DataFrame, experiment: str, *, stats: dict[str, Any] | None = None
) -> LoadedRun:
    """A run directory whose summary carries real aggregated scores.

    The scores come from `eval.aggregate`, exactly as a written run's do, so a
    figure reading them here reads what it would read from a real run.
    """
    directory.mkdir(parents=True, exist_ok=True)
    episodes = as_episodes(frame, experiment)
    episodes.to_parquet(directory / "episodes.parquet", index=False)

    cfg = load_config(f"experiment/{experiment}")
    assert isinstance(cfg, DictConfig)
    save_config(cfg, directory / "config.yaml")

    summary = {
        "meta": {"experiment": experiment},
        "scores": scores_block(episodes, n_boot=100),
        "stats": stats or {"model": None},
        "sanity": {"all_passed": True},
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return load_run(directory)


@pytest.fixture(scope="module")
def sweep_run(tmp_path_factory: pytest.TempPathFactory) -> LoadedRun:
    return build_run(tmp_path_factory.mktemp("lam"), sweep_frame(seeds=2), "lambda_sweep")


@pytest.fixture(scope="module")
def autonomy_run(tmp_path_factory: pytest.TempPathFactory) -> LoadedRun:
    return build_run(
        tmp_path_factory.mktemp("alpha"),
        alpha_frame(),
        "alpha_sweep",
        stats={"h3": {"applicable": True, "alpha_star": 0.6, "estimator": "grid"}},
    )


@pytest.fixture(scope="module")
def flat_run(tmp_path_factory: pytest.TempPathFactory) -> LoadedRun:
    """A run that varies nothing but the mapping, like the smoke run."""
    frame = sweep_frame(seeds=1)
    frame = frame[frame["quality_level"] == 0.0].copy()
    frame["kappa_offline"] = np.nan
    return build_run(tmp_path_factory.mktemp("flat"), frame, "lambda_sweep")


# --- the palette ---


def test_no_two_mappings_share_a_colour() -> None:
    """Two conditions in one colour is a figure that cannot be read at all."""
    colors = [style.color for style in MAPPING_STYLES.values()]
    assert len(set(colors)) == len(colors)


def test_every_mapping_also_differs_without_colour() -> None:
    """Printed in greyscale, the marker and the dash pattern are what is left."""
    signatures = [(style.linestyle, style.marker) for style in MAPPING_STYLES.values()]
    assert len(set(signatures)) == len(signatures)


def test_an_unknown_mapping_is_grey_rather_than_recoloured() -> None:
    """Silently reusing a mapping's colour would mislabel a condition."""
    assert style_for("s9_invented") is UNKNOWN_STYLE
    assert UNKNOWN_STYLE.color not in {style.color for style in MAPPING_STYLES.values()}


def test_an_unlabelled_column_falls_back_to_its_own_name() -> None:
    """Which tells the reader which column to go and look at."""
    assert label_for("uci_excluded_frac") == "uci_excluded_frac"
    assert label_for("success_rate") == "success rate"


def test_the_theme_keeps_pdf_text_as_text() -> None:
    """Type 42 so the publisher receives searchable text, not outlines."""
    import matplotlib as mpl

    apply_theme()
    assert mpl.rcParams["pdf.fonttype"] == 42


# --- reading, not recomputing ---


def test_the_cell_table_reads_the_summary_rather_than_the_episodes(
    sweep_run: LoadedRun,
) -> None:
    """A figure that recomputed a mean could disagree with the summary beside it."""
    table = viz.cell_table(sweep_run)
    entry = sweep_run.summary["scores"]["by_cell"][0]
    row = table.iloc[0]
    assert row["success_rate_mean"] == entry["success_rate"]["mean"]
    assert row["success_rate_lo"] == entry["success_rate"]["ci95"][0]


def test_the_cell_table_carries_the_axes_and_the_metrics(sweep_run: LoadedRun) -> None:
    table = viz.cell_table(sweep_run)
    assert "mapping" in table
    assert {"success_rate_mean", "success_rate_lo", "success_rate_hi"} <= set(table.columns)


def test_a_run_without_a_scores_block_says_to_analyse_it(tmp_path: Path) -> None:
    run = build_run(tmp_path / "run", sweep_frame(seeds=1), "lambda_sweep")
    stripped = LoadedRun(
        directory=run.directory,
        episodes=run.episodes,
        config=run.config,
        summary={"meta": {}},
    )
    with pytest.raises(KeyError, match="04_analyze"):
        viz.cell_table(stripped)


def test_a_missing_interval_is_left_blank_rather_than_filled(tmp_path: Path) -> None:
    """One subject has no between-subject interval, and the ribbon must not invent one."""
    frame = sweep_frame(n_subjects=1, seeds=2)
    run = build_run(tmp_path / "run", frame, "lambda_sweep")
    table = viz.cell_table(run)
    assert table["success_rate_lo"].isna().all()


# --- every figure renders ---


def test_all_six_figures_render_from_a_full_run(
    sweep_run: LoadedRun, autonomy_run: LoadedRun
) -> None:
    runs = {"lambda_sweep": sweep_run, "alpha_sweep": autonomy_run}
    produced = [
        viz.figure_quality_sweep(sweep_run),
        viz.figure_kappa_against_success(sweep_run),
        viz.figure_alpha_sweep(autonomy_run),
        viz.figure_command_tradeoff(sweep_run),
        viz.figure_ablation_grid(runs),
        viz.figure_intent_ablation(autonomy_run),
    ]
    assert all(isinstance(figure, Figure) for figure in produced)


def test_all_six_figures_render_from_a_run_that_varies_almost_nothing(
    flat_run: LoadedRun,
) -> None:
    """The smoke case. A figure that only renders once the real matrix exists is
    a figure nobody has run."""
    produced = [
        viz.figure_quality_sweep(flat_run),
        viz.figure_kappa_against_success(flat_run),
        viz.figure_alpha_sweep(flat_run),
        viz.figure_command_tradeoff(flat_run),
        viz.figure_ablation_grid({"lambda_sweep": flat_run}),
        viz.figure_intent_ablation(flat_run),
    ]
    assert all(isinstance(figure, Figure) for figure in produced)


def test_a_run_with_no_offline_kappa_says_so_on_the_panel(flat_run: LoadedRun) -> None:
    """Synthetic posteriors carry no kappa, and an empty panel would look like
    data that fell outside the axis limits."""
    figure = viz.figure_kappa_against_success(flat_run)
    written = [text.get_text() for axes in figure.axes for text in axes.texts]
    assert any("no offline kappa" in text for text in written)


def test_an_ablation_panel_names_the_run_it_is_missing() -> None:
    figure = viz.figure_ablation_grid({})
    written = [text.get_text() for axes in figure.axes for text in axes.texts]
    assert any("ablation_window" in text and "not supplied" in text for text in written)


def test_the_alpha_star_line_is_labelled_a_grid_estimate(autonomy_run: LoadedRun) -> None:
    """Never as a solved threshold: the true crossing lies between two grid points."""
    figure = viz.figure_alpha_sweep(autonomy_run)
    written = [text.get_text() for axes in figure.axes for text in axes.texts]
    assert any("grid estimate" in text for text in written)


def test_the_alpha_sweep_draws_one_line_per_decoder(autonomy_run: LoadedRun) -> None:
    """H3 is whether the decoders converge, so averaging them would erase it."""
    figure = viz.figure_alpha_sweep(autonomy_run)
    labels = [line.get_label() for line in figure.axes[0].lines]
    assert {"eegnet", "fbcsp", "riemann"} <= set(labels)


# --- output ---


def test_a_figure_is_written_as_both_vector_and_raster(
    sweep_run: LoadedRun, tmp_path: Path
) -> None:
    """Vector for the paper, raster for looking at. Both, always."""
    written = save_figure(viz.figure_quality_sweep(sweep_run), tmp_path, "fig")
    assert [path.name for path in written] == ["fig.pdf", "fig.png"]
    assert all(path.stat().st_size > 0 for path in written)
