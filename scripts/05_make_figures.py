"""Render the paper figures from one or more run directories.

    python scripts/05_make_figures.py artifacts/runs/smoke/20260901-033900-0adf8f2c
    python scripts/05_make_figures.py artifacts/runs/*/2026*
    python scripts/05_make_figures.py --out artifacts/figures/draft <run_dir> ...

Reads run directories and writes PDF and PNG to `artifacts/figures/`. It
simulates nothing and fits nothing: every number comes from `episodes.parquet`
or `summary.json`.

Runs are matched to figures by the `name` in each run's own config, so the order
they are given in does not matter and a directory renamed by hand is still
understood. All six figures are always written. One whose run was not supplied,
or whose axis the supplied run does not vary, is drawn with the reason written
across it rather than skipped, so a missing figure is never mistaken for a
figure that was never planned.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from micm.eval.writer import LoadedRun, load_run
from micm.utils import configure_logging, get_logger
from micm.viz.figures import (
    figure_ablation_grid,
    figure_alpha_sweep,
    figure_command_accuracy,
    figure_intent_ablation,
    figure_kappa_against_success,
    figure_quality_sweep,
)
from micm.viz.theme import save_figure

logger = get_logger(__name__)

# Figure name to the experiment it wants, in the order agents/05 §6 lists them.
# The fallback is used when that experiment was not supplied: every figure has
# to render from whatever is there, including a lone smoke run.
FIGURE_SOURCES: dict[str, tuple[str, ...]] = {
    "fig1_quality_sweep": ("lambda_sweep", "main"),
    "fig2_kappa_against_success": ("main", "lambda_sweep"),
    # The blind sweep first: `agents/05` §6 asks this figure to show the
    # crossing at alpha*, and only intent-blind has one. Intent-aware never
    # stops reading the decoder, so its slope never reaches zero.
    "fig3_alpha_sweep": ("alpha_sweep_blind", "alpha_sweep"),
    "fig4_command_accuracy": ("ablation_latency", "main"),
    "fig6_intent_ablation": ("ablation_intent_blind", "alpha_sweep"),
}


def pick(runs: dict[str, LoadedRun], preferred: tuple[str, ...]) -> LoadedRun:
    """The best available run for a figure, or any run at all.

    Never returns nothing. A figure given a run that does not vary its axis
    draws the panel and says so, which is more useful than a missing file,
    and is what makes the smoke run able to exercise all six.
    """
    for name in preferred:
        if name in runs:
            return runs[name]
    return next(iter(runs.values()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="run directories to read")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/figures"),
        help="where to write the figures",
    )
    args = parser.parse_args(argv)
    configure_logging(logging.INFO)

    runs: dict[str, LoadedRun] = {}
    for directory in args.run_dirs:
        run = load_run(directory)
        if run.experiment in runs:
            logger.warning(
                "two runs of %s were given; using %s and ignoring %s",
                run.experiment,
                runs[run.experiment].directory,
                directory,
            )
            continue
        runs[run.experiment] = run
    logger.info("read %d run(s): %s", len(runs), ", ".join(sorted(runs)))

    written: list[Path] = []
    written += save_figure(
        figure_quality_sweep(pick(runs, FIGURE_SOURCES["fig1_quality_sweep"])),
        args.out,
        "fig1_quality_sweep",
    )
    written += save_figure(
        figure_kappa_against_success(pick(runs, FIGURE_SOURCES["fig2_kappa_against_success"])),
        args.out,
        "fig2_kappa_against_success",
    )
    written += save_figure(
        figure_alpha_sweep(pick(runs, FIGURE_SOURCES["fig3_alpha_sweep"])),
        args.out,
        "fig3_alpha_sweep",
    )
    written += save_figure(
        figure_command_accuracy(pick(runs, FIGURE_SOURCES["fig4_command_accuracy"])),
        args.out,
        "fig4_command_accuracy",
    )
    written += save_figure(figure_ablation_grid(runs), args.out, "fig5_ablations")
    written += save_figure(
        figure_intent_ablation(pick(runs, FIGURE_SOURCES["fig6_intent_ablation"])),
        args.out,
        "fig6_intent_ablation",
    )

    print(f"\nwrote {len(written)} files to {args.out}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
