"""Fit the hypothesis models for one run and write them into its `summary.json`.

    python scripts/04_analyze.py artifacts/runs/lambda_sweep/20260901-142233-a3f9c1d2
    python scripts/04_analyze.py <run_dir> response=time_to_target_median
    python scripts/04_analyze.py --force <run_dir>

Flags come before the run directory. Overrides are a variable-length positional,
and argparse cannot split one around an option that appears in the middle of it.

Reads the run directory and nothing else: `episodes.parquet` for the numbers and
`config.yaml` for the environment the numbers came from. It recomputes no
episode, so an analysis can be redone without re-running twelve hours of
simulation, which is the point of separating this from the runner.

The models live in `configs/stats/lmm.yaml`, loaded separately from the run
config for the same reason: choosing a model is something you do after the
episodes exist.

**It updates `summary.json` in place.** The alternative, a second file beside it,
would mean two places to look for the same run's results and one of them stale.
The write is atomic and refuses to overwrite a `stats` block that has already
been fitted unless `--force`, so a re-analysis is always deliberate.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from micm.eval.aggregate import scores_block
from micm.eval.stats import NotFittableError, stats_block
from micm.eval.writer import load_run, write_summary
from micm.utils import configure_logging, get_logger, load_config

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a run directory under artifacts/runs/")
    parser.add_argument(
        "--config", default="stats/lmm", help="stats config name under configs/"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-fit even though this run already carries a stats block",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. response=effective_itr")
    args = parser.parse_args(argv)

    run = load_run(args.run_dir)
    frame, run_cfg, summary = run.episodes, run.config, run.summary
    configure_logging(getattr(logging, str(run_cfg.logging.level)))

    if summary.get("stats", {}).get("h1") is not None and not args.force:
        logger.error(
            "%s already carries a fitted stats block; pass --force to replace it",
            args.run_dir,
        )
        return 1

    stats_cfg = load_config(args.config, overrides=tuple(args.overrides))
    n_targets = int(run_cfg.env.n_targets)

    try:
        stats = stats_block(frame, stats_cfg, n_targets=n_targets, seed=int(run_cfg.seed))
    except NotFittableError as error:
        # Not a crash. The run is fine and the model is not supportable by it,
        # and saying which is more useful than a traceback.
        logger.error("cannot fit the primary model: %s", error)
        return 2

    summary["scores"] = scores_block(
        frame,
        n_boot=int(run_cfg.aggregate.n_boot),
        ci=float(run_cfg.aggregate.ci),
        min_subjects=int(run_cfg.aggregate.min_subjects),
        seed=int(run_cfg.seed),
    )
    summary["stats"] = stats
    write_summary(args.run_dir, summary)

    print(f"\nrun         {args.run_dir}")
    print(f"episodes    {len(frame)}")
    print(f"response    {stats['response']} ({stats['transform']})")
    for row in stats["response_diagnostics"]:
        print(f"  {row['column']:24s} sd={row['sd']:<10} distinct={row['n_distinct']}")
    for name in ("h1", "h2", "h3"):
        block = stats[name]
        if not block.get("applicable"):
            print(f"{name}          not applicable: {block['reason']}")
            continue
        print(f"{name}          {block.get('model', 'grid estimate')}")
        if name == "h2":
            for mapping, entry in block["trade_ratio"].items():
                print(f"  {mapping:24s} ratio={entry['estimate']} ci={entry['ci95']}")
        if name == "h3":
            print(f"  alpha_star={block['alpha_star']} (grid estimate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
