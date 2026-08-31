"""Execute an experiment matrix and write one run directory.

    python scripts/03_run_experiment.py experiment=smoke
    python scripts/03_run_experiment.py experiment=lambda_sweep

Reads cached posteriors, never EEG. Fails fast: an exception in one episode
aborts the run, because a matrix silently missing cells is worse than no
results.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from micm.eval.runner import run_all, sanity_block
from micm.eval.writer import write_run
from micm.utils import configure_logging, get_logger, load_config

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=None, help="config name under configs/, e.g. experiment/smoke"
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. experiment=smoke")
    args = parser.parse_args(argv)

    name = args.config
    passthrough = []
    for override in args.overrides:
        if name is None and override.startswith("experiment="):
            name = f"experiment/{override.split('=', 1)[1]}"
        else:
            passthrough.append(override)
    if name is None:
        parser.error("name the experiment, e.g. experiment=smoke")

    cfg = load_config(name, overrides=tuple(passthrough))
    configure_logging(getattr(logging, str(cfg.logging.level)))

    started = time.perf_counter()
    results = run_all(cfg)
    sanity = sanity_block(cfg)
    run = write_run(
        results, cfg, sanity=sanity, wall_time_s=time.perf_counter() - started
    )

    print(f"\nrun_id      {run.run_id}")
    print(f"directory   {run.directory}")
    print(f"episodes    {len(run.episodes)}")
    print(f"sanity      all_passed={sanity['all_passed']}")
    for key, entry in sanity.items():
        if not (isinstance(entry, dict) and entry.get("applicable")):
            continue
        numbers = " ".join(
            f"{field}={value}"
            for field, value in entry.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        print(f"  {key}: pass={entry['pass']}  {numbers}")
    return 0 if sanity["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
