"""Download BCI IV-2a through MOABB and verify what arrived.

Run this once before anything else:

    python scripts/00_download_data.py
    python scripts/00_download_data.py subjects=[1,2]

Idempotent. MOABB serves files it already has, so a second run only verifies.
Exits non-zero if any subject fails verification, so it can be chained.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from micm.data.download import download_and_verify
from micm.utils import configure_logging, get_logger, load_config

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="download", help="config name under configs/ (default: download)"
    )
    parser.add_argument(
        "--subject",
        type=int,
        action="append",
        dest="subjects",
        help="restrict to one subject; repeatable. Defaults to every configured subject.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. paths.data_raw=/mnt/data",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=tuple(args.overrides))
    configure_logging(getattr(logging, str(cfg.logging.level)))

    logger.info("raw data directory: %s", Path(cfg.paths.data_raw).resolve())
    report = download_and_verify(cfg, subjects=args.subjects)

    print(report.as_table())

    if not report.ok:
        logger.error("verification failed; the counts above do not match configs/data/bci2a.yaml")
        return 1

    logger.info("all %d subjects verified", len(report.subjects))
    return 0


if __name__ == "__main__":
    sys.exit(main())
