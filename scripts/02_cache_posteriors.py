"""Decode every window once and cache the posteriors.

    python scripts/02_cache_posteriors.py
    python scripts/02_cache_posteriors.py decoder=fbcsp --subject 1
    python scripts/02_cache_posteriors.py --force

This is the boundary of the project. Everything after it reads the `.npz` files
this writes and never touches EEG again, which is what makes the experiment
matrix affordable: all four mappings and all six lambda values read the same
cache. Skipping it would multiply the work by the number of mappings and make
iterating on a mapping unbearable.

The decoder is fitted on **windows**, not on whole trials. A trial is 3.5 s and a
window is 2 s, so a trial-fitted decoder would be applied to inputs it never saw
the length of. For the covariance decoders that is a silent statistics mismatch;
for EEGNet it is a shape error. Fitting on windows also matches the deployment
condition, since online the decoder only ever sees one window.

Cache invalidation is by config hash in the filename. An existing file with the
same name came from the same config, so it is skipped and logged rather than
recomputed. `--force` is the deliberate way round that.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from micm.data.constants import SESSION_TRAIN
from micm.data.download import build_dataset, load_subject_raws, set_download_dir
from micm.data.epoching import epoch_subject
from micm.data.splits import train_test_indices
from micm.decoders.registry import build_decoder
from micm.replay.cache import build_meta, posterior_path, write_posteriors
from micm.replay.stream import build_windows, extract_windows, round_half_up
from micm.utils import config_hash, configure_logging, get_logger, load_config

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="cache", help="config name under configs/")
    parser.add_argument(
        "--subject",
        type=int,
        action="append",
        dest="subjects",
        help="restrict to one subject; repeatable",
    )
    parser.add_argument(
        "--force", action="store_true", help="recompute and overwrite existing cache files"
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=tuple(args.overrides))
    configure_logging(getattr(logging, str(cfg.logging.level)))

    cfg_hash = config_hash(cfg)
    subjects = list(args.subjects if args.subjects else cfg.data.subjects)
    decoder_name = str(cfg.decoder.name)

    sfreq = float(cfg.data.sfreq)
    window_s, stride_s = float(cfg.replay.window_s), float(cfg.replay.stride_s)
    window_samples = round_half_up(window_s * sfreq)
    window_ms, stride_ms = round(window_s * 1000), round(stride_s * 1000)
    tmin, tmax = float(cfg.data.epoch.tmin), float(cfg.data.epoch.tmax)

    artifacts = Path(cfg.paths.artifacts)
    set_download_dir(Path(cfg.paths.data_raw))
    dataset = build_dataset(cfg)

    def windows_of(indices: np.ndarray, data) -> tuple:  # type: ignore[no-untyped-def]
        meta = data.trial_meta.iloc[indices].reset_index(drop=True)
        index = build_windows(
            data.X[indices],
            meta,
            protocol=str(cfg.replay.protocol),
            sfreq=sfreq,
            window_s=window_s,
            stride_s=stride_s,
            tmin=tmin,
            tmax=tmax,
            first_window=str(cfg.replay.first_window),
        )
        signal = extract_windows(
            data.X[indices],
            index,
            window_samples=window_samples,
            protocol=str(cfg.replay.protocol),
        )
        return index, signal

    written = 0
    for subject in subjects:
        logger.info("subject %d: loading", subject)
        data = epoch_subject(load_subject_raws(dataset, cfg, subject), cfg.data, subject=subject)
        train_idx, test_idx = train_test_indices(data.trial_meta)
        by_session = {SESSION_TRAIN: train_idx, "E": test_idx}

        train_index, train_signal = windows_of(train_idx, data)
        logger.info(
            "subject %d: fitting %s on %d windows from %d session %s trials",
            subject,
            decoder_name,
            len(train_signal),
            len(train_idx),
            SESSION_TRAIN,
        )
        decoder = build_decoder(cfg.decoder)
        decoder.fit(train_signal, train_index.label)

        for session in cfg.sessions:
            session = str(session)
            path = posterior_path(
                artifacts,
                subject=subject,
                session=session,
                decoder=decoder_name,
                window_ms=window_ms,
                stride_ms=stride_ms,
                cfg_hash=cfg_hash,
            )
            if path.exists() and not args.force:
                logger.info("subject %d session %s: %s exists, skipping", subject, session, path.name)
                continue

            in_sample = session == SESSION_TRAIN
            if in_sample:
                logger.warning(
                    "subject %d session %s is the training session: these posteriors are "
                    "in-sample and must not be used for a closed-loop result",
                    subject,
                    session,
                )

            index, signal = (
                (train_index, train_signal) if in_sample else windows_of(by_session[session], data)
            )
            posterior = decoder.predict_proba(signal)
            kappa = float(cohen_kappa_score(index.label, posterior.argmax(axis=1)))

            write_posteriors(
                path,
                posterior=posterior,
                label=index.label,
                burst_id=index.burst_id,
                t_rel=index.t_rel,
                burst_onset=index.burst_onset,
                burst_offset=index.burst_offset,
                meta=build_meta(
                    cfg,
                    subject=subject,
                    session=session,
                    decoder=decoder_name,
                    protocol=str(cfg.replay.protocol),
                    window_s=window_s,
                    stride_s=stride_s,
                    n_bursts=len(index.burst_onset),
                    n_windows=len(index),
                    boundary_windows=int(index.boundary.sum()),
                    # Window-level, which is what the control loop actually sees.
                    # 01_fit_decoders reports the trial-level figure, which is
                    # higher because a trial is 3.5 s and a window is 2 s.
                    kappa_window=kappa,
                    in_sample=in_sample,
                ),
                force=args.force,
            )
            logger.info(
                "subject %d session %s: %d windows, window-level kappa %.3f",
                subject,
                session,
                len(index),
                kappa,
            )
            written += 1

    logger.info("wrote %d cache file(s) with config hash %s", written, cfg_hash)
    return 0


if __name__ == "__main__":
    sys.exit(main())
