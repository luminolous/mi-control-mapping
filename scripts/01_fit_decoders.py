"""Fit one decoder per subject on session T and score it on session E.

    python scripts/01_fit_decoders.py
    python scripts/01_fit_decoders.py decoder=fbcsp
    python scripts/01_fit_decoders.py --subject 1 --save

Reports Cohen's kappa per subject against the range declared in the decoder
config, which says whether it is the published one or this implementation's own.
A kappa far below it means preprocessing is wrong. Tuning the classifier to
compensate would hide the cause, so don't.

Epoch-anchoring check (docs/decisions.md D13b). The annotations sit on the cue,
measured with an offset sweep rather than inferred from `dataset.interval`.
Shifting the window off the imagery period must cost kappa:

    python scripts/01_fit_decoders.py --subject 1
    python scripts/01_fit_decoders.py --subject 1 data.epoch.event_offset_s=2.0

On subject 1 that shift costs about 0.25 kappa. If the two runs score alike, the
anchoring is wrong and everything built on it is suspect.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

from micm.data.download import build_dataset, load_subject_raws, set_download_dir
from micm.data.epoching import epoch_subject
from micm.data.splits import train_test_indices
from micm.decoders.registry import build_decoder
from micm.utils import config_hash, configure_logging, get_logger, load_config

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="decode", help="config name under configs/")
    parser.add_argument(
        "--subject",
        type=int,
        action="append",
        dest="subjects",
        help="restrict to one subject; repeatable",
    )
    parser.add_argument(
        "--save", action="store_true", help="write each fitted decoder to artifacts/decoders/"
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=tuple(args.overrides))
    configure_logging(getattr(logging, str(cfg.logging.level)))

    cfg_hash = config_hash(cfg)
    subjects = list(args.subjects if args.subjects else cfg.data.subjects)
    decoder_name = str(cfg.decoder.name)
    low, high = (float(v) for v in cfg.decoder.expected_kappa)
    published = bool(cfg.decoder.kappa_is_published)

    set_download_dir(Path(cfg.paths.data_raw))
    dataset = build_dataset(cfg)

    rows: list[tuple[int, int, int, float, float]] = []
    for subject in subjects:
        logger.info("subject %d: loading", subject)
        data = epoch_subject(load_subject_raws(dataset, cfg, subject), cfg.data, subject=subject)
        train, test = train_test_indices(data.trial_meta)

        logger.info("subject %d: fitting %s on %d trials", subject, decoder_name, len(train))
        decoder = build_decoder(cfg.decoder)
        decoder.fit(data.X[train], data.y[train])

        predicted = decoder.predict_proba(data.X[test]).argmax(axis=1)
        truth = data.y[test]
        rows.append(
            (
                subject,
                len(train),
                len(test),
                float(accuracy_score(truth, predicted)),
                float(cohen_kappa_score(truth, predicted)),
            )
        )

        if args.save:
            out = Path(cfg.paths.artifacts) / "decoders"
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(decoder, out / f"{subject:02d}_{decoder_name}_{cfg_hash}.pkl")

    kappas = np.array([row[4] for row in rows])
    header = f"{'subj':>4}  {'train':>6}  {'test':>5}  {'acc':>6}  {'kappa':>6}"
    print(f"\ndecoder={decoder_name}  config_hash={cfg_hash}")
    print(f"event_offset_s={float(cfg.data.epoch.event_offset_s)}")
    print(header)
    print("-" * len(header))
    for subject, n_train, n_test, acc, kappa in rows:
        print(f"{subject:>4}  {n_train:>6}  {n_test:>5}  {acc:>6.3f}  {kappa:>6.3f}")

    # The published range describes the mean over subjects, not any one subject.
    # Between-subject spread on IV-2a is larger than most effects of interest,
    # so a single subject outside the range says nothing on its own.
    verdict = "within" if low <= kappas.mean() <= high else "OUTSIDE"
    origin = (
        "published cross-subject range"
        if published
        else "range reproduced by this implementation, NOT the published one"
    )
    print(
        f"\nmean kappa {kappas.mean():.3f} (sd {kappas.std(ddof=1) if len(kappas) > 1 else 0.0:.3f}) "
        f"over {len(kappas)} subject(s)"
    )
    print(f"{origin} {low:.2f} to {high:.2f}: {verdict}")
    if len(kappas) < 9:
        print("note: fewer than 9 subjects, the comparison to the range is not meaningful yet")

    if len(kappas) == 9 and kappas.mean() < low:
        logger.warning(
            "mean kappa is below the expected range. The cause is upstream of the "
            "classifier: check the epoch anchoring, the filter band, and the split"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
