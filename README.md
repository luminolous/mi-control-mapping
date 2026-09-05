<div align="center">

# Motor Imagery Control Mapping

<hr>

Four ways of turning the same motor-imagery posterior into robot motion, replayed
through a simulated planar arm on BCI Competition IV-2a, so the control mapping
can be measured against the decoder that feeds it.

[![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.5-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![MOABB](https://img.shields.io/badge/MOABB-1.6-4a7ebb?style=flat)](https://neurotechx.github.io/moabb)
[![MNE](https://img.shields.io/badge/MNE-1.12-6a5acd?style=flat)](https://mne.tools)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20cu124-ee4c2c?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

**[Method](#the-method)** · **[Reproducing](#reproducing)** ·
**[Design notes](#notes-on-the-design)** ·
**[Report an issue](https://github.com/luminolous/mi-control-mapping/issues)**

</div>

---

A decoder turns EEG into a posterior over four imagined movements. Something then
has to turn that posterior into a velocity for the robot, and that step is
usually an argmax nobody writes down. This repository makes it the independent
variable: one cached set of posteriors, four mappings reading it, and a
centre-out task that records what each one achieved.

Decoding happens once. `scripts/02_cache_posteriors.py` fits a decoder on windows
from session T, writes every session-E posterior to an `.npz`, and no experiment
touches EEG again. Nine subjects and three decoders take an hour; the 10,260
episodes that replay them take three. Every mapping sees identical windows, so a
difference between mappings is the mapping.

## The method

Each trial becomes a burst: 3.5 s of imagery, then a 2 s gap where the robot
receives nothing and holds still. A 2 s window slides by 250 ms, and the first
window waits for 2 s of real trial rather than padding from the pre-cue baseline.
Seven windows per burst survive that, which fixes the duty cycle everything else
is calibrated around: the robot carries a command for 1.25 s of each 5.5 s cycle.

The task is centre-out with eight targets. The dataset has four classes. Four of
the eight targets therefore line up with no single class, and a controller that
can only emit class directions has to reach them by alternating between two
neighbours and climbing a staircase. A controller that can blend directions goes
straight there. That gap is what the experiment is built to price.

Decoder quality is swept by mixing the posterior toward the truth,

```
p' = lam * onehot(y) + (1 - lam) * p
```

with `lam` solved per cell to hit a stated accuracy rather than fixed, because a
fixed `lam` buys different amounts of accuracy from differently peaked
posteriors.

### The four mappings

**S1, argmax.** Take the most likely class, emit its direction at full speed.

**S2, posterior-weighted.** Blend the class directions by their probabilities,

```
u = v_max * normalise(p @ D)
```

optionally scaled down by the posterior's normalised entropy, so an unsure
decoder drives slower instead of driving somewhere arbitrary.

**S3, evidence accumulation.** Integrate log-evidence with a leak, and commit
only once it clears a threshold,

```
e <- gamma * e + log(p + eps)          once per decoder update
if max(e) > theta:  emit d[argmax(e)] for hold_s, then reset
```

Between commitments the command is zero, and a burst that never crosses the
threshold is recorded as such rather than smoothed into a zero command.

**S4, shared control.** Arbitrate between S2 and an autonomous potential field,

```
u = alpha * u_auto(s) + (1 - alpha) * u_user(p)
```

`u_user` is S2 itself rather than a copy of its formula, so `alpha = 0` reduces
to S2 to bit equality. Two variants of `u_auto` differ in where they aim:
intent-aware picks the target most consistent with the decoded intent,
intent-blind aims at whichever target the task made active.

## Install

Python 3.12.

```
make setup
```

which creates `.venv` and runs `pip install -e ".[dev]"`. EEGNet needs the
optional deep-learning extra:

```
pip install -e ".[dl]"
```

On CUDA, install `torch`, `torchvision` and `torchaudio` in one command from the
same index. Installing one alone downgrades the others and every braindecode
import then fails with a Windows DLL error that names nothing useful.

## Run

```
make test
make smoke
```

`make smoke` drives the whole path on synthetic posteriors in about ten seconds,
writes a real run directory, and reports `all_passed=True`. It needs no dataset
and no GPU. Run it after any change to the pipeline.

## Reproducing

Each stage reads only what the one before it wrote.

```
make data          # download BCI IV-2a through MOABB, about 744 MB
make decoders      # fit per subject, report kappa against the published range
make decode        # cache posteriors; repeat per decoder
```

`make decode` prints a config hash. Copy it into
`configs/matrix_defaults.yaml` under
`replay_variants.burst_w2000.posterior_hash`, keyed by decoder. The hash covers
the whole caching config, so the three decoders get three different ones, and a
null entry raises with the decoder named rather than reading some other file.

The window and protocol ablations need three further caches:

```
python scripts/02_cache_posteriors.py decoder=riemann replay.window_s=1.0
python scripts/02_cache_posteriors.py decoder=riemann replay.window_s=3.0
python scripts/02_cache_posteriors.py decoder=riemann replay=stitched
```

Then the matrix, the models, and the figures:

```
make run EXP=main                              # nine experiments
make analyze RUN=artifacts/runs/main/<run_id>
make figures RUN="<run_dir> <run_dir> ..."
```

Nine experiments come to 10,260 episodes and about 3.4 h on eight cores. A run
directory is renamed into place only once it is complete, so an interrupted run
leaves nothing that a later analysis could read as finished, and every
`summary.json` records the commit that produced it.

## What the sanity checks guarantee

Every run computes four checks on synthetic decoders and writes them into
`summary.json`. A run whose checks fail is still written, with the failure
recorded and the exit code set.

**Oracle ceiling.** A perfect decoder reaches every target, for every mapping in
the grid. This one earns its keep: it caught obstacle placement that trapped
argmax at two of four feedback latencies, before any number reached a table.

**Chance floor.** A uniform posterior reaches almost nothing, so nothing can pass
by drifting into targets.

**alpha = 0 identity.** S4 at zero autonomy reproduces S2 to bit equality, which
catches an arbitration sign error that would otherwise read as an effect.

**alpha = 1 independence.** Intent-blind S4 at full autonomy produces the same
episode from a weak decoder and a strong one.

## What this does not measure

**Obstacle avoidance.** A controller restricted to four directions cannot route
around a round obstacle it meets head on. The obstacles here sit off the target
bearings and cost contacts rather than blocking a path.

**Online adaptation.** Every episode replays recorded data. No subject saw the
robot, so nobody learned to drive it.

**Decoder ranking.** The three decoders exist to span a quality axis, not to
compete. The controlled axis is the oracle mixing above.

## Notes on the design

**Obstacle placement is relative to the target ring.** Obstacles on a target's
bearing trap a four-direction controller: it walks the staircase into the
obstacle, both cardinal directions it alternates between push in, and the
collision resolver returns it to the surface each step. They sit 11.25 degrees
off, half a target spacing. Any config that changes `n_targets` has to move them.

**Episode seeds come from cell content, not from queue position.**
`generator_for(seed, *cell.key())` hashes what the cell is, so `n_workers` cannot
change a number and re-running part of a grid reproduces the whole. `agents/05`
asks for `SeedSequence.spawn` per worker; this is the stronger property that
requirement protects.

**The runner never reads `cfg.replay`.** Two ablations vary the protocol and the
window, and one composed `replay` group can describe one condition. Every
condition is a `replay_variants` entry filed under a key derived from its own
protocol and window, and a grid combination with no variant raises during
enumeration rather than on the first episode.

**Two module boundaries are enforced by tests.** `mapping/` and `env/` import
nothing from `decoders/` or `data/`; they read cached posteriors. The runner
never imports the renderer.

## Possible follow-ups

Not implemented here: a golden-file regression test for the decoding path, an
S3 parameter sweep tracing its latency and accuracy front, obstacle layouts that
obstruct without trapping a four-direction controller, other datasets, and any
online experiment.

## Credits

BCI Competition IV dataset 2a comes from the Institute for Knowledge Discovery,
Graz University of Technology, loaded through
[MOABB](https://github.com/NeuroTechX/moabb).

EEGNet is used through [braindecode](https://github.com/braindecode/braindecode),
the Riemannian decoder through [pyRiemann](https://github.com/pyRiemann/pyRiemann),
and the mixed models through [statsmodels](https://www.statsmodels.org).

This repository is MIT licensed; see `LICENSE`.
