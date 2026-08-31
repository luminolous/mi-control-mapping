"""Decoder tests, on synthetic trials only.

Real data is the user's job. What is checked here is the contract every decoder
has to satisfy before it is worth running on real data: posteriors that are
actually posteriors, a fit that is reproducible from its seed, and a fitted
object that survives a round trip through joblib.

The synthetic trials carry a class-specific oscillation on a class-specific
spatial pattern, which is the structure CSP and the Riemannian mean are built to
find. A decoder that cannot separate these has a bug; one that can is only
proven to run, not to be correct on EEG.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import cohen_kappa_score

from micm.decoders.base import (
    BaseDecoder,
    Decoder,
    validate_epochs,
    validate_labels,
    validate_posterior,
)
from micm.decoders.eegnet import EEGNetDecoder, resolve_device
from micm.decoders.fbcsp import FBCSPDecoder, band_edges, bandpass
from micm.decoders.registry import DECODERS, build_decoder
from micm.decoders.riemann import RiemannDecoder
from micm.utils import load_config

SFREQ = 250.0
N_CHANNELS = 8
N_TIMES = 375
N_CLASSES = 4


def synthetic_trials(
    *, n_per_class: int = 15, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Trials whose class shows up as oscillatory power on a spatial pattern."""
    rng = np.random.default_rng(seed)
    patterns = rng.normal(size=(N_CLASSES, N_CHANNELS))
    time = np.arange(N_TIMES) / SFREQ
    oscillation = np.sin(2.0 * np.pi * 10.0 * time)

    trials: list[np.ndarray] = []
    labels: list[int] = []
    for cls in range(N_CLASSES):
        for _ in range(n_per_class):
            noise = rng.normal(size=(N_CHANNELS, N_TIMES))
            amplitude = 2.0 + 0.3 * rng.normal()
            trials.append(noise + amplitude * np.outer(patterns[cls], oscillation))
            labels.append(cls)

    X = np.asarray(trials, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    order = rng.permutation(len(y))
    return X[order], y[order]


def make_fbcsp(seed: int = 1337) -> FBCSPDecoder:
    return FBCSPDecoder(
        seed=seed,
        sfreq=SFREQ,
        band_low=4.0,
        band_high=40.0,
        band_width=4.0,
        filter_order=4,
        n_components=4,
        csp_reg="ledoit_wolf",
        n_features=12,
        lda_solver="eigen",
        lda_shrinkage="auto",
    )


def make_riemann(seed: int = 1337) -> RiemannDecoder:
    return RiemannDecoder(
        seed=seed, cov_estimator="oas", metric="riemann", logreg_c=1.0, max_iter=500
    )


def make_eegnet(seed: int = 1337) -> EEGNetDecoder:
    """Small training budget, so the shared contract stays fast to check.

    The published settings live in configs/decoder/eegnet.yaml; the numbers here
    exist only to exercise the code path.
    """
    return EEGNetDecoder(
        seed=seed,
        device="cpu",
        sfreq=SFREQ,
        f1=8,
        depth_multiplier=2,
        f2=16,
        kernel_length=32,
        drop_prob=0.25,
        batch_norm_momentum=0.1,
        n_epochs=40,
        batch_size=16,
        lr=0.005,
        weight_decay=0.0,
        val_fraction=0.25,
        patience=40,
    )


DECODER_FACTORIES = {"fbcsp": make_fbcsp, "riemann": make_riemann, "eegnet": make_eegnet}


@pytest.fixture(scope="module")
def trials() -> tuple[np.ndarray, np.ndarray]:
    return synthetic_trials()


@pytest.fixture(scope="module", params=list(DECODER_FACTORIES), ids=list(DECODER_FACTORIES))
def fitted(request, trials):  # type: ignore[no-untyped-def]
    """One fitted decoder per implementation, shared by the read-only checks.

    Module scoped because fitting EEGNet is the slowest thing in the suite and
    none of the checks that use this fixture mutate the decoder.
    """
    X, y = trials
    return DECODER_FACTORIES[request.param]().fit(X, y)


# --- the shared contract ---


@pytest.mark.parametrize("factory", DECODER_FACTORIES.values(), ids=DECODER_FACTORIES)
def test_decoder_satisfies_the_protocol(factory) -> None:  # type: ignore[no-untyped-def]
    assert isinstance(factory(), Decoder)
    assert isinstance(factory(), BaseDecoder)


def test_predict_proba_returns_a_valid_posterior(fitted, trials) -> None:  # type: ignore[no-untyped-def]
    X, _ = trials
    posterior = fitted.predict_proba(X)

    assert posterior.shape == (len(X), N_CLASSES)
    assert posterior.dtype == np.float32
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-5)
    assert (posterior >= 0.0).all()


def test_decoder_separates_the_synthetic_classes(fitted, trials) -> None:  # type: ignore[no-untyped-def]
    """Not a claim about EEG. A decoder failing this cannot be debugged on real data."""
    X, y = trials
    predicted = fitted.predict_proba(X).argmax(axis=1)
    assert cohen_kappa_score(y, predicted) > 0.5


@pytest.mark.parametrize("factory", DECODER_FACTORIES.values(), ids=DECODER_FACTORIES)
def test_fitting_is_deterministic(factory, trials) -> None:  # type: ignore[no-untyped-def]
    """Same seed, same numbers. Without this the golden posterior file is worthless."""
    X, y = trials
    first = factory(7).fit(X, y).predict_proba(X)
    second = factory(7).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(first, second)


def test_fitted_decoder_round_trips_through_joblib(fitted, trials, tmp_path) -> None:  # type: ignore[no-untyped-def]
    X, _ = trials
    expected = fitted.predict_proba(X)

    path = tmp_path / "decoder.pkl"
    joblib.dump(fitted, path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(X), expected)


@pytest.mark.parametrize("factory", DECODER_FACTORIES.values(), ids=DECODER_FACTORIES)
def test_predict_before_fit_raises(factory, trials) -> None:  # type: ignore[no-untyped-def]
    X, _ = trials
    with pytest.raises(RuntimeError, match="not fitted"):
        factory().predict_proba(X)


def test_channel_count_change_between_fit_and_predict_raises(fitted, trials) -> None:  # type: ignore[no-untyped-def]
    """A changed montage would otherwise produce confident nonsense."""
    X, _ = trials
    with pytest.raises(ValueError, match="montage changed"):
        fitted.predict_proba(X[:, :-1])


@pytest.mark.parametrize("factory", DECODER_FACTORIES.values(), ids=DECODER_FACTORIES)
def test_decoder_does_not_touch_global_random_state(factory, trials) -> None:  # type: ignore[no-untyped-def]
    """Fitting must not move the global streams, or nothing else stays reproducible."""
    X, y = trials
    np.random.seed(0)
    numpy_before = np.random.rand()
    torch.manual_seed(0)
    torch_before = torch.rand(1).item()

    np.random.seed(0)
    torch.manual_seed(0)
    factory().fit(X, y)

    assert np.random.rand() == numpy_before
    assert torch.rand(1).item() == torch_before


# --- input validation ---


def test_validate_epochs_rejects_the_wrong_rank() -> None:
    with pytest.raises(ValueError, match="n_trials, n_channels, n_times"):
        validate_epochs(np.zeros((4, 5)))


def test_validate_epochs_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty trial batch"):
        validate_epochs(np.zeros((0, 4, 5)))


def test_validate_epochs_rejects_non_finite_values() -> None:
    X = np.zeros((2, 3, 4))
    X[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        validate_epochs(X)


def test_validate_labels_rejects_a_class_gap() -> None:
    """Class 2 missing would shift every posterior column by one."""
    with pytest.raises(ValueError, match="contiguous integers"):
        validate_labels(np.array([0, 1, 3, 3]), 4)


def test_validate_labels_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="3 trials but 4 labels"):
        validate_labels(np.array([0, 1, 0, 1]), 3)


def test_validate_posterior_rejects_rows_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_posterior(np.full((2, 4), 0.3), n_trials=2, n_classes=4)


def test_validate_posterior_rejects_negative_probabilities() -> None:
    p = np.array([[1.2, -0.2, 0.0, 0.0]])
    with pytest.raises(ValueError, match="negative probabilities"):
        validate_posterior(p, n_trials=1, n_classes=4)


def test_fit_rejects_labels_of_the_wrong_length(trials) -> None:  # type: ignore[no-untyped-def]
    X, y = trials
    with pytest.raises(ValueError, match="labels"):
        make_riemann().fit(X, y[:-1])


# --- filter bank ---


def test_band_edges_covers_the_range_contiguously() -> None:
    bands = band_edges(4.0, 40.0, 4.0)
    assert len(bands) == 9
    assert bands[0] == (4.0, 8.0)
    assert bands[-1] == (36.0, 40.0)
    assert all(bands[i][1] == bands[i + 1][0] for i in range(len(bands) - 1))


def test_band_edges_rejects_a_range_that_does_not_divide() -> None:
    with pytest.raises(ValueError, match=r"whole 4.0 Hz bands"):
        band_edges(4.0, 39.0, 4.0)


@pytest.mark.parametrize(("low", "high"), [(0.0, 8.0), (8.0, 4.0), (4.0, 200.0)])
def test_bandpass_rejects_an_invalid_band(low: float, high: float) -> None:
    with pytest.raises(ValueError):
        bandpass(np.zeros((2, 3, N_TIMES)), low, high, sfreq=SFREQ, order=4)


def test_bandpass_suppresses_out_of_band_power() -> None:
    time = np.arange(N_TIMES) / SFREQ
    signal = np.sin(2.0 * np.pi * 50.0 * time)[None, None, :]
    filtered = bandpass(signal, 4.0, 8.0, sfreq=SFREQ, order=4)
    assert filtered.std() < 0.05 * signal.std()


def test_fbcsp_rejects_more_features_than_the_bank_produces(trials) -> None:  # type: ignore[no-untyped-def]
    X, y = trials
    decoder = make_fbcsp()
    decoder.n_features = 999
    with pytest.raises(ValueError, match="exceeds the 36 features"):
        decoder.fit(X, y)


# --- registry ---


@pytest.mark.parametrize("name", ["fbcsp", "riemann", "eegnet"])
def test_registry_builds_each_decoder_from_its_config(name: str) -> None:
    cfg = load_config("decode", overrides=(f"decoder={name}",))
    decoder = build_decoder(cfg.decoder)
    assert isinstance(decoder, Decoder)
    assert decoder.name == name


def test_registry_config_seed_follows_the_global_seed() -> None:
    cfg = load_config("decode", overrides=("seed=99",))
    built = build_decoder(cfg.decoder)
    assert isinstance(built, BaseDecoder)
    assert built.seed == 99


@pytest.mark.parametrize("name", ["fbcsp", "riemann", "eegnet"])
def test_every_decoder_config_declares_a_kappa_range_and_its_origin(name: str) -> None:
    """The range tells the user a bad result is upstream of the classifier.

    `kappa_is_published` says whether the range is the literature's or this
    implementation's own. Without it, EEGNet scoring "within" its range would
    read as reproducing a published figure, which it does not.
    """
    cfg = load_config("decode", overrides=(f"decoder={name}",))
    low, high = (float(v) for v in cfg.decoder.expected_kappa)
    assert 0.0 < low < high < 1.0
    assert isinstance(cfg.decoder.kappa_is_published, bool)


def test_eegnet_range_is_marked_as_not_published() -> None:
    """EEGNet trains trial-wise; the published figures come from cropped training."""
    cfg = load_config("decode", overrides=("decoder=eegnet",))
    assert cfg.decoder.kappa_is_published is False


def test_registry_rejects_an_unknown_name() -> None:
    cfg = OmegaConf.create({"name": "nope", "params": {}})
    assert isinstance(cfg, DictConfig)
    with pytest.raises(KeyError, match="unknown decoder"):
        build_decoder(cfg)


def test_registry_rejects_a_node_without_params() -> None:
    cfg = OmegaConf.create({"name": "riemann"})
    assert isinstance(cfg, DictConfig)
    with pytest.raises(KeyError, match="no 'params' block"):
        build_decoder(cfg)


def test_registry_rejects_an_unexpected_param() -> None:
    """A stray config key must fail, not be ignored and vanish from the run record."""
    cfg = load_config("decode", overrides=("decoder=riemann",))
    node = OmegaConf.to_container(cfg.decoder, resolve=True)
    assert isinstance(node, dict)
    node["params"]["typo_key"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        build_decoder(OmegaConf.create(node))


def test_registry_lists_every_decoder() -> None:
    assert set(DECODERS) == {"fbcsp", "riemann", "eegnet"}


# --- EEGNet specifics ---


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_warns_loudly_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A run that meant to use the GPU and quietly did not has meaningless timings."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level("WARNING", logger="micm.micm.decoders.eegnet"):
        device = resolve_device("cuda")

    assert device.type == "cpu"
    assert any("Falling back to CPU" in record.message for record in caplog.records)


def test_resolve_device_honours_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda").type == "cuda"


def test_resolve_device_rejects_an_unknown_device() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        resolve_device("tpu")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("val_fraction", 0.0, "val_fraction"),
        ("val_fraction", 1.0, "val_fraction"),
        ("n_epochs", 0, "n_epochs"),
        ("patience", 0, "patience"),
    ],
)
def test_eegnet_rejects_bad_training_parameters(field: str, value: float, match: str) -> None:
    kwargs: dict[str, object] = {
        "seed": 1,
        "device": "cpu",
        "sfreq": SFREQ,
        "f1": 8,
        "depth_multiplier": 2,
        "f2": 16,
        "kernel_length": 32,
        "drop_prob": 0.25,
        "batch_norm_momentum": 0.1,
        "n_epochs": 5,
        "batch_size": 8,
        "lr": 0.01,
        "weight_decay": 0.0,
        "val_fraction": 0.25,
        "patience": 3,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        EEGNetDecoder(**kwargs)  # type: ignore[arg-type]


def test_eegnet_records_its_early_stopping_state(trials) -> None:  # type: ignore[no-untyped-def]
    """The selected epoch has to be inspectable, or early stopping cannot be reviewed."""
    X, y = trials
    decoder = make_eegnet().fit(X, y)

    assert decoder.best_epoch_ is not None
    assert 0 <= decoder.best_epoch_ < decoder.n_epochs
    assert decoder.best_val_loss_ is not None
    assert np.isfinite(decoder.best_val_loss_)


def test_eegnet_inner_split_is_stratified_and_stays_inside_its_input(trials) -> None:  # type: ignore[no-untyped-def]
    """Early stopping must never see session E. It only ever sees what fit was given."""
    _, y = trials
    decoder = make_eegnet()
    inner_train, inner_val = decoder._inner_split(np.asarray(y, dtype=np.int64))

    assert set(inner_train.tolist()).isdisjoint(inner_val.tolist())
    assert set(inner_train.tolist()) | set(inner_val.tolist()) == set(range(len(y)))
    assert set(np.unique(y[inner_val]).tolist()) == set(range(N_CLASSES))


def test_eegnet_checkpoint_loads_onto_cpu_regardless_of_fitted_device(
    trials, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    """A checkpoint fitted on a GPU box must open on a machine without one."""
    X, y = trials
    decoder = make_eegnet().fit(X, y)

    path = tmp_path / "eegnet.pkl"
    joblib.dump(decoder, path)
    restored = joblib.load(path)

    assert all(param.device.type == "cpu" for param in restored._model.parameters())
    np.testing.assert_array_equal(restored.predict_proba(X), decoder.predict_proba(X))


def test_eegnet_eval_mode_agrees_with_train_mode(trials) -> None:  # type: ignore[no-untyped-def]
    """Batch-norm running statistics must track the weights they normalise.

    braindecode's default momentum of 0.01 is faithful to the Keras original,
    which saw far more batches per epoch. On one subject an epoch is a handful of
    batches, so the running estimates lag the weights badly. Validation and
    prediction both run in eval mode and read those stale statistics, so the
    decoder reports chance while its training loss falls normally. Nothing about
    that raises, and the training loss curve looks healthy throughout.

    Comparing the two modes on the same data is the cheapest way to see it.
    """
    X, y = trials
    decoder = make_eegnet().fit(X, y)
    model = decoder._model
    assert model is not None

    batch = torch.from_numpy(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        model.eval()
        eval_accuracy = float((model(batch).argmax(1).numpy() == y).mean())
        model.train()
        train_accuracy = float((model(batch).argmax(1).numpy() == y).mean())
        model.eval()

    assert eval_accuracy > train_accuracy - 0.15, (
        f"eval mode {eval_accuracy:.3f} against train mode {train_accuracy:.3f}: "
        "the batch-norm running statistics do not match the weights"
    )
