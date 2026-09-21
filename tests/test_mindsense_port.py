"""Self-contained adapter tests using deterministic, untrained parameters."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

# Examples are source-tree tools, outside the installed snntorch package.
_directory = Path(__file__).resolve().parents[1] / "examples/mindsense_port"
_spec = importlib.util.spec_from_file_location(
    "mindsense_model", _directory / "model.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
TorchMindSense = _module.TorchMindSense


@pytest.fixture
def archive():
    import json

    rng = np.random.default_rng(17)
    n = 8
    return {
        "config": np.array(
            json.dumps({"neurons": n, "delta_threshold": 0.75})
        ),
        "smoothing": np.array(4),
        "recurrent": rng.normal(0, 0.08, (n, n)),
        "inputs": rng.uniform(0, 0.15, (n, 32)),
        "alpha": rng.uniform(0.5, 0.8, n),
        "beta": rng.uniform(0.6, 0.9, n),
        "thresholds": rng.uniform(0.8, 1.2, n),
        "mean": np.zeros(3 * n),
        "scale": np.ones(3 * n),
        "weights": rng.normal(0, 0.1, 3 * n + 1),
        "calibration": np.array([1.0, 0.0]),
    }


def test_current_voltage_and_trace_equations(archive):
    model = TorchMindSense(archive)
    model.inputs.zero_()
    model.recurrent.zero_()
    model.neuron.threshold.fill_(100)
    model.current.fill_(2)
    model.voltage.fill_(3)
    model.fast.fill_(0.4)
    model.slow.fill_(0.6)
    model.state(np.zeros(8))
    torch.testing.assert_close(model.current, 2 * model.neuron.alpha)
    torch.testing.assert_close(
        model.voltage, 3 * model.neuron.beta + 2 * model.neuron.alpha
    )
    torch.testing.assert_close(model.fast, torch.full_like(model.fast, 0.2))
    torch.testing.assert_close(model.slow, torch.full_like(model.slow, 0.54))


def test_encoder_crossings_levels_and_clipping(archive):
    model = TorchMindSense(archive)
    model.state(np.full(8, 0.75))
    np.testing.assert_array_equal(model.events[:8], np.ones(8))
    assert model.events[8:].count_nonzero() == 0
    model.state(np.full(8, 0.75))
    assert model.events.count_nonzero() == 0
    model.state(np.full(8, 0.75))
    np.testing.assert_array_equal(model.events[16:24], np.ones(8))
    model.state(np.full(8, -0.75))
    np.testing.assert_array_equal(model.events[8:16], np.full(8, 2))
    model.reset()
    model.state(np.full(8, 12))
    np.testing.assert_array_equal(model.events[:8], np.full(8, 8))
    np.testing.assert_array_equal(model.events[16:24], np.full(8, 4))


def test_threshold_boundary_refractory_and_reset(archive):
    model = TorchMindSense(archive)
    model.inputs.zero_()
    model.recurrent.zero_()
    model.neuron.alpha.fill_(1)
    model.neuron.beta.zero_()
    model.neuron.threshold.fill_(1)
    model.current.fill_(1)
    model.state(np.zeros(8))
    assert (model.spikes == 1).all()  # equality fires
    assert (model.voltage == 0).all()
    model.state(np.zeros(8))
    assert (model.spikes == 0).all()  # one refractory tick
    assert (model.current == 1).all()  # current still integrates
    model.state(np.zeros(8))
    assert (model.spikes == 1).all()
    model.reset()
    assert model.history_count == 0
    for name in (
        "current",
        "voltage",
        "spikes",
        "fast",
        "slow",
        "reference",
        "refractory",
        "history",
        "level_accumulator",
    ):
        assert getattr(model, name).count_nonzero() == 0


def test_chunk_continuity(archive):
    z = np.random.default_rng(11).normal(size=(60, 8))
    model = TorchMindSense(archive)
    expected = [model.step(x) for x in z]
    model.reset()
    actual = []
    for chunk in np.array_split(z, 7):
        actual.extend(model.step(x) for x in chunk)
    np.testing.assert_array_equal(expected, actual)


def test_causal_outputs_and_independent_instances(archive):
    z = np.random.default_rng(1).normal(size=(50, 8))
    a, b = TorchMindSense(archive), TorchMindSense(archive)
    scores = [a.step(x) for x in z]
    changed = z.copy()
    changed[25:] = 8
    np.testing.assert_array_equal(
        scores[:25], [b.step(x) for x in changed][:25]
    )
    a.reset()
    np.testing.assert_array_equal(scores, [a.step(x) for x in z])


def test_archive_and_checkpoint_roundtrip_without_session(archive, tmp_path):
    source = tmp_path / "toy.npz"
    np.savez(source, **archive)
    model = TorchMindSense.load(source)
    model.step(np.ones(8) * 3)
    checkpoint = tmp_path / "toy.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = TorchMindSense(archive)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert restored.history_count == 0
    assert restored.current.count_nonzero() == 0
    model.reset()
    for x in np.random.default_rng(0).normal(size=(40, 8)):
        assert model.step(x) == restored.step(x)


def test_invalid_inputs_do_not_advance_state(archive):
    model = TorchMindSense(archive)
    model.step(np.ones(8))
    before = model.current.clone()
    for z in ([1] * 7, [float("nan")] * 8, [float("inf")] * 8):
        with pytest.raises(ValueError):
            model.step(z)
        assert model.history_count == 1
        torch.testing.assert_close(model.current, before)


def test_dtype_conversion_and_frozen_weights(archive):
    model = TorchMindSense(archive).float()
    model.step(np.ones(8))
    assert model.current.dtype == torch.float32
    assert model.history.dtype == torch.float32
    assert model.neuron.alpha.dtype == torch.float32
    assert model.refractory.dtype == torch.long
    assert not list(model.parameters())
    with pytest.raises(ValueError):
        TorchMindSense(archive, dtype=torch.float16)


def test_same_input_depends_on_recent_history(archive):
    model = TorchMindSense(archive)
    cold = model.state(np.zeros(8))
    for _ in range(20):
        model.state(np.full(8, 4))
    warm = model.state(np.zeros(8))
    assert not torch.equal(cold, warm)
    model.reset()
    torch.testing.assert_close(model.state(np.zeros(8)), cold)


@pytest.mark.parametrize(
    "field",
    [
        "config",
        "smoothing",
        "recurrent",
        "inputs",
        "alpha",
        "beta",
        "thresholds",
        "mean",
        "scale",
        "weights",
        "calibration",
    ],
)
def test_missing_archive_fields_rejected(archive, field):
    del archive[field]
    with pytest.raises(ValueError, match="Missing archive field"):
        TorchMindSense(archive)


@pytest.mark.parametrize(
    "field,value",
    [
        ("smoothing", np.array(0)),
        ("smoothing", np.array(2.5)),
        ("smoothing", np.array(True)),
        ("smoothing", np.array([4])),
        ("inputs", np.zeros((8, 31))),
        ("recurrent", np.zeros((7, 8))),
        ("weights", np.zeros(24)),
        ("calibration", np.array([1])),
        ("scale", np.zeros(24)),
        ("thresholds", np.full(8, -1.0)),
        ("alpha", np.full(8, 1.1)),
        ("beta", np.full(8, -0.1)),
        ("mean", np.full(24, np.nan)),
        ("weights", np.full(25, np.inf)),
        ("inputs", np.full((8, 32), "0.1")),
        ("inputs", np.zeros((8, 32), dtype=complex)),
        ("config", np.array("[]")),
        ("config", np.array("{")),
    ],
)
def test_malformed_archive_rejected(archive, field, value):
    archive[field] = value
    with pytest.raises(ValueError):
        TorchMindSense(archive)


@pytest.mark.parametrize(
    "field,value",
    [
        ("neurons", 0),
        ("neurons", True),
        ("neurons", 8.5),
        ("delta_threshold", 0),
        ("delta_threshold", -1),
        ("delta_threshold", float("nan")),
        ("delta_threshold", float("inf")),
        ("delta_threshold", "0.75"),
    ],
)
def test_invalid_config_rejected(archive, field, value):
    import json

    config = json.loads(str(archive["config"]))
    config[field] = value
    archive["config"] = np.array(json.dumps(config))
    with pytest.raises(ValueError):
        TorchMindSense(archive)


@pytest.mark.parametrize(
    "field,value",
    [
        ("weights", np.full(25, 1e100)),
        ("scale", np.full(24, 1e-100)),
    ],
)
def test_float32_conversion_rejects_overflow_and_underflow(
    archive, field, value
):
    archive[field] = value
    TorchMindSense(archive, dtype=torch.float64)
    with pytest.raises(ValueError):
        TorchMindSense(archive, dtype=torch.float32)
