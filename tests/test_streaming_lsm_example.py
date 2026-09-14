"""Behavioral checks for the explicit-state reservoir example."""

import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).parents[1] / "examples" / "streaming_lsm.py"
spec = importlib.util.spec_from_file_location("streaming_lsm", path)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.fixture
def model():
    torch.manual_seed(3)
    return example.StreamingLSM()


def test_chunked_matches_full_sequence(model):
    x = torch.rand(17, 3, 4)
    full, final = model(x)
    state = None
    pieces = []
    for chunk in x.split(5):
        logits, state = model(chunk, state)
        pieces.append(logits)
    torch.testing.assert_close(full, torch.cat(pieces))
    for a, b in zip(final, state):
        torch.testing.assert_close(a, b)


def test_explicit_state_supports_interleaved_sessions(model):
    a, b = torch.rand(12, 2, 4), torch.rand(12, 2, 4)
    expected, _ = model(a)
    first, state = model(a[:5])
    model(b)
    rest, _ = model(a[5:], state)
    torch.testing.assert_close(expected, torch.cat([first, rest]))
    restarted, _ = model(a)
    torch.testing.assert_close(expected, restarted)


def test_future_inputs_cannot_change_prefix(model):
    x = torch.rand(15, 2, 4)
    before, _ = model(x)
    x[7:] = 10
    after, _ = model(x)
    torch.testing.assert_close(before[:7], after[:7])


def test_optimizer_updates_only_readout(model):
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    logits, state = model(torch.ones(12, 3, 4, requires_grad=True))
    torch.nn.functional.cross_entropy(
        logits[-1], torch.zeros(3).long()
    ).backward()
    optimizer.step()
    assert all(s.grad_fn is None for s in state)
    for name, p in model.named_parameters():
        if not name.startswith("readout."):
            assert p.grad is None
            torch.testing.assert_close(before[name], p)
    assert not torch.equal(before["readout.weight"], model.readout.weight)


def test_save_load_excludes_session_state(model, tmp_path):
    x = torch.rand(10, 2, 4)
    expected, _ = model(x)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = example.StreamingLSM()
    restored.load_state_dict(torch.load(path, weights_only=True))
    actual, _ = restored(x)
    torch.testing.assert_close(expected, actual)


def test_state_validation_and_dtype(model):
    model = model.double()
    x = torch.ones(4, 2, 4, dtype=torch.float64)
    _, state = model(x)
    assert all(s.dtype == torch.float64 for s in state)
    with pytest.raises(ValueError, match="State"):
        model(torch.ones(4, 3, 4, dtype=torch.float64), state)
    with pytest.raises(ValueError, match="State"):
        model(x, tuple(s.float() for s in state))
    with pytest.raises(ValueError, match="nonempty"):
        model(x[:0])


def test_recurrent_source_signs_and_diagonal(model):
    w = model.neuron.recurrent.weight
    split = int(0.8 * model.hidden_size)
    assert (w[:, :split] >= 0).all()
    assert (w[:, split:] <= 0).all()
    assert (w.diag() == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu(model):
    x = torch.rand(20, 2, 4)
    expected, _ = model(x)
    model.cuda()
    actual, state = model(x.cuda())
    torch.testing.assert_close(expected, actual.cpu())
    assert all(s.is_cuda for s in state)
