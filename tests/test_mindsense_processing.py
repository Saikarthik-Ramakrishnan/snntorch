"""Deterministic deadline accounting and local runner guard tests."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_directory = Path(__file__).resolve().parents[1] / "examples/mindsense_port"
_name = "_mindsense_lab_under_test"
_spec = importlib.util.spec_from_file_location(
    _name,
    _directory / "__init__.py",
    submodule_search_locations=[str(_directory)],
)
_package = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _package
_spec.loader.exec_module(_package)
processing = importlib.import_module(_name + ".processing")
runner = importlib.import_module(_name + ".run_lab")


def decision(status="GREEN", score=0.2):
    return {"status": status, "score": score}


def test_accounting_includes_late_abstention_and_warmup():
    results = [
        decision(),
        decision(),
        decision("ABSTAIN", None),
        decision("WARMUP", None),
    ]
    report = processing.summarize([0.001, 0.02, 0.01, 0.002], results, 10, 1)
    assert report["offered"] == report["completed"] == 4
    assert report["on_time_handling_ratio"] == 0.75
    assert report["on_time_scored_ratio"] == 0.25
    assert report["late"] == 1
    assert report["dropped"] == 0
    assert not report["handling_target_met"]
    assert report["status_counts"] == {"GREEN": 2, "ABSTAIN": 1, "WARMUP": 1}


def test_fast_abstentions_cannot_pass_scored_coverage():
    report = processing.summarize(
        [0.001] * 20, [decision("ABSTAIN", None)] * 20, 10, 1
    )
    assert report["handling_target_met"]
    assert not report["scored_target_met"]
    assert report["on_time_scored"] == 0


def test_deadline_and_target_boundaries_are_inclusive():
    report = processing.summarize(
        [0.01] * 19 + [0.011], [decision()] * 20, 10, 1
    )
    assert report["on_time_handling_ratio"] == 0.95
    assert report["handling_target_met"] and report["scored_target_met"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_fail_both_accounting_and_parity(score):
    with pytest.raises(ValueError):
        processing.summarize([0.001], [decision(score=score)], 10, 1)
    with pytest.raises(ValueError):
        runner.score_error(score, 0.2)
    with pytest.raises(ValueError):
        runner.score_error(0.2, score)


@pytest.mark.parametrize(
    "result",
    [
        decision("ABSTAIN", 0.2),
        decision("WARMUP", 0.2),
        decision("GREEN", None),
        decision("", 0.2),
    ],
)
def test_inconsistent_status_and_score_rejected(result):
    with pytest.raises(ValueError):
        processing.summarize([0.001], [result], 10, 1)


@pytest.mark.parametrize(
    "latencies", [[], [-1], [float("nan")], [[0.1]], [1, 2]]
)
def test_invalid_latency_arrays_rejected(latencies):
    with pytest.raises(ValueError):
        processing.summarize(latencies, [decision()], 10, 1)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


@pytest.mark.parametrize(
    "duration,expected_ms",
    [
        (0.001, [1, 1, 1]),
        (0.02, [20, 30, 40]),
    ],
)
def test_paced_replay_counts_backlog_and_drains_all_frames(
    duration, expected_ms
):
    clock = FakeClock()
    seen = []

    def handler(frame, index):
        seen.append((frame, index))
        clock.now += duration
        return decision()

    report = processing.paced_replay(
        handler, [10, 20, 30], clock=clock.clock, sleep=clock.sleep
    )
    np.testing.assert_allclose(report["response_ms"], expected_ms)
    assert seen == [(10, 0), (20, 1), (30, 2)]
    assert report["completed"] == 3
    assert report["late"] == (3 if duration == 0.02 else 0)
    if duration == 0.02:
        assert clock.sleeps == []


def test_scheduler_oversleep_counts_toward_latency():
    clock = FakeClock()

    def sleep(duration):
        clock.now += duration + 0.02

    def handler(frame, index):
        clock.now += 0.001
        return decision()

    report = processing.paced_replay(
        handler, [1, 2], clock=clock.clock, sleep=sleep
    )
    np.testing.assert_allclose(report["response_ms"], [1, 21])
    assert report["late"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"offered_hz": 0},
        {"offered_hz": float("nan")},
        {"deadline_ms": -1},
        {"deadline_ms": float("inf")},
    ],
)
def test_invalid_replay_config_fails_before_handler(kwargs):
    def handler(frame, index):
        pytest.fail("handler must not run")

    with pytest.raises(ValueError):
        processing.paced_replay(handler, [1], **kwargs)


def test_empty_replay_rejected():
    with pytest.raises(ValueError):
        processing.paced_replay(lambda frame, index: decision(), [])


def test_missing_private_project_has_clear_error(tmp_path):
    with pytest.raises(ValueError, match="Missing MindSense files"):
        runner.load_project(tmp_path)


def test_bad_report_is_not_written(tmp_path):
    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        runner.write_report(path, {"score": float("nan")})
    assert not path.exists()


def test_benchmark_uses_fresh_sessions_and_rotating_order(monkeypatch):
    loaded = []
    sessions = []
    replayed = []

    def load(path, dtype=None):
        loaded.append(dtype)
        return object()

    class Stream:
        def __init__(self, model, baseline):
            assert baseline.shape == (20, 8)
            self.timestamps = []
            sessions.append(self)

        def push(self, frame, timestamp):
            self.timestamps.append(timestamp)
            return decision()

    def replay(handler, frames, rate, deadline):
        replayed.append(frames.copy())
        results = [handler(frame, i) for i, frame in enumerate(frames)]
        return processing.summarize(
            [0.001] * len(frames), results, deadline, 1
        )

    project = SimpleNamespace(
        weights="unused",
        profile=lambda *args: object(),
        generate=lambda profile, length, **kwargs: (
            np.arange(length * 8).reshape(length, 8),
            None,
        ),
        reference=SimpleNamespace(load=load),
        stream=Stream,
    )
    monkeypatch.setattr(runner.TorchMindSense, "load", load)
    monkeypatch.setattr(runner, "paced_replay", replay)
    report = runner.processing_benchmark(project, frames=3, repeats=3)
    assert len(loaded) == len(sessions) == len(replayed) == 9
    assert len({tuple(order) for order in report["model_order"]}) == 3
    assert report["all_handling_targets_met"]
    assert report["all_scored_targets_met"]
    for session in sessions:
        assert session.timestamps == list(range(53))
    for frames in replayed:
        np.testing.assert_array_equal(frames, replayed[0])
