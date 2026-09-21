"""CPU replay timing with explicit deadlines and complete frame accounting."""

from collections import Counter
import time

import numpy as np


def positive_number(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def summarize(latencies_s, results, deadline_ms, elapsed_s, target=0.95):
    """Count every offered frame, including abstentions and late results."""
    positive_number(deadline_ms, "deadline_ms")
    positive_number(elapsed_s, "elapsed_s")
    if not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError("target must lie in (0, 1]")
    latencies = np.asarray(latencies_s, dtype=float)
    if (
        latencies.ndim != 1
        or not len(latencies)
        or len(latencies) != len(results)
        or not np.isfinite(latencies).all()
        or (latencies < 0).any()
    ):
        raise ValueError("Expected aligned, nonempty, finite latency/results")
    statuses, scored = [], []
    for result in results:
        status, score = result["status"], result["score"]
        if not isinstance(status, str) or not status:
            raise ValueError("Each result needs a nonempty status")
        has_score = score is not None
        if has_score and (not np.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Scores must be finite and lie in [0, 1]")
        if has_score and status in ("ABSTAIN", "WARMUP"):
            raise ValueError("ABSTAIN and WARMUP must have no score")
        if not has_score and status not in ("ABSTAIN", "WARMUP"):
            raise ValueError("A decision status requires a score")
        statuses.append(status)
        scored.append(has_score)
    on_time = latencies <= deadline_ms / 1000
    on_time_scored = on_time & np.asarray(scored)
    n = len(latencies)
    counts = Counter(statuses)
    return {
        "offered": n,
        "completed": n,
        "dropped": 0,
        "late": int((~on_time).sum()),
        "on_time_handled": int(on_time.sum()),
        "scored": int(sum(scored)),
        "on_time_scored": int(on_time_scored.sum()),
        "on_time_handling_ratio": float(on_time.mean()),
        "on_time_scored_ratio": float(on_time_scored.mean()),
        "handling_target_met": bool(on_time.mean() >= target),
        "scored_target_met": bool(on_time_scored.mean() >= target),
        "target_ratio": target,
        "deadline_ms": deadline_ms,
        "elapsed_s": elapsed_s,
        "completed_frames_per_second": n / elapsed_s,
        "response_p50_ms": float(np.quantile(latencies, 0.50) * 1000),
        "response_p95_ms": float(np.quantile(latencies, 0.95) * 1000),
        "response_p99_ms": float(np.quantile(latencies, 0.99) * 1000),
        "response_max_ms": float(latencies.max() * 1000),
        "status_counts": dict(counts),
        "response_ms": (latencies * 1000).tolist(),
        "statuses": statuses,
    }


def paced_replay(
    handler,
    frames,
    offered_hz=100,
    deadline_ms=10,
    *,
    clock=time.perf_counter,
    sleep=time.sleep,
):
    """Drain a preloaded FIFO; scheduled-arrival latency includes backlog.

    handler(frame, index) returns a Stream-style result. Logical sample times
    belong to the handler. There is no finite queue, transport or dropping.
    Injectable monotonic clock and sleep support deterministic overload tests.
    """
    positive_number(offered_hz, "offered_hz")
    positive_number(deadline_ms, "deadline_ms")
    if not len(frames):
        raise ValueError("At least one frame is required")
    origin = clock()
    latencies, results = [], []
    for index, frame in enumerate(frames):
        arrival = origin + index / offered_hz
        remaining = arrival - clock()
        while remaining > 0:
            sleep(remaining)
            remaining = arrival - clock()
        result = handler(frame, index)
        finished = clock()
        latencies.append(finished - arrival)
        results.append(result)
    report = summarize(latencies, results, deadline_ms, finished - origin)
    report["offered_frames_per_second"] = offered_hz
    return report
