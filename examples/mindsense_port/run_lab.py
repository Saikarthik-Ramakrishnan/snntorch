"""Compare this adapter with a trusted, local MindSense v3 checkout."""

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import snntorch
import torch

from .model import TorchMindSense
from .processing import paced_replay, positive_number


def load_project(root):
    """Import user-selected local code; no downloads or bundled weights."""
    root = root.expanduser().resolve()
    required = (
        "experiment.py",
        "robust.py",
        "streaming.py",
        "neurosensoros/data.py",
        "artifacts/robust_model.npz",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError("Missing MindSense files: " + ", ".join(missing))
    sys.path.insert(0, str(root))
    modules = {}
    for name in ("experiment", "robust", "streaming", "neurosensoros.data"):
        module = importlib.import_module(name)
        if root not in Path(module.__file__).resolve().parents:
            raise ValueError(f"{name} was imported from another checkout")
        modules[name] = module
    return SimpleNamespace(
        root=root,
        weights=root / "artifacts/robust_model.npz",
        sequences=modules["experiment"].sequences,
        reference=modules["robust"].RobustLiquid,
        stream=modules["streaming"].Stream,
        generate=modules["neurosensoros.data"].generate_sequence,
        profile=modules["neurosensoros.data"].make_user_profile,
    )


def compare_sequences(project, ids, dtype):
    reference = project.reference.load(project.weights)
    port = TorchMindSense.load(project.weights, dtype=dtype)
    maximum_error = decisions_differ = spike_differences = frames = 0
    metrics = {}
    for stress in (False, True):
        correct_reference = correct_port = endpoints = 0
        for seq in project.sequences(ids, stress=stress):
            reference.reset()
            port.reset()
            target = dict(zip(seq["ends"], seq["labels"]))
            for tick, z in enumerate(seq["z"], 1):
                a = reference.step(z)
                b = port.step(z)
                maximum_error = max(maximum_error, score_error(a, b))
                decisions_differ += (a >= 0.5) != (b >= 0.5)
                spike_differences += int(
                    np.count_nonzero(
                        reference.liquid.spikes != port.spikes.numpy()
                    )
                )
                frames += 1
                if tick in target:
                    correct_reference += (a >= 0.5) == target[tick]
                    correct_port += (b >= 0.5) == target[tick]
                    endpoints += 1
        metrics["stress" if stress else "clean"] = {
            "endpoints": endpoints,
            "numpy_accuracy": correct_reference / endpoints,
            "torch_accuracy": correct_port / endpoints,
        }
    return {
        "dtype": str(dtype),
        "frames": frames,
        "max_score_abs_error": maximum_error,
        "threshold_decision_differences": int(decisions_differ),
        "neuron_spike_differences": spike_differences,
        "classification": metrics,
    }


def score_error(a, b):
    """Reject invalid scores from either implementation before comparison."""
    if not np.isfinite([a, b]).all() or not (0 <= a <= 1 and 0 <= b <= 1):
        raise ValueError("Parity comparison requires finite scores in [0, 1]")
    return abs(a - b)


def compare_policy(project, dtype=torch.float64):
    result = {}
    for fault in (
        None,
        "eeg_dropout",
        "imu_flatline",
        "heart_rate_jump",
        "consent",
    ):
        raw, _ = project.generate(
            project.profile(900, 71),
            length=120,
            drift=fault is None,
            drift_onset=36,
            fault=fault,
            seed=81,
        )
        a = project.stream(project.reference.load(project.weights), raw[:20])
        b = project.stream(
            TorchMindSense.load(project.weights, dtype=dtype), raw[:20]
        )
        differences = 0
        error = 0
        statuses = {}
        for i, frame in enumerate(raw):
            consent = not (fault == "consent" and 50 <= i < 66)
            ra, rb = a.push(frame, i, consent), b.push(frame, i, consent)
            differences += (ra["status"], ra["reason"]) != (
                rb["status"],
                rb["reason"],
            )
            differences += (ra["score"] is None) != (rb["score"] is None)
            if ra["score"] is not None and rb["score"] is not None:
                error = max(error, score_error(ra["score"], rb["score"]))
            statuses[rb["status"]] = statuses.get(rb["status"], 0) + 1
        result[fault or "drift"] = {
            "decision_differences": int(differences),
            "max_score_abs_error": error,
            "statuses": statuses,
        }
    return result


def processing_benchmark(
    project, frames=500, repeats=3, offered_hz=100, deadline_ms=10
):
    """Compare the full CPU feature-processing path at a declared load."""
    for value, name in ((frames, "frames"), (repeats, "repeats")):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    positive_number(offered_hz, "offered_hz")
    positive_number(deadline_ms, "deadline_ms")
    raw, _ = project.generate(
        project.profile(8000, 19),
        length=frames + 70,
        drift=True,
        drift_onset=70 + frames // 2,
        seed=141,
    )
    factories = {
        "numpy": lambda: project.reference.load(project.weights),
        "torch_float64": lambda: TorchMindSense.load(project.weights),
        "torch_float32": lambda: TorchMindSense.load(
            project.weights, dtype=torch.float32
        ),
    }
    names = list(factories)
    runs = {name: [] for name in names}
    orders = []
    for repeat in range(repeats):
        # Rotate model order to reduce fixed-order bias between repeats.
        shift = repeat % len(names)
        order = names[shift:] + names[:shift]
        orders.append(order)
        for name in order:
            stream = project.stream(factories[name](), raw[:20])
            for tick, frame in enumerate(raw[20:70]):
                stream.push(frame, float(tick))

            def handle(frame, index):
                return stream.push(frame, float(index + 50))

            result = paced_replay(handle, raw[70:], offered_hz, deadline_ms)
            runs[name].append(result)
            print(
                f"{name} repeat {repeat + 1}: "
                f"on-time {result['on_time_handling_ratio']:.1%}, "
                f"scored {result['on_time_scored_ratio']:.1%}, "
                f"p99 {result['response_p99_ms']:.3f} ms",
                flush=True,
            )
    return {
        "measurement_scope": (
            "One CPU stream; normalization, quality checks, encoding, "
            "inference and policy. Synthetic engineered features. "
            "Model loading, baseline fit, 50 warmup frames, sensor "
            "acquisition, raw feature extraction, delivery and energy "
            "are excluded."
        ),
        "replay_scope": (
            "Preloaded FIFO drained in full, no finite queue or drops. "
            "Latency starts at scheduled arrival and includes backlog. "
            "Logical feature period stays 1 s; wall-clock replay is "
            "accelerated. Late and unscored frames stay in denominators."
        ),
        "target_is_provisional": True,
        "model_order": orders,
        "runs": runs,
        "all_handling_targets_met": all(
            run["handling_target_met"]
            for group in runs.values()
            for run in group
        ),
        "all_scored_targets_met": all(
            run["scored_target_met"]
            for group in runs.values()
            for run in group
        ),
    }


def memory_probe(project):
    model = TorchMindSense.load(project.weights)
    zero = np.zeros(8)
    pulse = np.ones(8) * 4
    model.reset()
    for _ in range(20):
        model.step(zero)
    cold = model.step(zero)
    model.reset()
    for _ in range(20):
        model.step(pulse)
    warm = model.step(zero)
    trace = [
        {
            "quiet_ticks": 1,
            "state_norm": float(
                torch.linalg.vector_norm(
                    torch.cat(
                        (model.current, model.voltage, model.fast, model.slow)
                    )
                )
            ),
            "score": warm,
        }
    ]
    for tick in range(2, 81):
        score = model.step(zero)
        if tick in (5, 10, 20, 40, 80):
            trace.append(
                {
                    "quiet_ticks": tick,
                    "state_norm": float(
                        torch.linalg.vector_norm(
                            torch.cat(
                                (
                                    model.current,
                                    model.voltage,
                                    model.fast,
                                    model.slow,
                                )
                            )
                        )
                    ),
                    "score": score,
                }
            )
    model.reset()
    reset = model.step(zero)
    return {
        "same_current_input": zero.tolist(),
        "after_quiet_history": cold,
        "after_pulse_history": warm,
        "after_reset": reset,
        "decay_trace": trace,
        "interpretation": (
            "Different recent histories can change the same-input score. "
            "Scores depend on synthetic training."
        ),
    }


def timing(project):
    z = np.random.default_rng(91).normal(size=(2000, 8))
    result = {}
    for name, model in (
        ("numpy", project.reference.load(project.weights)),
        ("torch_float64", TorchMindSense.load(project.weights)),
    ):
        for frame in z[:50]:
            model.step(frame)
        start = time.perf_counter()
        for frame in z:
            model.step(frame)
        elapsed = time.perf_counter() - start
        result[name] = {
            "frames": len(z),
            "elapsed_s": elapsed,
            "mean_model_step_ms": elapsed * 1000 / len(z),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mindsense-root",
        type=Path,
        required=True,
        help="Trusted local MindSense v3 checkout (executes its Python code)",
    )
    parser.add_argument("--output", type=Path, help="Local JSON report path")
    parser.add_argument(
        "--lesson", choices=("all", "memory", "processing"), default="all"
    )
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--offered-hz", type=float, default=100)
    parser.add_argument("--deadline-ms", type=float, default=10)
    args = parser.parse_args()
    try:
        positive_number(args.offered_hz, "offered_hz")
        positive_number(args.deadline_ms, "deadline_ms")
        if args.frames < 1 or args.repeats < 1:
            raise ValueError("frames and repeats must be positive integers")
        project = load_project(args.mindsense_root)
    except (ValueError, ImportError) as error:
        parser.error(str(error))
    filename = (
        "fork_processing_results.json"
        if args.lesson == "processing"
        else "fork_results.json"
    )
    output = (
        args.output or project.root / "experiments/snntorch_port" / filename
    )
    torch.set_num_threads(1)
    if args.lesson == "memory":
        print(json.dumps(memory_probe(project), indent=2))
        return
    source = Path(inspect.getfile(snntorch)).resolve()
    revision = subprocess.run(
        ["git", "-C", str(source.parent.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "purpose": (
            "Port verification and learning; "
            "existing synthetic holdouts reused without tuning"
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "snntorch": snntorch.__version__,
            "snntorch_source": str(source),
            "fork_revision": revision,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_threads": torch.get_num_threads(),
            "adapter_sha256": hashlib.sha256(
                Path(__file__).with_name("model.py").read_bytes()
            ).hexdigest(),
            "runner_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "processing_sha256": hashlib.sha256(
                Path(__file__).with_name("processing.py").read_bytes()
            ).hexdigest(),
        },
        "model_sha256": hashlib.sha256(
            project.weights.read_bytes()
        ).hexdigest(),
        "precision": {},
    }
    if args.lesson == "processing":
        report["purpose"] = "Synthetic CPU feature-processing deadline replay"
        report["processing"] = processing_benchmark(
            project,
            args.frames,
            args.repeats,
            args.offered_hz,
            args.deadline_ms,
        )
        write_report(output, report)
        # Record measurements even on failure; never tune a deadline to pass.
        if not (
            report["processing"]["all_handling_targets_met"]
            and report["processing"]["all_scored_targets_met"]
        ):
            raise SystemExit(1)
        return
    for dtype in (torch.float64, torch.float32):
        report["precision"][str(dtype)] = compare_sequences(
            project, list(range(5000, 5064)), dtype
        )
        print(
            json.dumps(report["precision"][str(dtype)], indent=2), flush=True
        )
    report["policy"] = compare_policy(project)
    report["policy_float32"] = compare_policy(project, dtype=torch.float32)
    report["memory"] = memory_probe(project)
    report["timing"] = timing(project)
    check = report["precision"]["torch.float64"]
    report["parity_passed"] = (
        check["max_score_abs_error"] < 1e-9
        and check["neuron_spike_differences"] == 0
        and check["threshold_decision_differences"] == 0
        and all(
            r["decision_differences"] == 0 and r["max_score_abs_error"] < 1e-9
            for r in report["policy"].values()
        )
    )
    write_report(output, report)
    print(f"Float64 parity passed: {report['parity_passed']}")
    if not report["parity_passed"]:
        raise SystemExit(1)


def write_report(output, report):
    """Keep reports local and reject nonstandard NaN/Infinity JSON values."""
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload)
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
