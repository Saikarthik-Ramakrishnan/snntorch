"""Run a tiny, untrained memory experiment without private project files."""

import json

import numpy as np
import torch

from .model import TorchMindSense


def toy_archive():
    """Return deterministic example parameters, with no trained weights."""
    n = 8
    inputs = np.zeros((n, 32))
    inputs[:, :8] = np.eye(n) * 0.3
    inputs[:, 16:24] = np.eye(n) * 0.2
    return {
        "config": np.array(
            json.dumps({"neurons": n, "delta_threshold": 0.75})
        ),
        "smoothing": np.array(4),
        "recurrent": np.roll(np.eye(n), 1, axis=0) * 0.12,
        "inputs": inputs,
        "alpha": np.full(n, 0.7),
        "beta": np.full(n, 0.8),
        "thresholds": np.ones(n),
        "mean": np.zeros(3 * n),
        "scale": np.ones(3 * n),
        "weights": np.full(3 * n + 1, 0.03),
        "calibration": np.array([1.0, 0.0]),
    }


def state_norm(model):
    return float(
        torch.linalg.vector_norm(
            torch.cat((model.current, model.voltage, model.fast, model.slow))
        )
    )


def main():
    torch.set_num_threads(1)
    model = TorchMindSense(toy_archive())
    zero = np.zeros(8)
    model.state(zero)
    print(f"Zero input, fresh session: state norm {state_norm(model):.6f}")
    for _ in range(20):
        model.state(np.full(8, 4.0))
    for tick in range(1, 81):
        model.state(zero)
        if tick in (1, 5, 20, 80):
            print(
                f"Zero input, {tick:2d} ticks after pulse: "
                f"state norm {state_norm(model):.6f}"
            )
    model.reset()
    model.state(zero)
    print(f"Zero input, after reset: state norm {state_norm(model):.6f}")
    print("Toy parameters illustrate memory. No health prediction is made.")


if __name__ == "__main__":
    main()
