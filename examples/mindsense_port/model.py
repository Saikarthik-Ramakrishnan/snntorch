"""Frozen MindSense v3 inference using the local snnTorch fork.

Synaptic performs current/voltage integration. MindSense threshold, reset,
refractory and trace rules are explicit so existing weights remain meaningful.
"""

import json

import numpy as np
import snntorch as snn
import torch
from torch import nn


def validate_archive(archive, dtype):
    """Validate the inference schema before creating any neuron state."""
    try:
        config = json.loads(str(archive["config"]))
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")
        n = config["neurons"]
        if type(n) is not int or n < 1:
            raise ValueError("neurons must be a positive integer")
        smoothing = np.asarray(archive["smoothing"])
        if (
            smoothing.shape != ()
            or smoothing.dtype.kind not in "iu"
            or smoothing.item() < 1
        ):
            raise ValueError("smoothing must be a positive integer scalar")
        delta = config["delta_threshold"]
        if type(delta) not in (int, float):
            raise ValueError("delta_threshold must be numeric")
        delta_tensor = torch.tensor(delta, dtype=dtype)
        if not torch.isfinite(delta_tensor) or delta_tensor <= 0:
            raise ValueError("delta_threshold must be positive and finite")
        shapes = {
            "recurrent": (n, n),
            "inputs": (n, 32),
            "alpha": (n,),
            "beta": (n,),
            "thresholds": (n,),
            "mean": (3 * n,),
            "scale": (3 * n,),
            "weights": (3 * n + 1,),
            "calibration": (2,),
        }
        tensors = {}
        for name, shape in shapes.items():
            array = np.asarray(archive[name])
            if array.shape != shape or array.dtype.kind not in "fiu":
                raise ValueError(
                    f"{name} must be a numeric array of shape {shape}"
                )
            tensor = torch.tensor(array, dtype=dtype)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite in {dtype}")
            tensors[name] = tensor
        for name in ("alpha", "beta"):
            if ((tensors[name] < 0) | (tensors[name] > 1)).any():
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("scale", "thresholds"):
            if (tensors[name] <= 0).any():
                raise ValueError(f"{name} must be strictly positive")
    except KeyError as error:
        raise ValueError(f"Missing archive field: {error.args[0]}") from error
    except (TypeError, OverflowError) as error:
        raise ValueError(f"Invalid archive value: {error}") from error
    return config, int(smoothing.item()), tensors


class TorchMindSense(nn.Module):
    """One mutable model per session; normalized eight-feature inputs.

    This adapter supports inference only. Float64 is the parity reference.
    Session buffers move with .to() and are excluded from state_dict().
    """

    def __init__(self, archive, dtype=torch.float64):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use float32 or float64")
        self.config, self.smoothing, tensors = validate_archive(archive, dtype)
        self.n = self.config["neurons"]
        self.delta_threshold = self.config["delta_threshold"]
        for name in (
            "recurrent",
            "inputs",
            "mean",
            "scale",
            "weights",
            "calibration",
        ):
            self.register_buffer(name, tensors[name])
        self.neuron = snn.Synaptic(
            alpha=tensors["alpha"],
            beta=tensors["beta"],
            threshold=tensors["thresholds"],
            reset_mechanism="none",
            surrogate_disable=True,
        )
        for name, size in (
            ("reference", 8),
            ("level_accumulator", 16),
            ("events", 32),
            ("current", self.n),
            ("voltage", self.n),
            ("spikes", self.n),
            ("fast", self.n),
            ("slow", self.n),
        ):
            self.register_buffer(
                name, torch.zeros(size, dtype=dtype), persistent=False
            )
        self.register_buffer(
            "refractory",
            torch.zeros(self.n, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "history",
            torch.zeros(self.smoothing, 3 * self.n, dtype=dtype),
            persistent=False,
        )
        self.history_count = 0
        self.eval()

    @classmethod
    def load(cls, path, dtype=torch.float64):
        with np.load(path, allow_pickle=False) as archive:
            return cls(archive, dtype=dtype)

    def reset(self):
        for name in (
            "reference",
            "level_accumulator",
            "events",
            "current",
            "voltage",
            "spikes",
            "fast",
            "slow",
            "refractory",
            "history",
        ):
            getattr(self, name).zero_()
        self.history_count = 0
        self.neuron.reset_mem()

    @torch.no_grad()
    def state(self, values):
        z = torch.as_tensor(
            values, dtype=self.inputs.dtype, device=self.inputs.device
        )
        if z.shape != (8,) or not torch.isfinite(z).all():
            raise ValueError("Expected eight finite normalized features")
        z = z.clamp(-8, 8)
        residual = z - self.reference
        signed = residual.sign() * (
            residual.abs() / self.delta_threshold
        ).floor().clamp(max=8)
        self.reference += signed * self.delta_threshold
        self.level_accumulator += (
            torch.cat((z.clamp(min=0), (-z).clamp(min=0))) / 2
        )
        level = self.level_accumulator.floor().clamp(max=8)
        self.level_accumulator -= level
        self.events = torch.cat(
            (signed.clamp(min=0), (-signed).clamp(min=0), level)
        )

        incoming = self.inputs @ self.events + self.recurrent @ self.spikes
        active = self.refractory == 0
        # Ignore Synaptic's spike output: MindSense uses >= at threshold,
        # immediate reset and an explicit refractory gate.
        _, self.current, integrated = self.neuron(
            incoming, self.current, self.voltage
        )
        voltage = torch.where(active, integrated, torch.zeros_like(integrated))
        fired = active & (voltage >= self.neuron.threshold)
        self.spikes = fired.to(self.inputs.dtype)
        self.voltage = torch.where(
            fired, torch.zeros_like(voltage), voltage
        ).clamp(min=-5)
        self.refractory = torch.where(
            fired, 1, (self.refractory - 1).clamp(min=0)
        )
        self.fast = 0.5 * self.fast + 0.5 * self.spikes
        self.slow = 0.9 * self.slow + 0.1 * self.spikes
        features = torch.cat((self.fast, self.slow, self.voltage.clamp(-2, 2)))
        self.history = torch.roll(self.history, shifts=-1, dims=0)
        self.history[-1] = features
        self.history_count = min(self.history_count + 1, self.smoothing)
        start = -self.history_count
        return self.history[start:].mean(dim=0)

    @torch.no_grad()
    def forward(self, values):
        features = self.state(values)
        score = ((features - self.mean) / self.scale) @ self.weights[
            :-1
        ] + self.weights[-1]
        logit = (self.calibration[0] * score + self.calibration[1]).clamp(
            -30, 30
        )
        return torch.sigmoid(logit)

    def step(self, values):
        """Adapter for the existing MindSense Stream quality/policy wrapper."""
        return float(self(values))
