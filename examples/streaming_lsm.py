"""Train a readout over a frozen RSynaptic reservoir with explicit state.

Run from a checkout with torch and snntorch installed:
    python examples/streaming_lsm.py
"""

import argparse

import torch
from torch import nn

import snntorch as snn


class StreamingLSM(nn.Module):
    """Example reservoir with an exponentially filtered spike readout.

    Inputs have shape [time, batch, input_size]. State is a tuple containing
    spikes, synaptic current, membrane potential and spike trace, each with
    shape [batch, hidden_size]. Pass the returned state to continue a stream;
    pass None to start a new one. Reservoir computation uses no_grad, so only
    the readout receives gradients. Sparse weights use dense torch operators.
    """

    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super().__init__()
        if min(input_size, hidden_size, output_size) < 1:
            raise ValueError("Layer sizes must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input = nn.Linear(input_size, hidden_size, bias=False)
        self.neuron = snn.RSynaptic(
            alpha=0.7,
            beta=0.85,
            linear_features=hidden_size,
            learn_recurrent=False,
        )
        self.readout = nn.Linear(hidden_size, output_size)
        with torch.no_grad():
            self.input.weight.uniform_(0.4, 1.2)
            recurrent = self.neuron.recurrent.weight
            mask = torch.rand_like(recurrent) < 0.1
            mask.fill_diagonal_(False)
            signs = torch.ones(hidden_size)
            split = int(0.8 * hidden_size)
            signs[split:] = -4
            # Rows are destinations; each source column has a fixed sign.
            recurrent.copy_(
                torch.rand_like(recurrent) * mask * signs.unsqueeze(0) * 0.15
            )
            self.neuron.recurrent.bias.zero_()
        self.input.requires_grad_(False)

    def forward(self, inputs, state=None):
        """Return per-timestep logits and detached state for the next chunk."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("Expected [time, batch, input_size] inputs")
        if inputs.shape[0] == 0 or inputs.shape[1] == 0:
            raise ValueError("Time and batch dimensions must be nonempty")
        if not inputs.is_floating_point():
            raise ValueError("Inputs must be floating-point tensors")
        shape = (inputs.shape[1], self.hidden_size)
        if state is None:
            state = tuple(inputs.new_zeros(shape) for _ in range(4))
        if len(state) != 4 or any(
            s.shape != shape
            or s.device != inputs.device
            or s.dtype != inputs.dtype
            for s in state
        ):
            raise ValueError("State must match batch size, device and dtype")
        spk, syn, mem, trace = state
        traces = []
        with torch.no_grad():
            for frame in inputs:
                spk, syn, mem = self.neuron(self.input(frame), spk, syn, mem)
                trace = 0.9 * trace + 0.1 * spk
                traces.append(trace)
        logits = self.readout(torch.stack(traces))
        return logits, tuple(s.detach() for s in (spk, syn, mem, trace))


def make_data(generator, samples=256, steps=32, input_size=4):
    """Generate two Bernoulli firing-rate classes for a small CPU example."""
    labels = torch.randint(2, (samples,), generator=generator)
    rates = 0.05 + 0.25 * labels.float()
    events = torch.rand(steps, samples, input_size, generator=generator)
    return (events < rates[None, :, None]).float(), labels


def main():
    """Fit only the readout, then verify full and chunked inference agree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.manual_seed(7)
    generator = torch.Generator().manual_seed(11)
    train_x, train_y = make_data(generator)
    test_x, test_y = make_data(generator)
    model = StreamingLSM()
    optimizer = torch.optim.Adam(model.readout.parameters(), lr=0.03)
    for _ in range(args.epochs):
        optimizer.zero_grad()
        logits, _ = model(train_x)
        loss = nn.functional.cross_entropy(logits[-1], train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        full, _ = model(test_x)
        state = None
        chunks = []
        for chunk in test_x.split(8):
            logits, state = model(chunk, state)
            chunks.append(logits)
        torch.testing.assert_close(full, torch.cat(chunks))
        accuracy = (full[-1].argmax(-1) == test_y).float().mean().item()
    print(f"Synthetic held-out accuracy: {accuracy:.1%}")
    print("Full-sequence and chunked outputs agree.")


if __name__ == "__main__":
    main()
