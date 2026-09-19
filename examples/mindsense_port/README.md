# MindSense inference and memory lab

A fork-local experiment that runs the existing MindSense v3 liquid state
machine through `snntorch.Synaptic`. The adapter preserves its encoder,
refractory timing, state smoothing and frozen readout.

## Run the standalone lesson

From the fork root, with Python and PyTorch installed:

```sh
python -m pip install -e . pytest
python -m examples.mindsense_port.demo
python -m pytest tests/test_mindsense_port.py -q
```

The demo uses eight neurons and hand-set, untrained parameters. A pulse leaves
current, voltage and spike traces behind. Subsequent zero inputs produce
different states until activity decays. `reset()` clears that history.

## Check the actual MindSense model locally

```sh
python -m examples.mindsense_port.run_lab \
  --mindsense-root /path/to/your/local/mindsense
```

This optional check requires a trusted MindSense v3 checkout containing
`experiment.py`, `robust.py`, `streaming.py`, `neurosensoros/` and
`artifacts/robust_model.npz`. It imports and executes that checkout's code.
Install that project's dependencies in the same environment first.

The runner compares float64 and float32 outputs with the NumPy model on the
existing synthetic holdouts. It checks spikes, score differences, threshold
decisions and quality-policy responses. Float64 parity is the exit-status gate:
score error below `1e-9`, identical spikes and identical policy decisions.
Float32 differences are reported separately. A memory probe and a single-run
CPU step-timing measurement are also included.

Reports default to `experiments/snntorch_port/fork_results.json` inside the
private checkout. They include a model hash and local environment paths.
Keep them private. No trained weights, private data or recorded results are
included here. The runner makes no network requests of its own.

## How the adapter works

Eight normalized features become 32 signed delta and level event channels.
Input events and the previous tick's spikes drive the recurrent reservoir:

```text
current[t] = alpha * current[t-1] + input_drive + recurrent_drive
voltage[t] = beta * voltage[t-1] + current[t]
```

`Synaptic` performs these integrations. The adapter applies MindSense's
`>= threshold` firing rule, immediate voltage reset and one-tick refractory
gate. Current continues integrating during the refractory tick. Fast and slow
spike traces plus clipped voltage form the readout features; a rolling mean
smooths them before the frozen linear readout and sigmoid calibration.

An LSM is a recurrent SNN. Both can retain short-term state. Here memory comes
from decaying currents, voltages, recurrent spikes, traces and smoothing.
Use one instance per session. Preserve state across chunks and call `reset()`
for a new session or invalid input stream. The private `Stream` wrapper owns
normalization, input-quality checks and alert policy.

## Limits

This is an inference-only learning and parity experiment. It consumes engineered
features. Raw EEG processing, real-person validation and training are outside
its scope. Synthetic accuracy gives no clinical guarantee. CPU step timing
does not establish power savings or an end-to-end throughput target.
The core snnTorch library is unchanged.
