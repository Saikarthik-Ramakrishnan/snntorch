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

The loader checks archive fields, shapes, finite weights, positive thresholds
and scales, decay ranges and representability in the selected precision.
Invalid reference or port scores fail the comparison. Policy replay also
reports float32 behavior. Parameters and training remain unchanged.

## Measure processing deadlines

```sh
python -m examples.mindsense_port.run_lab \
  --mindsense-root /path/to/your/local/mindsense --lesson processing \
  --frames 500 --repeats 3 --offered-hz 100 --deadline-ms 10
python -m pytest tests/test_mindsense_port.py tests/test_mindsense_processing.py -q
```

This takes about 45 seconds for three CPU implementations: NumPy, float64 and
float32. Each run starts a fresh session with the same inputs. Model order
rotates across repeats. Normalization, quality checks, inference and policy
are timed. Loading, baseline fitting and 50 warmup frames are excluded.

- On-time handling: all responses within the deadline / all offered frames.
- On-time scored coverage: usable scores within the deadline / all offered frames.
- Tail latency: p95 and p99 time from scheduled arrival to result.

The example target is 95% for both ratios. A run of fast abstentions fails
scored coverage. Reports retain every latency, status, late-frame count and
per-repeat result. Exit status is nonzero if any repeat misses either target.
These are engineering test settings; the founder's final workload still needs
confirmation.

The preloaded FIFO drains every frame, including backlog. Queue overflow is
outside this test. Sample timestamps remain one logical second apart while
replay runs at the requested wall-clock rate. This does not change the model's
time constants or establish a native 100 Hz sensing pipeline. Raw sensor
acquisition, feature extraction, transport and energy remain unmeasured.

Reports default to `experiments/snntorch_port/fork_results.json` inside the
private checkout. Processing uses `fork_processing_results.json` in the same
directory. Use `--output /path/to/new-report.json` to preserve previous runs.
Reports include model/code hashes, hardware metadata and local environment paths.
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
