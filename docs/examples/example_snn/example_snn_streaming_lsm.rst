Streaming liquid state machine
==============================

A liquid state machine uses a recurrent spiking reservoir to retain recent
input history. A trained readout maps the reservoir state to a prediction.
This example combines ``RSynaptic(learn_recurrent=False)`` with a frozen
input projection and a linear readout over filtered spikes.

Run the example from a checkout with PyTorch and snnTorch installed::

    python examples/streaming_lsm.py

It generates two synthetic firing-rate classes, trains the readout, and checks
that processing eight timesteps at a time matches a single full-sequence call.
The data serves as a reproducible demonstration of the training workflow.

Continuing a stream
-------------------

Inputs have shape ``[time, batch, input_size]``. Pass the returned state into
the next call to retain history::

    from examples.streaming_lsm import StreamingLSM

    model = StreamingLSM(input_size=4, hidden_size=64)
    logits_a, state = model(first_chunk)
    logits_b, state = model(next_chunk, state)

State contains spikes, synaptic current, membrane potential and a filtered
spike trace. Every tensor has shape ``[batch, hidden_size]``. Each batch lane
must continue the same stream in the same order. Use a separate state tuple
for each session; passing ``None`` starts all lanes with zero state.

State must share the input dtype and device. Move the model, input and state
together when changing devices. ``state_dict()`` stores model weights and
neuron parameters; callers manage session state separately.

Training and computation
------------------------

The reservoir runs under ``torch.no_grad()``. Input gradients and reservoir
gradients are disabled, while the readout remains trainable. Returned state
is detached, bounding the graph across streaming calls. For end-to-end
surrogate-gradient training, see the recurrent neuron tutorials.

The example uses 10% recurrent connection probability, a zero diagonal and
source columns with excitatory or inhibitory signs. These weights are stored
and executed with dense PyTorch operators. Hardware efficiency requires
separate measurement. Decay parameters are expressed per timestep, so the
input cadence determines their duration in physical time.

Source
------

.. literalinclude:: ../../../examples/streaming_lsm.py
    :language: python
