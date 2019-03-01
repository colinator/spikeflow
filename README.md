# spikeflow

Spiking neural networks in tensorflow.

***Installation:*** pip install coming soon. I'll wait until it achieves just a little bit of stability... some feedback'd help here...



**Hypothesis:** Biological plausibility in neural networks is not an obstacle, but a benefit, for machine learning applications.

**Proof:** None. Yet.

The purpose of this library is to explore the connection between computational neuroscience and machine learning, and to practice implementing efficient and fast models in tensorflow.

Spikeflow makes it easy to create arbitrarily connected layers of spiking neurons into a tensorflow graph, and then to 'step time'.

Spikeflow will concentrate on facilitating the exploration of spiking neuron models, local-only learning rules, dynamic network construction, the role of inhibition in the control of attractor dynamics, etc.

The library will implement:
- multiple models of spiking neurons, including
  - [x] Leaky Integrate-and-fire
  - [x] Izhikevich
  - [x] Poisson (can convert things like images to spike trains)
  - [ ] Hodgkin-Huxley? Taking requests...
- arbitrary connections between layers of similar neurons, including
  - [x] feed-foward
  - [x] recurrent
  - [x] inhibitory
  - [x] layer compositing (layers can have sub-layers with a single input and output)
- multiple models of synapses, including
  - [x] simple weights
  - [x] synapses with decay
  - [x] synapses with failure
  - [x] synapses with post-synaptic reset
  - [x] synapses with delay
- out-of-graph and in-graph learning rules, including
  - [x] out-of-graph weight modification
  - [x] out-of-graph STDP, compatible with model.step_time's end timestep callback
  - [ ] in-graph STDP
  - [ ] others? taking requests!
- forms of dynamic neural network construction, including
  - [ ] Synaptic pruning and synaptogenesis
  - [ ] Neuron pruning and neurogenesis
  - [ ] Other structure modification

#### The basic modeling idea is this:

```
model = BPNNModel.compiled_model(input_shape,
  [ neuronlayer1, neuronlayer2, ...],
  [ connections1, connections2, ...])

model.run_time(data_generator, end_time_step_callback)
```

See the examples in the `jupyter_examples` directory.

Feedback and collaborators welcome!
