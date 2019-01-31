import tensorflow as tf
import numpy as np
from collections import namedtuple
from spikeflow.core.utils import floats_uniform, floats_normal
from spikeflow.core.computation_layer import ComputationLayer



Synapse = namedtuple('Synapse', ['from_neuron_index', 'to_neuron_index', 'v'])


def weights_from_synapses(from_layer, to_layer, synapses):
    """ Constructs a weight matrix from the given synapses.
    Args:
        from_layer: the pre-synaptic neuron layer
        to_neuron: the post-synaptic neuron layer
        synapses: iterable of Synapse
    """
    w = np.zeros((from_layer.n, to_layer.n), dtype=np.float32)
    for s in synapses:
        w[s.from_neuron_index, s.to_neuron_index] = s.v
    return w

def weights_connecting_from_to(from_layer, to_layer, connectivity, v_sampler, from_range=None, to_range=None):
    """ Constructs a weight matrix by connecting each neuron in from_layer to
    a random set of neurons in to_layer.
    Args:
        from_layer: the pre-synaptic neuron layer
        to_layer: the post-synaptic neuron layer
        connectivity: float [0,1], each pre-synaptic neuron should connect to
        this fraction of post-synaptic neurons
        v_sampler: weight distribution
        from_range: tuple of (from: int index, to: int index), or None. Use this
        to control connectivity from a subset of pre-synaptic neurons
        to_range: tuple of (from: int index, to: int index), or None. Use this
        to control connectivity to a subset of post-synaptic neurons
    """
    w = np.zeros((from_layer.n, to_layer.n), dtype=np.float32)
    fr = (0, from_layer.n) if from_range is None else from_range
    tr = (0, to_layer.n) if to_range is None else to_range
    n = int(connectivity * (tr[1] - tr[0]))
    for from_i in range(fr[0], fr[1]):
        v_arr = v_sampler(n)
        to_neuron_indexes = np.random.choice(tr[1]-tr[0], n, replace=False) + tr[0]
        for to_i, v in zip(to_neuron_indexes, v_arr):
            w[from_i, to_i] = v
    return w


class ConnectionLayer(ComputationLayer):

    def compile_output_node(self):
        pass


class AbstractSynapseLayer(ConnectionLayer):

    def __init__(self, from_layer, to_layer):
        super().__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer


class SimpleSynapseLayer(AbstractSynapseLayer):

    def __init__(self, from_layer, to_layer, weights):
        super().__init__(from_layer, to_layer)
        self.w = weights
        self.output_op = None

    def compile_output_node(self):
        self.output = tf.Variable(np.zeros((self.w.shape[1],), dtype=np.float32), name='SSL_Input')

    def _ops(self):
        return [self.input, self.output, self.output_op]

    def _compile(self):
        self.weights = tf.Variable(self.w)
        input_f = tf.to_float(self.input)
        o = tf.matmul(tf.expand_dims(input_f, 0), self.weights)
        o_reshaped = tf.reshape(o, [-1])
        self.output_op = self.output.assign(o_reshaped)


class DecaySynapseLayer(SimpleSynapseLayer):

    def __init__(self, from_layer, to_layer, decay, weights):
        super().__init__(from_layer, to_layer, weights)
        self.decay = decay

    def _compile(self):
        self.weights = tf.Variable(self.w)
        input_f = tf.to_float(self.input)
        o = tf.matmul(tf.expand_dims(input_f, 0), self.weights)
        o_decayed = tf.reshape(o, [-1]) + self.output * self.decay
        self.output_op = self.output.assign(o_decayed)
