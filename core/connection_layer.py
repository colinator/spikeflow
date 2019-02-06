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
    Return:
        weights: np.array((from_layer.n, to_layer.n), float32)
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
    Return:
        weights: np.array((from_layer.n, to_layer.n), float32)
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
    """ Base class for all connection layers.
    """

    def __init__(self, from_layer, to_layer):
        """ Constructs a connection layer.
        Args:
            from_layer: NeuronLayer
            to_layer: NeuronLayer
        """
        super().__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer


    def _compile_output_node(self):
        """ Child classes must implement this. Will be called in the context
        of a graph during network construction phase. This method will be called
        before compile(), and must create an output tensor, which will be
        connected to to_layer's input.
        """
        pass


class SynapseLayer(ConnectionLayer):
    """ A connection layer of simple synapses (just a weight matrix that multiplies
    the input).
    """

    def __init__(self, from_layer, to_layer, weights):
        """ Constructs a (simple) SynapseLayer.
        Args:
            from_layer: from neuron layer
            to_layer: to neuron layer
            weights: np.array((from_layer.n, to_layer.n), float32)
        """
        super().__init__(from_layer, to_layer)
        self.w = weights
        self.output_op = None

    def _compile_output_node(self):
        self.output = tf.Variable(np.zeros((self.w.shape[1],), dtype=np.float32), name='Synapse_Layer_Output')

    def _ops(self):
        return [self.input, self.output, self.output_op]

    def _compile(self):
        self.weights = tf.Variable(self.w)
        input_f = tf.to_float(self.input)
        o = tf.matmul(tf.expand_dims(input_f, 0), self.weights)
        o_reshaped = tf.reshape(o, [-1])
        self.output_op = self.output.assign(o_reshaped)



class ComplexSynapseLayer(SynapseLayer):
    """ A connection layer of synapses with additional capabilities:
    
    decay: synaptic sum values will decay in time: single layer-wide float
    failure_prob: some synapses might fail to transmit: single layer-wide float
    post_synaptic_reset_factor: when post synaptic neuron fires, scale synaptic
        sum values immediately: single layer-wide float
    """

    def __init__(self, from_layer, to_layer, weights, decay=None, failure_prob=None, post_synaptic_reset_factor=None):
        """ Constructs a ComplexSynapseLayer.
        Args:
            from_layer: from neuron layer
            to_layer: to neuron layer
            weights: np.array((from_layer.n, to_layer.n), float32)
            decay: float [0,1], decay to apply to post-synaptic summation over time
            failure_prob: float [0,1], probability of synaptic failure
            post_synaptic_reset_factor: float (likely [0,1] but not necessarily),
                amount to scale post synaptic summation in the event post-synaptic
                neuron fires
        """

        super().__init__(from_layer, to_layer, weights)
        self.decay = decay
        self.failure_prob = failure_prob
        self.post_synaptic_reset_factor = post_synaptic_reset_factor


    def _compile(self):

        # define variables
        self.weights = tf.Variable(self.w)
        input_f = tf.to_float(self.input)

        if self.post_synaptic_reset_factor is not None:
            pstf_inv = tf.to_float(1.0 - self.post_synaptic_reset_factor)
        else:
            pstf_inv = None

        # make some synapses fail (just bring their weights to 0)
        if self.failure_prob is not None:
            random_noise = tf.random_uniform(self.weights.shape, 0, 1.0)
            succeeding_weights = tf.to_float(tf.greater(random_noise, self.failure_prob))
            result_weights = self.weights * succeeding_weights
        else:
            result_weights = self.weights

        # compute post-synaptic sums
        o = tf.matmul(tf.expand_dims(input_f, 0), result_weights)
        o_resh = tf.reshape(o, [-1])

        # apply decay
        if self.decay is not None:
            o_decayed = o_resh + self.output * self.decay
        else:
            o_decayed = o_resh

        # if the post-synaptic layer fired, scale my output by post-synaptic reset factor
        if pstf_inv is not None:
            o_reset = o_decayed - tf.to_float(self.to_layer.output) * pstf_inv * o_decayed
        else:
            o_reset = o_decayed

        # top of graph
        self.output_op = self.output.assign(o_reset)
