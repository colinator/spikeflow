import tensorflow as tf
import numpy as np
from collections import namedtuple
from spikeflow.core.utils import floats_uniform, floats_normal, identical_ints_sampler
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


def samples_for_weights(weights, sampler):
    """ Creates a matrix of the same shape as weights,
    with values drawn from sampler everywhere weights is non-zero,
    and zero otherwise. Used, for instance, for creating a delay
    matrix to match a weight matrix.
    Args:
        weights: a 2d weight matrix
        sampler: a function that return n random values, or a scalar
    Return:
        a matrix of values drawn from sampler, same shape and
        topology as weights
    """
    try:
        return np.float32(operator.index(sampler))
    except:
        indexes = np.nonzero(weights)
        delays = np.zeros(weights.shape)
        delays[indexes] = sampler(len(indexes[0]))
        return delays

def delays_for_weights(weights, delay_sampler):
    """ Creates a delay matrix for a corresponding weight matrix, using a sampler """
    return samples_for_weights(weights, delay_sampler)

def decays_for_weights(weights, decay_sampler):
    """ Creates a decay matrix for a corresponding weight matrix, using a sampler """
    return samples_for_weights(weights, decay_sampler)


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
    delay: synaptic-level delay matrix. Each value is int: number of timesteps
        of delay to apply to that synapse. Must match shape and topology of weights.
        NOTE: using this feature will create a tensor of shape n x t, where n is the
        number of synapses, and t is the maximum delay, in number of timesteps.
    """

    def __init__(self, from_layer, to_layer, weights, decay=None, failure_prob=None, post_synaptic_reset_factor=None, delay=None, max_delay=None):
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
            delay: int (single layer-wide delay) or np.array(ints), which
                must match shape and topology of weights (same indexes of non-zero
                weights).
            max_delay: int, maximum possible value for delay
        """

        super().__init__(from_layer, to_layer, weights)
        self.decay = decay
        self.failure_prob = failure_prob
        self.post_synaptic_reset_factor = post_synaptic_reset_factor
        self.delay = delay if not np.isscalar(delay) else delays_for_weights(weights, identical_ints_sampler(delay))
        self.max_delay = max_delay

    def _ops(self):
        return [self.input, self.output, self.output_op]

    def _compile(self):

        # define object variable for weights
        self.weights = tf.Variable(self.w)

        input_f = tf.to_float(self.input)

        if self.post_synaptic_reset_factor is None:
            pstf_inv = None
        else:
            pstf_inv = tf.to_float(1.0 - self.post_synaptic_reset_factor)

        # make some synapses fail (just bring their weights to 0)
        if self.failure_prob is None:
            succeeding_weights = self.weights
        else:
            random_noise = tf.random_uniform(self.weights.shape, 0, 1.0)
            succeeding_weights = self.weights * tf.to_float(tf.greater(random_noise, self.failure_prob))

        # synaptic-level delay
        inputs_flat = tf.expand_dims(input_f, 0)
        if self.delay is None:
            o = tf.matmul(inputs_flat, succeeding_weights)
            o_delayed = tf.reshape(o, [-1])
        else:
            # Create a tensor variable to hold the delays (so we can change them)
            self.delays = tf.Variable(self.delay)

            n = self.w.shape[0]*self.w.shape[1]
            d_max = self.max_delay if self.max_delay is not None else int(np.max(self.delay))

            # Create a variable: n (input*output neurons) by d_max tensor
            # to hold propagating synaptic values
            propagating_values = tf.Variable(np.zeros((n, d_max+1)), dtype=tf.float32)

            # get 2d tensor of scatter indices from delays
            delays_flat = tf.cast(tf.reshape(self.delays, [-1]), tf.int32)
            indices = tf.range(0, n)
            delay_scatter_indices = tf.transpose(tf.reshape(tf.concat([indices, delays_flat], axis=0), (2, n)))

            # multiply inputs by weights element-wise, and flatten
            multed_inputs = tf.transpose(tf.multiply(tf.transpose(succeeding_weights), input_f))
            flat_inputs = tf.reshape(multed_inputs, (-1,))

            # roll the propagation tensor by 1 timestep down
            propagation_rolled = tf.roll(propagating_values, shift=-1, axis=1)
            roll_op = propagating_values.assign(propagation_rolled)

            # scatter input into the propagation tensor
            k = tf.scatter_nd_update(roll_op, delay_scatter_indices, flat_inputs)
            prop_op = propagating_values.assign(k)

            # get the 0th timestep's values
            final_out = prop_op[:,0]

            # reshape back into same shape as weight matrix
            r_out = tf.reshape(final_out, self.weights.shape)

            # sum per output neuron
            o_delayed = tf.reduce_sum(r_out, axis=0)


        # apply decay
        if self.decay is None:
            o_decayed = o_delayed
        else:
            o_decayed = o_delayed + self.output * self.decay

        # if the post-synaptic layer fired, scale my output by post-synaptic reset factor
        if pstf_inv is None:
            o_reset = o_decayed
        else:
            o_reset = o_decayed - tf.to_float(self.to_layer.output) * pstf_inv * o_decayed

        # top of graph
        self.output_op = self.output.assign(o_reset)
