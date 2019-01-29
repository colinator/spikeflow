import tensorflow as tf
import numpy as np
from collections import namedtuple
from spikeflow.core.utils import floats_uniform, floats_normal
from spikeflow.core.computation_layer import ComputationLayer


class ConnectionLayer(ComputationLayer):

    def compile_output_node(self):
        pass



Synapse = namedtuple('Synapse', ['from_neuron_index', 'to_neuron_index', 'v'])


class AbstractSynapseLayer(ConnectionLayer):

    def __init__(self, from_layer, to_layer, synapses):
        super().__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.synapses = synapses


class SimpleSynapseLayer(AbstractSynapseLayer):

    def __init__(self, from_layer, to_layer, synapses):
        super().__init__(from_layer, to_layer, synapses)
        self.w = np.zeros((self.from_layer.n, self.to_layer.n), dtype=np.float32)
        for s in self.synapses:
            self.w[s.from_neuron_index, s.to_neuron_index] = s.v
        self.output_op = None

    @classmethod
    def layer_with_random_connectivity(cls, from_layer, to_layer, fraction_connected, v_sampler):
        n = int(fraction_connected *  from_layer.n)
        synapses = []
        for to_i in range(to_layer.n):
            v_arr = v_sampler(n)
            from_i_arr = np.random.choice(from_layer.n, n, replace=False)
            for from_i, v in zip(from_i_arr, v_arr):
                synapses.append(Synapse(from_neuron_index=from_i, to_neuron_index=to_i, v=v))
        return cls(from_layer, to_layer, synapses)

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

    def __init__(self, decay, from_layer, to_layer, synapses):
        super().__init__(from_layer, to_layer, synapses)
        self.decay = decay

    @classmethod
    def layer_with_random_connectivity(cls, decay, from_layer, to_layer, fraction_connected, v_sampler):
        n = int(fraction_connected *  from_layer.n)
        synapses = []
        for to_i in range(to_layer.n):
            v_arr = v_sampler(n)
            from_i_arr = np.random.choice(from_layer.n, n, replace=False)
            for from_i, v in zip(from_i_arr, v_arr):
                synapses.append(Synapse(from_neuron_index=from_i, to_neuron_index=to_i, v=v))
        return cls(decay, from_layer, to_layer, synapses)

    def _compile(self):
        self.weights = tf.Variable(self.w)
        input_f = tf.to_float(self.input)
        o = tf.matmul(tf.expand_dims(input_f, 0), self.weights)
        o_decayed = tf.reshape(o, [-1]) + self.output * self.decay
        self.output_op = self.output.assign(o_decayed)
