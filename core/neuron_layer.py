import tensorflow as tf
import numpy as np
import pandas as pd
from collections import namedtuple
from spikeflow.core.computation_layer import ComputationLayer
from spikeflow.core.utils import *


class NeuronLayer(ComputationLayer):

    """ A computation layer that contains neurons.
    Really, just adds a single operation: add_input.
    """

    def add_input(self, new_input):
        """ Adds the new_input to the summation operation tree.
        Args:
            new_input: tensor with same shape as self.input
        """
        self.input = new_input if self.input is None else tf.add(self.input, new_input)


class IdentityNeuronLayer(NeuronLayer):
    """ Implements a layer of pass-through neurons. Output = Input """

    def _ops(self):
        return [self.input, self.assign_op]

    def _compile(self):
        self.output = tf.Variable(np.zeros(self.input.shape, dtype=np.float32), dtype=np.float32, name='output')
        self.assign_op = self.output.assign(self.input)


class LIFNeuronLayer(NeuronLayer):

    """ Implements a layer of Leaky Integrate-and-fire Neurons.
    Contains multiple convenience creation functions.

    Configuration values can be different for each neuron:
        r: resistance
        c: capacitance
        t: threshhold
        refrac: refactory periods in number of timesteps
    """

    C = namedtuple('LIFNeuronConfig', ['r', 'c', 't', 'refrac'])

    def __init__(self, neuron_configuration, dt=0.5):
        super().__init__()
        self.n = neuron_configuration.shape[0]
        self.r = neuron_configuration[:,0] # resistance
        self.c = neuron_configuration[:,1] # capacitance
        self.t = neuron_configuration[:,2] # threshhold
        self.refrac = neuron_configuration[:,3].astype(np.int32) # refractory periods
        self.dt = np.float32(dt)
        self.tau = self.r * self.c

    @classmethod
    def layer_from_tuples(cls, neuron_configuration_tuples, dt=0.5):
        """ Creates a layer from the given configuration tuples.
        Args:
            neuron_configuration_tuples: [LIFNeuronLayer.C]
            dt: state update timestep, float
        """
        r = np.array([nn.r for nn in neuron_configuration_tuples], dtype=np.float32)
        c = np.array([nn.c for nn in neuron_configuration_tuples], dtype=np.float32)
        t = np.array([nn.t for nn in neuron_configuration_tuples], dtype=np.float32)
        refrac = np.array([nn.refrac for nn in neuron_configuration_tuples], dtype=np.float32)
        configuration = np.array([r, c, t, refrac]).T
        return cls(configuration, dt)

    @classmethod
    def layer_with_n_identical_neurons(cls, n, r, c, t, refrac, dt):
        """ Creates a layer of n identical neurons.
        Args:
            r, c, t, refrac: floats
            dt: state update timestep, float
        """
        configuration = np.tile(np.array([r, c, t, refrac]), (n,1))
        return cls(configuration, dt)

    @classmethod
    def layer_with_n_distributions(cls, n, r_dist, c_dist, t_dist, refrac_dist, dt):
        """ Creates a layer of n neurons, whose configuration values are pulled from
        distribution generation functions.
        Args:
            r_dist, c_dist, t_dist, refrac_dist: functions that generate n float values according to any distribution.
            dt: state update timestep, float
        """
        configuration = np.array([r_dist(n), c_dist(n), t_dist(n), refrac_dist(n)]).T
        return cls(configuration, dt)

    def to_dataframe(self):
        """ Returns a pandas Dataframe containing the neuron configurations.
        """
        return pd.DataFrame(np.array([self.r, self.c, self.t, self.refrac]).T,
                            index = range(len(self.r)),
                            columns = ['r', 'c', 't', 'refrac'])


    def _ops(self):
        return [self.input, self.fired_op, self.recovery_op, self.v_op]

    def _compile(self):
        nintzeros = np.zeros((self.n,), dtype=np.int32)
        nfloatzeros = np.zeros((self.n,), dtype=np.float32)

        # refractory recovery times
        self.recovery_times = tf.Variable(nintzeros, dtype=tf.int32, name='recoverytimes')

        # Create tensor variables for the neuron states
        v = tf.Variable(nfloatzeros, name='v')
        self.output = tf.Variable(np.zeros(v.shape, dtype=bool), dtype=bool, name='fired')

        # Reset any neurons that spiked last timestep
        # NOTE: Does the WARNING: tf.Variable from the tensorflow Variable
        # documentation apply here? Is tf.where safe here, and below?
        v_1 = tf.where(self.output, nfloatzeros, v)

        # State update equations: update the 'voltage', but
        # only do it if the neuron is not in its recovery period.
        v_2 = v_1 + (self.input * self.r - v_1) / self.tau * self.dt
        recovering = tf.greater(self.recovery_times, nintzeros)
        v_3 = tf.where(recovering, nfloatzeros, v_2)

        # Spikes:
        # Limit anything above threshold to threshold value
        # We are saving which fired to use again in the next iteration
        threshhold = tf.constant(self.t, tf.float32, v.shape)
        self.fired_op = self.output.assign(tf.greater_equal(v_3, threshhold))
        v_f = tf.where(self.fired_op, threshhold, v_3)

        # Set the new recovery time - if the neuron fired, set to the refactory
        # period - otherwise decrement by one (floor of 0).
        recovery_decremented = tf.maximum(nintzeros, self.recovery_times - 1)
        recovery_new = tf.where(self.output, self.refrac, recovery_decremented)
        self.recovery_op = self.recovery_times.assign(recovery_new)

        # Operations to update the state
        self.v_op = v.assign(v_f)


class IzhikevichNeuronLayer(NeuronLayer):

    """ Implements a layer of Izhikevich Neurons.
    Contains multiple convenience creation functions.
    """

    C = namedtuple('IzhikevichNeuronConfig', ['a', 'b', 'c', 'd', 't', 'v0'])

    def __init__(self, neuron_configuration, dt=0.5):
        """Creates an IzhikevichNeuronLayer with the given neuron configurations.

        Args:
            neuron_configuration: np.array of size n by 6. Each row represents
            one neuron. Columns 0 through 5 represent values of a, b, c, d, t, and v0.
            dt: python float value for dt.
        """
        super().__init__()

        self.n = neuron_configuration.shape[0]
        self.a = neuron_configuration[:,0]
        self.b = neuron_configuration[:,1]
        self.c = neuron_configuration[:,2]
        self.d = neuron_configuration[:,3]
        self.t = neuron_configuration[:,4]
        self.v0 = neuron_configuration[:,5]
        self.dt = np.float32(dt)

        self.v_op = None
        self.u_op = None
        self.fired_op = None

    @classmethod
    def layer_from_tuples(cls, neuron_configuration_tuples, dt=0.5):
        """ Creates a layer from the given configuration tuples.
        Args:
            neuron_configuration_tuples: [IzhikevichNeuronLayer.C]
            dt: state update timestep, float
        """
        a = np.array([nn.a for nn in neuron_configuration_tuples], dtype=np.float32)
        b = np.array([nn.b for nn in neuron_configuration_tuples], dtype=np.float32)
        c = np.array([nn.c for nn in neuron_configuration_tuples], dtype=np.float32)
        d = np.array([nn.d for nn in neuron_configuration_tuples], dtype=np.float32)
        t = np.array([nn.t for nn in neuron_configuration_tuples], dtype=np.float32)
        v0 = np.array([nn.v0 for nn in neuron_configuration_tuples], dtype=np.float32)
        configuration = np.array([a, b, c, d, t, v0]).T
        return cls(configuration, dt)

    @classmethod
    def layer_with_n_identical_neurons(cls, n, a, b, c, d, t, v0, dt):
        """ Creates a layer of n identical neurons.
        Args:
            a, b, c, d, t, v0: floats
            dt: state update timestep, float
        """
        configuration = np.tile(np.array([a, b, c, d, t, v0]), (n,1))
        return cls(configuration, dt)

    @classmethod
    def layer_with_n_distributions(cls, n, a_dist, b_dist, c_dist, d_dist, t_dist, v0_dist, dt):
        """ Creates a layer of n neurons, whose configuration values are pulled from
        distribution generation functions.
        Args:
            a_dist, b_dist, c_dist, d_dist, t_dist, v0_dist: functions that
            generate n float values according to any distribution.
            dt: state update timestep, float
        """
        configuration = np.array([a_dist(n), b_dist(n), c_dist(n), d_dist(n), t_dist(n), v0_dist(n)]).T
        return cls(configuration, dt)


    def to_dataframe(self):
        """ Returns a pandas Dataframe containing the neuron configurations.
        """
        return pd.DataFrame(np.array([self.a, self.b, self.c, self.d, self.t, self.v0]).T,
                            index = range(len(self.a)),
                            columns = ['a', 'b', 'c', 'd', 't', 'v0'])


    def _ops(self):
        return [self.input, self.v_op, self.u_op, self.fired_op]

    def _compile(self):

        n = self.n
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        t = self.t

        # Create tensor variables for the neuron states
        v = tf.Variable(np.ones((n,), dtype=np.float32) * c, name='v')
        u = tf.Variable(np.zeros((n,), dtype=np.float32), name='u')
        self.output = tf.Variable(np.zeros(v.shape, dtype=bool), dtype=bool, name='fired')

        # Reset any neurons that spiked last timestep
        # NOTE: Does the WARNING: tf.Variable from the tensorflow Variable
        # documentation apply here? Is tf.where safe here, and below?
        v_1 = tf.where(self.output, c, v)
        u_1 = tf.where(self.output, tf.add(u, d), u)

        # State update equations
        v_2 = v_1 + (0.04 * v_1 * v_1 + 5.0 * v_1 + 140.0 - u_1 + self.input) * self.dt
        u_f = u_1 + a * (b * v_1 - u_1) * self.dt

        # Spikes:
        # Limit anything above threshold to threshold value
        # We are saving which fired to use again in the next iteration
        threshhold = tf.constant(t, tf.float32, v.shape)
        self.fired_op = self.output.assign(tf.greater_equal(v_2, threshhold))
        v_f = tf.where(self.fired_op, threshhold, v_2)

        # Operations to update the state
        self.v_op = v.assign(v_f)
        self.u_op = u.assign(u_f)
