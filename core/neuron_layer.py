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

    def __init__(self, name):
        super().__init__(name)

    @property
    def input_n(self):
        pass

    @property
    def output_n(self):
        pass

    def add_input(self, new_input):
        """ Adds the new_input to the summation operation tree.
        Args:
            new_input: tensor with same shape as self.input
        """
        self.input = new_input if self.input is None else tf.add(self.input, new_input)


class IdentityNeuronLayer(NeuronLayer):
    """ Implements a layer of pass-through neurons. Output = Input """

    def __init__(self, name, n):
        super().__init__(name)
        self.n = n

    @property
    def input_n(self):
        return self.n

    @property
    def output_n(self):
        return self.n

    def _ops(self):
        return [self.input, self.assign_op]

    def _compile(self):
        self.output = tf.Variable(np.zeros(self.input.shape, dtype=np.float32), dtype=np.float32, name='output')
        self.assign_op = self.output.assign(self.input)


class LIFNeuronLayer(NeuronLayer):

    """ Implements a layer of Leaky Integrate-and-fire Neurons.
    Contains multiple convenience creation functions.

    Configuration values can be different for each neuron:
        resistance: np.array(floats)
        tau: time constant, np.array(floats)
        treshhold: threshhold, np.array(floats)
        n_refrac: refactory periods in number of timesteps, np.array(ints)
    """

    C = namedtuple('LIFNeuronConfig', ['resistance', 'tau', 'threshhold', 'n_refrac'])

    def __init__(self, name, neuron_configuration, dt=0.5):
        """ LIFNeuronLayer constructor
        Args:
            neuron_configuration: 2d numpy array; columns are
            resistance, tau, capacitance, n_refrac
            dt: single timestep dt value.
        """
        super().__init__(name)

        self.n = neuron_configuration.shape[0]
        self.resistance = neuron_configuration[:,0]
        self.tau = neuron_configuration[:,1]
        self.threshhold = neuron_configuration[:,2]
        self.n_refrac = neuron_configuration[:,3].astype(np.int32)
        self.dt = np.float32(dt)

    @property
    def input_n(self):
        return self.n

    @property
    def output_n(self):
        return self.n

    @classmethod
    def layer_from_tuples(cls, name, neuron_configuration_tuples, dt=0.5):
        """ Creates a layer from the given configuration tuples.
        Args:
            neuron_configuration_tuples: [LIFNeuronLayer.C]
            dt: state update timestep, float
        """
        nct = neuron_configuration_tuples
        res = np.array([nn.resistance for nn in nct], dtype=np.float32)
        tau = np.array([nn.tau for nn in nct], dtype=np.float32)
        thresh = np.array([nn.threshhold for nn in nct], dtype=np.float32)
        refrac = np.array([nn.n_refrac for nn in nct], dtype=np.float32)
        configuration = np.array([res, tau, thresh, refrac]).T
        return cls(name, configuration, dt)

    @classmethod
    def layer_with_n_identical_neurons(cls, name, n, resistance, tau, threshhold, n_refrac, dt):
        """ Creates a layer of n identical neurons.
        Args:
            resistance, tau, threshhold, n_refrac: floats
            dt: state update timestep, float
        """
        configuration = np.tile(np.array([r, c, t, refrac]), (n,1))
        return cls(name, configuration, dt)

    @classmethod
    def layer_with_n_distributions(cls, name, n, resistance_dist, tau_dist, threshhold_dist, n_refrac_dist, dt):
        """ Creates a layer of n neurons, whose configuration values are pulled from
        distribution generation functions.
        Args:
            *_dist: functions that generate n float values according to any distribution.
            dt: state update timestep, float
        """
        configuration = np.array([resistance_dist(n), tau_dist(n), threshhold_dist(n), n_refrac_dist(n)]).T
        return cls(name, configuration, dt)

    def to_dataframe(self):
        """ Returns a pandas Dataframe containing the neuron configurations.
        """
        return pd.DataFrame(np.array([self.resistance, self.tau, self.threshhold, self.n_refrac]).T,
                            index = range(len(self.resistance)),
                            columns = ['resistance', 'tau', 'threshhold', 'n_refrac'])


    def _ops(self):
        return [self.input, self.recovery_op, self.fired_op, self.v_op]

    def _compile(self):
        nintzeros = np.zeros((self.n,), dtype=np.int32)
        nfloatzeros = np.zeros((self.n,), dtype=np.float32)

        # internal tensor variables for the neuron states
        v = tf.Variable(nfloatzeros, name='v')
        recovery_times = tf.Variable(nintzeros, dtype=tf.int32, name='recoverytimes')

        # define output variable
        self.output = tf.Variable(np.zeros(nfloatzeros.shape, dtype=bool), dtype=bool, name='fired')

        # Reset any neurons that spiked last timestep
        # NOTE: Does the WARNING: tf.Variable from the tensorflow Variable
        # documentation apply here? Is tf.where safe here, and below?
        v_1 = tf.where(self.output, nfloatzeros, v)

        # Reset recovery durations for neurons that spiked last timestep.
        recovery_decremented = tf.maximum(nintzeros, recovery_times - 1)
        recovery_new = tf.where(self.output, self.n_refrac, recovery_decremented)
        self.recovery_op = recovery_times.assign(recovery_new)

        # State update equations: update the 'voltage', but
        # only do it if the neuron is not in its recovery period.
        v_2 = v_1 + (self.input * self.resistance - v_1) / self.tau * self.dt
        recovering = tf.greater(self.recovery_op, nintzeros)
        v_3 = tf.where(recovering, nfloatzeros, v_2)

        # Compute spikes that fired, assign to output
        self.fired_op = self.output.assign(tf.greater_equal(v_3, self.threshhold))

        # Update the state: where spiked, set to threshhold
        v_f = tf.where(self.fired_op, self.threshhold, v_3)
        self.v_op = v.assign(v_f)


class IzhikevichNeuronLayer(NeuronLayer):

    """ Implements a layer of Izhikevich Neurons.
    Contains multiple convenience creation functions.
    """

    C = namedtuple('IzhikevichNeuronConfig', ['a', 'b', 'c', 'd', 't', 'v0'])

    def __init__(self, name, neuron_configuration, dt=0.5):
        """Creates an IzhikevichNeuronLayer with the given neuron configurations.

        Args:
            neuron_configuration: np.array of size n by 6. Each row represents
            one neuron. Columns 0 through 5 represent values of a, b, c, d, t, and v0.
            dt: state update timestep, float
        """
        super().__init__(name)

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

    @property
    def input_n(self):
        return self.n

    @property
    def output_n(self):
        return self.n

    @classmethod
    def layer_from_tuples(cls, name, neuron_configuration_tuples, dt=0.5):
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
        return cls(name, configuration, dt)

    @classmethod
    def configuration_with_n_identical_neurons(cls, n, a, b, c, d, t, v0):
        """ Creates a configuration for the constructor for n identical izhikevich neurons
        Args:
            a, b, c, d, t, v0: floats
        """
        return np.tile(np.array([a, b, c, d, t, v0]), (n,1))

    @classmethod
    def configuration_with_n_distributions(cls, n, a_dist, b_dist, c_dist, d_dist, t_dist, v0_dist):
        """ Creates a configuration for the constructor forn neurons, whose configuration
        values are pulled from distribution generation functions.
        Args:
            a_dist, b_dist, c_dist, d_dist, t_dist, v0_dist: functions that
            generate n float values according to any distribution.
        """
        return np.array([a_dist(n), b_dist(n), c_dist(n), d_dist(n), t_dist(n), v0_dist(n)]).T

    @classmethod
    def layer_with_configurations(cls, name, configurations, dt):
        """ Creates a layer from a list of configurations
        Args:
            configuration: list of configurations
            dt: state update timestep, float
        """
        return cls(name, np.concatenate(configurations), dt)

    @classmethod
    def layer_with_n_identical_neurons(cls, name, n, a, b, c, d, t, v0, dt):
        """ Creates a layer of n identical neurons.
        Args:
            a, b, c, d, t, v0: floats
            dt: state update timestep, float
        """
        return cls(name, cls.configuration_with_n_identical_neurons(n, a, b, c, d, t, v0), dt)

    @classmethod
    def layer_with_n_distributions(cls, name, n, a_dist, b_dist, c_dist, d_dist, t_dist, v0_dist, dt):
        """ Creates a layer of n neurons, whose configuration values are pulled from
        distribution generation functions.
        Args:
            a_dist, b_dist, c_dist, d_dist, t_dist, v0_dist: functions that
            generate n float values according to any distribution.
            dt: state update timestep, float
        """
        configuration = cls.configuration_with_n_distributions(n, a_dist, b_dist, c_dist, d_dist, t_dist, v0_dist)
        return cls(name, configuration, dt)

    def to_dataframe(self):
        """ Returns a pandas Dataframe containing the neuron configurations.
        """
        return pd.DataFrame(np.array([self.a, self.b, self.c, self.d, self.t, self.v0]).T,
                            index = range(len(self.a)),
                            columns = ['a', 'b', 'c', 'd', 't', 'v0'])


    def _ops(self):
        #return { 'input': self.input, 'v': self.v_op, 'u': self.u_op, 'output': self.fired_op }
        return [self.input, self.v_op, self.u_op, self.fired_op]

    # @classmethod
    # def ops_result_tensor(cls, ops_result):
    #     return [ ops_result['input'], ops_result['v'], ops_result['u'], ops_result['output']]

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
