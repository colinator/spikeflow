import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from spikeflow.core.learning.learning_rule import LearningRule
from spikeflow.core.learning.weight_bounds import WeightBounds, WeightBounds_Enforcer
from spikeflow.core.spike_process import *


class STDPParams:

    """ Parameters governing stdp, as described here:
    http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity """

    def __init__(self, APlus=1.0, TauPlus=10.0, AMinus=1.0, TauMinus=10.0, all_to_all=True):
        self.APlus = APlus
        self.TauPlus = TauPlus
        self.AMinus = AMinus
        self.TauMinus = TauMinus
        self.all_to_all = all_to_all

    def __str__(self):
        return 'STDPParams A:{0:1.2f} A-:{1:1.2f} T+:{2:1.2f} T-:{3:1.2f}'.format(self.APlus, self.AMinus, self.TauPlus, self.TauMinus)


class STDP_Tracer:

    """ Computes pre- and post-synaptic trace contributions, according to STDP.

    Should we inherit from ComputationLayer? This is basically a computation kernel.
    But who compiles it? Probably not the top model layer - instead, the LearningRule,
    or some higher layer.

    Implements the online trace model of STDP described here:
    http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity
    """

    def __init__(self, stdp_params):
        self.stdp_params = stdp_params

    def _compile(self, W, input, output, pre_synaptic_traces, post_synaptic_traces):

        stdp_params = self.stdp_params

        # --- decay traces, raise by A+ or A- for pre- and post-synaptic spikes ---

        # decay all traces exponentially
        pre_synaptic_traces_decayed = pre_synaptic_traces / stdp_params.TauPlus
        post_synaptic_traces_decayed = post_synaptic_traces / stdp_params.TauMinus

        # for each pre- or post-synaptic spike, change the appropriate trace by a factor
        pre_synaptics_activated = input * stdp_params.APlus
        post_synaptics_activated = output * stdp_params.AMinus * -1.0

        # update the traces
        new_pre_synaptic_traces = pre_synaptic_traces_decayed + pre_synaptics_activated
        new_post_synaptic_traces = post_synaptic_traces_decayed + post_synaptics_activated

        # if it's not 'all-to-all', then cap traces at the appropriate level
        if not stdp_params.all_to_all:
            new_pre_synaptic_traces = tf.minimum(new_pre_synaptic_traces, stdp_params.APlus)
            new_post_synaptic_traces = tf.minimum(new_post_synaptic_traces, stdp_params.AMinus)

        # assign new traces
        pre_synaptic_traces = tf.assign(pre_synaptic_traces, new_pre_synaptic_traces)
        post_synaptic_traces = tf.assign(post_synaptic_traces, new_post_synaptic_traces)

        return pre_synaptic_traces, post_synaptic_traces


    def test(self, W, input_firings, output_firings):
        """ Compiles a computation graph, runs it through input and output firings
        (which should represent pre0 and post-synaptic neuron firings), returns
        recordings of pre and post trace contributions.
        """

        graph = tf.Graph()
        with graph.as_default():

            pre_synaptic_traces = tf.Variable(np.zeros((W.shape[0],)), dtype=tf.float32)
            post_synaptic_traces = tf.Variable(np.zeros((W.shape[1],)), dtype=tf.float32)

            input = tf.placeholder(tf.float32, shape=(W.shape[0],), name='input')
            output = tf.placeholder(tf.float32, shape=(W.shape[1],), name='output')

            trace_contributions = self._compile(W, input, output, pre_synaptic_traces, post_synaptic_traces)
            pre_synaptic_trace_contributions, post_synaptic_trace_contributions = trace_contributions

        pre_trace_contributions = []
        post_trace_contributions = []

        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()

            runnables = [pre_synaptic_trace_contributions, post_synaptic_trace_contributions]

            for input_v, output_v in zip(input_firings, output_firings):

                results = sess.run(runnables, feed_dict={input: input_v, output: output_v})

                pre_trace_contributions.append(results[0])
                post_trace_contributions.append(results[1])

        return np.array(pre_trace_contributions), np.array(post_trace_contributions)


class STDPLearningRule(LearningRule):

    """ The first (and, currently, only) concrete LearningRule.
    Implements online stdp, with optional weight-bounds, as described here:

    http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity

    Like all LearningRules, compiles two computation sub-graphs; one for 'accumulation',
    which is executed by the model with every time-step, and one for 'learning',
    which is executed at any time by the model (in turn by the client).
    """

    def __init__(self, name, stdp_params, weight_bounds=None, connection_layer=None, uses_teaching_signal=False):
        """ Creates an stdp learning rule applied to some connection layer, as described here:
        http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity

        Args:
            name: String, name of the learning rule. Must be globally unique.
            stdp_params: STDPParams
            weight_bounds: WeightBounds, optional
            connection_layer: the ConnectionLayer to apply to, optional (maybe don't set for testing)
            uses_teaching_signal: whether to use the teaching signal. MUST be
                set to True here if you ever want to use this feature. Can be
                subsequently toggled off and on.
        """
        super().__init__(name, connection_layer, uses_teaching_signal)

        self.stdp_tracer = STDP_Tracer(stdp_params)
        self.weight_bounds_enforcer = WeightBounds_Enforcer(weight_bounds) if weight_bounds else None

        # Tensorflow graph nodes we'll need to store

        # variables
        self.pre_dw = None
        self.post_dw = None

        # trace contributions
        self.accumulate_pre_dw = None
        self.accumulate_post_dw = None
        self.assignments = None

        # modify W according to weight bounds, and zero out
        self.assign_W = None
        self.zero_out = None


    def __compile_accumulate(self, W, input, output):

        # -- define variables --
        self.pre_dw = tf.Variable(np.zeros(W.shape), dtype=tf.float32)
        self.post_dw = tf.Variable(np.zeros(W.shape), dtype=tf.float32)

        pre_synaptic_trace_conts = tf.Variable(np.zeros((W.shape[0],)), dtype=tf.float32)
        post_synaptic_trace_conts = tf.Variable(np.zeros((W.shape[1],)), dtype=tf.float32)

        # compute trace inputs
        input_cast = tf.cast(input, tf.float32)
        output_cast = tf.cast(output, tf.float32)
        input_reshaped = tf.reshape(input_cast, [input_cast.shape[0], 1])
        output_reshaped = tf.reshape(output_cast, [output_cast.shape[0], 1])
        pre_reshaped = tf.reshape(pre_synaptic_trace_conts, [pre_synaptic_trace_conts.shape[0], 1])
        post_reshaped = tf.reshape(post_synaptic_trace_conts, [post_synaptic_trace_conts.shape[0], 1])
        step_pre_dw = tf.transpose(output_reshaped * tf.transpose(pre_reshaped))
        step_post_dw = input_reshaped * tf.transpose(post_reshaped)

        acc_pre_dw = self.pre_dw + step_pre_dw
        acc_post_dw = self.post_dw + step_post_dw

        self.accumulate_pre_dw = self.pre_dw.assign(acc_pre_dw)
        self.accumulate_post_dw = self.post_dw.assign(acc_post_dw)

        with tf.control_dependencies([self.accumulate_pre_dw, self.accumulate_post_dw]):

            # instantiate partial graph for stdp traces
            pre_conts, post_conts = self.stdp_tracer._compile(W, input_cast, output_cast, pre_synaptic_trace_conts, post_synaptic_trace_conts)

            # assign new trace contributions
            a1 = pre_synaptic_trace_conts.assign(pre_conts)
            a2 = post_synaptic_trace_conts.assign(post_conts)
            self.assignments = tf.group(a1, a2)


    def __compile_apply(self, W):

        # calculate new W
        if self.weight_bounds_enforcer is not None:
            new_W = self.weight_bounds_enforcer._compile(W, self.pre_dw, self.post_dw)
        else:
            new_W = W + tf.add(self.pre_dw, self.post_dw)

        # assign new weights
        self.assign_W = W.assign(new_W)

        # zero out the pre and post dw accumulations
        with tf.control_dependencies([self.assign_W]):
            zero_pre_dw = self.pre_dw.assign(np.zeros(W.shape))
            zero_post_dw = self.post_dw.assign(np.zeros(W.shape))
            self.zero_out = tf.group(zero_pre_dw, zero_post_dw)


    # --- overridden from LearningRule ---

    def _compile(self, W, input, output):
        self.__compile_accumulate(W, input, output)
        self.__compile_apply(W)

    def accumulation_ops(self):
        return [self.accumulate_pre_dw, self.accumulate_post_dw, self.assignments]

    def accumulation_ops_format(self):
        return ['accumulate_pre_dw', 'accumulate_post_dw', 'assignments']

    def learning_ops(self):
        return [self.assign_W, self.zero_out]

    def learning_ops_format(self):
        return ['assign_W', 'zero_out']



    # custom testing method!

    def apply_to_W(self, session):
        r = session.run(self.learning_ops())
        return r[0]

    def test(self, Wi, input_firings, output_firings, reps=1):
        graph = tf.Graph()
        with graph.as_default():
            W = tf.Variable(Wi, dtype=tf.float32)
            input = tf.placeholder(tf.float32, shape=(W.shape[0],), name='input')
            output = tf.placeholder(tf.float32, shape=(W.shape[1],), name='output')

            self.compile(W, input, output)

        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()

            # show data to network, accumulate pre and post dw
            runnables1 = self.accumulation_ops()

            acc_pre, acc_post, last_w = (None, None, W.eval())
            for i in range(reps):

                # run batch
                for input_v, output_v in zip(input_firings, output_firings):
                    acc_pre, acc_post, assi = sess.run(runnables1, feed_dict={input: input_v, output: output_v})

                # apply learning rule to weights after every batch
                last_w = self.apply_to_W(sess)

        # return the last accumulated pre, last accumulated post, and final W
        return acc_pre, acc_post, last_w



# ---
# Methods to calculate STDP 'offline': will be significantly slower than
# the above, 'online' (or 'in-graph') version. Not yet benchmarked!


def stdp_offline_dw(delta_times, stdp_params):
    """ Calculates the changes in weights due to deltas of firing times,
    according to the stdp learning rule:

    dW := 0 for dt == 0
          APlus * exp(-1.0 * delta_times / TauPlus) for dt > 0
          AMinus * exp(1.0 * delta_times / TauMinus) for dt < 0

    Args:
        delta_times: np.array(int)
        stdp_params: STDPParams
    Returns:
        np.array(delta_times.shape, float32)
    """
    return np.where(delta_times == 0,
                    0,
                    np.where(delta_times > 0,
                             stdp_params.APlus * np.exp(-1.0 * delta_times / stdp_params.TauPlus),
                             -1.0 * stdp_params.AMinus * np.exp(delta_times / stdp_params.TauMinus)))


def stdp_offline_dw_process(input_spike_process, output_spike_process, stdp_params):
    """ Calculates the sum total change in weight from input and output spike processes as subject to the stdp learning rule, 'stdp_dw'.

    Args:
        w: weight matrix (np.array(input_n, output_n) of floats32)
        input_spike_processes: np.array(int)
        output_spike_process: np.array(int)
        stdp_params: STDPParams
    Returns:
        Change in weight (np.float), subject to the stdp learning rule, 'stdp_dw'.
    """
    delta_times = spike_process_delta_times(input_spike_process, output_spike_process)
    all_dws = stdp_offline_dw(delta_times, stdp_params)
    return np.sum(all_dws)


def stdp_offline_dw_processes(w, input_spike_processes, output_spike_processes, stdp_params):
    """ Calculates change in weights from spike processes as subject to the stdp learning rule, 'stdp_dw'. Returns a dw matrix that corresponds to the weight matrix w - where there is a non-zero weight, there will be a dw.

    NOTE: This might be slow! Does not use a tensorflow computation graph (yet).

    Args:
        w: weight matrix (np.array(input_n, output_n) of floats32)
        input_spike_processes: [ np.array(int) ]
        output_spike_processes: [ np.array(int) ]
        stdp_params: STDPParams
    Returns:
        Change in weights (np.array(w.shape), float32), subject to the stdp learning rule, 'stdp_dw'.
    """
    dW = np.zeros(w.shape, dtype=np.float32)

    w_indices_i, w_indices_o = np.nonzero(w)

    for i, o in zip(w_indices_i, w_indices_o):
        dW[i, o] = stdp_offline_dw_process(input_spike_processes[i],
                                           output_spike_processes[o],
                                           stdp_params)

    return dW


def stdp_offline_dw_firings(w, input_firings, output_firings, stdp_params):
    """ Calculates change in weights from firing matrices as subject to the stdp learning rule, 'stdp_dw'.

    Args:
        w: weight matrix (np.array(input_n, output_n) of floats32)
        input_firings: firings
        output_firings: firings
        stdp_params: STDPParams
    Returns:
        Change in weights, subject to the stdp learning rule, 'stdp_dw'.
    """
    return stdp_offline_dw_processes(w,
                                     firings_to_spike_processes(input_firings),
                                     firings_to_spike_processes(output_firings),
                                     stdp_params)
