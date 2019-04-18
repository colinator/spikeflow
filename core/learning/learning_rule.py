import tensorflow as tf

class LearningRule:

    """ The base class for all online-style learning rules, including STDP.

    Premature abstraction, you say? Hmph. Right now there's just a single child:
    STDPLearningRule.

    Learning rules are applied to a connection layer, and should accumulate some
    value over time, given the connection layers' input and output, in order to
    apply this value to adjusting something (like weights) after a batch of input.

    Learning rules need to compile two computation sub-graphs: the 'accumulator'
    sub-graph, and the 'learning' sub-graph. The 'accumulator' sub-graph will be
    executed within the full time-stepping graph computation; the method 'accumulation_ops'
    needs to return the accumulator graph nodes.

    Any time during the main time step callback (at the end of a batch?), clients
    may call 'model.learn(session)', which will execute the learning sub-graph of
    all LearningRules, whose nodes must be returned by 'learning_ops'.

    Can be set to use a 'teaching signal': in this case, the 'output' fed to the
    learning rule will come from the teaching signal, rather than the output of
    the output neurons of the connection layer. This feature can be toggled.
    """

    def __init__(self, name, connection_layer, uses_teaching_signal=False):
        """ Constructs a LearningRule

        Args:
            name: String, name of the learning rule. Must be globally unique.
            connection_layer: the ConnectionLayer to apply to
            uses_teaching_signal: whether to use the teaching signal. MUST be
                set to True here if you ever want to use this feature. Can be
                subsequently toggled off and on.
        """
        self.name = name
        self.connection_layer = connection_layer
        self._uses_teaching_signal = uses_teaching_signal

    @property
    def teaching_signal_key(self):
        """ Returns the tensorflow node name of the teaching signal """
        return self.name + '_teaching_signal'

    @property
    def uses_teaching_signal(self):
        """ Returns whether the teaching signal is used or not. """
        return self._uses_teaching_signal

    def set_uses_teaching_signal(self, uses, session):
        """ Sets whether to use the teaching signal. Only works if
        _uses_teaching_signal was set to True during compilation!

        Args:
            uses: boolean, whether to use the teaching signal
            session: session to execute in
        """
        self._uses_teaching_signal = uses
        if self.uses_teaching_signal_var is not None:
            session.run(self.uses_teaching_signal_var.assign(uses))

    def compile(self, W, input, output):
        """ Compiles all computation graph nodes. Creates a teaching-signal bypass
        if necessary: output can be set to the teaching signal variable.

        Then calls _compile: child classes must implement to actually compile!

        Args:
            W: weight tensor variable to apply learning to
            input: input tensor (input Neurons output)
            output: output tensor (output Neurons output)
        """
        if self._uses_teaching_signal:
            self.uses_teaching_signal_var = tf.Variable(self._uses_teaching_signal, dtype=tf.bool)
            self.teaching_signal = tf.placeholder(tf.float32, shape=(W.shape[1],), name=self.teaching_signal_key)
            final_output = tf.cond(self.uses_teaching_signal_var, lambda: self.teaching_signal, lambda: output)
        else:
            final_output = output

        self._compile(W, input, final_output)


    # called by the model:

    def _compile_into_model(self):
        """ Called by the model; compiles this learning rule fully. """
        self.compile(self.connection_layer.weights, self.connection_layer.input, self.connection_layer.output)

    def _ops(self):
        """ Convenience function for the model; the _ops to run with every timestep
        are the 'accumulation' nodes. """
        return self.accumulation_ops()

    def ops_format(self):
        """ Get the names of the accumulation ops """
        return self.accumulation_ops_format()


    # subclasses must override:

    def _compile(self, W, input, output):
        """ Subclasses must implement to compile all accumulation step tensorflow
        nodes, as well as all learning step tensorflow nodes. """
        raise NotImplementedError

    def accumulation_ops(self):
        """ Gets a list of tensorflow nodes for the accumulation step. """
        raise NotImplementedError

    def accumulation_ops_format(self):
        """ Gets a list of tensorflow node names for the accumulation step. """
        raise NotImplementedError

    def learning_ops(self):
        """ Gets a list of tensorflow nodes for the learning step. """
        raise NotImplementedError

    def learning_ops_format(self):
        """ Gets a list of tensorflow node names for the learning step. """
        raise NotImplementedError
