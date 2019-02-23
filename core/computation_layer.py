class ComputationLayer:

    """ Base class for computation graph layers.
    Inherited by NeuronLayer and ConnectionLayer.

    Defines:
    2 nodes the subclasses must create, and
    2 abstract methods the subclasses must implement

    Variables:
        name: name of the layer. Must be unique within sibling layers.
        input: tf.Tensor operation, such as tf.Variable
        output: tf.Tensor operation, such as tf.Variable
    """

    def __init__(self, name):
        self.name = name
        self.input = None
        self.output = None

    def _ops(self):
        """ Subclasses need to return [tf.Tensor operation]
        An array of tensor operations that session.run should evaluate.
        Or: a dictionary of names to ops values - basically anything that
        can be fed to tensorflow's session.run method's fetches parameter.
        """
        pass

    def _compile(self):
        """ Compile the tensorflow graph operations. Herein may construct
        output and input.
        """
        pass

    def ops_format(self):
        """ Gets a description of what ops returns. """
        pass
