class ComputationLayer:

    """ Base class for computation graph layers.
    Inherited by NeuronLayer and ConnectionLayer.

    Defines:
    2 nodes the subclasses must create, and
    2 abstract methods the subclasses must implement

    Variables:
        input: tf.Tensor operation, such as tf.Variable
        output: tf.Tensor operation, such as tf.Variable
    """

    def __init__(self):
        self.input = None
        self.output = None

    def _ops(self):
        """ Subclasses need to return [tf.Tensor operation]
        An array of tensor operations that session.run should evaluate.
        """
        pass

    def _compile(self):
        """ Compile the tensorflow graph operations. Herein may construct
        output and input.
        """
        pass
