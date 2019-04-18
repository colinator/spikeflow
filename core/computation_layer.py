class ComputationLayer:

    """ Base class for computation graph layers.
    Inherited by NeuronLayer and ConnectionLayer.

    Basically, creates a computation graph kernel (or subset thereof).

    Defines:
    - computation graph nodes the subclasses must have (and possibly create): input and output
    - abstract methods the subclasses must implement in order to compile to a computation graph

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
        """ Subclasses need to return [tf.Tensor operation] or similar
        Returns an array of tensor operations that session.run should evaluate.
        Or: a dictionary of names to ops values - basically anything that
        can be fed to tensorflow's session.run method's 'fetches' parameter.
        """
        raise NotImplementedError

    def _compile(self):
        """ Compile the tensorflow graph operations. Herein may construct
        output and input.
        """
        raise NotImplementedError

    def ops_format(self):
        """ Gets a description of what ops returns. """
        raise NotImplementedError
