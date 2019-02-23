import tensorflow as tf
from spikeflow.core.neuron_layer import NeuronLayer


class CompositeLayer(NeuronLayer):

    """ A layer that looks like a neuron layer, but that can have any arbitrary
    internal structure, containing layers and connections. There is a single
    input, which is the input to the first neuron layer, and a single output,
    which is the output of the last neuron layer. In between: anything can happen.
    """

    def __init__(self, name, neuron_layers, connections):
        """ Creates a composite layer.
        Args:
            neuron_layers: list of NeuronLayer
            connections: list of ConnectionLayer
        """
        super().__init__(name)
        self.neuron_layers = neuron_layers
        self.connections = connections

    @property
    def input_n(self):
        """ The number of inputs is the number of inputs in the first neuron layer. """
        return self.neuron_layers[0].input_n

    @property
    def output_n(self):
        """ The number of outputs is the number of outputs in the last neuron layer. """
        return self.neuron_layers[-1].output_n

    def _ops(self):
        computation_layers = self.neuron_layers + self.connections
        return { clayer.name: clayer._ops() for clayer in computation_layers }

    def ops_format(self):
        computation_layers = self.neuron_layers + self.connections
        return { clayer.name: clayer.ops_format() for clayer in computation_layers }

    def _compile(self):

        if len(self.neuron_layers) == 0:
            raise ValueError("Composites and models must contain at least one neuron layer.")

        # get connection outputs
        connection_tos = {}
        for connection in self.connections:
            connection._compile_output_node()
            to_i = self.neuron_layers.index(connection.to_layer)
            connection_tos.setdefault(to_i, []).append(connection.output)

        # first neuron layer gets model inputs
        self.neuron_layers[0].add_input(self.input)

        # all neuron layers can get synaptic inputs
        for i, neuron_layer in enumerate(self.neuron_layers):
            for connection_input in connection_tos.get(i, []):
                neuron_layer.add_input(connection_input)

        # compile neuron layers
        for neuron_layer in self.neuron_layers:
            neuron_layer._compile()

        # hook up synapse layer inputs
        for connection in self.connections:
            connection.input = connection.from_layer.output
            connection._compile()

        # finally, my output is the last neuron layer output
        self.output = self.neuron_layers[-1].output


class BPNNModel(CompositeLayer):
    """ Top-level biologically plausible neural network model runner.
    Contains neuron and connection layers. Can compile to tensorflow graph, and
    then run through time, feeding input.
    """

    def __init__(self, input_shape):
        super().__init__('top', [], [])
        self.input_shape = input_shape
        self.input = None
        self.graph = None

    @classmethod
    def compiled_model(cls, input_shape, neuron_layers, connections):
        """ Convenience creation method. Creates and returns a compiled model.
        Args:
            input_shape: tuple of int; shape of input
            neuron_layers: [ neuron_layer.NeuronLayer ]
            connections: [ connection_layer.ConnectionLayer ]
        """
        model = cls(input_shape)
        for nl in neuron_layers:
            model.add_neuron_layer(nl)
        for conn in connections:
            model.add_connection_layer(conn)
        model.compile()
        return model

    def add_neuron_layer(self, neuron_layer):
        self.neuron_layers.append(neuron_layer)

    def add_connection_layer(self, connection_layer):
        self.connections.append(connection_layer)

    def compile(self):
        """ Creates and connects tensor float graph and graph node operations,
        including neuron and connection computation layers.
        """

        self.graph = tf.Graph()
        with self.graph.as_default():

            # create input tensor
            self.input = tf.placeholder(tf.float32, shape=self.input_shape, name='I')

            # compile neuron layers and connections
            self._compile()


    def run_time(self, data_generator, post_batch_callback):
        """ Runs the model through time as long as the data_generator produces data.

        After each time step, calls post_batch_callback, allowing data collection
        and graph modification (to an extent).

        NOTE: The callback method will necessarily drop back into python after each
        timestep. This will obviously slow it down. Solutions being considered.
        NOTE: It's called a post_batch_callback, but for now, batch size must be 1.
        This will be optimized in the future. Sort of the point of this library.

        Args:
            data_generator: a generator that must produce data with shape self.input_shape
            post_batch_callback: function (i: incrementing integer index
                                           graph: the tensorflow graph
                                           sess: the tensorflow session
                                           results: { layer_index: layer _ops output})

                results: Dictionary of keys to results, where keys are the indexes of
                computation layers (in order neuron layers, then connection layers,
                by addition), and results are outputs of the layer _ops
                method calls, which was just run in session.run.
                Called after every step of data generation.
        """

        runnables = self._ops()

        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()
            for i, data in enumerate(data_generator):
                results = sess.run(runnables, feed_dict={self.input: data})
                post_batch_callback(i, self.graph, sess, results)
