import tensorflow as tf

class BPNNModel:

    """ Top-level biologically plausible neural network model runner.
    Contains neuron and connection layers. Can compile to tensorflow graph, and
    then run through time, feeding input.
    """

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.input = None
        self.neuron_layers = []
        self.connections = []
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

            # get connection outputs
            connection_tos = {}
            for connection in self.connections:
                connection._compile_output_node()
                to_i = self.neuron_layers.index(connection.to_layer)
                connection_tos.setdefault(to_i, []).append(connection.output)

            # first neuron layer gets model inputs
            if len(self.neuron_layers) > 0:
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
            post_batch_callback: function (results: { layer_index: layer _ops output})
                Dictionary of keys to results, where keys are the indexes of
                computation layers (in order neuron layers, then connection layers,
                by addition), and results are outputs of the layer _ops
                array, which was just run in session.run.
                Called after every step of data generation.
        """

        computation_layers = self.neuron_layers + self.connections

        # NOTE! This is weak: why do I actually need to use all _ops? Why can't
        # I just run the top-level (final?) one? (tried, doesn't work).
        runnables = { i: clayer._ops() for i, clayer in enumerate(computation_layers) }

        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()
            for data in data_generator:
                results = sess.run(runnables, feed_dict={self.input: data})
                post_batch_callback(results)
