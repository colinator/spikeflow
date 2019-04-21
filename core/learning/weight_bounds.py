import tensorflow as tf

class WeightBounds:

    """ Contains weight-bounding parameters, as described here:
    http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity
    """

    def __init__(self, WMax, EtaPlus, EtaMinus, Soft=True):
        self.WMax = WMax
        self.EtaPlus = EtaPlus
        self.EtaMinus = EtaMinus
        self.Soft = Soft
        self.Hard = not self.Soft

    def __str__(self):
        return 'WeightBounds Wmax:{0:1.2f} η+:{1:1.2f} η-:{2:1.2f} Soft:{3}'.format(self.WMax, self.EtaPlus, self.EtaMinus, self.Soft)


def _Heaviside(v):
    fv = tf.cast(v, tf.float32)
    return tf.sign(fv) * 0.5 + 0.5

def _Constrain(v, min, max):
    return tf.minimum(tf.maximum(v, min), max)

class WeightBounds_Enforcer:

    """ Compiles weight-dependent weight-bounding, as described here:
    http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity
    """

    def __init__(self, weight_bounds):
        """ Constructs a WeightBounds_Enforcer.

        Args:
            weight_bounds: WeightBounds
        """
        self.weight_bounds = weight_bounds

    def _APlus(self, W):
        wd = self.weight_bounds.WMax - W
        bwd = wd if self.weight_bounds.Soft else _Heaviside(wd)
        return bwd * self.weight_bounds.EtaPlus

    def _AMinus(self, W):
        # seems to be an error in the scholarpedia article: NOT '-W'; use positive instead
        bw = W if self.weight_bounds.Soft else _Heaviside(W)
        return bw * self.weight_bounds.EtaMinus


    def _compile(self, W, trace_activation_totals):

        # OK: technically we need to do this:
        #   a_plus_pre = self._APlus(W) * pre_synaptic_trace_activations
        #   a_minus_post = self._AMinus(W) * post_synaptic_trace_activations # posts are already negative!
        #   new_W = W + a_plus_pre + a_minus_post
        #
        # BUT: we use a trick. All the negative values in the sum will treated as
        # depression, and all positive as potentiation. This is basically the same
        # as above.

        step_W = tf.where(
            trace_activation_totals > 0,
            self._APlus(W) * trace_activation_totals,
            self._AMinus(W) * trace_activation_totals)

        new_W = W + step_W

        # a minor nit:
        # the scholarpedia article implies that this step: the 'hard' limiting
        # of the resultant weight to [0,Wmax] should only be done if 'hard bounds'
        # is used. But it should ALWAYS be done: with soft bounds, the a_plus_pre
        # and a_minus_post terms can grow arbitrarily large (or small), and the
        # dW algorithm can make a single jump even with soft bounds that will
        # surpass [0,WMax]. So: always perform hard bounding.
        # Is this because this is not a continuous system, but uses discrete
        # timesteps?
        return _Constrain(new_W, 0.0, self.weight_bounds.WMax)
