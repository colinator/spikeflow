import tensorflow as tf
import numpy as np

def logsafe(v):
    return tf.where(tf.equal(v, 0.0), tf.zeros_like(v), tf.log(v))

def ricker_wavelet(v, sigma):
    """ Returns the ricker wavelet, or mexican hat wavelet, like this:
    https://en.wikipedia.org/wiki/Mexican_hat_wavelet
    """

    sigma_cast = tf.cast(sigma, tf.float32)
    sigmasquared = tf.pow(sigma_cast, 2.0)
    f = 3.0 * np.power(np.pi, 0.25)
    p1 = 2.0 / (tf.sqrt(sigma_cast * f))
    p2 = 1.0 - tf.pow((v / sigma_cast), 2.0)
    p3 = tf.exp(-1.0 * v*v / 2.0 * sigmasquared)
    return p1 * p2 * p3
