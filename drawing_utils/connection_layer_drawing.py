import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from spikeflow import ConnectionLayer, SynapseLayer, ComplexSynapseLayer

"""
The point of this library is not really these convenience rendering functions.
But maybe they'll be quick and dirty help.
"""

def draw_synapse_layer(synapse_layer, dpi=100):
    """ Draws a synapse layer as image of weights
    """
    min = np.min(synapse_layer.w)
    colors = 'gnuplot2' if min == 0 else 'RdBu'
    plt.imshow(synapse_layer.w * -1 + 1, colors, aspect='auto')
    plt.ylabel('from neuron')
    plt.xlabel('to neuron')
    plt.box(on=False)
    plt.show()
