import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from spikeflow import ConnectionLayer, SimpleSynapseLayer, DecaySynapseLayer

"""
The point of this library is not really these convenience rendering functions.
But maybe they'll be quick and dirty help.
"""

def draw_synapse_layer(synapse_layer, dpi=100):
    """ Draws a synapse layer as image of weights
    """

    plt.imshow(synapse_layer.w, 'RdBu', aspect='auto')
    plt.ylabel('from neuron')
    plt.xlabel('to neuron')
    plt.box(on=False)
    plt.show()
