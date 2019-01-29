import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from spikeflow import IzhikevichNeuronLayer

"""
The point of this library is not really these convenience rendering functions.
But maybe they'll be quick and dirty help.
"""

def draw_izhikevich_neuron_layer_abcdt_distributions(neuron_layer, dpi=100):
    """ Draws distributions of all a, b, c, d, and t values.
    """

    df = neuron_layer.to_dataframe()

    fig = plt.figure(num=None, figsize=(10, 0.6), dpi=dpi)
    fig.text(0 , 1, '{} neurons'.format(df.shape[0],))

    gs = gridspec.GridSpec(1, 5, hspace=0.2)
    for i, p in enumerate(['a', 'b', 'c', 'd', 't']):
        ax = fig.add_subplot(gs[i])
        ax.set_title(p)
        plt.box(on=None)
        bins = max(9, df[p].nunique() // 7)
        df[p].hist(bins=bins, grid=False, xlabelsize=7, ylabelsize=7, ax=ax)

    plt.show()


def draw_izhikevich_neuron_layer_ab_cd_distributions(neuron_layer, dpi=100,
    ab_ranges=((-0.2, 1.2), (-1.1, 1.1)),
    cd_ranges=((-70.0, -40.0), (-22.0, 10.0))):

    """ Draws a vs b and c vs d distributions of neuron layer values.
    """

    df = neuron_layer.to_dataframe()

    fig = plt.figure(num=None, figsize=(10, 2.0), dpi=dpi)
    fig.text(0 , 1, '{} neurons'.format(df.shape[0],))
    gs = gridspec.GridSpec(1, 5, hspace=0.3, width_ratios=[1, 3, 1, 3, 2])

    for (i, xname, yname, r) in [ (1, 'a', 'b', ab_ranges), (3, 'c', 'd', cd_ranges)]:
        ax = fig.add_subplot(gs[i])
        ax.set_title(xname+' vs '+yname)
        #plt.box(on=None)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.xticks(r[0])
        plt.yticks(r[1])
        ax.set_xlim(r[0])
        ax.set_ylim(r[1])
        plt.plot(df[xname], df[yname], 'o', alpha=0.1)

    plt.show()
