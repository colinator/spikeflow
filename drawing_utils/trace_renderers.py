import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from spikeflow.core.analysis_utils import *


"""
The point of this library is not really these convenience rendering functions.
But maybe they'll be quick and dirty help.
"""

def render_signal(ax, name, signal, colorCode, show_yticks=True, alpha=1.0, linewidth=1.0):
    plt.box(on=None)
    ax.set_ylabel(name, color=colorCode)
    ax.tick_params('y', colors=colorCode)
    plt.xlim(xmax=signal.shape[0])
    plt.xticks([])
    if not show_yticks:
        plt.yticks([])
    plt.plot(signal, colorCode, alpha=alpha, linewidth=linewidth)


class TraceRenderer:

    def __init__(self, traces, name):
        self.traces = traces
        self.name = name

    def subfigure_height_ratios(self, dpi):
        return []

    def height(self, dpi):
        pass

    def render(self, fig, gs, gs_start, start_time, end_time):
        pass


class IdentityNeuronTraceRenderer(TraceRenderer):

    def height(self, dpi):
        return 0.8 * self.traces.shape[2]

    def subfigure_height_ratios(self, dpi):
        return [1, 1] + [3, 1, 1] * (self.traces.shape[2]-1)

    def render(self, fig, gs, gs_start, start_time, end_time):
        for i in range(self.traces.shape[2]):
            for t in range(2):
                ax = fig.add_subplot(gs[gs_start+(i*3)+t])
                if t == 0:
                    ax.set_title(self.name + ' ' + str(i))
                color = { 0: 'r', 1: 'b'}[t]
                name = { 0: 'in', 1: 'out'}[t]
                render_signal(ax, name, self.traces[start_time:end_time,t,i], color, show_yticks=t<1)


class LIFNeuronTraceRenderer(TraceRenderer):

    def height(self, dpi):
        return 1.0 * self.traces.shape[2]

    def subfigure_height_ratios(self, dpi):
        return [1, 2, 1] + [3, 1, 2, 1] * (self.traces.shape[2]-1)

    def render(self, fig, gs, gs_start, start_time, end_time):
        for i in range(self.traces.shape[2]):
            for j, t in enumerate([0, 3, 2]):
                ax = fig.add_subplot(gs[gs_start+(i*4)+j])
                if t == 0:
                    ax.set_title(self.name + ' ' + str(i))
                color = { 0: 'r', 1: 'g', 2:'b'}[j]
                name = { 0: 'in', 1: 'v', 2: 'out'}[j]
                render_signal(ax, name, self.traces[start_time:end_time,t,i], color, show_yticks=t<2)


class IzhikevichNeuronTraceRenderer(TraceRenderer):

    def height(self, dpi):
        return 1.4 * self.traces.shape[2]

    def subfigure_height_ratios(self, dpi):
        return [1, 2, 1, 1] + [3, 1, 2, 1, 1] * (self.traces.shape[2]-1)

    def render(self, fig, gs, gs_start, start_time, end_time):
        for i in range(self.traces.shape[2]):
            for t in range(4):
                ax = fig.add_subplot(gs[gs_start+(i*5)+t])
                if t == 0:
                    ax.set_title(self.name + ' ' + str(i))
                color = { 0: 'r', 1: 'g', 2:'y', 3:'b'}[t]
                name = { 0: 'in', 1: 'v', 2: 'u', 3: 'out'}[t]
                render_signal(ax, name, self.traces[start_time:end_time,t,i], color, show_yticks=t<3)


class NeuronFiringsRenderer(TraceRenderer):

    def __init__(self, traces, firing_height, name, groups=None):
        super().__init__(traces, name)
        self.firing_height = firing_height
        self.groups = groups

    def height(self, dpi):
        return (2.0 / dpi) * self.traces.shape[1] * self.firing_height + 1.0

    def subfigure_height_ratios(self, dpi):
        return [(2.0 / dpi) * self.traces.shape[1] * self.firing_height + 1.0]

    def render(self, fig, gs, gs_start, start_time, end_time):
        data = self.traces[start_time:end_time,:].T
        data2 = np.repeat(data, 3).reshape(data.shape[0], data.shape[1], 3)
        data3 = data2 * (np.array([0.0, 0.0, 1.0]) * -1.0 + 0.95) * -1.0 + 0.95
        if self.groups is not None:
            for (fromn, ton, color) in self.groups:
                k = data2[fromn:ton,:,:] * (np.array(color) * -1.0 + 0.95) * -1.0 + 0.95
                data3[fromn:ton,:,:] = k
        data_to_render = np.flipud(np.repeat(data3, self.firing_height, axis=0))
        ax = fig.add_subplot(gs[gs_start])
        ax.set_title(self.name)
        plt.box(on=None)
        ax.set_ylabel('neuron i')
        ax.set_xlabel('timestep t')
        ax.set_ylim((-5, data_to_render.shape[0]+5))
        ax.set_xlim((0, end_time-start_time))
        plt.yticks([])
        plt.xticks([])
        plt.imshow(data_to_render, aspect='auto', interpolation='nearest')
        #plt.imshow(data_to_render, 'Blues', aspect='auto', interpolation='nearest')
        #plt.imshow(data_to_render, 'RdBu', aspect='auto', interpolation='nearest')


def render_figure(renderers, start_time, end_time, dpi=100):
    height = sum([r.height(dpi) for r in renderers]) + (len(renderers)-1)*0.2

    height_ratios = []
    for r in renderers:
        height_ratios += r.subfigure_height_ratios(dpi) + [6]

    fig = plt.figure(num=None, figsize=(14, height), dpi=dpi)
    gs = gridspec.GridSpec(len(height_ratios), 1, hspace=0.1, height_ratios=height_ratios)

    gs_start = 0
    for renderer in renderers:
        renderer.render(fig, gs, gs_start, start_time, end_time)
        gs_start += len(renderer.subfigure_height_ratios(dpi)) + 1

    plt.xticks([i for i in range(0, end_time-start_time, 100)])
    plt.show()


def draw_firing_distributions(layer_firings, groups=None, dpi=100):
    fig = plt.figure(num=None, figsize=(2 * len(layer_firings), 0.6), dpi=dpi)
    gs = gridspec.GridSpec(1, len(layer_firings), hspace=1.0)
    for i, firings in enumerate(layer_firings):
        rates = firing_rates(firings)
        df = pd.DataFrame(rates, index = range(len(rates)), columns = ['r'])
        ax = fig.add_subplot(gs[i])
        ax.spines['bottom'].set_color('grey')
        ax.spines['top'].set_color('grey')
        ax.spines['left'].set_color('grey')
        ax.spines['right'].set_color('grey')
        ax.set_xlim((0, df['r'].max() + 0.01))
        ax.set_title('Layer ' + str(i) + ": " + str(len(rates)) + ' neurons', fontsize=9)
        bins = max(17, df['r'].nunique() // 7)
        df['r'].hist(bins=bins, grid=False, xlabelsize=7, ylabelsize=7, ax=ax)

    plt.show()
