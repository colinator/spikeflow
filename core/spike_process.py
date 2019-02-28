import numpy as np

"""
Defines types for firings and spike processes, and functions to convert
between them.


firing: np.array((num_timesteps,), dtype=bool)
    A 'firing' is firing record for a single neuron: a numpy array of bools, one
    for each timestep.

firings: np.array((num_timesteps, num_neurons), dtype=bool)
    A 'firings' is a record of many neuron firings, in which each column is a neuron and
    each row is a timestep, of booleans: True means a neuron fired at a timestep.

spike_process: np.array(int)
    A 'spike_process' is a numpy array of integer indexes at which a neuron spiked.

spike_processes: [ spike_process ]
    A list of spike_process, one for each neuron


This is where typed-python might make sense.
"""

def firing_to_spike_process(firing):
    """ Converts a firing record for a single neuron to a spike process:
    a list of time step indexes at which the neuron fired.
    Args:
        firing: firing record for a single neuron
    Returns:
        spike_process for a single neuron
    """
    # return np.argwhere(firing).ravel()
    return np.reshape(np.argwhere(firing), (-1,))


def firings_to_spike_processes(firings):
    """ Converts firings numpy tensor to array of spike processes. These are
    simply numpy arrays of timestep indexes for which each neuron fired.
    Args:
        firings: firing records for multiple neurons
    Returns:
        spike_processes, one for each neuron
    """
    return [ firing_to_spike_process(firings[:,i]) for i in range(firings.shape[1]) ]


def spike_process_to_firing(spike_process, max_length=None):
    """ Converts a single neuron spike processes to a firing record.
    Args:
        spike_process: spike_process for a single neuron
        max_length: if highest timestep is < max_length, pads to max_length with 0s
    Returns:
        firing record for one neuron
    """
    length = max(np.max(spike_process)+1, 0 if max_length is None else max_length)
    firing = np.zeros((length,), dtype=bool)
    firing[spike_process] = True
    return firing


def spike_processes_to_firings(spike_processes, max_length=None):
    """ Converts spike processes to firing records.
    Args:
        spike_processes: list of spike processes for multiple neurons
        max_length: if highest timestep is < max_length, pads to max_length with 0s
    Returns:
        firing record for multiple neurons
    """
    length = max(np.max(spike_processes)+1, 0 if max_length is None else max_length)
    firings = np.zeros((length, len(spike_processes)), dtype=bool)
    for i, spike_process in enumerate(spike_processes):
        firings[spike_process,i] = True
    return firings


def spike_process_delta_times(pre_spike_process, post_spike_process):
    """ Calculates the delta times for all combinations of pre and post
    spike times, in one big array.
    Args:
        pre_spike_processes: spike_process of presynaptic neuron
        post_spike_processes: spike_process of postsynaptic neuron
    Returns:
        list of np.array(varying n, float32): for each spike process, the
            array of time differences; post_spike_times - each pre spike time,
            for each pre spike time
    """
    return np.array([ post_spike_process - pre_spike_time for pre_spike_time in pre_spike_process ]).ravel()
