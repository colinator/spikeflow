import numpy as np

"""

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
    return np.reshape(np.argwhere(firing), (-1,))


def firings_to_spike_processes(firings):
    """ Converts firings numpy tensor to array of spike processes. These are
    simply numpy arrays of timestep indexes for which each neuron fired.
    Args:
        firings: np.array((timesteps, # neurons)) of bools; True = spike
    Returns:
        [ np.array([indexes of firings (where firings==True)]) ]
    """
    return [ firing_to_spike_process(firings[:,i]) for i in range(firings.shape[1]) ]


def spike_process_to_firing(spike_process, max_length=None):
    """ Converts
    Args:
        spike_process: numpy array of integers, where each value is the time of a spike.
    Returns:
        numpy array of booleans: False for no spike, True for spike, length maximum
        of spike_process
    """
    length = max(np.max(spike_process)+1, 0 if max_length is None else max_length)
    firing = np.zeros((length,), dtype=bool)
    firing[spike_process] = True
    return firing


def spike_processes_to_firings(spike_processes, max_length=None):
    """ Converts a list of spike processes to a single n x t numpy array of booleans
    Args:
        spike_processes: list of spike processes
        max_time: max time to pad out the resulting array to
    Returns:
        numpy array of n x t, where n is the number of spike processes, and
        t is either the largest value in spike processes, or max_time, if max_time
        is not None. Values in the result are either 0 (no spike) or 1 (spike)
    """
    length = max(np.max(spike_processes)+1, 0 if max_length is None else max_length)
    firings = np.zeros((length, len(spike_processes)), dtype=bool)
    for i, spike_process in enumerate(spike_processes):
        firings[spike_process,i] = True
    return firings
