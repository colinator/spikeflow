import numpy as np


"""
A 'firings' is a record of neuron firings: np.array of shape (num_timesteps, num_neurons)
where each element is a boolean: True means a neuron fired at a timestep.

A 'spike process' is a list of arrays of integer indexes at which a neuron spiked,
one list for each neuron. [ np.array(ints) ] where each int is a timestep index.

This is where typed-python might make sense.
"""


def firings_to_processes(firings):
    """ Converts firings numpy tensor to array of spike processes. These are
    simply numpy arrays of timestep indexes for which each neuron fired.
    Args:
        firings: np.array((timesteps, # neurons)) of bools; True = spike
    Returns:
        [ np.array([indexes of firings (where firings==True)]) ]
    """
    return [ np.reshape(np.argwhere(firings[:,i]), (-1,)) for i in range(firings.shape[1]) ]


def spike_counts(spike_processes):
    """ Gets the number of times each neuron fired.
    Args:
        spike_processes: [ np.array([indexes of spikes]) ]
    Returns:
        [ ints: counts of firings ]
    """
    return np.array([ p.shape[0] for p in spike_processes ])


def num_inactive_processes(spike_processes):
    """ Gets the number of neurons that didn't fire at all.
    Args:
        spike_processes: [ np.array([indexes of spikes]) ]
    Returns:
        int
    """
    return sum([0 if len(fp) > 1 else 0 for fp in spike_processes])


def num_zero_firing(firings):
    """ Gets the number of neurons that didn't fire at all.
    Args:
        firings: np.array((timesteps, # neurons)) of bools; True = spike
    Returns:
        int
    """
    return num_inactive_processes(firings_to_processes(firings))


def spiking_rates(spike_processes, n_timesteps):
    """ Gets the spiking rate of each neuron
    Args:
        spike_processes: [ np.array([indexes of spikes]) ]
        n_timesteps: int
    Returns:
        [ float ]
    """
    return np.array([ c / n_timesteps for c in spike_counts(spike_processes) ])


def firing_rates(firings):
    """ Gets the firing rates of all neurons.
    Args:
        firings: np.array((timesteps, # neurons)) of bools; True = spike
    Returns:
        [ float ]
    """
    return spiking_rates(firings_to_processes(firings), firings.shape[0])


def average_firing_rates(rates):
    """ Gets the total average firing rate of all neurons.
    Args:
        rates:
    Returns:
        float
    """
    return np.sum(rates) / len(rates)
