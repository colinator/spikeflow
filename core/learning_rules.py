from collections import namedtuple
from spikeflow.core.spike_process import *


STDP_Params = namedtuple('STDP_Params', ['APlus', 'AMinus', 'TauPlus', 'TauMinus'])


def stdp_dw(delta_times, stdp_params):
    """ Calculates the changes in weights due to deltas of firing times,
    according to the stdp learning rule:

    dW := 0 for dt == 0
          APlus * exp(-1.0 * delta_times / TauPlus) for dt > 0
          AMinus * exp(1.0 * delta_times / TauMinus) for dt < 0

    Args:
        delta_times: np.array(int)
        stdp_params: STDP_Params
    Returns:
        np.array(delta_times.shape, float32)
    """
    return np.where(delta_times == 0,
                    0,
                    np.where(delta_times > 0,
                             stdp_params.APlus * np.exp(-1.0 * delta_times / stdp_params.TauPlus),
                             -1.0 * stdp_params.AMinus * np.exp(delta_times / stdp_params.TauMinus)))


def stdp_dw_process(input_spike_process, output_spike_process, stdp_params):
    """ Calculates the sum total change in weight from input and output spike processes as subject to the stdp learning rule, 'stdp_dw'.

    Args:
        w: weight matrix (np.array(input_n, output_n) of floats32)
        input_spike_processes: np.array(int)
        output_spike_process: np.array(int)
        stdp_params: STDP_Params
    Returns:
        Change in weight (np.float), subject to the stdp learning rule, 'stdp_dw'.
    """
    delta_times = spike_process_delta_times(input_spike_process, output_spike_process)
    all_dws = stdp_dw(delta_times, stdp_params)
    return np.sum(all_dws)


def stdp_dw_processes(w, input_spike_processes, output_spike_processes, stdp_params):
    """ Calculates change in weights from spike processes as subject to the stdp learning rule, 'stdp_dw'. Returns a dw matrix that corresponds to the weight matrix w - where there is a non-zero weight, there will be a dw.

    NOTE: This might be slow! Does not use a tensorflow computation graph (yet).

    Args:
        w: weight matrix (np.array(input_n, output_n) of floats32)
        input_spike_processes: [ np.array(int) ]
        output_spike_processes: [ np.array(int) ]
        stdp_params: STDP_Params
    Returns:
        Change in weights (np.array(w.shape), float32), subject to the stdp learning rule, 'stdp_dw'.
    """
    dW = np.zeros(w.shape, dtype=np.float32)

    w_indices_i, w_indices_o = np.nonzero(w)

    for i, o in zip(w_indices_i, w_indices_o):
        dW[i, o] = stdp_dw_process(input_spike_processes[i],
                                   output_spike_processes[o],
                                   stdp_params)

    return dW


def stdp_dw_firings(w, input_firings, output_firings, stdp_params):
    """ Calculates change in weights from firing matrices as subject to the stdp learning rule, 'stdp_dw'.

    Args:
        w: weight matrix (np.array(input_n, output_n) of floats32)
        input_firings: firings
        output_firings: firings
        stdp_params: STDP_Params
    Returns:
        Change in weights, subject to the stdp learning rule, 'stdp_dw'.
    """
    return stdp_dw_processes(w,
                             firings_to_spike_processes(input_firings), firings_to_spike_processes(output_firings),
                             stdp_params)
