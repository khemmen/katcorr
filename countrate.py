from __future__ import annotations
import typing

import numpy as np
import numba as nb
import pylab as p
import tttrlib


# jit = just in time compiler, compiles code before execution to speed up algorithm
@nb.jit(
    nopython=True
)
# Slices the full trace of the data into pieces of 2 seconds
# Determines and saves the event-ID of the "start" and "end" of these slices
# change the number after time_window_size_seconds to slice data in larger pieces
def get_indices_of_time_windows(
        macro_times: np.ndarray,
        selected_indices: np.ndarray,
        macro_time_calibration: float,
        time_window_size_seconds: float = 2.0,
) -> typing.List[np.ndarray]:
    """Determines a list of start and stop indices for a TTTR object with
    selected indices and that correspond to the indices of the start and stop
    of time-windows.

    :param macro_times: numpy array of macro times
    :param macro_time_calibration: the macro time clock in milliseconds
    :param selected_indices: A preselected list of indices that defines which events
    in the TTTR event stream are considered
    :param time_window_size_seconds: The size of the time windows
    :return:
    """
    print("Getting indices of time windows")
    time_window_size_idx = int(time_window_size_seconds / macro_time_calibration * 1000.0)
    # time 1000 to bring macro_time_calibration from ms to sec
    returned_indeces = list()
    macro_time_start_idx = 0
    current_list = [macro_time_start_idx]
    macro_time_start = macro_times[macro_time_start_idx]
    for idx in selected_indices[1:]:
        current_list.append(idx)
        macro_time_current = macro_times[idx]
        dt = macro_time_current - macro_time_start
        if dt >= time_window_size_idx:
            macro_time_start = macro_time_current
            returned_indeces.append(
                np.array(current_list)
            )
            current_list = [idx]
    return returned_indeces

# Returned indices = indices_ch1/ch2 or a nested list/array
# in total for this example it is an arrays of 129 array
# the sub-array contain the ID of selected events
# I need to count the number of these events and dvide by the size of my time slice in seconds
# this will give me the average countrate for each slice in counts/sec
# how to address getting the length of this subarray?

def calculate_countrate(
        start_stop_indices: np.ndarray,
        time_window_size_seconds: float = 2.0,
):
    print("Calculating the average countrate...")
    avg_countrate = list()
    for idx in start_stop_indices[1:]:
        nr_of_photons = len(indices_ch1[:])
        # nr_of_photons = tttrlib.get_n_events()
        avg_countrate = nr_of_photons/time_window_size_seconds


########################################################
#  Here the actual data input starts
########################################################

data = tttrlib.TTTR('1_20min_1.ptu', 'PTU')
# rep rate = 80 MHz
header = data.get_header()
macro_time_calibration = header.macro_time_resolution  # unit nanoseconds
macro_time_calibration /= 1e6  # macro time calibration in milliseconds
macro_times = data.get_macro_time()
time_window_size = 5.0  # time window size in seconds (overwrites selection above)

green_1_indices = data.get_selection_by_channel(np.array([2]))
indices_ch1 = get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_1_indices,
    macro_time_calibration=macro_time_calibration,
    time_window_size_seconds=time_window_size
)

green_2_indices = data.get_selection_by_channel(np.array([2]))
indices_ch2 = get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_2_indices,
    macro_time_calibration=macro_time_calibration,
    time_window_size_seconds=time_window_size
)
