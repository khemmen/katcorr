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
    print("time windows size (sec):", time_window_size_seconds)
    time_window_size_idx = int(time_window_size_seconds / macro_time_calibration * 1000.0)
    # times 1000 to bring macro_time_calibration from ms to sec
    returned_indices = list()
    macro_time_start_idx = 0
    current_list = [macro_time_start_idx]
    macro_time_start = macro_times[macro_time_start_idx]
    for idx in selected_indices[1:]:
        current_list.append(idx)
        macro_time_current = macro_times[idx]
        dt = macro_time_current - macro_time_start
        if dt >= time_window_size_idx:
            macro_time_start = macro_time_current
            returned_indices.append(
                np.array(current_list)
            )
            current_list = [idx]
    return returned_indices

# based on the sliced timewindows the average countrate for each slice is calculated
# input are the returned indices (list of arrays) from getting_idices_of_time_windows
# count rate in counts per seconds is returned
def calculate_countrate(
        timewindows: typing.List[np.ndarray],
        time_window_size_seconds: float = 2.0,
) -> List[float]:
    print("Calculating the average count rate...")
    avg_count_rate = list()
    index = 0
    while index < len(timewindows):
        nr_of_photons = len(timewindows[index])  # determines number of photons in a time slice
        avg_countrate = nr_of_photons / time_window_size_seconds  # division by length of time slice in seconds
        avg_count_rate.append(avg_countrate)
#        print(avg_countrate)
        index += 1
    return avg_count_rate


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

green_1_indices = data.get_selection_by_channel(np.array([0]))
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

avg_countrate_ch1 = calculate_countrate(
    timewindows=indices_ch1,
    time_window_size_seconds=time_window_size
)

avg_countrate_ch2 = calculate_countrate(
    timewindows=indices_ch2,
    time_window_size_seconds=time_window_size
)

g_factor = 0.8
total_countrate = np.array(avg_countrate_ch2) + np.array(avg_countrate_ch2)
parallel_channel = np.array(avg_countrate_ch2)
perpendicular_channel = np.array(avg_countrate_ch1)
rss = (parallel_channel - g_factor * perpendicular_channel)/(parallel_channel + 2 * g_factor * perpendicular_channel)

filename = 'avg_countrate.txt'
np.savetxt(
    filename,
    np.vstack(
        [
            total_countrate,
            avg_countrate_ch1,
            avg_countrate_ch2,
            rss
         ]
    ).T,
    delimiter='\t'
)

p.plot(avg_countrate_ch1)
p.plot(avg_countrate_ch2)
p.show()
p.plot(rss)
p.show()
print("Done.")