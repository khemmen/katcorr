from __future__ import annotations

import numpy as np
import numba as nb
import pylab as p
import tttrlib


########################################################
#  Data input
########################################################

data = tttrlib.TTTR('A488_1.ptu', 'PTU')
# rep rate = 80 MHz
header = data.get_header()
macro_time_calibration = header.macro_time_resolution  # unit nanoseconds
macro_time_calibration /= 1e6  # macro time calibration in milliseconds
macro_times = data.get_macro_time()
time_window_size = 1.0  # time window size in seconds (overwrites selection above)

green_1_indices = data.get_selection_by_channel(np.array([0]))
indices_ch1 = functions.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_1_indices,
    macro_time_calibration=macro_time_calibration,
    time_window_size_seconds=time_window_size
)

green_2_indices = data.get_selection_by_channel(np.array([2]))
indices_ch2 = functions.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_2_indices,
    macro_time_calibration=macro_time_calibration,
    time_window_size_seconds=time_window_size
)

avg_countrate_ch1 = functions.calculate_countrate(
    timewindows=indices_ch1,
    time_window_size_seconds=time_window_size
)

avg_countrate_ch2 = functions.calculate_countrate(
    timewindows=indices_ch2,
    time_window_size_seconds=time_window_size
)

g_factor = 0.8  # please change according to your setup calibration
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
