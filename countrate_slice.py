from __future__ import annotations

import numpy as np
import pylab as p
import tttrlib
import functions_slice

########################################################
#  Data input
########################################################

data = tttrlib.TTTR('A88_1.ptu', 'PTU')
# rep rate = 80 MHz
header = data.get_header()
macro_time_calibration = header.macro_time_resolution  # unit nanoseconds
macro_time_calibration /= 1e6  # macro time calibration in milliseconds
macro_times = data.get_macro_time()
time_window_size = 1.0  # time window size in seconds

green_1_indices = np.array(data.get_selection_by_channel([0]), dtype=np.int64)
indices_ch1 = functions_slice.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_1_indices,
    macro_time_calibration=macro_time_calibration,
    time_window_size_seconds=time_window_size
)

green_2_indices = np.array(data.get_selection_by_channel([2]), dtype=np.int64)
indices_ch2 = functions_slice.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_2_indices,
    macro_time_calibration=macro_time_calibration,
    time_window_size_seconds=time_window_size
)

avg_countrate_ch1 = functions_slice.calculate_countrate(
    timewindows=indices_ch1,
    time_window_size_seconds=time_window_size
)

avg_countrate_ch2 = functions_slice.calculate_countrate(
    timewindows=indices_ch2,
    time_window_size_seconds=time_window_size
)

g_factor = 0.8  # please change according to your setup calibration
total_countrate = np.array(avg_countrate_ch2) + np.array(avg_countrate_ch2)
parallel_channel = np.array(avg_countrate_ch2)
perpendicular_channel = np.array(avg_countrate_ch1)
rss = (parallel_channel - g_factor * perpendicular_channel) / (parallel_channel + 2 * g_factor * perpendicular_channel)

filename = 'avg_countrate_A88_1.txt'
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

print("Done.")

########################################################
#  Plotting
########################################################

fig, ax = p.subplots(nrows=1, ncols=2, constrained_layout=True)

ax[0].plot(avg_countrate_ch1, label='CR Ch1(perpendicular)')
ax[0].plot(avg_countrate_ch2, label='CR Ch2(parallel)')
ax[1].plot(rss, label='rss')

ax[0].set_xlabel('slice #')
ax[0].set_ylabel('countrate [Hz]')
ax[1].set_xlabel('slice #')
ax[1].set_ylabel('steady-state anisotropy')

legend = ax[0].legend()
legend = ax[1].legend()
p.savefig('A88_1.svg', dpi=150)
p.show()
