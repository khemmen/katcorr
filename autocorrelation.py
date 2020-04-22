# Autocorrelation of selected events from single detector channel
# inspired from Ries...Schwille 2010 Optics Express "Automated suppression of sample-related artifacts in FCS"

from __future__ import annotations

import numpy as np
import pylab as p
import tttrlib
import functions
import functionsPIE

########################################################
#  Here the actual data input starts
########################################################

data = tttrlib.TTTR('A488_1.ptu', 'PTU')
# rep rate = 80 MHz
header = data.get_header()
macro_time_calibration = header.macro_time_resolution  # unit nanoseconds
macro_time_calibration /= 1e6  # macro time calibration in milliseconds
macro_times = data.get_macro_time()
time_window_size = 1.0  # time window size in seconds (overwrites selection above)

# the dtype to int64 otherwise numba jit has hiccups
green_1_indices = np.array(data.get_selection_by_channel([0]), dtype=np.int64)
indices_ch1 = functions.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_1_indices,
    macro_time_calibration=macro_time_calibration,
    time_window_size_seconds=time_window_size
)

green_2_indices = np.array(data.get_selection_by_channel([0]), dtype=np.int64)
indices_ch2 = functions.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_2_indices,
    macro_time_calibration=macro_time_calibration,
    time_window_size_seconds=time_window_size
)

correlation_curves = functions.correlate_pieces(
    macro_times=macro_times,
    indices_ch1=indices_ch1,
    indices_ch2=indices_ch2
)

correlation_amplitudes = correlation_curves[:, 1, :]
average_correlation_amplitude = correlation_amplitudes.mean(axis=0)

# adjust comparison_start & stop according to your diffusion time
# the selected values here encompass 1 ms -> 100 ms
deviation_from_mean = functions.calculate_deviation(
    correlation_amplitudes=correlation_amplitudes,
    comparison_start=120,
    comparison_stop=180,
    n=1
)

# select the curves with a small enough deviation to be considered in the further analysis
selected_curves_idx = functions.select_by_deviation(
    deviations=deviation_from_mean,
    d_max=2e-5
)

########################################################
#  Option: get average count rate per slice
########################################################
avg_countrate_ch1 = functions.calculate_countrate(
    timewindows=indices_ch1,
    time_window_size_seconds=time_window_size
)

########################################################
#  Average selected curves
########################################################
selected_curves = list()
for curve_idx in selected_curves_idx:
    selected_curves.append(
        correlation_curves[curve_idx]
    )
selected_curves = np.array(selected_curves)
avg_curve = np.mean(selected_curves, axis=0)
std_curve = np.std(selected_curves, axis=0)

########################################################
#  Save correlation curve
########################################################
time_axis = avg_curve[0] * macro_time_calibration  # calculates the correct time axis by multiplication of x-axis with macro_time
avg_correlation_amplitude = avg_curve[1]  # 2nd column contains the average correlation amplitude calculated above
suren_column = np.zeros_like(time_axis)  # fill 3rd column with 0's for compatibility with ChiSurf
std_avg_correlation_amplitude = std_curve[1]  # 4th column contains standard deviation from the average curve calculated above
filename = 'ch2_acf.cor'  # change file name!
np.savetxt(
    filename,
    np.vstack(
        [
            time_axis,
            average_correlation_amplitude,
            suren_column,
            std_avg_correlation_amplitude
         ]
    ).T,
    delimiter='\t'
)

filename = 'avg_countrate.txt'
np.savetxt(
    filename,
    np.vstack(
        [
            avg_countrate_ch1
         ]
    ).T,
    delimiter='\t'
)

########################################################
#  Save deviations
########################################################
deviations = np.array(deviation_from_mean)

filename = 'deviations.txt'  # change file name!
np.savetxt(
    filename,
    np.vstack(
        [
            deviations,
         ]
    ).T,
    delimiter='\t'
)

print("Done.")

########################################################
#  Plotting
########################################################
fig, ax = p.subplots(nrows=1, ncols=3, constrained_layout=True)

devx = np.arange(len(deviation_from_mean[0]))

ax[0].semilogy(devx, deviation_from_mean[0], label='deviations')
ax[1].semilogx(time_axis, avg_correlation_amplitude, label='ACF Ch1')
ax[2].plot(avg_countrate_ch1, label='CR Ch1')

ax[0].set_xlabel('slice #')
ax[0].set_ylabel('deviation')
ax[1].set_xlabel('correlation time [ms]')
ax[1].set_ylabel('correlation amplitude')
ax[2].set_xlabel('slice #')
ax[2].set_ylabel('countrate [Hz]')

legend = ax[0].legend()
legend = ax[1].legend()
legend = ax[2].legend()
p.show()
