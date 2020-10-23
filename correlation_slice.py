from __future__ import annotations

import numpy as np
import pylab as p
import tttrlib
import functions_slice

########################################################
#  Data input & optional selection process
########################################################

#data = tttrlib.TTTR(r"\\HC1008\Users\AG Heinze\DATA\FCSSetup\2020\20200717_FK_ LANAP\cellsptw\A47 730nm 100um_cell2.ptu", 'PTU')
data = tttrlib.TTTR("A488_1.ptu", 'PTU')
# rep rate = 80 MHz
header = data.get_header()
macro_time_calibration_ns = header.macro_time_resolution  # unit nanoseconds
macro_time_calibration_ms = macro_time_calibration_ns / 1e6  # macro time calibration in milliseconds
macro_times = data.get_macro_time()
time_window_size = 1.0  # time window size in seconds
print("macro_time_calibration_ns:", macro_time_calibration_ns)
print("macro_time_calibration_ms:", macro_time_calibration_ms)
print("time_window_size:", time_window_size)
print("duration:", len(macro_times))

# the dtype to int64 otherwise numba jit has hiccups
green_s_indices = np.array(data.get_selection_by_channel([0]), dtype=np.int64)
indices_ch1 = functions_slice.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_s_indices,
    macro_time_calibration=macro_time_calibration_ms,
    time_window_size_seconds=time_window_size
)

green_p_indices = np.array(data.get_selection_by_channel([2]), dtype=np.int64)
indices_ch2 = functions_slice.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_p_indices,
    macro_time_calibration=macro_time_calibration_ms,
    time_window_size_seconds=time_window_size
)

crosscorrelation_curves = functions_slice.correlate_pieces(
    macro_times=macro_times,
    indices_ch1=indices_ch1,
    indices_ch2=indices_ch2,
)

########################################################
#  Option: get autocorrelation curves
########################################################

autocorr_curve_ch1 = functions_slice.correlate_pieces(
    macro_times=macro_times,
    indices_ch1=indices_ch1,
    indices_ch2=indices_ch1,
    n_casc=25
)

autocorr_curve_ch2 = functions_slice.correlate_pieces(
    macro_times=macro_times,
    indices_ch1=indices_ch2,
    indices_ch2=indices_ch2,
    n_casc=25
)

########################################################
#  Option: get average count rate per slice
########################################################
avg_countrate_ch1 = functions_slice.calculate_countrate(
    timewindows=indices_ch1,
    time_window_size_seconds=time_window_size
)

avg_countrate_ch2 = functions_slice.calculate_countrate(
    timewindows=indices_ch2,
    time_window_size_seconds=time_window_size
)

# comparison is only made for the crosscorrelation curves
# autocorrelation curves are calculated based on the curve_ids selected by crosscorr
correlation_amplitudes = crosscorrelation_curves[:, 1, :]
average_correlation_amplitude = correlation_amplitudes.mean(axis=0)

# adjust comparison_start & stop according to your diffusion time
# the selected values here encompass 1 ms -> 100 ms
deviation_from_mean = functions_slice.calculate_deviation(
    correlation_amplitudes=correlation_amplitudes,
    comparison_start=40,
    comparison_stop=60,
    n=1
)

# select the curves with a small enough deviation to be considered in the further analysis
selected_curves_idx = functions_slice.select_by_deviation(
    deviations=deviation_from_mean,
    d_max=0.5
)

# sort squared deviations (get the indices)
# sorted_indices = np.argsort(d, axis=0)
# curve_closest_to_mean = correlation_curves[sorted_indices[0], 0, :], correlation_curves[sorted_indices[0], 1, :]
# curve_farstest_from_mean = correlation_curves[sorted_indices[-1], 0, :], correlation_curves[sorted_indices[-1], 1, :]

# alternative selection
# selected_curves_idx = select_by_half_height_time(
#     correlation_amplitudes=correlation_amplitudes,
#     number_of_stdev=0.2
# )

########################################################
#  Average selected curves
########################################################
selected_curves = list()
for curve_idx in selected_curves_idx:
    selected_curves.append(
        crosscorrelation_curves[curve_idx]
    )

selected_curves = np.array(selected_curves)
avg_curve = np.mean(selected_curves, axis=0)
std_curve = np.std(selected_curves, axis=0)

########################################################
#  Average selected autocorrelation curves
########################################################
selected_curves_ch1 = list()
for curve_idx in selected_curves_idx:
    selected_curves_ch1.append(
        autocorr_curve_ch1[curve_idx]
    )
selected_curves_ch1 = np.array(selected_curves_ch1)
avg_curve_ch1 = np.mean(selected_curves_ch1, axis=0)
std_curve_ch1 = np.std(selected_curves_ch1, axis=0)

selected_curves_ch2 = list()
for curve_idx in selected_curves_idx:
    selected_curves_ch2.append(
        autocorr_curve_ch2[curve_idx]
    )
selected_curves_ch2 = np.array(selected_curves_ch2)
avg_curve_ch2 = np.mean(selected_curves_ch2, axis=0)
std_curve_ch2 = np.std(selected_curves_ch2, axis=0)

########################################################
#  Save correlation curve
########################################################
time_axis = avg_curve[0] * macro_time_calibration_ms  # calculates the correct time axis by multiplication of x-axis with macro_time
time_axis_acf = avg_curve_ch1[0] * macro_time_calibration_ms
avg_correlation_amplitude = avg_curve[1]  # 2nd column contains the average correlation amplitude calculated above
avg_correlation_amplitude_ch1 = avg_curve_ch1[1]
avg_correlation_amplitude_ch2 = avg_curve_ch2[1]
suren_column = np.zeros_like(time_axis)  # fill 3rd column with 0's for compatibility with ChiSurf
suren_column_acf = np.zeros_like(time_axis_acf)
std_avg_correlation_amplitude = std_curve[1]/np.sqrt(len(selected_curves))
std_avg_correlation_amplitude_ch1 = std_curve_ch1[1]/np.sqrt(len(selected_curves))
std_avg_correlation_amplitude_ch2 = std_curve_ch2[1]/np.sqrt(len(selected_curves))
# 4th column contains standard deviation from the average curve calculated above
filename_cc = '10s_ch0_ch2_cross.cor'  # change file name!
np.savetxt(
    filename_cc,
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

filename_acf1 = '10s_ch0_auto.cor'  # change file name!
np.savetxt(
    filename_acf1,
    np.vstack(
        [
            time_axis_acf,
            avg_correlation_amplitude_ch1,
            suren_column_acf,
            std_avg_correlation_amplitude_ch1
         ]
    ).T,
    delimiter='\t'
)

filename_acf2 = '10s_ch2_auto.cor'  # change file name!
np.savetxt(
    filename_acf2,
    np.vstack(
        [
            time_axis_acf,
            avg_correlation_amplitude_ch2,
            suren_column_acf,
            std_avg_correlation_amplitude_ch2
         ]
    ).T,
    delimiter='\t'
)

########################################################
#  Calculate steady-state anisotropy & save count rate per slice
########################################################

g_factor = 1  # please change according to your setup calibration
total_countrate = np.array(avg_countrate_ch2) + np.array(avg_countrate_ch2)
parallel_channel = np.array(avg_countrate_ch2)
perpendicular_channel = np.array(avg_countrate_ch1)
rss = (parallel_channel - g_factor * perpendicular_channel)/(parallel_channel + 2 * g_factor * perpendicular_channel)

filename = '10s_avg_countrate.txt'  # change file name!
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

########################################################
#  Save deviations
########################################################
deviations = np.array(deviation_from_mean)

filename = '10s_deviations.txt'  # change file name!
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
fig, ax = p.subplots(nrows=2, ncols=2, constrained_layout=True)

devx = np.arange(len(deviation_from_mean[0]))

ax[0, 0].semilogy(devx, deviation_from_mean[0], label='deviations')
ax[0, 1].semilogx(time_axis, avg_correlation_amplitude, label='gs-gp')
ax[0, 1].semilogx(time_axis, avg_correlation_amplitude_ch1, label='gs-gs')
ax[0, 1].semilogx(time_axis, avg_correlation_amplitude_ch2, label='gp-gp')
ax[1, 0].plot(avg_countrate_ch1, label='CR gs(perpendicular)')
ax[1, 0].plot(avg_countrate_ch2, label='CR gp(parallel)')
ax[1, 1].plot(rss, label='rss')

ax[0, 0].set_xlabel('slice #')
ax[0, 0].set_ylabel('deviation')
ax[0, 1].set_xlabel('correlation time [ms]')
ax[0, 1].set_ylabel('correlation amplitude')
ax[1, 0].set_xlabel('slice #')
ax[1, 0].set_ylabel('countrate [Hz]')
ax[1, 1].set_xlabel('slice #')
ax[1, 1].set_ylabel('steady-state anisotropy')

legend = ax[0, 0].legend()
legend = ax[0, 1].legend()
legend = ax[1, 0].legend()
legend = ax[1, 1].legend()
p.savefig("result.svg", dpi=150)
p.show()