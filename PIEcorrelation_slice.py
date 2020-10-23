from __future__ import annotations

import numpy as np
import pylab as p
import tttrlib
import functionsPIE_slice
import functions_slice

########################################################
#  Data input & optional selection process
########################################################

data = tttrlib.TTTR('A4+5-10bpPIE.ptu', 'PTU')
# rep rate = 80 MHz
header = data.get_header()
macro_time_calibration_ns = header.macro_time_resolution  # unit nanoseconds
macro_time_calibration_ms = macro_time_calibration_ns / 1e6  # macro time calibration in milliseconds
micro_time_resolution = header.micro_time_resolution
macro_times = data.get_macro_time()
micro_times = data.get_micro_time()
time_window_size = 1.0  # time window size in seconds
number_of_bins = macro_time_calibration_ns/micro_time_resolution
PIE_windows_bins = int(number_of_bins/2)

print("macro_time_calibration_ns:", macro_time_calibration_ns)
print("macro_time_calibration_ms:", macro_time_calibration_ms)
print("micro_time_resolution_ns:", micro_time_resolution)
print("number_of_bins:", number_of_bins)
print("PIE_windows_bins:", PIE_windows_bins)
print("time_window_size:", time_window_size)

# the dtype to int64 otherwise numba jit has hiccups
green_indices = np.array(data.get_selection_by_channel([0, 2]), dtype=np.int64)
indices_ch1 = functions_slice.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_indices,
    macro_time_calibration=macro_time_calibration_ms,
    time_window_size_seconds=time_window_size
)

red_indices = np.array(data.get_selection_by_channel([1, 3]), dtype=np.int64)
indices_ch2 = functions_slice.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=red_indices,
    macro_time_calibration=macro_time_calibration_ms,
    time_window_size_seconds=time_window_size
)

PIEcrosscorrelation_curves = functionsPIE_slice.correlate_piecesPIE(
    micro_times=micro_times,
    macro_times=macro_times,
    indices_ch1=indices_ch1,
    indices_ch2=indices_ch2,
    PIE_windows_bins=PIE_windows_bins,
    n_casc=25
)


########################################################
#  Option: get autocorrelation curves
########################################################
prompt_indices1 = np.array(data.get_selection_by_channel([0]), dtype=np.int64)
prompt_indices_ch1 = functions_slice.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=prompt_indices1,
    macro_time_calibration=macro_time_calibration_ms,
    time_window_size_seconds=time_window_size
)

prompt_indices2 = np.array(data.get_selection_by_channel([2]), dtype=np.int64)
prompt_indices_ch2 = functions_slice.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=prompt_indices2,
    macro_time_calibration=macro_time_calibration_ms,
    time_window_size_seconds=time_window_size
)

autocorr_prompt_g = functionsPIE_slice.correlate_pieces_prompt(
    micro_times=micro_times,
    macro_times=macro_times,
    indices_ch1=prompt_indices_ch1,
    indices_ch2=prompt_indices_ch2,
    PIE_windows_bins=PIE_windows_bins,
    n_casc=25
)

delay_indices1 = np.array(data.get_selection_by_channel([1]), dtype=np.int64)
delay_indices_ch1 = functions_slice.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=delay_indices1,
    macro_time_calibration=macro_time_calibration_ms,
    time_window_size_seconds=time_window_size
)

delay_indices2 = np.array(data.get_selection_by_channel([3]), dtype=np.int64)
delay_indices_ch2 = functions_slice.get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=delay_indices2,
    macro_time_calibration=macro_time_calibration_ms,
    time_window_size_seconds=time_window_size
)

autocorr_delay_r = functionsPIE_slice.correlate_pieces_delay(
    micro_times=micro_times,
    macro_times=macro_times,
    indices_ch1=delay_indices_ch1,
    indices_ch2=delay_indices_ch2,
    PIE_windows_bins=PIE_windows_bins,
    n_casc=25
)

# comparison is only made for the crosscorrelation curves
# autocorrelation curves are calculated based on the curve_ids selected by crosscorr
correlation_amplitudes = PIEcrosscorrelation_curves[:, 1, :]
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
    d_max=1
)

########################################################
#  Average selected PIE curves
########################################################
selected_curves = list()
for curve_idx in selected_curves_idx:
    selected_curves.append(
        PIEcrosscorrelation_curves[curve_idx]
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
        autocorr_prompt_g[curve_idx]
    )
selected_curves_ch1 = np.array(selected_curves_ch1)
avg_curve_ch1 = np.mean(selected_curves_ch1, axis=0)
std_curve_ch1 = np.std(selected_curves_ch1, axis=0)

selected_curves_ch2 = list()
for curve_idx in selected_curves_idx:
    selected_curves_ch2.append(
        autocorr_delay_r[curve_idx]
    )
selected_curves_ch2 = np.array(selected_curves_ch2)
avg_curve_ch2 = np.mean(selected_curves_ch2, axis=0)
std_curve_ch2 = np.std(selected_curves_ch2, axis=0)

########################################################
#  Calculate average count rates
########################################################

avg_cr_prompt_green = functionsPIE_slice.calculate_cr_prompt(
    timewindows=indices_ch1,
    time_window_size_seconds=time_window_size,
    micro_times=micro_times,
    macro_times=macro_times,
    PIE_windows_bins=PIE_windows_bins
)

avg_cr_prompt_red = functionsPIE_slice.calculate_cr_prompt(
    timewindows=indices_ch2,
    time_window_size_seconds=time_window_size,
    micro_times=micro_times,
    macro_times=macro_times,
    PIE_windows_bins=PIE_windows_bins
)

avg_cr_delay_red = functionsPIE_slice.calculate_cr_delay(
    timewindows=indices_ch2,
    time_window_size_seconds=time_window_size,
    micro_times=micro_times,
    macro_times=macro_times,
    PIE_windows_bins=PIE_windows_bins
)

########################################################
#  Save correlation curve
########################################################
time_axis = avg_curve[0] * macro_time_calibration_ms
# calculates the correct time axis by multiplication of x-axis with macro_time
avg_correlation_amplitude = avg_curve[1]  # 2nd column contains the average correlation amplitude calculated above
avg_correlation_amplitude_ch1 = avg_curve_ch1[1]
avg_correlation_amplitude_ch2 = avg_curve_ch2[1]
suren_column = np.zeros_like(time_axis)  # fill 3rd column with 0's for compatibility with ChiSurf
std_avg_correlation_amplitude = std_curve[1]/np.sqrt(len(selected_curves))
std_avg_correlation_amplitude_ch1 = std_curve_ch1[1]/np.sqrt(len(selected_curves))
std_avg_correlation_amplitude_ch2 = std_curve_ch2[1]/np.sqrt(len(selected_curves))
# 4th column contains standard deviation from the average curve calculated above

filename_cc = 'CCF_PIE.cor'  # change file name!
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


filename_acf_prompt = 'ACF_prompt_g.cor'  # change file name!
np.savetxt(
    filename_acf_prompt,
    np.vstack(
        [
            time_axis,
            avg_correlation_amplitude_ch1,
            suren_column,
            std_avg_correlation_amplitude_ch1
         ]
    ).T,
    delimiter='\t'
)

filename_acf_delay = 'ACF_delay_r.cor'  # change file name!
np.savetxt(
    filename_acf_delay,
    np.vstack(
        [
            time_axis,
            avg_correlation_amplitude_ch2,
            suren_column,
            std_avg_correlation_amplitude_ch2
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

########################################################
#  Save average countrates
########################################################

filename_avg_cr = 'avg_countrates.txt'  # change file name!
np.savetxt(
    filename_avg_cr,
    np.vstack(
        [
            avg_cr_prompt_green,
            avg_cr_prompt_red,
            avg_cr_delay_red
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
ax[1].semilogx(time_axis, avg_correlation_amplitude, label='gp-rd')
ax[1].semilogx(time_axis, avg_correlation_amplitude_ch2, label='rd-rd')
ax[1].semilogx(time_axis, avg_correlation_amplitude_ch1, label='gp-gp')
ax[2].plot(avg_cr_prompt_red, label='CR rp')
ax[2].plot(avg_cr_prompt_green, label='CR gp')
ax[2].plot(avg_cr_delay_red, label='CR rd')

ax[0].set_xlabel('slice #')
ax[0].set_ylabel('deviation')
ax[1].set_xlabel('correlation time [ms]')
ax[1].set_ylabel('correlation amplitude')
ax[2].set_xlabel('slice #')
ax[2].set_ylabel('countrate [Hz]')

legend = ax[0].legend()
legend = ax[1].legend()
legend = ax[2].legend()
p.savefig("result.svg", dpi=150)
p.show()
