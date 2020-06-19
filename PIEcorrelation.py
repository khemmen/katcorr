from __future__ import annotations

import numpy as np
import pylab as p
import tttrlib
import functions

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
red_indices = np.array(data.get_selection_by_channel([1, 3]), dtype=np.int64)

PIEcrosscorrelation_curve = functions.correlatePIE(
    micro_times=micro_times,
    macro_times=macro_times,
    indices_ch1=green_indices,
    indices_ch2=red_indices,
    PIE_windows_bins=PIE_windows_bins,
    n_casc=25
)

FRETcrosscorrelation_curve = functions.correlate_prompt(
    micro_times=micro_times,
    macro_times=macro_times,
    indices_ch1=green_indices,
    indices_ch2=red_indices,
    PIE_windows_bins=PIE_windows_bins,
    n_casc=25
)
########################################################
#  Option: get autocorrelation curves
########################################################
prompt_indices_ch1 = np.array(data.get_selection_by_channel([0]), dtype=np.int64)
prompt_indices_ch2 = np.array(data.get_selection_by_channel([2]), dtype=np.int64)

autocorr_prompt_g = functions.correlate_prompt(
    micro_times=micro_times,
    macro_times=macro_times,
    indices_ch1=prompt_indices_ch1,
    indices_ch2=prompt_indices_ch2,
    PIE_windows_bins=PIE_windows_bins,
    n_casc=25
)

prompt_indices_ch3 = np.array(data.get_selection_by_channel([1]), dtype=np.int64)
prompt_indices_ch4 = np.array(data.get_selection_by_channel([3]), dtype=np.int64)

autocorr_prompt_r = functions.correlate_prompt(
    micro_times=micro_times,
    macro_times=macro_times,
    indices_ch1=prompt_indices_ch3,
    indices_ch2=prompt_indices_ch4,
    PIE_windows_bins=PIE_windows_bins,
    n_casc=25
)


delay_indices_ch1 = np.array(data.get_selection_by_channel([1]), dtype=np.int64)
delay_indices_ch2 = np.array(data.get_selection_by_channel([3]), dtype=np.int64)

autocorr_delay_r = functions.correlate_delay(
    micro_times=micro_times,
    macro_times=macro_times,
    indices_ch1=delay_indices_ch1,
    indices_ch2=delay_indices_ch2,
    PIE_windows_bins=PIE_windows_bins,
    n_casc=25
)


########################################################
#  Save correlation curve
########################################################
time_axis = PIEcrosscorrelation_curve[0] * macro_time_calibration_ms
# calculates the correct time axis by multiplication of x-axis with macro_time
PIEcrosscorrelation = PIEcrosscorrelation_curve[1]  # 2nd column contains the average correlation amplitude calculated above
FRETcrosscorrelation = FRETcrosscorrelation_curve[1]
autocorrelation_green_prompt = autocorr_prompt_g[1]
autocorrelation_red_prompt = autocorr_prompt_r[1]
autocorrelation_red_delay = autocorr_delay_r[1]

suren_column = np.zeros_like(time_axis)  # fill 3rd column with 0's for compatibility with ChiSurf

# 4th column will contain uncertainty

filename_ccf = 'CCF_PIE.cor'  # change file name!
np.savetxt(
    filename_ccf,
    np.vstack(
        [
            time_axis,
            PIEcrosscorrelation,
            suren_column,
            #std_avg_correlation_amplitude
         ]
    ).T,
    delimiter='\t'
)

filename_fret = 'CCF_FRET.cor'  # change file name!
np.savetxt(
    filename_fret,
    np.vstack(
        [
            time_axis,
            FRETcrosscorrelation,
            suren_column,
            #std_avg_correlation_amplitude
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
            autocorrelation_green_prompt,
            suren_column,
            #std_avg_correlation_amplitude_ch1
         ]
    ).T,
    delimiter='\t'
)

filename_acf_prompt_red = 'ACF_prompt_r.cor'  # change file name!
np.savetxt(
    filename_acf_prompt_red,
    np.vstack(
        [
            time_axis,
            autocorrelation_red_prompt,
            suren_column,
            #std_avg_correlation_amplitude_ch2
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
            autocorrelation_red_delay,
            suren_column,
            #std_avg_correlation_amplitude_ch2
         ]
    ).T,
    delimiter='\t'
)

########################################################
#  Plotting
########################################################

p.semilogx(time_axis, PIEcrosscorrelation, label='gp-rd')
p.semilogx(time_axis, autocorrelation_red_prompt, label='rp-rp')
p.semilogx(time_axis, autocorrelation_green_prompt, label='gp-gp')
p.semilogx(time_axis, autocorrelation_red_delay, label='rd-rd')
p.semilogx(time_axis, FRETcrosscorrelation, label='gp-rp')

p.xlabel('correlation time [ms]')
p.ylabel('correlation amplitude')
p.legend()
p.savefig("correlation.svg", dpi=150)
p.show()
