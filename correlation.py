#!/usr/bin/env python

from __future__ import annotations

import numpy as np
import pylab as p
import tttrlib
import functions

########################################################
#  Actual data input & optional selections
########################################################

data = tttrlib.TTTR('A488_1.ptu', 'PTU')
# rep rate = 80 MHz
header = data.get_header()
macro_time_calibration_ns = header.macro_time_resolution  # unit nanoseconds
macro_time_calibration_ms = macro_time_calibration_ns / 1e6  # macro time calibration in milliseconds
macro_times = data.get_macro_time()

# the dtype to int64 otherwise numba jit has hiccups
green_s_indices = np.array(data.get_selection_by_channel([0]), dtype=np.int64)
green_p_indices = np.array(data.get_selection_by_channel([2]), dtype=np.int64)

crosscorrelation_curve = functions.correlate(
    macro_times=macro_times,
    indices_ch1=green_s_indices,
    indices_ch2=green_p_indices,
    n_casc=25
)

########################################################
#  Option: get autocorrelation curves
########################################################

autocorr_curve_ch1 = functions.correlate(
    macro_times=macro_times,
    indices_ch1=green_s_indices,
    indices_ch2=green_s_indices,
    n_casc=25
)

autocorr_curve_ch2 = functions.correlate(
    macro_times=macro_times,
    indices_ch1=green_p_indices,
    indices_ch2=green_p_indices,
    n_casc=25
)

########################################################
#  Save correlation curve
########################################################
time_axis = crosscorrelation_curve[0]* macro_time_calibration_ms
crosscorrelation_curve = crosscorrelation_curve[1]  # 2nd column contains the average correlation amplitude calculated above
autocorrelation_ch1 = autocorr_curve_ch1[1]
autocorrelation_ch2 = autocorr_curve_ch1[1]
suren_column = np.zeros_like(time_axis)  # fill 3rd column with 0's for compatibility with ChiSurf
suren_column_acf = np.zeros_like(time_axis_acf)

# 4th column will contain standard deviation
# How to get errors for single curve?

filename_cc = 'ch0_ch2_cross.cor'  # change file name!
np.savetxt(
    filename_cc,
    np.vstack(
        [
            time_axis,
            crosscorrelation_curve,
            suren_column,
            #std_avg_correlation_amplitude
        ]
    ).T,
    delimiter='\t'
)

filename_acf1 = 'ch0_auto.cor'  # change file name!
np.savetxt(
    filename_acf1,
    np.vstack(
        [
            time_axis,
            autocorrelation_ch1,
            suren_column,
            #std_avg_correlation_amplitude_ch1
        ]
    ).T,
    delimiter='\t'
)

filename_acf2 = 'ch2_auto.cor'  # change file name!
np.savetxt(
    filename_acf2,
    np.vstack(
        [
            time_axis,
            autocorrelation_ch2,
            suren_column,
            #std_avg_correlation_amplitude_ch2
        ]
    ).T,
    delimiter='\t'
)

########################################################
#  Plotting
########################################################
p.semilogx(time_axis, crosscorrelation_curve, label='gs-gp')
p.semilogx(time_axis, autocorrelation_ch1, label='gs-gs')
p.semilogx(time_axis, autocorrelation_ch2, label='gp-gp')

p.xlabel('correlation time [ms]')
p.ylabel('correlation amplitude')
p.legend()
p.savefig("correlation.svg", dpi=150)
p.show()
