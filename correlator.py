# crosscorrelation of selected events from two detector channels
# inspired from Ries...Schwille 2010 Optics Express "Automated suppression of sample-related artifacts in FCS"

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

# correlation of the full trace
def correlate(
        macro_times: np.ndarray,
        indices_ch1: np.ndarray,
        indices_ch2: np.ndarray,
        B: int = 9,
        n_casc: int = 25
        # B and n_casc define the spacing of the logarithmic correlation axis
        # increase n_casc if you need to correlate to longer time intervals
) -> (np.ndarray, np.ndarray):
    # Create correlator
    mt = macro_times
    correlator = tttrlib.Correlator()
    correlator.set_n_bins(B)
    correlator.set_n_casc(n_casc)
    # Select the green channels (channel number 1 and 2)
    # w1 & w2 are weights, which will be 1 by default if not defined elsewhere
    # use w1 & w2 e.g. for filtered FCS or when only selected events should be correlated
    t1 = mt[indices_ch1]
    w1 = np.ones_like(t1, dtype=np.float)
    t2 = mt[indices_ch2]
    w2 = np.ones_like(t2, dtype=np.float)
    correlator.set_events(t1, w1, t2, w2)
    correlator.run()
    x = correlator.get_x_axis_normalized()
    y = correlator.get_corr_normalized()
    return x, y

# correlation of the 2-sec-pieces
def correlate_pieces(
        macro_times: np.ndarray,
        indices_ch1: typing.List[np.ndarray],
        indices_ch2: typing.List[np.ndarray],
        B: int = 9,
        n_casc: int = 25
) -> np.ndarray:
    print("Correlating pieces...")
    n_correlations = min(len(indices_ch1), len(indices_ch2))
    # returns nr of slices, minimum of ch1 or ch2 is reported in case they have different size
    correlation_curves = list()
    for i in range(n_correlations):
        print("%i / %i" % (i, n_correlations))  # gives the progress, how many pieces have already been evaluated
        # no weights are used!
        x, y = correlate(
            macro_times=macro_times,
            indices_ch1=indices_ch1[i],
            indices_ch2=indices_ch2[i],
            B=B,
            n_casc=n_casc
        )
        correlation_curves.append([x, y])
    correlation_curves = np.array(
        correlation_curves
    )
    return correlation_curves


########################################################
#  Option 2: Select time slices based on their deviation from the mean beginning curve
########################################################

def select_by_deviation(
        correlation_amplitudes: typing.List[np.ndarray],
        d_max: float = 2e-5,  # selection criterion
        comparison_start: int = 100,  # only a specified range of the curve is compared as beginning will be too noisy
        comparison_stop: int = 170  # comparison of this range will be done to the first 30 curves
) -> typing.List[int]:
    print("Selecting indices of curves by deviations.")
    print("d_max:", d_max)
    print("Comparison time range:", comparison_start, "-", comparison_stop)
    print("compared to the first 30 curves")
    if (comparison_start is None) or (comparison_stop is None):
        # calculates from every curve the difference to the mean of the first 30 curves, squares this value
        # and divides this value by the number of curves
        ds = np.mean(
            (correlation_amplitudes - correlation_amplitudes[:30].mean(axis=0))**2.0, axis=1
        ) / len(correlation_amplitudes)
    else:
        ca = correlation_amplitudes[:, comparison_start: comparison_stop]
        ds = np.mean(
            (ca - ca[:30].mean(axis=0)) ** 2.0, axis=1
        ) / (comparison_stop - comparison_start)
    print(ds)
    logdev = np.log10(ds)
    p.plot(logdev)
    p.show()
    selected_curves_idx = list()
    for i, d in enumerate(ds):
        if d < d_max:
            selected_curves_idx.append(i)
    print("Total number of curves: ", len(ds))
    print("Selected curves: ", len(selected_curves_idx))
    return selected_curves_idx

########################################################
#  Here the actual data input starts
########################################################

data = tttrlib.TTTR('1_20min_1.ptu', 'PTU')
# rep rate = 80 MHz
header = data.get_header()
macro_time_calibration = header.macro_time_resolution  # unit nanoseconds
macro_time_calibration /= 1e6  # macro time calibration in milliseconds
macro_times = data.get_macro_time()
time_window_size = 10.0  # time window size in seconds (overwrites selection above)

green_s_indices = data.get_selection_by_channel(np.array([0]))
indices_ch1 = get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_s_indices,
    macro_time_calibration=macro_time_calibration,
    time_window_size_seconds=time_window_size
)

green_p_indices = data.get_selection_by_channel(np.array([2]))
indices_ch2 = get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_p_indices,
    macro_time_calibration=macro_time_calibration,
    time_window_size_seconds=time_window_size
)

correlation_curves = correlate_pieces(
    macro_times=macro_times,
    indices_ch1=indices_ch1,
    indices_ch2=indices_ch2
)

correlation_amplitudes = correlation_curves[:, 1, :]
average_correlation_amplitude = correlation_amplitudes.mean(axis=0)

# sort squared deviations (get the indices)
# sorted_indices = np.argsort(d, axis=0)
# curve_closest_to_mean = correlation_curves[sorted_indices[0], 0, :], correlation_curves[sorted_indices[0], 1, :]
# curve_farthest_from_mean = correlation_curves[sorted_indices[-1], 0, :], correlation_curves[sorted_indices[-1], 1, :]
# selected_curves_idx = select_by_half_height_time(
#     correlation_amplitudes=correlation_amplitudes,
#     number_of_stdev=0.2
# )

selected_curves_idx = select_by_deviation(
    correlation_amplitudes=correlation_amplitudes,
    d_max=1.5e-6  # selection criterion (overwrites the above value)
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
filename = 'test.cor'
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
print("Done.")
p.semilogx(time_axis, avg_correlation_amplitude)
p.show()

########################################################
#  Save log file --> to do
########################################################
# filename = 'log.txt'



########################################################
#  Save deviation values --> to do
########################################################
# filename = 'ds.txt'
# curve_number = avg_curve[0]
# deviation = ds
# np.savetxt(
#     filename,
#     np.vstack(
#         [
#             curve_number,
#             deviation,
#          ]
#     ).T,
#     delimiter='\t'
# )

# ########################################################
# #  Correlate again with photons of selected curves
# ########################################################
#
# # get indices of events which belong to selected curves
# # this approach as suggested in the Fries-paper was not successful for my type of data
# # too long measurements, too diverse count rate when beginning and end are compared
# selected_ch1_events = list()
# for i in selected_curves_idx:
#     selected_ch1_events.append(indices_ch1[i])
# selected_ch1_events = np.hstack(selected_ch1_events)
#
# selected_ch2_events = list()
# for i in selected_curves_idx:
#     selected_ch2_events.append(indices_ch2[i])
# selected_ch2_events = np.hstack(selected_ch2_events)
#
# # correlated with selected indices
# x, y = correlate(
#     macro_times=macro_times,
#     indices_ch1=selected_ch1_events,
#     indices_ch2=selected_ch2_events,
#     B=9,
#     n_casc=25
# )
#
# after the slices have been correlated, the slices have to be compared und judged
# Option 1: determine an apparent diffusion time by the half height of the curves

# this did not work due to the noise in the curves at low correlation times
# def select_by_half_height_time(
#         correlation_amplitudes: typing.List[np.ndarray],
#         number_of_stdev: float = 1.0
# ) -> typing.List[int]:
#     #  calculate mean of the first correlation points
#     #  use these correlation points as the initial amplitude
#     initial_amplitudes = np.mean(correlation_amplitudes[:, 5:30], axis=1)
#     # get the baseline of the correlation amplitudes
#     offsets = np.array(correlation_amplitudes[:, -1])
#     # calculate the average between initial_amplitude and offset
#
#     # find the indices of half the initial amplitude searching
#     # backwards from the offset to half the maximum
#     i_half_max = list()
#     for correlation_amplitude, initial_amplitude, offset in zip(
#             correlation_amplitudes,
#             initial_amplitudes,
#             offsets
#     ):
#         imaxhalf = np.max(
#             np.nonzero(
#                 correlation_amplitude > (initial_amplitude - offset) / 2 + offset
#             )
#         )
#         i_half_max.append(imaxhalf)
#
#     # calculate mean and stdev of i_half_max
#     mean_half_max = np.mean(i_half_max)
#     stdev_half_max = number_of_stdev * np.std(i_half_max)
#
#     # filter curves for those within 1 sigma
#     selected_curves_idx = list()
#     for i, hm in enumerate(i_half_max):
#         if mean_half_max - stdev_half_max < hm < mean_half_max + stdev_half_max:
#             selected_curves_idx.append(i)
#     print("Total number of curves: ", len(i_half_max))
#     print("Selected curves: ", len(selected_curves_idx))
#     return selected_curves_idx

# weights based on countrate and measurement duration, Suren-style
# empirically developed, performed best in comparison to either weighting methods
# tries to balance uncertainties from photon noise & measurement noise
# not used so far
# def fcs_weights(
#         t: np.ndarray,
#         g: np.ndarray,
#         dur: float,
#         cr: float
# ):
#     """
#     :param t: correlation times [ms]
#     :param g: correlation amplitude
#     :param dur: measurement duration [s]
#     :param cr: count-rate [kHz]
#     """
#     dt = np.diff(t)
#     dt = np.hstack([dt, dt[-1]])
#     ns = dur * 1000.0 / dt
#     na = dt * cr
#     syn = (t < 10) + (t >= 10) * 10 ** (-np.log(t + 1e-12) / np.log(10) + 1)
#     b = np.mean(g[1:5]) - 1
#     imaxhalf = np.min(np.nonzero(g < b / 2 + 1))
#     tmaxhalf = t[imaxhalf]
#     A = np.exp(-2 * dt / tmaxhalf)
#     B = np.exp(-2 * t / tmaxhalf)
#     m = t / dt
#     S = (b * b / ns * ((1 + A) * (1 + B) + 2 * m * (1 - A) * B) / (1 - A) + 2 * b / ns / na * (1 + B) + (1 + b * np.sqrt(B)) / (ns * na * na)) * syn
#     S = np.abs(S)
#     return 1. / np.sqrt(S)

