from __future__ import annotations
import typing

import numpy as np
import numba as nb
import pylab as p
import tttrlib

########################################################
#  Definition of required functions
########################################################

# jit = just in time compiler, compiles code before execution to speed up algorithm
@nb.jit(
    nopython=True
)

def get_indices_of_time_windows(
        macro_times: np.ndarray,
        selected_indices: np.ndarray,
        macro_time_calibration: float,
        time_window_size_seconds: float = 2.0,
) -> typing.List[np.ndarray]:
    """Determines a list of start and stop indices for a TTTR object with
    selected indices and that correspond to the indices of the start and stop
    of time-windows.
    - Slices the full trace of the data into pieces of seconds
    - change the number after time_window_size_seconds to slice data in larger pieces
    :param macro_times: numpy array of macro times
    :param macro_time_calibration: the macro time clock in milliseconds
    :param selected_indices: A preselected list of indices that defines which events
    in the TTTR event stream are considered
    :param time_window_size_seconds: The size of the time windows
    :return: list of arrays, where each array contains the indices of detection events for a time window
    """
    print("Getting indices of time windows")
    print("Time window size [sec]: ", time_window_size_seconds)
    time_window_size_idx = int(time_window_size_seconds / macro_time_calibration * 1000.0)
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

# correlation of the full trace
def correlate(
        macro_times: np.ndarray,
        indices_ch1: np.ndarray,
        indices_ch2: np.ndarray,
        B: int = 9,
        n_casc: int = 37,
        micro_times: np.ndarray = None,
        micro_time_resolution: float = None,
        macro_time_clock: float = None
) -> (np.ndarray, np.ndarray):
    """actual correlator

    :param macro_times: numpy array of macro times
    :param indices_ch1: numpy array of indices based on the selected indices for the first channel
    :param indices_ch2: numpy array of indices based on the selected indices for the second channel
    :param B: Base of the logarithmic correlation axis
    :param n_casc: nr of cascades of the logarithmic time axis, increase for longer correlation times
    :param micro_times: numpy array of micro times
    :param micro_time_resolution: micro time resolution in ns
    :param macro_time_clock: macro time clock in in ns
    :return: list of two arrays (time, correlation amplitude)
    """
    # macro_time_clock in nanoseconds
    if micro_times is not None:
        n_micro_times = int(macro_time_clock / micro_time_resolution)
        times = macro_times * n_micro_times + micro_times
        time_factor = micro_time_resolution / 1e6
    else:
        times = macro_times
        time_factor = macro_time_clock / 1e6
    correlator = tttrlib.Correlator()
    correlator.set_n_bins(B)
    correlator.set_n_casc(n_casc)
    # Select the green channels (channel number 1 and 2)
    # w1 & w2 are weights, which will be 1 by default if not defined elsewhere
    # use w1 & w2 e.g. for filtered FCS or when only selected events should be correlated
    t1 = times[indices_ch1]
    w1 = np.ones_like(t1, dtype=np.float)
    t2 = times[indices_ch2]
    w2 = np.ones_like(t2, dtype=np.float)
    correlator.set_events(t1, w1, t2, w2)
    correlator.run()
    x = correlator.get_x_axis_normalized()
    y = correlator.get_corr_normalized()
    t = x * time_factor
    # p.semilogx(t, y)
    # p.show()
    return t, y

def correlate_pieces(
        macro_times: np.ndarray,
        indices_ch1: typing.List[np.ndarray],
        indices_ch2: typing.List[np.ndarray],
        B: int = 9,
        n_casc: int = 25,
        micro_times: np.ndarray = None,
        micro_time_resolution: float = None,
        macro_time_clock: float = None
) -> np.ndarray:
    """ times slices are selected one after another based on the selected indices
        and then transferred to the correlator

        :param macro_times: numpy array of macro times
        :param indices_ch1: numpy array of indices based on the selected indices for the first channel
        :param indices_ch2: numpy array of indices based on the selected indices for the second channel
        :param B: Base of the logarithmic correlation axis
        :param n_casc: nr of cascades of the logarithmic time axis, increase for longer correlation times
        :return: array of correlation curves (y-values), which are then transferred to the correlator
        """
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
            n_casc=n_casc,
            micro_time_resolution=micro_time_resolution,
            micro_times=micro_times,
            macro_time_clock=macro_time_clock
        )
        correlation_curves.append([x, y])
    correlation_curves = np.array(
        correlation_curves
    )
    return correlation_curves

def calculate_deviation(
        correlation_amplitudes: np.ndarray,
        comparison_start: int = 220,
        comparison_stop: int = 280
) -> typing.List[float]:
    """Determines the similarity of the individual curves towards the first n curves
        The values of each correlation amplitude are averaged over a time range defined by start and stop
        This time range usually encompasses the diffusion time, i.e. is sample-specific
        The calculated average is compared to the mean of the first n curves

        :param correlation_amplitudes: array of correlation amplitudes
        :param comparison_start: index within the array of correlation amplitude which marks the start of comparison range
        :param comparison_stop: index within the array of correlation amplitude which marks the end of comparison range
        :return: list of deviations calculated as difference to the starting amplitudes
        """
    print("Calculating deviations.")
    print("Comparison time range:", comparison_start, "-", comparison_stop)
    deviation = list()
    n = 5  # select the time range/nr of curves to which the similarity comparison should be made
    print("compared to the first", n, "curves")
    # calculates from every curve the difference to the mean of the first N curves, squares this value
    # and divides this value by the number of curves
    if (comparison_start is None) or (comparison_stop is None):
        ds = np.mean(
            (correlation_amplitudes - correlation_amplitudes[:n].mean(axis=0))**2.0, axis=1
        ) / len(correlation_amplitudes)
        deviation.append(ds)
    else:
        ca = correlation_amplitudes[:, comparison_start: comparison_stop]
        ds = np.mean(
            (ca - ca[:n].mean(axis=0)) ** 2.0, axis=1
        ) / (comparison_stop - comparison_start)
        deviation.append(ds)
    # print(ds)
    logdev = np.log10(ds)
    p.plot(logdev)
    p.show()
    return deviation

def select_by_deviation(
        deviations: typing.List[float],
        d_max: float = 2e-5,
) -> typing.List[int]:
    """ The single correlated time windows are now selected for further analysis based on their deviation
        to the first n curves

        :param deviations: list of deviations, calculated in the calculate_deviation function
        :param d_max: threshold, all curves which have a deviation value smaller than this are selected for further analysis
        :return: list of indices, the indices corresponds to the curves' number/time window
        """
    print("Selecting indices of curves by deviations.")
    devs = deviations[0]
    selected_curves_idx = list()
    for i, d in enumerate(devs):
        if d < d_max:
            selected_curves_idx.append(i)
    print("Total number of curves: ", len(devs))
    print("Selected curves: ", len(selected_curves_idx))
    return selected_curves_idx

def calculate_countrate(
        timewindows: typing.List[np.ndarray],
        time_window_size_seconds: float = 2.0,
) -> List[float]:
    """based on the sliced timewindows the average countrate for each slice is calculated

        :param timewindows: list of numpy arrays, the indices which have been returned from getting_indices_of_time_windows
        :param time_window_size_seconds: The size of the time windows
        :return: list of average countrate (counts/sec) for the individual time windows
        """
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
#  Here the actual data input & optional selection process starts
########################################################

data = tttrlib.TTTR('1_20min_1.ptu', 'PTU')
# rep rate = 80 MHz
header = data.get_header()
macro_time_calibration_ns = header.macro_time_resolution  # unit nanoseconds
macro_time_calibration_ms = macro_time_calibration_ns / 1e6  # macro time calibration in milliseconds
macro_times = data.get_macro_time()
micro_times = data.get_micro_time()
time_window_size = 60.0  # time window size in seconds (overwrites selection above)
micro_time_resolution = header.micro_time_resolution

green_s_indices = data.get_selection_by_channel(np.array([0]))
indices_ch1 = get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_s_indices,
    macro_time_calibration=macro_time_calibration_ms,
    time_window_size_seconds=time_window_size
)

green_p_indices = data.get_selection_by_channel(np.array([2]))
indices_ch2 = get_indices_of_time_windows(
    macro_times=macro_times,
    selected_indices=green_p_indices,
    macro_time_calibration=macro_time_calibration_ms,
    time_window_size_seconds=time_window_size
)

correlation_curves_fine = correlate_pieces(
    macro_times=macro_times,
    indices_ch1=indices_ch1,
    indices_ch2=indices_ch2,
    micro_times=micro_times,
    micro_time_resolution=micro_time_resolution,
    macro_time_clock=macro_time_calibration_ns,
    n_casc=37
)

########################################################
#  Option: get autocorrelation curves
########################################################

autocorr_curve_ch1 = correlate_pieces(
    macro_times=macro_times,
    indices_ch1=indices_ch1,
    indices_ch2=indices_ch1,
    micro_times=None,
    micro_time_resolution=None,
    macro_time_clock=macro_time_calibration_ns,
    n_casc=25
)

autocorr_curve_ch2 = correlate_pieces(
    macro_times=macro_times,
    indices_ch1=indices_ch2,
    indices_ch2=indices_ch2,
    micro_times=None,
    micro_time_resolution=None,
    macro_time_clock=macro_time_calibration_ns,
    n_casc=25
)

########################################################
#  Option: get average count rate per slice
########################################################
avg_countrate_ch1 = calculate_countrate(
    timewindows=indices_ch1,
    time_window_size_seconds=time_window_size
)

avg_countrate_ch2 = calculate_countrate(
    timewindows=indices_ch2,
    time_window_size_seconds=time_window_size
)

# comparison is only made for the crosscorrelation curves
# autocorrelation curves are calculated based on the curve_ids selected by crosscorr
correlation_amplitudes = correlation_curves_fine[:, 1, :]
average_correlation_amplitude = correlation_amplitudes.mean(axis=0)

# adjust comparison_start & stop according to your diffusion time
# the selected values here encompass 1 ms -> 100 ms
deviation_from_mean = calculate_deviation(
    correlation_amplitudes=correlation_amplitudes,
    comparison_start=220,
    comparison_stop=280
)

# select the curves with a small enough deviation to be considered in the further analysis
selected_curves_idx = select_by_deviation(
    deviations=deviation_from_mean,
    d_max=2e-5
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
        correlation_curves_fine[curve_idx]
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
time_axis = avg_curve[0]  # calculates the correct time axis by multiplication of x-axis with macro_time
time_axis_acf = avg_curve_ch1[0]
avg_correlation_amplitude = avg_curve[1]  # 2nd column contains the average correlation amplitude calculated above
avg_correlation_amplitude_ch1 = avg_curve_ch1[1]
avg_correlation_amplitude_ch2 = avg_curve_ch2[1]
suren_column = np.zeros_like(time_axis)  # fill 3rd column with 0's for compatibility with ChiSurf
suren_column_acf = np.zeros_like(time_axis_acf)
std_avg_correlation_amplitude = std_curve[1]/np.sqrt(len(selected_curves))
std_avg_correlation_amplitude_ch1 = std_curve_ch1[1]/np.sqrt(len(selected_curves))
std_avg_correlation_amplitude_ch2 = std_curve_ch2[1]/np.sqrt(len(selected_curves))
# 4th column contains standard deviation from the average curve calculated above
filename_cc = '60s_ch0_ch2_cross.cor'  # change file name!
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

filename_acf1 = '60s_ch0_auto.cor'  # change file name!
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

filename_acf2 = '60s_ch2_auto.cor'  # change file name!
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

g_factor = 0.8  # please change according to your setup calibration
total_countrate = np.array(avg_countrate_ch2) + np.array(avg_countrate_ch2)
parallel_channel = np.array(avg_countrate_ch2)
perpendicular_channel = np.array(avg_countrate_ch1)
rss = (parallel_channel - g_factor * perpendicular_channel)/(parallel_channel + 2 * g_factor * perpendicular_channel)

filename = 'avg_countrate.txt'  # change file name!
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

p.semilogx(time_axis, avg_correlation_amplitude)
p.semilogx(time_axis_acf, avg_correlation_amplitude_ch1)
p.semilogx(time_axis_acf, avg_correlation_amplitude_ch2)
p.show()

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

# ########################################################
# #  Option: Correlate again with photons of selected curves
# ########################################################
#
# # get indices of events which belong to selected curves
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
# ########################################################
# #  Option: Select by half height instead of averaged diffusion part
# ########################################################
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

# ########################################################
# #  Option: Introduce weights while correlating
# ########################################################
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
#     Suren-way of weighting (when compared to other methods of weighting this was advantageous)
#     defined by try-and-error
#     S = (b * b / ns * ((1 + A) * (1 + B) + 2 * m * (1 - A) * B) / (1 - A) + 2 * b / ns / na * (1 + B) + (1 + b * np.sqrt(B)) / (ns * na * na)) * syn
#     S = np.abs(S)
#     return 1. / np.sqrt(S)
