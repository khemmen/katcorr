# Autocorrelation of selected events from single detector channel
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
        time_window_size_seconds: float = 10.0,
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
    print("Time window size [sec]: ", time_window_size_seconds)
    time_window_size_idx = int(time_window_size_seconds / macro_time_calibration * 1000.0)
    # time 1000 to bring macro_time_calibration from ms to sec
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

# correlation of the 10 sec-pieces
def correlate_pieces(
        macro_times: np.ndarray,
        indices_ch1: typing.List[np.ndarray],
        indices_ch2: typing.List[np.ndarray],
        B: int = 9,
        n_casc: int = 25
) -> np.ndarray:
    print("Correlating pieces...")
    n_correlations = (len(indices_ch1))
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

# calculates for each curve the average within a certain time range
# usually this time range (comparison_start, comparison_stop) encompasses the diffusion time range
# the calculated value is compared to the mean of the first N curves
def calculate_deviation(
        correlation_amplitudes: np.ndarray,
        comparison_start: int = 120,
        comparison_stop: int = 180
) -> typing.List[float]:
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
    logdev = np.log10(ds)  # better would be to use semilogy(x, ds), what would be x?
    p.plot(logdev)
    p.show()
    return deviation

# based on the above calculated deviations from the mean of the first curves
# now the curves which will be used for further analysis are selected
# saved/returned are the indices of those selected curves
def select_by_deviation(
        deviations: typing.List[float],
        d_max: float = 2e-5,
) -> typing.List[int]:
    print("Selecting indices of curves by deviations.")
    devs = deviations[0]
    selected_curves_idx = list()
    for i, d in enumerate(devs):
        if d < d_max:
            selected_curves_idx.append(i)
    print("Total number of curves: ", len(devs))
    print("Selected curves: ", len(selected_curves_idx))
    return selected_curves_idx

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

green_1_indices = data.get_selection_by_channel(np.array([2]))
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

correlation_curves = correlate_pieces(
    macro_times=macro_times,
    indices_ch1=indices_ch1,
    indices_ch2=indices_ch2
)

correlation_amplitudes = correlation_curves[:, 1, :]
average_correlation_amplitude = correlation_amplitudes.mean(axis=0)

# adjust comparison_start & stop according to your diffusion time
# the selected values here encompass 1 ms -> 100 ms
deviation_from_mean = calculate_deviation(
    correlation_amplitudes=correlation_amplitudes,
    comparison_start=120,
    comparison_stop=180
)

# select the curves with a small enough deviation to be considered in the further analysis
selected_curves_idx = select_by_deviation(
    deviations=deviation_from_mean,
    d_max=2e-5
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

p.plot(avg_countrate_ch1)
p.show()
p.semilogx(time_axis, avg_correlation_amplitude)
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
