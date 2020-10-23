import tttrlib
import pylab as p
import numpy as np

########################################################
#  Data input & reading
########################################################

data = tttrlib.TTTR('A488_1.ptu', 'PTU')
header = data.get_header()
macro_time_calibration = header.macro_time_resolution  # unit nanoseconds
micro_times = data.get_micro_time()
micro_time_resolution = header.micro_time_resolution
jordi = False

########################################################
#  Data rebinning (native resolution often too high, 16-32 ps sufficient)
########################################################

binning = 4  # Binning factor
# This is the max nr of bins the data should contain:
expected_nr_of_bins = int(macro_time_calibration//micro_time_resolution)
# After binning the nr of bins is reduced:
binned_nr_of_bins = int(expected_nr_of_bins//binning)

########################################################
#  Histogram creation
########################################################

# Select the channels & get the respective microtimes
green_s_indices = np.array(data.get_selection_by_channel([0]), dtype=np.int64)
green_p_indices = np.array(data.get_selection_by_channel([2]), dtype=np.int64)

green_s = micro_times[green_s_indices]
green_p = micro_times[green_p_indices]

# Build the histograms
green_s_counts = np.bincount(green_s // binning, minlength=binned_nr_of_bins)
green_p_counts = np.bincount(green_p // binning, minlength=binned_nr_of_bins)

#  observed problem: data contains more bins than possible, rounding errors?
#  cut down to expected length:
green_s_counts_cut = green_s_counts[0:binned_nr_of_bins:]
green_p_counts_cut = green_p_counts[0:binned_nr_of_bins:]

# Build the time axis
dt = header.micro_time_resolution
x_axis = np.arange(green_s_counts_cut.shape[0]) * dt * binning  # identical for data from same time window

########################################################
#  Saving & plotting
########################################################

output_filename = 'Decay_s.txt'
np.savetxt(
    output_filename,
    np.vstack([x_axis, green_s_counts_cut]).T
)

output_filename = 'Decay_p.txt'
np.savetxt(
    output_filename,
    np.vstack([x_axis, green_p_counts_cut]).T
)

p.semilogy(x_axis, green_s_counts_cut, label='gs')
p.semilogy(x_axis, green_p_counts_cut, label='gp')

p.xlabel('time [ns]')
p.ylabel('Counts')
p.legend()
p.savefig("Decay.svg", dpi=150)
p.show()

# Optional: jordi format for direct reading in FitMachine & ChiSurf(2015-2017)
if jordi:
    jordi_counts_green = np.concatenate([green_s_counts_cut, green_p_counts_cut])

    output_filename = 'Jordi.txt'
    np.savetxt(
        output_filename,
        np.vstack([jordi_counts_green]).T
    )
