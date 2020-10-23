import tttrlib
import pylab as p
import numpy as np

########################################################
#  Data input & reading
########################################################

data = tttrlib.TTTR('DNA10bpPIE.ptu', 'PTU')
header = data.get_header()
macro_time_calibration = header.macro_time_resolution  # unit nanoseconds
micro_times = data.get_micro_time()
micro_time_resolution = header.micro_time_resolution
jordi = True

########################################################
#  Data rebinning (native resolution often too high, 16-32 ps sufficient)
########################################################
binning = 2  # Binning factor
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
red_s_indices = np.array(data.get_selection_by_channel([1]), dtype=np.int64)
red_p_indices = np.array(data.get_selection_by_channel([3]), dtype=np.int64)

green_s = micro_times[green_s_indices]
green_p = micro_times[green_p_indices]
red_s = micro_times[red_s_indices]
red_p = micro_times[red_p_indices]

# Generate PIE weights
w_prompt_green_s = np.ones_like(green_s, dtype=np.int64)
w_prompt_green_s[np.where(green_s > expected_nr_of_bins//2)] *= 0
w_prompt_green_p = np.ones_like(green_p, dtype=np.int64)
w_prompt_green_p[np.where(green_p > expected_nr_of_bins//2)] *= 0
w_prompt_red_s = np.ones_like(red_s, dtype=np.int64)
w_prompt_red_s[np.where(red_s > expected_nr_of_bins//2)] *= 0
w_prompt_red_p = np.ones_like(red_p, dtype=np.int64)
w_prompt_red_p[np.where(red_p > expected_nr_of_bins//2)] *= 0
w_delay_red_s = np.ones_like(red_s, dtype=np.int64)
w_delay_red_s[np.where(red_s < expected_nr_of_bins//2)] *= 0
w_delay_red_p = np.ones_like(red_p, dtype=np.int64)
w_delay_red_p[np.where(red_p < expected_nr_of_bins//2)] *= 0

# Build the histograms
green_s_counts = np.bincount(green_s // binning, weights=w_prompt_green_s, minlength=binned_nr_of_bins)
green_p_counts = np.bincount(green_p // binning, weights=w_prompt_green_p, minlength=binned_nr_of_bins)
red_s_counts_prompt = np.bincount(red_s // binning, weights=w_prompt_red_s,  minlength=binned_nr_of_bins)
red_p_counts_prompt = np.bincount(red_p // binning, weights= w_prompt_red_p, minlength=binned_nr_of_bins)
red_s_counts_delay = np.bincount(red_s // binning, weights=w_delay_red_s, minlength=binned_nr_of_bins)
red_p_counts_delay = np.bincount(red_p // binning, weights=w_delay_red_p, minlength=binned_nr_of_bins)

#  observed problem: data contains more bins than possible, rounding errors?
#  cut down to expected length:
green_s_counts_cut = green_s_counts[0:binned_nr_of_bins//2:]
green_p_counts_cut = green_p_counts[0:binned_nr_of_bins//2:]
red_s_counts_cut_prompt = red_s_counts_prompt[0:binned_nr_of_bins//2:]
red_p_counts_cut_prompt = red_p_counts_prompt[0:binned_nr_of_bins//2:]
red_s_counts_cut_delay = red_s_counts_delay[binned_nr_of_bins//2:binned_nr_of_bins:]
red_p_counts_cut_delay = red_p_counts_delay[binned_nr_of_bins//2:binned_nr_of_bins:]

# Build the time axis
dt = header.micro_time_resolution
x_axis = np.arange(green_s_counts.shape[0]) * dt * binning  # identical for data from same time window
x_axis_prompt = x_axis[0:binned_nr_of_bins//2:]
x_axis_delay = x_axis[binned_nr_of_bins//2::]

########################################################
#  Saving & plotting
########################################################
output_filename = 'Decay_s_green.txt'
np.savetxt(
    output_filename,
    np.vstack([x_axis_prompt, green_s_counts_cut]).T
)

output_filename = 'Decay_p_green.txt'
np.savetxt(
    output_filename,
    np.vstack([x_axis_prompt, green_p_counts_cut]).T
)

output_filename = 'Decay_s_red_prompt.txt'
np.savetxt(
    output_filename,
    np.vstack([x_axis_prompt, red_s_counts_cut_prompt]).T
)

output_filename = 'Decay_p_red_prompt.txt'
np.savetxt(
    output_filename,
    np.vstack([x_axis_prompt, red_p_counts_cut_prompt]).T
)

output_filename = 'Decay_s_red_delay.txt'
np.savetxt(
    output_filename,
    np.vstack([x_axis_delay, red_s_counts_cut_delay]).T
)

output_filename = 'Decay_p_red_delay.txt'
np.savetxt(
    output_filename,
    np.vstack([x_axis_delay, red_p_counts_cut_delay]).T
)

p.semilogy(x_axis_prompt, green_s_counts_cut, label='gs')
p.semilogy(x_axis_prompt, green_p_counts_cut, label='gp')
p.semilogy(x_axis_prompt, red_s_counts_cut_prompt, label='rs(prompt)')
p.semilogy(x_axis_prompt, red_p_counts_cut_prompt, label='rp(prompt)')
p.semilogy(x_axis_delay, red_s_counts_cut_delay, label='rs(delay)')
p.semilogy(x_axis_delay, red_p_counts_cut_delay, label='rp(delay)')

p.xlabel('time [ns]')
p.ylabel('Counts')
p.legend()
p.savefig("Decay.svg", dpi=150)
p.show()

# Optional: jordi format for direct reading in FitMachine & ChiSurf(2015-2017)
if jordi:
    jordi_counts_green = np.concatenate([green_s_counts_cut, green_p_counts_cut])
    jordi_counts_red_prompt = np.concatenate([red_s_counts_cut_prompt, red_p_counts_cut_prompt])
    jordi_counts_red_delay = np.concatenate([red_s_counts_cut_delay, red_p_counts_cut_delay])

    output_filename = 'Jordi_green.txt'
    np.savetxt(
        output_filename,
        np.vstack([jordi_counts_green]).T
    )

    output_filename = 'Jordi_red_prompt.txt'
    np.savetxt(
        output_filename,
        np.vstack([jordi_counts_red_prompt]).T
    )

    output_filename = 'Jordi_red_delay.txt'
    np.savetxt(
        output_filename,
        np.vstack([jordi_counts_red_delay]).T
    )


