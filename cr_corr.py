# Copied from tttrlib webpage as starting point

import pylab as p
import tttrlib
import numpy as np

fig, ax = p.subplots(nrows=1, ncols=2)

#  Read the data data

data = tttrlib.TTTR('1_20min_1.ptu', 'PTU')

# Create correlator
B = 9
n_casc = 25

correlator = tttrlib.Correlator()
correlator.set_n_bins(B)
correlator.set_n_casc(n_casc)


# Select the green channels (channel number 0 and 8)

ch1_indeces = data.get_selection_by_channel(np.array([0]))
ch2_indeces = data.get_selection_by_channel(np.array([2]))

mt = data.get_macro_time()

t1 = mt[ch1_indeces]
w1 = np.zeros_like(t1, dtype=np.float)
cr_selection = tttrlib.selection_by_count_rate(t1, 1200000, 300)
w1[cr_selection] += 1.0

t2 = mt[ch2_indeces]
w2 = np.zeros_like(t2, dtype=np.float)
cr_selection = tttrlib.selection_by_count_rate(t2, 1200000, 300)
w2[cr_selection] += 1.0

correlator.set_events(t1, w1, t2, w2)
correlator.run()

x = correlator.get_x_axis_normalized()
y = correlator.get_corr_normalized()

ax[0].semilogx(x, y)


p.show()