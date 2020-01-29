# katcorr
## Katherinas correlator

The katcorr is a collection of scripts and correlators that are inspired by
the paper Ries...Schwille 2010 Optics Express "Automated suppression of
sample-related artifacts in FCS".

This scripts have different scopes use the library [tttrlib](https://github.com/Fluorescence-Tools/tttrlib) 
to read and correlate time-tagged time resolved (TTTR) files.


### full_correlator
Input files are two channel FCS data which was recorded in continuous-wave mode.
Many parameter are automatically read from the header of measurement data, 
others you have to give based on your preferences and/or the data properties.
It slices the full trace into user-defined pieces of e.g. several seconds and 
correlates this pieces. For each slice in the next step its similarity to the
first (n) slices is calculated based on a predefined correlation time range.
Here: I adjusted the time range (correlation_start, correlation_stop) such that 
the diffusion range of membrane protein is averaged and compared. 

Note: If you change the correlating options (base B and number of cascades
ncasc), you have to adjust the comparison range accordingly.
Next, based on the calculated deviations the curves (i.e. respective time 
slices of your trace) are selected, averaged and saved.
Finally, also the mean countrates, steady-state anisotropy and deviation per 
slice is saved. Remember to define your parallel and perpendicular channel
correctly for steady-state anisotropy calculation.

### Autocorrelation
Use this script, when you are interested in the autocorrelation of a single 
channel only. When the input is a cw-mode measured sample, the microtimes are
ignored and only the macrotimes are used for correlation. Selection
procedures works as above.

### Correlator
Same functionality as the full_correlator except that only macrotimes are used 
as correlation. When the input is a cw-mode measured sample, the microtimes
are ignored.

### Countrate
This script slices your data in user-defined pieces of e. g. several seconds 
and calculates the average countrate (in counts per second) for each of the
two channels. Remember to define your parallel and perpendicular channel 
correctly for steady-state anisotropy calculation.

## References

1.  Ries J, Bayer M, Csucs G, Dirkx R, Solimena M, Ewers H, Schwille P, 
Automated suppression of sample-related artifacts in Fluorescence Correlation 
Spectroscopy . Optics Express Vol. 18, Issue 11, pp. 11073-11082 (2010)

