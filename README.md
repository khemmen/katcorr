# katcorr
## Katherinas correlator

The katcorr is a collection of scripts and correlators that are inspired by
the paper Ries...Schwille 2010 Optics Express "Automated suppression of
sample-related artifacts in FCS".

This scripts have different scopes and use the library [tttrlib](https://github.com/Fluorescence-Tools/tttrlib) 
to read and correlate time-tagged time resolved (TTTR) files.


### functions & functionsPIE
These two scripts define the functions called in the correlating scripts:
+ get_indices_of_time_windows
+ calculate_countrate
+ correlate
+ correlate_pieces
+ calculate_deviation
+ select_by_deviation

### full_correlation
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

If you are correlating two different colors instead of two channels with different polarization,
you can remove the anisotropy calculation or maybe replace it with an apparent FRET efficiency
or FD/FA ratio calculation.

### Autocorrelation
Use this script, when you are interested in the autocorrelation of a single 
channel only. When the input is a cw-mode measured sample, the microtimes are
ignored and only the macrotimes are used for correlation. Selection
procedures works as above.

### Correlation
Same functionality as the full_correlation except that only macrotimes are used 
as correlation. When the input is a cw-mode measured sample, the microtimes
are ignored.

### PIEcorrelation
Same functionality as the correlation except that the microtimes are used to sort the photons
into "prompt" and "delay" windows of the Pulsed Interleaved Excitation.
Correlations made are "prompt-delay" of the green to red channel, "prompt-prompt" and "delay-delay"
autocorrelation of the green and red channels, respectively.


### Countrate
This script slices your data in user-defined pieces of e. g. several seconds 
and calculates the average countrate (in counts per second) for each of the
two channels. Remember to define your parallel and perpendicular channel 
correctly for steady-state anisotropy calculation. Or you might modify it to calculate an apparent FRET-efficiency
or FD/FA ratio.

## References

1.  Ries J, Bayer M, Csucs G, Dirkx R, Solimena M, Ewers H, Schwille P, 
Automated suppression of sample-related artifacts in Fluorescence Correlation 
Spectroscopy . Optics Express Vol. 18, Issue 11, pp. 11073-11082 (2010)

