# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import astropy as ap
import astropy.units
from astropy.table import Table
import astropy.constants as apc
import time
import sys
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sqlalchemy
import sqlalchemy.ext.declarative
import sqlalchemy.orm
import sqlalchemy.orm.query
import matplotlib
from sqlalchemy import and_, or_
from astropy import units as u
from astropy.coordinates import Angle

seconds_per_day = 60*60*24
batch_size = 1e8


# ==============================================================================
def calculate_FWHM(X,Y):
    """Calculate FWHM from arrays"""
    # Taken from http://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
    # Create 'difference' array by subtracting half maximum
    d = Y - (np.amax(Y) / 2)
    # Find the points where the difference is positive
    indexes = np.where(d>0)[0]
    # The first and last positive points are the edges of the peak
    return abs(X[indexes[-1]] - X[indexes[0]])

def calculate_centroid(X,Y, bounds=None):
    """Returns the centroid position, with optional flux interval"""
    bins = X
    vals = Y
    print('bins',bins)
    print('vals',vals)
    centroid_total = np.sum(vals)
    centroid_position = np.sum(np.multiply(bins,vals))/centroid_total

    bounds = None #TEMPORARY FIX

    if bounds is not None:
        bound_width = bounds/2
        bound_min = -1
        bound_max = -1
        value_total = 0
        for index, value in enumerate(vals):
            value_total += value
            if value_total/centroid_total >= 0.5+bound_width:
                bound_max = bins[index]
                break
        value_total = centroid_total
        for index,value in enumerate(vals[::-1]):
            value_total -= value
            if value_total/centroid_total <= 0.5-bound_width:
                bound_min = bins[len(bins)-1-index]
                break
        print(centroid_position, bound_min, bound_max)
        return centroid_position, bound_min, bound_max
    else:
        print(centroid_position)
        return centroid_position

def calculate_modal_value(X,Y):
    """Find the modal delay"""
    return X[np.argmax(Y)]

def calculate_midpoints(X):
    X_midp = np.zeros(shape=len(X)-1)
    for i in range(0,len(X)-1):
        X_midp[i] = (X[i] + X[i+1])/ 2
    return X_midp

# ==============================================================================
def calculate_delay(angle, phase, radius, timescale):
    """Delay relative to continuum observed at angle for emission at radius"""
    # Calculate delay compared to continuum photons
    # Draw plane at r_rad_min out. Find x projection of disk position.
    # Calculate distance travelled to that plane from the current disk position
    # Delay relative to continuum is thus (distance from centre to plane)
    # + distance from centre to point
    # vr_disk     = np.array([r_rad.value*np.cos(r_phase), 0.0]) * u.m
    vr_disk     = np.array([radius*np.cos(phase), 0.0])
    vr_normal   = np.array([np.cos(angle), np.sin(angle)])
    vr_plane    = radius * vr_normal
    # return (np.dot((vr_plane - vr_disk), vr_normal) * u.m / apc.c.value).to(timescale)
    return (np.dot((vr_plane - vr_disk), vr_normal) / apc.c.value) / seconds_per_day


def keplerian_velocity(mass, radius):
    """Calculates Keplerian velocity at radius"""
    return np.sqrt(ap.constants.G.value * mass / radius)

def path_to_delay(path):
    """Converts path to time delay"""
    return path / apc.c.value

def doppler_shift_wave(line, vel):
    """Converts passed line and velocity into red/blueshifted wavelength"""
    return line * apc.c.value / (apc.c.value - vel)

def doppler_shift_vel(line, wave):
    """Converts passed red/blueshifted wave into velocity"""
    if wave > line:
        return -1*apc.c.value * (1 - (line / wave))
    else:
        return apc.c.value * ((line / wave) - 1)

def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))

# ==============================================================================
class TransferFunction:
    # FAKE INIT
    def __init__(self, database, filename, luminosity=None, dimensions=None, template=None, template_different_line=False, template_different_spectrum=False):
        """Initialises the TF, taking the query it is to execute, the dimensions to bin by, and the delay range bins"""
        assert dimensions is not None or template is not None,\
            "Must provide either dimensions or another TF to copy them from!"
        # self._query = database.query(Photon.Wavelength, Photon.Delay, Photon.Weight, Photon.X, Photon.Z)
        start = time.clock()

        self._database = database
        Session = sqlalchemy.orm.sessionmaker(bind=self._database)
        self._session = Session()
        self._query = self._session.query(Photon.Wavelength, Photon.Delay, Photon.Weight)

        self._delay_dynamic_range = None
        self._velocity = None
        self._line_list = None
        self._line_wave = None
        self._line_num = None
        self._delay_range = None
        self._luminosity=None
        self._filename = filename
        self._dimensions = dimensions
        self._bins_vel = None
        self._bins_wave = None
        self._bins_delay = None
        self._flux = None
        self._flux_w = None
        self._count = None
        self._response_map = None
        self._wave_range = None
        self._spectrum = None

        if template is not None:
            print("Templating '{}' off of '{}'...".format(self._filename, template._filename))
            self._dimensions = template._dimensions
            self._delay_dynamic_range = template._delay_dynamic_range
            self._bins_vel = template._bins_vel
            self._bins_delay = template._bins_delay
            self._luminosity= template._luminosity

            # If we are templating off of the same line, we want the same wavelength bins
            if template_different_line is False:
                self._bins_wave = template._bins_wave
            if template._line_wave is not None and template_different_line is False:
                self.line(template._line_num, template._line_wave)
            if template._velocity is not None:
                self.velocities(template._velocity)
            if template._wave_range is not None and template_different_line is False:
                self.wavelengths(self._wave_range[0], self._wave_range[1])
            if template._line_list is not None and template_different_line is False:
                self.lines(template._line_list)
            if template._spectrum is not None and template_different_spectrum is False:
                self.spectrum(template._spectrum)

        if luminosity is not None:
            self._luminosity=luminosity
        # print("'{}' successfully created ({:.1f}s)".format(self._filename, time.clock()-start))

    def close_query(self):
        self._session.close()
        del(self._query)
        del(self._session)
        Session = sqlalchemy.orm.sessionmaker(bind=self._database)
        self._session = Session()
        self._query = self._session.query(Photon.Wavelength, Photon.Delay, Photon.Weight)
        return(self)

    def spectrum(self, number):
        self._spectrum = number
        self._query = self._query.filter(Photon.Spectrum == number)
        return self
    def line(self, number, wavelength):
        self._line_wave = wavelength
        self._line_num = number
        self._query = self._query.filter(Photon.Resonance == number)
        return self
    def velocities(self, velocity):
        assert self._line_wave is not None,\
            "Cannot limit doppler shift around a line without specifying a line!"
        self._velocity = velocity
        self._query = self._query.filter(Photon.Wavelength >= doppler_shift_wave(self._line_wave, -velocity),
                                         Photon.Wavelength <= doppler_shift_wave(self._line_wave,  velocity))
        return self
    def wavelengths(self, wave_min, wave_max):
        assert wave_min < wave_max,\
            "Minimum wavelength must be lower than maximum wavelength!"
        self._wave_range = [wave_min, wave_max]
        self._query = self._query.filter(Photon.Wavelength >= wave_min, Photon.Wavelength <= wave_max)
        return self
    def lines(self, line_list):
        assert len(lines) > 1,\
            "For a single line, use the 'line()' filter rather than 'lines()'!"
        self._line_list = lines
        self._query = self._query.filter(Photon.Resonance.in_(line_list))
        return self
    def delays(self, delay_min, delay_max, unit='d'):
        assert delay_min < delay_max,\
            "Minimum delay must be below maximum delay!"

        if unit in ['d','D','day','Day','days','Days']:
            self._delay_range = [delay_min * seconds_per_day, delay_max * seconds_per_day]
        else:
            self._delay_range = [delay_min, delay_max]
        self._query=self._query.filter(Photon.Delay > self._delay_range[0], Photon.Delay < self._delay_range[1])
        return self
    def cont_scatters(self, scat_min, scat_max=None):
        if scat_max is not None:
            assert scat_min < scat_max,\
                "Minimum continuum scatters must be below maximum scatters!"
        assert scat_min >= 0,\
            "Must select a positive number of continuum scatters"

        if scat_max is not None:
            self._query = self._query.filter(Photon.ContinuumScatters >= scat_min, Photon.ContinuumScatters <= scat_max)
        else:
            self._query = self._query.filter(Photon.ContinuumScatters == scat_min)
        return self
    def res_scatters(self, scat_min, scat_max=None):
        if scat_max is not None:
            assert scat_min < scat_max,\
                "Minimum resonant scatters must be below maximum scatters!"
        assert scat_min >= 0,\
            "Must select a positive number of resonant scatters"

        if scat_max is not None:
            self._query = self._query.filter(Photon.ResonantScatters >= scat_min, Photon.ResonantScatters <= scat_max)
        else:
            self._query = self._query.filter(Photon.ResonantScatters == scat_min)
        return self

    def filter(self, *args):
        self._query=self._query.filter(args)
        return self

    def response_map_by_tf(self, low_state, high_state, plot=False):
        """Creates a response map from two other transfer functions, to be applied during plotting"""
        # The other two TFs ***must*** have identical bins and both provide ionising luminosity information
        assert self._flux is not None,\
            "You must run the TF query with '.run()' before response mapping it!"
        assert low_state._flux is not None and high_state._flux is not None,\
            "You must run the low and high state TF queries with '.run()' before response mapping using them!"
        assert np.array_equal(self._bins_wave, low_state._bins_wave) and np.array_equal(self._bins_delay, low_state._bins_delay),\
            "Low state TF is binned differently to target TF! Cannot rescale using it."
        assert np.array_equal(self._bins_wave, high_state._bins_wave) and np.array_equal(self._bins_delay, high_state._bins_delay),\
            "High state TF is binned differently to target TF! Cannot rescale using it."
        assert self._luminosity != None,\
            "TF missing continuum luminosity information!"
        assert low_state._luminosity != None,\
            "Low state TF missing continuum luminosity information!"
        assert high_state._luminosity != None,\
            "High state TF missing continuum luminosity information!"
        assert low_state._luminosity <= self._luminosity,\
            "Low state ionising luminosity greater than target TF ionising luminosity!"
        assert high_state._luminosity >= self._luminosity,\
            "High state ionising luminosity lower than target TF ionising luminosity!"

        # If that is true, the map is trivial to construct. We divide the difference in TFs by the luminosity difference
        luminosity_difference = high_state._luminosity - low_state._luminosity
        response_map = ((high_state._flux*high_state._luminosity) - (low_state._flux*low_state._luminosity)) / luminosity_difference
        self._response_map = response_map
        return self

    def calc_mass(self):
        data_time = np.sum(data_plot, 0)
        data_vel = np.sum(data_plot, 1)

        bins_vel_midp = np.zeros(shape=self._dimensions[0])
        for i in range(0,self._dimensions[1]):
            bins_vel_midp[i] = (bins_vel[i] + bins_vel[i+1])/ 2
        bins_time_midp = np.zeros(shape=self._dimensions[1])
        for i in range(0,self._dimensions[1]):
            bins_time_midp[i] = (bins_time[i] + bins_time[i+1])/ 2

        velocity = FWHM(bins_vel_midp, data_vel)/2
        delay = Centroid(bins_time_midp, data_time, threshold=0.8*calculate_modal_delay(bins_time_midp, data_time))
        # WORK IN PROGRESS

    def modal_delay(self, response=False):
        """Calculates the modal delay for the current data"""
        if response:
            return calculate_modal_value(calculate_midpoints(self._bins_delay), np.sum(self._response_map, 1))
        else:
            return calculate_modal_value(calculate_midpoints(self._bins_delay), np.sum(self._flux, 1))
    def centroid_delay(self):
        """Calculates the centroid delay for the current data"""
        return calculate_centroid(calculate_midpoints(self._bins_delay), np.sum(self._flux, 1), bounds=0.9545)

    def run(self, response_map=None, line=None, scaling_factor=1.0, delay_dynamic_range=None, limit=None):
        """Performs a query on the photon DB and bins it"""
        assert response_map is None or line is not None,\
            "Passing a response map but no line information for it!"
        assert response_map is None or self._flux_w is None,\
            "A response map has already been built!"
        assert response_map is None,\
            "Response mapping by location not yet implemented!"
        assert self._flux is None,\
            "TF has already been run!"
        start = time.clock()

        # If we're not in limit mode, fetch all
        data = None
        if limit is None:
            data = np.asarray(self._query.all())
        else:
            print("Limited to {} results...".format(limit))
            data = np.asarray(self._query.limit(limit).all())

        assert len(data) > 0,\
            "No records found!"

        if self._spectrum is not None:
            print("Fetched {} records from '{}' for spectrum {}...".format(len(data), self._filename, self._spectrum))
        else:
            print("Fetched {} records from '{}'...".format(len(data), self._filename))


        # Check if we've already got delay bins from another TF
        if self._bins_delay is None:
            # Data returned as Wavelength, Delay, Weight. Find min and max delays
            if delay_dynamic_range is not None:
                self._delay_dynamic_range = delay_dynamic_range
                range_delay = [0,np.percentile(data[:,1],(1 - (10**(-delay_dynamic_range)))*100)]
                print("Delays up to the {} percentile value, {}d".format((1 - (10**(-delay_dynamic_range)))*100, range_delay[1]/seconds_per_day))
            else:
                range_delay = [0,np.amax(data[:,1])]
            self._bins_delay = np.linspace(range_delay[0], range_delay[1],
                                           self._dimensions[1]+1, endpoint=True, dtype=np.float64)

        # Check if we've already got wavelength bins from another TF
        if self._bins_wave is None:
            # If we have no velocity bins, this is a factory-fresh TF
            if self._bins_vel is None:
                # Data returned as Wavelength, Delay, Weight. Find min and max delays and wavelengths
                range_wave = [np.amin(data[:,0]), np.amax(data[:,0])]
            # If we do have velocity bins, this was templated off a different line and we need to copy the velocities (but bins are in km! not m!)
            else:
                range_wave = [doppler_shift_wave(self._line_wave, self._bins_vel[0]*1000), doppler_shift_wave(self._line_wave, self._bins_vel[-1]*1000)]
                print("Creating new wavelength bins from template, velocities from {:.2e}-{:.2e} to waves: {:.2f}-{:.2f}".format(self._bins_vel[0], self._bins_vel[-1], range_wave[0], range_wave[1]))

            # Now create the bins for each dimension
            self._bins_wave  = np.linspace(range_wave[0], range_wave[1],
                                           self._dimensions[0]+1, endpoint=True, dtype=np.float64)

        # Check if we've already got velocity bins from another TF
        if self._bins_vel is None:
            # If this is a line-based TF, we can set velocity bins up
            if self._line_wave is not None:
                self._bins_vel  = np.linspace(doppler_shift_vel(self._line_wave, range_wave[1]),
                                    doppler_shift_vel(self._line_wave, range_wave[0]),
                                    self._dimensions[0]+1, endpoint=True, dtype=np.float64)
                # Convert speed from m/s to km/s
                self._bins_vel = np.true_divide(self._bins_vel, 1000.0)


        # Now we bin the photons, weighting them by their photon weights for the luminosity
        self._flux, junk, junk = np.histogram2d(data[:,1], data[:,0], weights=data[:,2],
                                                    bins=[self._bins_delay, self._bins_wave])
        # Keep an unweighted photon count for statistical error purposes
        self._count, junk, junk = np.histogram2d(data[:,1], data[:,0],
                                                    bins=[self._bins_delay, self._bins_wave])


        # Scaling factor! Each spectral cycle outputs L photons. If we do 50 cycles, we want a factor of 1/50
        self._flux *= scaling_factor
        # Scale to continuum luminosity
        self._flux /= self._luminosity

        print("'{}' successfully run ({:.1f}s)".format(self._filename,time.clock()-start))
        # Make absolutely sure this data is wiped as it's *HUGE*
        del(data)
        return self

    def flux(self, wave, delay):
        """Returns the photon luminosity in this bin"""
        return(self._flux[np.searchsorted(self._bins_delay, delay),
                          np.searchsorted(self._bins_wave, wave)])
    def count(self, wave, delay):
        """Returns the photon count in this bin"""
        return(self._count[np.searchsorted(self._bins_delay, delay),
                           np.searchsorted(self._bins_wave, wave)])

    def plot(self, log=False, normalised=False, rescaled=False, velocity=False, name=None, days=True,
            response_map=False, keplerian=None, dynamic_range=None):
        """Takes the data gathered by calling 'run' and outputs a plot"""
        assert response_map is False or self._response_map is not None,\
            "No data available for response map!"
        assert log is False or response_map is False,\
            "Cannot plot a logarithmic response map!"
        assert normalised is False or rescaled is False,\
            "Cannot be both normalised and rescaled!"
        assert self._bins_wave is not None,\
            "You must run the TF query with '.run()' before plotting it!"

        start = time.clock()
        if name is not None:
            print("Plotting to file '"+self._filename+"_"+name+".eps'...")
        else:
            print("Plotting to file '"+self._filename+".eps'...")

        if dynamic_range is not None:
            log_range = dynamic_range
        elif self._delay_dynamic_range is not None:
            log_range = self._delay_dynamic_range
        else:
            log_range = 3

        fig = None
        ax_spec = None
        ax_tf = None
        ax_resp = None
        ax_rms = None
        # Set up the multiplot figure and axis
        fig, ((ax_spec, ax_none), (ax_tf, ax_resp)) = plt.subplots(2,2,sharex='col', sharey='row',
            gridspec_kw={'width_ratios':[3,1], 'height_ratios':[1,3]})
        ax_none.axis('off')
        fig.subplots_adjust(hspace=0, wspace=0)

        # Set the properties that depend on log and wave/velocity status
        cb_label = None
        cb_label_vars = r""
        cb_label_units = r""
        cb_label_scale= r""
        cb_map = "afmhot_r"

        # Copy the data for later modification.
        data_plot = None
        if response_map:
            data_plot = np.copy(self._response_map)
            print("Total response: {:.3e}".format(np.sum(data_plot)))
            cb_label = r"$\Psi_{resp}$"
        else:
            data_plot = np.copy(self._flux)
            print("Total line: {:.3e}".format(np.sum(data_plot)))
            cb_label = r"$\Psi$"

        # Set the xlabel and colour bar label - these differ if velocity or not
        x_bin_mult = 1
        bins_x = np.zeros(shape=self._dimensions[0])
        if velocity:
            # We're rescaling the axis to e.g. 10^3 km/s but the colorbar is still in km/s
            # So when we scale by bin width, we need a multiplier on the bin widths
            oom = np.log10(np.amax(self._bins_vel))
            oom = oom - oom%3
            bins_x = self._bins_vel/(10**oom)
            x_bin_mult = 10**oom
            ax_tf.set_xlabel(r'Velocity ($10^{:.0f}$ km s$^{}$)'.format(oom, '{-1}'))
            cb_label_vars = r"($v, \tau$)"
            cb_label_units = r"/ km s$^{-1}$"
        else:
            bins_x = self._bins_wave
            ax_tf.set_xlabel(r'Wavelength $\lambda$ ($\AA$)')
            cb_label_vars += r"($\lambda, \tau$)"
            cb_label_units = r"$\AA$"

        bins_x_midp = np.zeros(shape=self._dimensions[0])
        for i in range(0,self._dimensions[0]):
            bins_x_midp[i] = (bins_x[i] + bins_x[i+1])/ 2

       # Set the ylabel and y bins for whether it's in days or seconds
        if days:
            bins_y = np.true_divide(self._bins_delay, float(seconds_per_day))
            data_plot *= seconds_per_day
            ax_tf.set_ylabel(r'Delay $\tau$ (days)')
            cb_label_units += r' d'
        else:
            bins_y = self._bins_delay
            ax_tf.set_ylabel(r'Delay $\tau$ (seconds)')
            cb_label_units += r' s'

        bins_y_midp = np.zeros(shape=self._dimensions[1])
        for i in range(0,self._dimensions[1]):
            bins_y_midp[i] = (bins_y[i] + bins_y[i+1])/ 2

        # Rescale the values to be luminosity/km s^-1 d or /A d
        for i in range(0, self._dimensions[0]):
            width_x = bins_x[i+1] - bins_x[i]
            for j in range(0, self._dimensions[1]):
                width_y = bins_y[j+1] - bins_y[j]
                data_plot[i][j] /= (width_x * x_bin_mult * width_y)

        # Plot the spectrum and light curve, normalised
        data_plot_spec = np.sum(data_plot, 0)
        data_plot_resp = np.sum(data_plot, 1)
        resp = ax_resp.plot(data_plot_resp, bins_y_midp, c='m')
        ax_spec.set_ylabel(r'$\Psi$(v) ((km s$^{-1}$)$^{-1}$)')
        if days:
            ax_resp.set_xlabel(r'$\Psi$($\tau$) (d^{-1})')
        else:
            ax_resp.set_xlabel(r'$\Psi$($\tau$) (s^{-1})')

        if response_map:
            ax_spec.axhline(0, color='grey')
            ax_resp.axvline(0, color='grey')
            #ax_spec.set_yticks([0])
            #ax_resp.set_xticks([0])
            #ax_spec.tick_params(axis='x', labelbottom='off', labelleft='off', labeltop='off', labelright='off', left='off', bottom='off')
            #ax_resp.tick_params(axis='y', labelbottom='off', labelleft='off', labeltop='off', labelright='off', left='off', bottom='off')
            data_plot_rms = np.sum(np.sqrt(np.square(data_plot)), 0) / np.sum(np.sqrt(np.square(data_plot)))
            rms = ax_spec.plot(bins_x_midp, data_plot_rms, c='c', label=r'RMS $\Psi$(v)')
            spec = ax_spec.plot(bins_x_midp, data_plot_spec, c='m', label=r'$\Psi$(v)')
            lg_orig = ax_spec.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

            #delays = self.centroid_delay(response=True)
            #delay_x = np.asarray([np.amax(data_plot_resp)/2])
            #delay_y = np.asarray([delays[0]])/seconds_per_day
            #delay_y_min = np.asarray([delays[1]])/seconds_per_day
            #delay_y_max = np.asarray([delays[2]])/seconds_per_day
            #ax_resp.errorbar(delay_x, delay_y, yerr=[delay_y_min, delay_y_max], capsize=3, fmt='o', c='k')

        else:
            #ax_spec.tick_params(labelbottom='off', labelleft='off', labeltop='off', labelright='off', left='off', bottom='off')
            #ax_spec.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(''))
            #ax_resp.tick_params(labelbottom='off', labelleft='off', labeltop='off', labelright='off', bottom='off', left='off')
            #ax_resp.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(''))
            spec = ax_spec.plot(bins_x_midp, data_plot_spec, c='m')

            # delays = self.centroid_delay(response=False)
            # delay_x = np.asarray([np.amax(data_plot_resp)/2])
            # delay_y = np.asarray([delays[0]])/seconds_per_day
            # delay_y_min = np.asarray([delays[1]])/seconds_per_day
            # delay_y_max = np.asarray([delays[2]])/seconds_per_day
            # ax_resp.errorbar(delay_x, delay_y, yerr=[delay_y_min, delay_y_max], capsize=3, fmt='o', c='k')


        # If this is a log plot, take log and correct label and limits
        if log:
            cb_max = np.log10(np.amax(data_plot))
            cb_min = cb_max-log_range
            cb_label = r"Log "+cb_label
            data_plot = np.ma.log10(data_plot)
         # Else just scale the data
        else:
            maxval = np.floor(np.log10(np.amax(data_plot)))
            data_plot /= np.power(10,maxval)
            cb_max = np.amax(data_plot)
            cb_min = np.amin(data_plot)
            dummy = "{}{:.0f}{}".format("{",maxval,"}")
            cb_label_scale = r" 10$^{}$".format(dummy)

        # If this is a response map, it may have a negative component and need a different plot
        if response_map:
            cb_max = np.amax([cb_max, np.abs(cb_min)])
            cb_min = -cb_max
            cb_map = 'seismic'

        # Normalise or rescale the data. If doing neither, put units on cb.
        if normalised:
            data_plot /= np.sum(data_plot)
            cb_label_units = r""
            cb_label_scale = r""
        elif rescaled:
            data_plot /= np.amax(data_plot)
            cb_label_units = r""
            cb_label_scale = r""

        # Plot the main colourplot for the transfer function
        tf = None
        tf = ax_tf.pcolor(bins_x, bins_y, data_plot,
                          vmin=cb_min, vmax=cb_max, cmap=cb_map)
        ax_tf.set_ylim(top=bins_y[-1])
        ax_tf.set_aspect('auto')

        # Add lines for keplerian rotational outflows
        if keplerian is not None:
            resolution  = 1000
            r_angle     = np.radians(keplerian["angle"])
            r_mass_bh   = keplerian["mass"] * apc.M_sun.value
            r_rad_grav  = (6 * apc.G.value * r_mass_bh / np.power(apc.c.value, 2))
            ar_wave     = np.zeros(resolution) # * u.angstrom
            ar_delay    = np.zeros(resolution) # * u.s
            ar_phase    = np.linspace(0, np.pi*2, resolution)
            ar_rad      = np.linspace(keplerian["radius"][0]*r_rad_grav, keplerian["radius"][1]*r_rad_grav,resolution)
            ar_vel      = np.zeros(resolution)

            # ITERATE OVER BLUE BOUND
            for r_rad, r_wave, r_delay, r_vel in np.nditer([ar_rad, ar_wave, ar_delay, ar_vel], op_flags=['readwrite']):
                r_rad           = r_rad # * u.m
                r_vel[...]      = keplerian_velocity( r_mass_bh, r_rad ) * np.sin(r_angle) / (1e3 * x_bin_mult)
                r_wave[...]     = doppler_shift_wave( self._line_wave, r_vel )
                r_delay[...]    = calculate_delay( r_angle, np.pi/2, r_rad, u.day )
            ax_tf.plot(ar_vel, ar_delay, '-', c='m')

            # ITERATE OVER RED BOUND
            for r_rad, r_wave, r_delay, r_vel in np.nditer([ar_rad, ar_wave, ar_delay, ar_vel], op_flags=['readwrite']):
                r_rad           = r_rad # * u.m
                r_vel[...]      = -keplerian_velocity( r_mass_bh, r_rad ) * np.sin(r_angle) / (1e3 * x_bin_mult)
                r_wave[...]     = doppler_shift_wave( self._line_wave, r_vel )
                r_delay[...]    = calculate_delay( r_angle, np.pi/2, r_rad, u.day )
            ax_tf.plot(ar_vel, ar_delay, '-', c='m')

        cbar = plt.colorbar(tf, orientation="vertical")
        cbar.set_label(cb_label+cb_label_vars+cb_label_scale+cb_label_units)

        if name is None:
            plt.savefig("{}.eps".format(self._filename),bbox_inches='tight')
        else:
            plt.savefig("{}_{}.eps".format(self._filename, name),bbox_inches='tight')
        print("Successfully plotted ({:.1f}s)".format(time.clock()-start))
        plt.close(fig)
        return self

# ==============================================================================
def open_database(s_file, s_user, s_password):
    ### TRY OPENING THE DATABASE ###
    print ("Opening database '{}'...".format(s_file))

    db_engine = None
    try:
        db_engine = sqlalchemy.create_engine("sqlite:///{}.db".format(s_file))
    except sqlalchemy.exc.SQLAlchemyError as e:
        print(e)
        sys.exit(1)

    ### DOES IT ALREADY EXIST? ###
    # print ("Searching for table 'Photons'...")
    Session = sqlalchemy.orm.sessionmaker(bind=db_engine)
    session = Session()

    start = time.clock()

    try:
        session.query(Photon.Weight).first()
        # If so, we go with what we've found.
        print("Found existing filled photon database '{}'".format(s_file))
    except sqlalchemy.exc.SQLAlchemyError as e:
        # If not, we populate from the delay dump file. This bit is legacy!
        print("No existing filled photon database, reading from file '{}.delay_dump'".format(s_file))
        Base.metadata.create_all(db_engine)

        added = 0
        delay_dump = open("{}.delay_dump".format(s_file), 'r')
        for line in delay_dump:
            if line.startswith('#'):
                continue

            try:
                values = [float(i) for i in line.split()]
            except:
                print("Malformed line: '{}'".format(line))
                continue

            if len(values) is not 13:
                print("Malformed line: '{}'".format(line))
                continue

            #del values[0]
            #del values[8]
            #matom_bool = False
            #if(values[11]>= 10):
            #    values[11] = values[11] - 10
            #    matom_bool = True

            session.add(Photon(Wavelength=values[1], Weight=values[2], X=values[3], Y=values[4], Z=values[5],
                            ContinuumScatters=values[6]-values[7], ResonantScatters=values[7], Delay=values[8],
                            Spectrum=values[10], Origin=(values[11]%10), Resonance=values[12], Origin_matom = (values[11]>9)))
            added += 1
            if added > 25000:
                added = 0
                session.commit()

        session.commit()
        session.close()
        del(session)
        print("Successfully read in ({:.1f}s)".format(time.clock()-start))

    return db_engine
# ==============================================================================

s_user = "root"
s_password = "password"

Base = sqlalchemy.ext.declarative.declarative_base()
class Spectrum(Base):
    __tablename__ = "Spectra"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    angle = sqlalchemy.Column(sqlalchemy.Float)

class Origin(Base):
    __tablename__ = "Origins"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    name = sqlalchemy.Column(sqlalchemy.String)

class Photon(Base):
    __tablename__ = "Photons"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    Wavelength = sqlalchemy.Column(sqlalchemy.Float)
    Weight = sqlalchemy.Column(sqlalchemy.Float)
    X = sqlalchemy.Column(sqlalchemy.Float)
    Y = sqlalchemy.Column(sqlalchemy.Float)
    Z = sqlalchemy.Column(sqlalchemy.Float)
    ContinuumScatters = sqlalchemy.Column(sqlalchemy.Integer)
    ResonantScatters = sqlalchemy.Column(sqlalchemy.Integer)
    Delay = sqlalchemy.Column(sqlalchemy.Float)
    Spectrum = sqlalchemy.Column(sqlalchemy.Integer)
    Origin = sqlalchemy.Column(sqlalchemy.Integer)
    Resonance = sqlalchemy.Column(sqlalchemy.Integer)
    Origin_matom = sqlalchemy.Column(sqlalchemy.Boolean)
    #__table_args__ = (sqlalchemy.Index('spec_res', "Spectrum", "Resonance"),)


# agn_spec0_engine, agn_spec0_db = open_database("/Users/swm1n12/python_runs/paper1_fiducial/agn_obs_0", "root", "password")
# agn_spec1_engine, agn_spec1_db = open_database("/Users/swm1n12/python_runs/paper1_fiducial/agn_obs_1", "root", "password")
# agn_spec2_engine, agn_spec2_db = open_database("/Users/swm1n12/python_runs/paper1_fiducial/agn_obs_2", "root", "password")
# agn_spec3_engine, agn_spec3_db = open_database("/Users/swm1n12/python_runs/paper1_fiducial/agn_obs_3", "root", "password")
# agn_spec4_engine, agn_spec4_db = open_database("/Users/swm1n12/python_runs/paper1_fiducial/agn_obs_4", "root", "password")
# agn_spec5_engine, agn_spec5_db = open_database("/Users/swm1n12/python_runs/paper1_fiducial/agn_obs_5", "root", "password")


#sey095_engine, sey095_db = open_database("/Users/swm1n12/python_runs/paper1_5548_resp/sey_095", "root", "password")
#sey105_engine, sey105_db = open_database("/Users/swm1n12/python_runs/paper1_5548_resp/sey_105", "root", "password")

dims = [50, 50]
kep_sey = {"angle":40, "mass":1e7, "radius":[50,2000]}
kep_agn = {"angle":40, "mass":1e9, "radius":[50,20000]}

def do_tf_plots(tf_list_inp, dynamic_range=None, keplerian=None, name=''):
    tf_delay = []
    for tf_inp in tf_list_inp:
        tf_inp.plot(velocity=True, keplerian=keplerian, log=False, name=(None if name is '' else name))
        tf_inp.plot(velocity=True, keplerian=keplerian, log=True,  name=name+"_log", dynamic_range=dynamic_range)
        tf_delay.append(tf_inp.centroid_delay())
    return
def do_rf_plots(tf_min, tf_mid, tf_max, keplerian=None, name=''):
    rf_delay = []
    tf_mid.response_map_by_tf(tf_min, tf_mid).plot(velocity=True, response_map=True, keplerian=keplerian, name=name+"_resp_low")
    rf_delay.append(tf_mid.modal_delay(response=True))
    tf_mid.response_map_by_tf(tf_min, tf_max).plot(velocity=True, response_map=True, keplerian=keplerian, name=name+"_resp_mid")
    rf_delay.append(tf_mid.modal_delay(response=True))
    tf_mid.response_map_by_tf(tf_mid, tf_max).plot(velocity=True, response_map=True, keplerian=keplerian, name=name+"_resp_high")
    rf_delay.append(tf_mid.modal_delay(response=True))
    return np.array(rf_delay,dtype='float')

# # ==============================================================================
# # RUN FOR SEYFERT
# # ==============================================================================
sey100_db = open_database("/Users/amsys/paper_tss_qso/qso_100", "root", "password")
sey110_db = open_database("/Users/amsys/paper_tss_qso/qso_110", "root", "password")
sey090_db = open_database("/Users/amsys/paper_tss_qso/qso_090", "root", "password")
