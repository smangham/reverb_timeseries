# -*- coding: utf-8 -*-
import numpy as np
import astropy as ap
import astropy.constants as apc
import time
import sys
import matplotlib.pyplot as plt
import sqlalchemy
import sqlalchemy.ext.declarative
import sqlalchemy.orm
import sqlalchemy.orm.query
import matplotlib
from astropy import units as u

# Constant used for rescaling data.
# Probably already exists in apc but I don't want to faff around with units
seconds_per_day = 60*60*24


# ==============================================================================
# MATHS FUNCTIONS
# ==============================================================================
def calculate_FWHM(X, Y):
    """
    Calculate FWHM from arrays

    Taken from http://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
    I don't think this can cope with being passed a doublet or an array with no
    peak within it. Doublets will calculate FWHM from the HM of both!

    Args:
        X (numpy array):    Array of bin midpoints
        Y (numpy array):    Array of bin values

    Returns:
        float:              FWHM of the peak (should it exist!)
    """
    # Create 'difference' array by subtracting half maximum
    d = Y - (np.amax(Y) / 2)
    # Find the points where the difference is positive
    indexes = np.where(d > 0)[0]
    # The first and last positive points are the edges of the peak
    return abs(X[indexes[-1]] - X[indexes[0]])


def calculate_centroid(bins, vals, bounds=None):
    """
    Returns the centroid position, with optional percentile bounds.

    Args:
        bins (numpy array): Array of bin bounds
        vals (numpy array): Array of bin values
        bounds (float):     Fraction from 0-0.5. Percentile either side of the
                            centroid to find (e.g. .2 -> 30%, 70%)

    Returns:
        float:              Flux-weighted centroid
        float (optional):   Lower percentile centroid, if 'bounds' passed
        float (optional):   Upper percentile centroid, if 'bounds' passed
    """
    centroid_total = np.sum(vals)
    centroid_position = np.sum(np.multiply(bins, vals))/centroid_total

    if bounds is not None:
        # If we're finding bounds
        bound_width = bounds/2
        bound_min = -1
        bound_max = -1
        # Find the upper bound
        value_total = 0
        for index, value in enumerate(vals):
            # Starting at 0, add the value in this bin to the running total
            value_total += value
            if value_total/centroid_total >= 0.5+bound_width:
                # If this total is > the bound we're looking for, record the bin and stop
                bound_max = bins[index]
                break
        # Find the lower bound
        value_total = centroid_total
        for index, value in enumerate(vals[::-1]):
            # Starting at the total value, subtract the value in this bin from the running total
            value_total -= value
            if value_total/centroid_total <= 0.5-bound_width:
                # If this total is < the bound we're looking for, record the bin and stop
                bound_min = bins[len(bins)-1-index]
                break
        # On reflection, they could both sum since I'm just iterating backwards.
        # Also, I could use zip() even though they're numpy arrays as zip works fine
        # if you don't want to modify the array entries.
        # Maybe go over this later, should be easy enough to test.

        # Return the centroid and the bins.
        # NOTE: If the value exceeds the bound range midway through a cell, it'll just return the min/max
        # for that cell as appropriate. This will overestimate the error on the centroid.
        return centroid_position, bound_min, bound_max
    else:
        return centroid_position


def calculate_midpoints(X):
    """
    Converts bin boundaries into midpoints

    Args:
        X (numpy array):    Array of bin boundaries

    Returns:
        numpy array:        Array of bin midpoints (1 shorter!)
    """
    X_midp = np.zeros(shape=len(X)-1)
    for i in range(0, len(X)-1):
        X_midp[i] = (X[i] + X[i+1]) / 2
    return X_midp


# ==============================================================================
# PHYSICS FUNCTIONS
# ==============================================================================
def calculate_delay(angle, phase, radius, days=True):
    """
    Delay relative to continuum for emission from a point on the disk.

    Calculate delay for emission from a point on a keplerian disk, defined by
    its radius and disk angle, to an observer at a specified angle.

    Draw plane at r_rad_min out. Find x projection of disk position.
    Calculate distance travelled to that plane from the current disk position
    Delay relative to continuum is thus (distance from centre to plane)
    + distance from centre to point

    Args:
        angle (float):  Observer angle to disk normal, in radians
        phase (float):  Rotational angle of point on disk, in radians. 0 = in line to observer
        radius (float): Radius of the point on the disk, in m
        days (bool):    Whether the timescale should be seconds or days

    Returns:
        float:          Delay relative to continuum
    """
    vr_disk   = np.array([radius*np.cos(phase), 0.0])
    vr_normal = np.array([np.sin(angle), np.cos(angle)])
    vr_plane  = radius * vr_normal
    if days:
        return (np.dot((vr_plane - vr_disk), vr_normal) / apc.c.value) / seconds_per_day
    else:
        return (np.dot((vr_plane - vr_disk), vr_normal) / apc.c.value)


def keplerian_velocity(mass, radius):
    """
    Calculates Keplerian velocity at given radius

    Args:
        mass (float):   Object mass in kg
        radius (float): Orbital radius in m

    Returns:
        float:          Orbital velocity in m/s
    """
    return np.sqrt(ap.constants.G.value * mass / radius)


def doppler_shift_wave(line, vel):
    """
    Converts passed line and velocity into red/blueshifted wavelength

    Args:
        line (float):   Line wavelength (any length unit)
        vel (float):    Doppler shift velocity (m/s)

    Returns:
        float:          Doppler shifted line wavelength (as above)
    """
    return line * apc.c.value / (apc.c.value - vel)


def doppler_shift_vel(line, wave):
    """
    Converts passed red/blueshifted wave into velocity

    Args:
        line (float):   Base line wavelength (any length unit)
        wave (float):   Doppler shifted line wavelength (as above)

    Returns:
        float:          Speed of Doppler shift
    """
    if wave > line:
        return -1*apc.c.value * (1 - (line / wave))
    else:
        return apc.c.value * ((line / wave) - 1)


# ==============================================================================
# TRANSFER FUNCTION DEFINITION
# ==============================================================================
class TransferFunction:
    """
    Used to create, store and query emissivity and response functions
    """
    def __getstate__(self):
        """
        Removes invalid data before saving to disk

        Returns:
            dict: Updated internal dict, with references to external,
                  session-specific database things, removed.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_session']
        del state['_query']
        del state['_database']
        return state

    def __setstate__(self, state):
        """
        Restores the data from disk, and sets a flag to show this is a frozen TF.

        Args:
            state (dict): The unpickled object dict..
        """
        self.__dict__.update(state)
        self._unpickled = True

    def __init__(self, database, filename, continuum, wave_bins=None, delay_bins=None, template=None,
                 template_different_line=False, template_different_spectrum=False):
        """
        Initialises the TF, optionally by templating off another TF.

        Sets up all the basic properties of the TF that are required to create
        it. It must be '.run()' to query the DB before it can itself be queried.
        If templating, it applies all the same filters that were applied to the
        template TF, unless explicitly told not to. Filters don't overwrite!
        They stack. So you can't simply call '.line()' to change the line the TF
        corresponds to if its template was a different line, unless you specify
        thhat the template was of a different line.

        Args:
            database (sqlalchemy connection):
                                The database to be queried for this TF
            filename (string):  The root filename for plots created for this TF
            continuum (float):  The continuum value associated with this TF
            wave_bins (int):    Number of wavelength/velocity bins
            delay_bins (int):   Number of delay time bins
            template (TransferFunction):
                                Other TF to copy all filter settings from. Will
                                match delay, wave and velocity bins exactly
            template_different_line (bool):
                                Is this TF going to share delay & velocity bins
                                but have different wavelength bins?
            template_different_spectrum (bool):
                                Is this TF going to share all specified bins but
                                be taken on photons from a different observer

        Returns:
            TransferFunction:   The created TF

        """
        assert (delay_bins is not None and wave_bins is not None) or template is not None,\
            "Must provide either resolutions or another TF to copy them from!"
        # self._query = database.query(Photon.Wavelength, Photon.Delay, Photon.Weight, Photon.X, Photon.Z)

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
        self._continuum = continuum
        self._filename = filename
        self._bins_wave_count = wave_bins
        self._bins_delay_count = delay_bins
        self._bins_vel = None
        self._bins_wave = None
        self._bins_delay = None
        self._emissivity = None
        self._response = None
        self._count = None
        self._wave_range = None
        self._spectrum = None
        self._unpickled = False

        if template is not None:
            # If we're templating off a pre-existing transfer function, copy over all the shared properties
            print("Templating '{}' off of '{}'...".format(self._filename, template._filename))
            # Regardless of what line we're templating off, we want to share the velocity and delay bins
            self._bins_wave_count = template._bins_wave_count
            self._bins_delay_count = template._bins_delay_count
            self._bins_vel = template._bins_vel
            self._bins_delay = template._bins_delay

            # Now we want to call all the same filter functions that've been applied to the template
            # (where appropriate)
            if template_different_line is False:
                # If we're templating off of the same line, we want the same wavelength bins
                self.wavelength_bins(template._bins_wave)
            if template._line_wave is not None and template_different_line is False:
                # If we're templating off the same line, record we're using that line
                self.line(template._line_num, template._line_wave)
            if template._velocity is not None:
                # If we're templating off a TF with velocity, record we're doing so
                self.velocities(template._velocity)
            if template._line_list is not None and template_different_line is False:
                # If we want the same bins for the same list of lines, record so
                self.lines(template._line_list)
            if template._spectrum is not None and template_different_spectrum is False:
                # If we want the same bins for the same spectrum, record so
                self.spectrum(template._spectrum)

    def spectrum(self, number):
        """
        Constrain the TF to photons from a specific observer

        Args:
            number (int):       Observer number from Python run
        Returns:
            TransferFunction:   Self, so filters can be stacked
        """
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun, filters cannot be applied."
        self._spectrum = number
        self._query = self._query.filter(Photon.Spectrum == number)
        return self
    def line(self, number, wavelength):
        """
        Constrain the TF to only photons last interacting with a given line

        This includes being emitted in the specified line, or scattered off it

        Args:
            number (int):       Python line number. Will vary based on data file!
            wavelength (float): Wavelength of the line in angstroms
        Returns:
            TransferFunction:   Self, so filters can be stacked
        """
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun, filters cannot be applied."
        self._line_wave = wavelength
        self._line_num = number
        self._query = self._query.filter(Photon.Resonance == number)
        return self
    def velocities(self, velocity):
        """
        Constrain the TF to only photons with a range of Doppler shifts

        Args:
            velocity (float):   Maximum doppler shift velocity in m/s. Applies
                                to both positive and negative Doppler shift
        Returns:
            TransferFunction:   Self, so filters can be stacked
        """
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun, filters cannot be applied."
        assert self._line_wave is not None,\
            "Cannot limit doppler shift around a line without specifying a line!"
        self._velocity = velocity
        self._query = self._query.filter(Photon.Wavelength >= doppler_shift_wave(self._line_wave, -velocity),
                                         Photon.Wavelength <= doppler_shift_wave(self._line_wave,  velocity))
        return self
    def wavelengths(self, wave_min, wave_max):
        """
        Constrain the TF to only photons with a range of wavelengths

        Args:
            wave_min (float):   Minimum wavelength in angstroms
            wave_max (float):   Maximum wavelength in angstroms
        Returns:
            TransferFunction:   Self, so filters can be stacked
        """
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun, filters cannot be applied."
        assert wave_min < wave_max,\
            "Minimum wavelength must be lower than maximum wavelength!"
        self._wave_range = [wave_min, wave_max]
        self._query = self._query.filter(Photon.Wavelength >= wave_min, Photon.Wavelength <= wave_max)
        return self
    def wavelength_bins(self, wave_range):
        """
        Constrain the TF to only photons with a range of wavelengths, and to a specific set of bins

        Args:
            wave_range (numpy array):   Array of bins to use
        Returns:
            TransferFunction:   Self, so filters can be stacked
        """
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun, filters cannot be applied."
        assert len(wave_range) > 2,\
            "When providing an array, it must be of more than 2 entries! Use wavelength(min, max)."
        self._bins_wave = wave_range
        self._bins_wave_count = len(wave_range)-1
        self.wavelengths(self._bins_wave[0], self._bins_wave[-1])
        return self

    def lines(self, line_list):
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun, filters cannot be applied."
        assert len(line_list) > 1,\
            "For a single line, use the 'line()' filter rather than 'lines()'!"
        self._line_list = line_list
        self._query = self._query.filter(Photon.Resonance.in_(line_list))
        return self
    def delays(self, delay_min, delay_max, days=True):
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun, filters cannot be applied."
        assert delay_min < delay_max,\
            "Minimum delay must be below maximum delay!"

        if days:
            self._delay_range = [delay_min * seconds_per_day, delay_max * seconds_per_day]
        else:
            self._delay_range = [delay_min, delay_max]
        self._query = self._query.filter(Photon.Delay > self._delay_range[0], Photon.Delay < self._delay_range[1])
        return self
    def delay_dynamic_range(self, delay_dynamic_range):
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun, filters cannot be applied."
        assert delay_dynamic_range > 0,\
            "Cannot have a negative dynamic range!"
        self._delay_dynamic_range = delay_dynamic_range
        return self

    def cont_scatters(self, scat_min, scat_max=None):
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun, filters cannot be applied."
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
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun, filters cannot be applied."
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
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun, filters cannot be applied."
        self._query = self._query.filter(args)
        return self

    def response_map_by_tf(self, low_state, high_state, cf_min=1, cf_max=1):
        """Creates a response map from two other transfer functions, to be applied during plotting"""
        # The other two TFs ***must*** have identical bins and both provide ionising luminosity information
        assert self._emissivity is not None,\
            "You must run the TF query with '.run()' before response mapping it!"
        assert low_state._emissivity is not None and high_state._emissivity is not None,\
            "You must run the low and high state TF queries with '.run()' before response mapping using them!"
        assert np.array_equal(self._bins_wave, low_state._bins_wave) and np.array_equal(self._bins_delay, low_state._bins_delay),\
            "Low state TF is binned differently to target TF! Cannot rescale using it."
        assert np.array_equal(self._bins_wave, high_state._bins_wave) and np.array_equal(self._bins_delay, high_state._bins_delay),\
            "High state TF is binned differently to target TF! Cannot rescale using it."
        assert self._continuum is not None,\
            "TF missing continuum luminosity information!"
        assert low_state._continuum is not None,\
            "Low state TF missing continuum luminosity information!"
        assert high_state._continuum is not None,\
            "High state TF missing continuum luminosity information!"
        assert low_state._continuum <= self._continuum,\
            "Low state ionising luminosity greater than target TF ionising luminosity!"
        assert high_state._continuum >= self._continuum,\
            "High state ionising luminosity lower than target TF ionising luminosity!"

        continuum_difference = high_state._continuum - low_state._continuum

        # We divide the difference in TFs by the luminosity difference
        self._response = ((high_state._emissivity*high_state._continuum*cf_max) - (low_state._emissivity*low_state._continuum*cf_min)) / continuum_difference
        return self


    def FWHM(self, response=False, velocity=True):
        """Calculates the full width half maximum of the TF"""

        if velocity:
            midpoints = calculate_midpoints(self._bins_vel)
        else:
            midpoints = calculate_midpoints(self._bins_wave)

        if response:
            return calculate_FWHM(midpoints, np.sum(self._response, 0))
        else:
            return calculate_FWHM(midpoints, np.sum(self._emissivity, 0))

    def delay(self, response=False, threshold=0, bounds=None, days=False):
        """
        Calculates the centroid delay for the current data

        Args:
            response (Bool):    Whether or not to calculate the delay from the response
            threshold (Float):  Exclude all bins with value < threshold
            bounds (Float):     Return the percentile bounds (i.e. bounds=0.25,
                                the function will return [0.5, 0.25, 0.75])
        Returns:
            Float:              Centroid delay
            Float[]:            Centroid and lower and upper bounds
        """
        assert threshold < 1 or threshold >= 0,\
            "Threshold is a multiplier to the peak flux! It must be between 0 and 1"

        data = None
        if response:
            data = np.sum(self._response, 1)
        else:
            data = np.sum(self._emissivity, 1)
        value_threshold = np.amax(data) * threshold
        delay_midp = calculate_midpoints(self._bins_delay)

        delay_weighted = 0
        value_total = 0
        for value, delay in zip(data, delay_midp):
            if value >= value_threshold:
                delay_weighted += value * delay
                value_total += value

        if days:
            return delay_weighted/(value_total * seconds_per_day)
        else:
            return delay_weighted/value_total

    def delay_peak(self, response: bool=False, days: bool=False) -> float:
        """
        Calculates the peak delay for the current data
        """
        data = self.transfer_function_1d(response=response, days=days)
        peak = data[np.argmax(data[:, 1]), 0]
        return peak

    def run(self, scaling_factor=1.0, limit=None, verbose=False):
        """
        Performs a query on the photon DB and bins it

        A TF must be run *after* all filters are applied and before any attempts
        to retrieve or process data from it. This can be a time-consuming call,
        on the order of 1 minute per GB of input file.

        Args:
            scaling_factor (float): 1/Number of cycles in the spectra file
            limit (int):            Number of photons to limit the TF to, for testing
            verbose (bool):         Whether to output exactly what the query is
        Returns:
            TransferFunction:   Self, for chaining commands
        """
        assert self._unpickled is False,\
            "TF restored from pickle! It cannot be rerun."
        assert self._emissivity is None,\
            "TF has already been run!"
        assert scaling_factor > 0,\
            "Negative scaling factors make no sense!"
        assert limit is None or limit > 0,\
            "Limit must either be zero or a positive number!"
        start = time.clock()

        # If we're not in limit mode, fetch all
        data = None

        if verbose:
            if limit is not None:
                print("Limited to {} results...".format(limit))
            if self._velocity is not None:
                print("Limited to velocities -{} to +{}".format(self._velocity, self._velocity))
            if self._bins_wave is not None:
                print("Limited to preset wavelength bins from {} to {}".format(self._bins_wave[0], self._bins_wave[-1]))
            elif self._wave_range is not None:
                print("Limited to wavelengths {} to {}".format(self._wave_range[0], self._wave_range[1]))
            if self._line_num is not None:
                print("Limited to line {}, wavelength {}".format(self._line_num, self._line_wave))
            if self._spectrum is not None:
                print("Limited to spectrum {}".format(self._spectrum))
            if self._delay_range is not None:
                print("Limited to delays {} to {}".format(self._delay_range[0], self._delay_range[1]))

        if limit is None:
            data = np.asarray(self._query.all())
        else:
            data = np.asarray(self._query.limit(limit).all())

        assert len(data) > 0,\
            "No records found!"

        if verbose:
            print("Fetched {} records from '{}'...".format(len(data), self._filename))

        # Check if we've already got delay bins from another TF
        if self._bins_delay is None:
            # Data returned as Wavelength, Delay, Weight. Find min and max delays
            if self._delay_dynamic_range is not None:
                percentile = (1 - (10**(-self._delay_dynamic_range)))*100
                range_delay = [0, np.percentile(data[:, 1], percentile)]
                if verbose:
                    print("Delays up to the {} percentile value, {}d".format(percentile, range_delay[1]/seconds_per_day))
            else:
                range_delay = [0, np.amax(data[:, 1])]
            self._bins_delay = np.linspace(range_delay[0], range_delay[1],
                                           self._bins_delay_count+1, endpoint=True, dtype=np.float64)

        # Check if we've already got wavelength bins from another TF
        if self._bins_wave is None:
            # If we have no velocity bins, this is a factory-fresh TF
            if self._bins_vel is None:
                # Data returned as Wavelength, Delay, Weight. Find min and max delays and wavelengths
                range_wave = [np.amin(data[:, 0]), np.amax(data[:, 0])]
            # If we do have velocity bins, this was templated off a different line and we need to copy the velocities (but bins are in km! not m!)
            else:
                range_wave = [doppler_shift_wave(self._line_wave, self._bins_vel[0]*1000), doppler_shift_wave(self._line_wave, self._bins_vel[-1]*1000)]
                print("Creating new wavelength bins from template, velocities from {:.2e}-{:.2e} to waves: {:.2f}-{:.2f}".format(self._bins_vel[0], self._bins_vel[-1], range_wave[0], range_wave[1]))

            # Now create the bins for each dimension
            self._bins_wave = np.linspace(range_wave[0], range_wave[1],
                                          self._bins_wave_count+1, endpoint=True, dtype=np.float64)

        # Check if we've already got velocity bins from another TF and we have a line to center around
        if self._bins_vel is None and self._line_wave is not None:
            range_wave = [self._bins_wave[0], self._bins_wave[-1]]
            self._bins_vel = np.linspace(doppler_shift_vel(self._line_wave, range_wave[1]),
                                         doppler_shift_vel(self._line_wave, range_wave[0]),
                                         self._bins_wave_count+1, endpoint=True, dtype=np.float64)
            # Convert speed from m/s to km/s
            self._bins_vel = np.true_divide(self._bins_vel, 1000.0)

        # Now we bin the photons, weighting them by their photon weights for the luminosity
        self._emissivity, junk, junk = np.histogram2d(data[:, 1], data[:, 0], weights=data[:, 2],
                                                      bins=[self._bins_delay, self._bins_wave])
        # Keep an unweighted photon count for statistical error purposes
        self._count, junk, junk = np.histogram2d(data[:, 1], data[:, 0],
                                                 bins=[self._bins_delay, self._bins_wave])

        # Scaling factor! Each spectral cycle outputs L photons. If we do 50 cycles, we want a factor of 1/50
        self._emissivity *= scaling_factor
        # Scale to continuum luminosity
        self._emissivity /= self._continuum

        print("'{}' successfully run ({:.1f}s)".format(self._filename, time.clock()-start))
        # Make absolutely sure this data is wiped as it's *HUGE*
        del(data)
        return self

    def _return_array(self, array, delay, wave, delay_index):
        """
        Internal function used by response(), emissivity() and count()

        Args:
            array (numpy array):    Array to return value from
            delay (float):          Delay to return value for
            delay_index (int):      Delay index to return value for
            wave (float):           Wavelength to return value for
        Returns:
            int:                    If array == count
            float:                  If delay/delay_index and wave provided
            numpy.Array:            If delay but not wave provided
        """
        if delay is None and delay_index is None and wave is None:
            return array

        if delay is not None:
            if delay < self._bins_delay[0] or delay > self._bins_delay[-1]:
                if wave is None:
                    return np.zeros(self._bins_wave_count)
                else:
                    return 0
            delay_index = np.searchsorted(self._bins_delay, delay)
        elif delay_index is not None:
            if delay_index < 0 or delay_index > self._bins_delay_count:
                return 0

        if wave is None:
            return(array[delay_index, :])
        else:
            return(array[delay_index, np.searchsorted(self._bins_wave, wave)])

    def response_total(self):
        """Returns the total response"""
        # total = 0
        # for i in range(0, self._bins_wave_count):
        #     for j in range(0, self._bins_delay_count):
        #         total += self._response[j][i] \
        #               * (self._bins_delay[j+1] - self._bins_delay[j]) \
        #               * (self._bins_wave[i+1] - self._bins_wave[i])
        # return total
        return np.sum(self._response)

    def delay_bins(self):
        """Returns the range of delays covered by this TF"""
        return self._bins_delay
    def response(self, delay=None, wave=None, delay_index=None):
        """Returns the response in this bin"""
        assert self._response is not None,\
            "No response map has been built!"
        return self._return_array(self._response, delay=delay, wave=wave, delay_index=delay_index)
    def emissivity(self, delay=None, wave=None, delay_index=None):
        """Returns the emissivity in this bin"""
        assert self._emissivity is not None,\
            "The TF has not been run! Use .run() to query the DB first."
        return self._return_array(self._emissivity, delay=delay, wave=wave, delay_index=delay_index)
    def count(self, delay=None, wave=None, delay_index=None):
        """Returns the photon count in this bin"""
        assert self._count is not None,\
            "The TF has not been run! Use .run() to query the DB first."
        assert delay_index is not None or delay is not None,\
            "You must provide a delay, or a delay index!"
        return self._return_array(self._count, delay=delay, wave=wave, delay_index=delay_index)

    def transfer_function_1d(self, response=False, days=True):
        """Returns a 1d transfer  function"""
        if response:
            if days:
                return np.column_stack((calculate_midpoints(self._bins_delay/seconds_per_day), np.sum(self._response, 1)))
            else:
                return np.column_stack((calculate_midpoints(self._bins_delay), np.sum(self._response, 1)))
        else:
            if days:
                return np.column_stack((calculate_midpoints(self._bins_delay/seconds_per_day), np.sum(self._emissivity, 1)))
            else:
                return np.column_stack((calculate_midpoints(self._bins_delay), np.sum(self._emissivity, 1)))


    def plot(self, log=False, normalised=False, rescaled=False, velocity=False, name=None, days=True,
             response_map=False, keplerian=None, dynamic_range=None, RMS=False, show=False,
             max_delay=None):
        """Takes the data gathered by calling 'run' and outputs a plot"""
        assert response_map is False or self._response is not None,\
            "No data available for response map!"
        assert log is False or response_map is False,\
            "Cannot plot a logarithmic response map!"
        assert normalised is False or rescaled is False,\
            "Cannot be both normalised and rescaled!"
        assert self._bins_wave is not None,\
            "You must run the TF query with '.run()' before plotting it!"

        # matplotlib.rcParams["text.usetex"] = "True"
        matplotlib.rcParams.update({'font.size': 14})

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
        # Set up the multiplot figure and axis
        fig, ((ax_spec, ax_none), (ax_tf, ax_resp)) = plt.subplots(2, 2, sharex='col', sharey='row',
            gridspec_kw={'width_ratios': [3,1], 'height_ratios': [1,3]})
        ax_none.axis('off')
        ax_resp.invert_xaxis()
        fig.subplots_adjust(hspace=0, wspace=0)

        if response_map:
            ratio = np.sum(self._response)/np.sum(self._emissivity)
            ratio_exp = np.floor(np.log10(ratio))
            ratio_text = '\n'

            if ratio_exp < -1 or ratio_exp > 1:
                ratio_text_exp = r"{}{:.0f}{}".format("{", ratio_exp, "}")
                ratio_text += r"${:.2f}\times 10^{}$".format(ratio/(10**ratio_exp), ratio_text_exp)
            else:
                ratio_text += r"${:.3g}$".format(ratio)

            ax_tf.text(0.05, 0.95, r"$\frac{\Delta L}{L}/\frac{\Delta C}{C}=$"+ratio_text,
                transform=ax_tf.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left')

        # Set the properties that depend on log and wave/velocity status
        cb_label = None
        cb_label_vars = r""
        cb_label_units = r""
        cb_label_scale = r""
        cb_map = "afmhot_r"

        # Copy the data for later modification.
        data_plot = None
        if response_map:
            data_plot = np.copy(self._response)
            print("Total response: {:.3e}".format(np.sum(data_plot)))
            psi_label = r"$\Psi_{R}$"
        else:
            data_plot = np.copy(self._emissivity)
            print("Total line: {:.3e}".format(np.sum(data_plot)))
            psi_label = r"$\Psi_{T}$"
        cb_label = psi_label

        # Set the xlabel and colour bar label - these differ if velocity or not
        x_bin_mult = 1
        bins_x = np.zeros(shape=self._bins_wave_count)
        if velocity:
            # We're rescaling the axis to e.g. 10^3 km/s but the colorbar is still in km/s
            # So when we scale by bin width, we need a multiplier on the bin widths
            oom = np.log10(np.amax(self._bins_vel))
            oom = oom - oom % 3
            bins_x = self._bins_vel/(10**oom)
            x_bin_mult = 10**oom
            ax_tf.set_xlabel(r'Velocity ($10^{:.0f}$ km s$^{}$)'.format(oom, '{-1}'))
            cb_label_vars = r"($v, \tau$)"
            cb_label_units = r"/ km s$^{-1}$"
        else:
            bins_x = self._bins_wave
            ax_tf.set_xlabel(r'Wavelength $\lambda$ ($\AA$)')
            cb_label_vars += r"($\lambda, \tau$)"
            cb_label_units = r"/ $\AA$"

        bins_x_midp = np.zeros(shape=self._bins_wave_count)
        for i in range(0, self._bins_wave_count):
            bins_x_midp[i] = (bins_x[i] + bins_x[i+1]) / 2

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

        bins_y_midp = np.zeros(shape=self._bins_delay_count)
        for bin_y in range(0, self._bins_delay_count):
            bins_y_midp[bin_y] = (bins_y[bin_y] + bins_y[bin_y+1]) / 2

        # Rescale the values to be luminosity/km s^-1 d or /A d
        for bin_y in range(0, self._bins_delay_count):
            width_y = bins_y[bin_y+1] - bins_y[bin_y]
            for bin_x in range(0, self._bins_wave_count):
                width_x = bins_x[bin_x+1] - bins_x[bin_x]
                data_plot[bin_y][bin_x] /= (width_x * x_bin_mult * width_y)

        # Plot the spectrum and light curve, normalised
        data_plot_spec = np.sum(data_plot, 0)
        data_plot_resp = np.sum(data_plot, 1)
        exponent_spec = np.floor(np.log10(np.amax(data_plot_spec)))
        exponent_resp = np.floor(np.log10(np.amax(data_plot_resp)))
        exponent_resp_text = "{}{:.0f}{}".format("{", exponent_resp, "}")
        exponent_spec_text = "{}{:.0f}{}".format("{", exponent_spec, "}")

        ax_resp.plot(data_plot_resp/(10**exponent_resp), bins_y_midp, c='m')

        if velocity:
            ax_spec.set_ylabel(r'{}(v) $10^{}/$'.format(psi_label, exponent_spec_text)+r'km s$^{-1}$', fontsize=12)
        else:
            ax_spec.set_ylabel(r'{}($\lambda$) $10^{}/$'.format(psi_label, exponent_spec_text)+r'$\AA$', fontsize=12)

        if days:
            ax_resp.set_xlabel(r'{}($\tau$) $10^{}$ / d'.format(psi_label, exponent_resp_text))
        else:
            ax_resp.set_xlabel(r'{}($\tau$) $10^{}$ / s'.format(psi_label, exponent_resp_text))

        if response_map and RMS:
            ax_spec.axhline(0, color='grey')
            ax_resp.axvline(0, color='grey')

            data_plot_rms = np.sqrt(np.sum(np.square(data_plot), 0) / self._bins_wave_count)
            exponent_rms = np.floor(np.log10(np.amax(data_plot_rms)))
            exponent_rms_text = "{}{:.0f}{}".format("{", exponent_rms, "}")
            maximum_spec = np.amax(data_plot_spec)/np.power(10, exponent_spec)
            maximum_rms = np.amax(data_plot_rms)/np.power(10, exponent_rms)
            data_plot_rms /= np.amax(data_plot_rms)
            data_plot_spec /= np.amax(data_plot_spec)

            ax_spec.plot(bins_x_midp, data_plot_rms, c='c', label=r'RMS {}(v)/{:.2f}$x10^{}$'.format(psi_label, maximum_rms, exponent_rms_text))
            ax_spec.plot(bins_x_midp, data_plot_spec, c='m', label=r'{}(v)/{:.2f}$x10^{}$'.format(psi_label, maximum_spec, exponent_spec_text))
            ax_spec.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

        elif response_map:
            ax_spec.axhline(0, color='grey')
            ax_resp.axvline(0, color='grey')
            ax_spec.plot(bins_x_midp, data_plot_spec/(10**exponent_spec), c='m')

        else:
            ax_spec.plot(bins_x_midp, data_plot_spec/(10**exponent_spec), c='m')

        # If this is a log plot, take log and correct label and limits
        if log:
            cb_max = np.log10(np.amax(data_plot))
            cb_min = cb_max-log_range
            cb_label = r"Log "+cb_label
            data_plot = np.ma.log10(data_plot)
        # Else just scale the data
        else:
            maxval = np.floor(np.log10(np.amax(data_plot)))
            data_plot /= np.power(10, maxval)
            cb_max = np.amax(data_plot)
            cb_min = np.amin(data_plot)
            dummy = "{}{:.0f}{}".format("{", maxval, "}")
            cb_label_scale = r" 10$^{}$".format(dummy)

        # If this is a response map, it may have a negative component and need a different plot
        if response_map:
            cb_max = np.amax([cb_max, np.abs(cb_min)])
            cb_min = -cb_max
            cb_map = 'RdBu_r'

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
        tf = ax_tf.pcolor(bins_x, bins_y, data_plot,
                          vmin=cb_min, vmax=cb_max, cmap=cb_map)
        if not max_delay:
            ax_tf.set_ylim(bottom=bins_y[0], top=bins_y[-1])
        else:
            ax_tf.set_ylim(bottom=bins_y[0], top=max_delay)
        ax_tf.set_xlim(left=bins_x[0], right=bins_x[-1])
        ax_tf.set_aspect('auto')

        # Add lines for keplerian rotational outflows
        if keplerian is not None:
            print('Keplerian!?', keplerian)
            resolution = 1000
            scale_factor = keplerian.get('rescale', 1)
            r_angle    = np.radians(keplerian["angle"])
            r_mass_bh  = keplerian["mass"] * apc.M_sun.value
            r_rad_grav = (6 * apc.G.value * r_mass_bh / np.power(apc.c.value, 2))
            ar_wave    = np.zeros(resolution)  # * u.angstrom
            ar_delay   = np.zeros(resolution)  # * u.s
            ar_phase   = np.linspace(0, np.pi*2, resolution)
            ar_rad     = np.linspace(keplerian["radius"][0]*r_rad_grav, 20*keplerian["radius"][1]*r_rad_grav, resolution)
            ar_vel     = np.zeros(resolution)
            r_rad_min  = r_rad_grav * keplerian["radius"][0]
            r_rad_max  = r_rad_grav * keplerian["radius"][1]
            r_vel_min  = keplerian_velocity(r_mass_bh, r_rad_max)
            r_vel_max  = keplerian_velocity(r_mass_bh, r_rad_min)

            # ITERATE OVER INNER EDGE
            for r_phase, r_wave, r_delay, r_vel in np.nditer([ar_phase, ar_wave, ar_delay, ar_vel], op_flags=['readwrite']):
                r_vel[...]   = r_vel_max * np.sin(r_phase) * np.sin(r_angle) / (1e3 * x_bin_mult)
                # r_vel[...]   = r_vel_max * np.sin(r_phase) * 1 / (1e3 * x_bin_mult)
                r_wave[...]  = doppler_shift_wave(self._line_wave, r_vel * 1e3 * x_bin_mult)
                r_delay[...] = calculate_delay(r_angle, r_phase, r_rad_min, u.day)
            if velocity:
                ax_tf.plot(ar_vel, ar_delay, '-', c='m')
            else:
                ax_tf.plot(ar_wave, ar_delay, '-', c='m')

            # # ITERATE OVER OUTER EDGE
            # for r_phase, r_wave, r_delay, r_vel in np.nditer([ar_phase, ar_wave, ar_delay, ar_vel], op_flags=['readwrite']):
            #     r_vel[...]   = r_vel_min * np.sin(r_phase) * np.sin(r_angle) / (1e3 * x_bin_mult)
            #     # r_vel[...]   = r_vel_min * np.sin(r_phase) * 1 / (1e3 * x_bin_mult)
            #     r_wave[...]  = doppler_shift_wave(self._line_wave, r_vel * 1e3 * x_bin_mult)
            #     r_delay[...] = calculate_delay(r_angle, r_phase, r_rad_max, u.day)
            # if velocity:
            #     ax_tf.plot(ar_vel, ar_delay, '-', c='m')
            # else:
            #     ax_tf.plot(ar_wave, ar_delay, '-', c='m')

            # ITERATE OVER BLUE BOUND
            for r_rad, r_wave, r_delay, r_vel in np.nditer([ar_rad, ar_wave, ar_delay, ar_vel], op_flags=['readwrite']):
                r_rad        = r_rad  # * u.m
                r_vel[...]   = keplerian_velocity(r_mass_bh, r_rad) * np.sin(r_angle) / (1e3 * x_bin_mult)
                # r_vel[...]   = keplerian_velocity(r_mass_bh, r_rad) * 1 / (1e3 * x_bin_mult)
                r_wave[...]  = doppler_shift_wave(self._line_wave, r_vel * 1e3 * x_bin_mult)
                r_delay[...] = calculate_delay(r_angle, np.pi/2, r_rad, u.day)
            if velocity:
                ax_tf.plot(ar_vel, ar_delay, '-', c='m')
            else:
                ax_tf.plot(ar_wave, ar_delay, '-', c='m')

            # ITERATE OVER RED BOUND
            for r_rad, r_wave, r_delay, r_vel in np.nditer([ar_rad, ar_wave, ar_delay, ar_vel], op_flags=['readwrite']):
                r_rad        = r_rad  # * u.m
                r_vel[...]   = -keplerian_velocity(r_mass_bh, r_rad) * np.sin(r_angle) / (1e3 * x_bin_mult)
                # r_vel[...]   = -keplerian_velocity(r_mass_bh, r_rad) * 1 / (1e3 * x_bin_mult)
                r_wave[...]  = doppler_shift_wave(self._line_wave, r_vel * 1e3 * x_bin_mult)
                r_delay[...] = calculate_delay(r_angle, np.pi/2, r_rad, u.day)
            if velocity:
                ax_tf.plot(ar_vel, ar_delay, '-', c='m')
            else:
                ax_tf.plot(ar_wave, ar_delay, '-', c='m')

        cbar = plt.colorbar(tf, orientation="vertical")
        cbar.set_label(cb_label+cb_label_vars+cb_label_scale+cb_label_units)

        if name is None:
            plt.savefig("{}.eps".format(self._filename), bbox_inches='tight')
            print("Successfully plotted '{}.eps'({:.1f}s)".format(self._filename, time.clock()-start))
        else:
            plt.savefig("{}_{}.eps".format(self._filename, name), bbox_inches='tight')
            print("Successfully plotted '{}_{}.eps'({:.1f}s)".format(self._filename, name, time.clock()-start))

        if show:
            fig.show()

        plt.close(fig)
        return self


# ==============================================================================
def open_database(file_root: str, user:str=None, password:str=None, batch_size:int=25000):
    """
    Open or create a SQL database

    Will open a SQL DB if one already exists, otherwise will create one from
    file. Note, though, that if the process is interrupted the code cannot
    intelligently resume- you must delete the half-written DB!

    Args:
        file_root (string): Root of the filename (no '.db' or '.delay_dump')
        user (string):      Username. Here in case I change to PostgreSQL
        password (string):  Password. Here in case I change to PostgreSQL
        batch_size (int):   Number of photons to stage before committing. If
                            too low, file creation is slow. If too high, get
                            out-of-memory errors.

    Returns:
        sqlalchemy engine:  Connection to the database opened
    """

    print("Opening database '{}'...".format(file_root))

    db_engine = None
    try:
        db_engine = sqlalchemy.create_engine("sqlite:///{}.db".format(file_root))
    except sqlalchemy.exc.SQLAlchemyError as e:
        print(e)
        sys.exit(1)

    # DOES IT ALREADY EXIST? ###
    Session = sqlalchemy.orm.sessionmaker(bind=db_engine)
    session = Session()

    start = time.clock()

    try:
        session.query(Photon.Weight).first()
        # If so, we go with what we've found.
        print("Found existing filled photon database '{}'".format(file_root))
    except sqlalchemy.exc.SQLAlchemyError as e:
        # If not, we populate from the delay dump file. This bit is legacy!
        print("No existing filled photon database, reading from file '{}.delay_dump'".format(file_root))
        Base.metadata.create_all(db_engine)

        added = 0
        delay_dump = open("{}.delay_dump".format(file_root), 'r')
        for line in delay_dump:
            # For each line in this file, if it is not a comment
            if line.startswith('#'):
                continue
            try:
                # Try reading it in as a series of values.
                values = [float(i) for i in line.split()]
            except:
                print("Malformed line: '{}'".format(line))
                continue

            if len(values) is not 13:
                # There should be 13 values per line in our base formatting!
                print("Malformed line: '{}'".format(line))
                continue

            # Add the photo using the values. Some must be modified here; ideally, this would be done in Python.
            session.add(Photon(Wavelength=values[1], Weight=values[2], X=values[3], Y=values[4], Z=values[5],
                               ContinuumScatters=values[6]-values[7], ResonantScatters=values[7], Delay=values[8],
                               Spectrum=values[10], Origin=(values[11]%10), Resonance=values[12], Origin_matom=(values[11] > 9)))
            added += 1
            if added > batch_size:
                # We commit in batches in order to avoid out-of-memory errors
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


dims = [50, 50]
kep_sey = {"angle": 40, "mass": 1e7, "radius": [50, 2000]}
kep_qso = {"angle": 40, "mass": 1e9, "radius": [50, 20000]}


def do_tf_plots(tf_list_inp, dynamic_range=None, keplerian=None, name=None, file=None):
    tf_delay = []
    for tf_inp in tf_list_inp:
        tf_inp.plot(velocity=True, keplerian=keplerian, log=False, name=name)
        tf_inp.plot(velocity=True, keplerian=keplerian, log=True,  name=('log' if name is None else name+"_log"), dynamic_range=dynamic_range)
        tf_delay.append(tf_inp.delay(threshold=0.8))

    if file is not None:
        print("Saving TF plots to file: {}".format(file+"_tf_delay.txt"))
        np.savetxt(file+"_tf_delay.txt", np.array(tf_delay, dtype='float'), header="Delay")


def do_rf_plots(tf_min, tf_mid, tf_max, keplerian=None, name=None, file=None):
    if name is not None:
        name += '_'
    else:
        name = ''

    total_min = np.sum(tf_min._emissivity).item()
    total_mid = np.sum(tf_mid._emissivity).item()
    total_max = np.sum(tf_max._emissivity).item()

    calibration_factor = total_mid / ((total_min + total_max) / 2)

    tf_mid.response_map_by_tf(tf_min, tf_max, cf_min=1, cf_max=1).plot(velocity=True, response_map=True, keplerian=keplerian, name=name+"resp_mid")
    rf_mid = tf_mid.delay(response=True, threshold=0.8)

    tf_mid.response_map_by_tf(tf_min, tf_mid, cf_min=calibration_factor, cf_max=1).plot(velocity=True, response_map=True, keplerian=keplerian, name=name+"resp_low")
    rf_min = tf_mid.delay(response=True, threshold=0.8)
    tf_mid.response_map_by_tf(tf_mid, tf_max, cf_min=1, cf_max=calibration_factor).plot(velocity=True, response_map=True, keplerian=keplerian, name=name+"resp_high")
    rf_max = tf_mid.delay(response=True, threshold=0.8)

    if file is not None:
        print("Saving RF plots to file: {}".format(file+"_rf_delay.txt"))
        np.savetxt(file+"_rf_delay.txt", np.array([rf_min, rf_mid, rf_max], dtype='float'), header="Delay")
