import sys

import astropy as ap
import astropy.constants as apc
import astropy.table as apt
from astropy import units as u
from astropy.units import cds as ucds

import numpy as np

import matplotlib.pyplot as plt


def spectra_times(spectra_times_file, time_units=None, time_name='MJD'):
    """
    Imports a ASCII list of spectrum times

    Args:
        spectra_times_file (string):    Name of the file with times
        time_units (astropy.Unit):      Unit the times are in (e.g. u.s, u.d)

    Returns:
        astropy.QTable:                 Single-column table of time in given units
    """
    spectra_times = ap.table.Table.read(spectra_times_file, format='ascii', names=['time'])
    spectra_times['time'].unit = time_units
    spectra_times['time'].meta['name'] = time_name

    return spectra_times


def lightcurve(lightcurve_file, time_units=None, value_units=None, time_name='MJD', value_name=None,
               bolometric_correction=None, error_ratio=None, delta_continuum_range=None,
               target_bolometric_luminosity=None):
    """
    Inputs a two- or three-column ascii file of times, continuum values and errors.

    Args:
        lightcurve_file (string):   Name of the file to read
        time_units (astropy.Unit):  Unit the times are in (e.g. u.s, u.d)
        value_units (astropy.Unit): Unit the values are in (e.g. )
        bolometric_correction(astropy.Quantity):
                                    Conversion factor from e.g. monochromatic flux to bolometric
        target_bolometric_luminosity (astropy.Quantity):
                                    Target mean bolometric luminosity to rescale to.
        error_ratio (float):        F/F_error ratio to create errors at

    Returns:
        astropy.Table:              Two/three column
    """
    assert bolometric_correction is None or target_bolometric_luminosity is None,\
        "Rescale either by correction factor or by giving target luminosity, not both!"
    assert bolometric_correction is not None or target_bolometric_luminosity is not None,\
        "The lightcurve must be converted to bolometric luminosity! Either provide a correction" + \
        "factor, or provide a mean bolometric luminosity to rescale to."

    if error_ratio is None:
        # If we've not been given a S/N ratio, the input file should have one
        try:
            lightcurve = apt.Table.read(lightcurve_file, format='ascii', names=['time', 'value', 'error'])
            lightcurve['error'].unit = value_units
        except:
            print("The input file does not have errors! Provide S/N ratio via 'error_ratio' argument")
            sys.exit(1)
    else:
        # Otherwise, construct a fake error from the S/N ratio
        lightcurve = apt.Table.read(lightcurve_file, format='ascii', names=['time', 'value'])
        lightcurve['error'] = lightcurve['value']/error_ratio

    lightcurve['time'].unit = time_units
    lightcurve['value'].unit = value_units
    lightcurve['time'].meta['name'] = time_name
    lightcurve['value'].meta['name'] = value_name

    value_orig = lightcurve['value']

    if delta_continuum_range is not None:
        # If we're correcting the continuum range
        lc_min = np.amin(lightcurve['value'])
        lc_max = np.amax(lightcurve['value'])
        lc_mean = np.mean(lightcurve['value'])
        lc_dc_range = (lc_max - lc_min) / (lc_mean * 2)
        print("Rescaling ΔC. Current range: {:.2g}, ({:.3g}:{:.3g}:{:.3g} {})".format(
            lc_dc_range, lc_min, lc_mean, lc_max, lightcurve['value'].unit))
        lightcurve['value'] -= lc_mean
        lightcurve['value'] *= delta_continuum_range / lc_dc_range
        lightcurve['error'] *= delta_continuum_range / lc_dc_range
        lightcurve['value'] += lc_mean
        lc_dc_range = (np.amax(lightcurve['value']) - np.amin(lightcurve['value'])) / (np.mean(lightcurve['value']) * 2)
        print("Rescaled ΔC. New: {:.2g}, ({:.3g}:{:.3g}:{:.3g} {})".format(lc_dc_range,
              np.amin(lightcurve['value']), np.mean(lightcurve['value']), np.amax(lightcurve['value']), lightcurve['value'].unit))

    if bolometric_correction:
        # If we're correcting e.g. from monochromatic to bolometric
        lightcurve['value'] *= bolometric_correction.value
        lightcurve['error'] *= bolometric_correction.value
        lightcurve['value'].unit *= bolometric_correction.unit
        lightcurve['error'].unit *= bolometric_correction.unit
    elif target_bolometric_luminosity:
        # If we're rescaling this to a given bolometric luminosity
        rescale_factor = target_bolometric_luminosity.value / np.mean(lightcurve['value'])
        lightcurve['value'] *= rescale_factor
        lightcurve['error'] *= rescale_factor
        lightcurve['value'].unit = target_bolometric_luminosity.unit
        lightcurve['error'].unit = target_bolometric_luminosity.unit

    # Calculate the bounds of the lightcurve for use later
    lightcurve.meta['min'] = np.amin(lightcurve['value']) * lightcurve['value'].unit
    lightcurve.meta['mean'] = np.mean(lightcurve['value']) * lightcurve['value'].unit
    lightcurve.meta['max'] = np.amax(lightcurve['value']) * lightcurve['value'].unit

    fig, ax1 = plt.subplots(1)
    ax2 = ax1.twinx()
    ax1.set_title("Continuum Rescaling")
    ax1.set_xlabel("Time (MJD)")
    ax1.set_ylabel("Flux (original)")
    ax2.set_ylabel("Flux (rescaled)")
    l_orig = ax1.plot(lightcurve["time"], value_orig, '-', c='r', label='Original')
    l_resc = ax2.plot(lightcurve["time"], lightcurve["value"], '--', c='b', label='Rescaled')
    lns = l_orig+l_resc
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    return lightcurve


def spectrum(spectrum_file, bins, values, frequency=True, limits=None,
             wave_units=None, value_units=None,
             wave_name=None, value_name=None,
             error=None,
             subtract_continuum_with_mask=None, rebin_to=None):
    """
    Imports a spectrum, and converts to target units

    Returns:
        astropy.Table:  Table of input file. Key columns are 'wave', 'value' and 'error'

    """

    # Import a spectrum with bins
    spectrum = ap.table.Table.read(spectrum_file, format='ascii')
    assert type(wave_units) is u.Quantity or type(wave_units) is u.Unit, \
        "Please provide the units that wavelength is to be taken or produced in!"

    # Rename value column and assign units
    spectrum.rename_column(values, 'value')
    spectrum['value'].unit = value_units
    if error:
        # If there are errors, set the value
        spectrum.rename_column(error, 'error')
    else:
        # If there's no ratio, there's zero
        spectrum['error'] = 0
    spectrum['error'].unit = value_units

    # Sort out which way around the array goes
    if frequency:
        # If we're using frequency, not wavelength
        spectrum.rename_column(bins, 'freq')
        if spectrum['freq'][0] < spectrum['freq'][-1]:
            # We want to go from high frequency to low frequency
            spectrum.reverse()
        if limits:
            # If we're removing data outside of certain limits
            to_remove = []
            for i in range(0, len(spectrum)):
                # For each line
                if spectrum['freq'][i] > limits[1].value or spectrum['freq'][i] < limits[0].value:
                    # If the frequency is outside of our range, remove it
                    to_remove.append(i)
            spectrum.remove_rows(to_remove)

    else:
        spectrum.rename_column(bins, 'wave')
        # We want to go from low wavelength to high wavelength
        if spectrum['wave'][0] > spectrum['wave'][-1]:
            spectrum.reverse()
        if limits:
            # If we're removing data outside of certain limits
            to_remove = []
            for i in range(0, len(spectrum)):
                # For each line
                if spectrum['wave'][i] > limits[1].value or spectrum['wave'][i] < limits[0].value:
                    # If the wavelength is outside of our range, remove it
                    to_remove.append(i)
            spectrum.remove_rows(to_remove)

    # If we're taking in frequency and converting to wavelength
    if frequency is True:
        # Rename to internal names and calculate minima and maxima (a la voronoi?)
        bin_min_data = np.zeros(len(spectrum))
        bin_max_data = np.zeros(len(spectrum))
        for i in range(0, len(spectrum['freq'])-1):
            bin_max_data[i] = (spectrum['freq'][i+1] + spectrum['freq'][i]) / 2
        for i in range(1, len(spectrum['freq'])):
            bin_min_data[i] = (spectrum['freq'][i-1] + spectrum['freq'][i]) / 2

        # Assume end bins either side are symmetrical about the midpoint
        # This is VERY NOT TRUE for log spectra; TODO add log mode that assumes even spacing in logspace
        bin_min_data[0] = spectrum['freq'][0] - (spectrum['freq'][1]-spectrum['freq'][0])
        bin_max_data[-1] = spectrum['freq'][-1] + (spectrum['freq'][-1]-spectrum['freq'][-2])

        # Assign bin bound arrays
        spectrum["freq_min"] = bin_min_data
        spectrum["freq_max"] = bin_max_data

        # Add units to everything
        # Calculate wavelength bins from the frequencies we've been given
        spectrum['freq'].unit = 1/u.s
        spectrum['freq_min'].unit = 1/u.s
        spectrum['freq_max'].unit = 1/u.s
        spectrum['wave'] = (apc.c / spectrum['freq'].quantity).to(wave_units)
        spectrum['wave_max'] = (apc.c / spectrum['freq_min'].quantity).to(wave_units)
        spectrum['wave_min'] = (apc.c / spectrum['freq_max'].quantity).to(wave_units)

    else:
        bin_min_data = np.zeros(len(spectrum))
        bin_max_data = np.zeros(len(spectrum))
        for i in range(0, len(spectrum)-1):
            bin_max_data[i] = (spectrum['wave'][i+1] + spectrum['wave'][i]) / 2
        bin_max_data[-1] = spectrum['wave'][-1] + (spectrum['wave'][-1]-spectrum['wave'][-2])

        for i in range(1, len(spectrum)):
            bin_min_data[i] = (spectrum['wave'][i-1] + spectrum['wave'][i]) / 2
        bin_min_data[0] = spectrum['wave'][0] - (spectrum['wave'][1]-spectrum['wave'][0])
        spectrum["wave_min"] = bin_min_data
        spectrum["wave_max"] = bin_max_data

        spectrum['wave'].unit = wave_units
        spectrum['wave_min'].unit = wave_units
        spectrum['wave_max'].unit = wave_units
        spectrum['freq'] = (apc.c / spectrum['wave'].quantity).to(1/u.s)
        spectrum['freq_max'] = (apc.c / spectrum['wave_min'].quantity).to(1/u.s)
        spectrum['freq_min'] = (apc.c / spectrum['wave_max'].quantity).to(1/u.s)

    continuum_fit = None
    if subtract_continuum_with_mask is not None:
        bins = np.array(spectrum['wave'])
        values = np.array(spectrum['value'])
        masked_bins = np.ma.masked_inside(bins, subtract_continuum_with_mask[0].value,
                                          subtract_continuum_with_mask[1].value)
        masked_values = np.ma.array(values, mask=np.ma.getmaskarray(masked_bins), copy=True)
        continuum_fit = np.poly1d(np.ma.polyfit(masked_bins, masked_values, 1))
        spectrum['value'] -= continuum_fit(spectrum['wave'])

        spectrum.remove_rows(slice(np.searchsorted(spectrum["wave"], subtract_continuum_with_mask[1]), len(spectrum)+1))
        spectrum.remove_rows(slice(0, np.searchsorted(spectrum["wave"], subtract_continuum_with_mask[0])))

        fig, ax = plt.subplots(1, 1)
        ax2 = ax.twinx()
        ax.set_title("Continuum Subtraction")
        l_unmod = ax.plot(bins, values, label="Original", c='k')
        l_masked = ax.plot(bins, masked_values, label="Masked original", c='g')
        l_fit = ax.plot(spectrum['wave'], continuum_fit(spectrum['wave']), label="Fit to mask", c='b')
        # No longer necessary now we output to many DP
        # l_fit_step = ax.plot(spectrum['wave'], np.around(continuum_fit(spectrum['wave']), 2), label="Fit (stepped)", c='b')
        l_mod = ax2.plot(spectrum['wave'], spectrum['value'], label="Subtracted", c='r')
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux (non-subtracted)")
        ax2.set_ylabel("Flux (subtracted)")

        lns = l_unmod+l_masked+l_fit+l_mod
        # lns = l_unmod+l_masked+l_fit+l_fit_step+l_mod
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)
        # plt.show()

    if rebin_to:
        # If we're rebinning to X bins
        wave_bounds = np.linspace(spectrum['wave'][0], spectrum['wave'][-1], rebin_to+1)
        wave_midpoints = np.zeros(rebin_to)
        values = np.zeros(rebin_to)
        errors = np.zeros(rebin_to)

        for i in range(0, rebin_to):
            wave_midpoints[i] = (wave_bounds[i] + wave_bounds[i+1]) / 2
            full_bin_width = wave_bounds[i+1] - wave_bounds[i]
            for j in range(0, len(spectrum)):
                if spectrum["wave_min"][j] > wave_bounds[i+1] or spectrum["wave_max"][j] < wave_bounds[i]:
                    continue
                elif spectrum["wave_min"][j] > wave_bounds[i] and spectrum["wave_max"][j] < wave_bounds[i+1]:
                    bin_width = spectrum["wave_max"][j] - spectrum["wave_min"][j]
                elif spectrum["wave_min"][j] < wave_bounds[i]:
                    bin_width = spectrum["wave_max"][j] - wave_bounds[i]
                elif spectrum["wave_max"][j] > wave_bounds[i+1]:
                    bin_width = wave_bounds[i+1] - spectrum["wave_min"][j]

                values[i] += spectrum["value"][j] * bin_width / full_bin_width
                errors[i] += spectrum["error"][j] * bin_width / full_bin_width

            if values[i] < 0:
                values[i] = 0

        freq_bounds = (apc.c / (wave_bounds * wave_units)).to(1/u.s).value
        freq_midpoints = (apc.c / (wave_midpoints * wave_units)).to(1/u.s).value
        freq_min = freq_bounds[1:]
        freq_max = freq_bounds[:-1]

        fig, ax = plt.subplots(1)
        ax.set_title("Rebinning from {} to {} bins".format(len(spectrum), rebin_to))
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux")
        ax.plot(spectrum["wave"], spectrum["value"], '-', c='r', zorder=1, label="Original")
        ax.errorbar(wave_midpoints, values, errors, c='b', label="Rebinned")
        ax.legend()
        # plt.show()

        # Replace the existing spectrum with rebinned values and errors
        spectrum.remove_rows(slice(rebin_to, len(spectrum)+1))
        spectrum["value"] = values
        spectrum["error"] = errors
        spectrum["value"].unit = value_units
        spectrum["error"].unit = value_units

        spectrum["wave"] = wave_midpoints
        spectrum["wave_min"] = wave_bounds[:-1]
        spectrum["wave_max"] = wave_bounds[1:]
        spectrum["wave"].unit = wave_units
        spectrum["wave_min"].unit = wave_units
        spectrum["wave_max"].unit = wave_units

        spectrum["freq"] = freq_midpoints
        spectrum["freq_min"] = freq_min
        spectrum["freq_max"] = freq_max
        spectrum["freq"].unit = 1/u.s
        spectrum["freq_min"].unit = 1/u.s
        spectrum["freq_max"].unit = 1/u.s

    # Set names
    spectrum['wave'].meta['name'] = wave_name
    spectrum['wave_min'].meta['name'] = wave_name
    spectrum['wave_max'].meta['name'] = wave_name
    spectrum['value'].meta['name'] = value_name
    spectrum['error'].meta['name'] = value_name

    if subtract_continuum_with_mask:
        return [spectrum, continuum_fit]
    else:
        return spectrum


def import_caramel(caramel_line_file, caramel_spectra_file):
    """
    Routine to import CARAMEL output spectra into the same spectra format as tss_process creates.
    """
    print("Importing CARAMEL files")

    caramel_line = ap.table.Table.read(caramel_line_file, format='ascii',
                                       names=['time', 'value', 'error'])
    caramel_line['time'].unit = u.s

    caramel_spectra_stream = open(caramel_spectra_file)

    lines = caramel_spectra_stream.readlines()
    np_lines = []
    for line in lines[1:]:
        np_lines.append(np.array([float(x) for x in line.strip().split()]))

    # CARAMEL spectra all start at an arbibtary value, they have no zeroes. So remove this.
    # caramel_min = 999999999
    # for time_index in range(len(caramel_line)):
    #     if np.amin(np_lines[1+time_index*2]) < caramel_min:
    #         caramel_min = np.amin(np_lines[1+time_index*2])
    caramel_min = 1

    caramel_spectra = ap.table.Table()  # Line 0 is a comment
    caramel_spectra['wave'] = np_lines[0]  # Wavelengths
    caramel_spectra['wave'].unit = u.angstrom
    caramel_spectra['value'] = np.zeros(len(np_lines[0]))  # Dummy for mean
    caramel_spectra['error'] = np_lines[2]  # Dummy for error
    caramel_spectra['value_min'] = np_lines[1]  # Dummy for min
    caramel_spectra['value_max'] = np_lines[1]  # Dummy for max

    # Read the spectra for each time-step
    for time_index in range(len(caramel_line)):
        print(time_index, '/', len(caramel_line))
        time = caramel_line['time'].quantity[time_index].to(ucds.MJD)
        spectra_index = 1 + (time_index * 2)  # The first line is wave, rest are spectra
        caramel_spectra['value {}'.format(time)] = np_lines[spectra_index] - caramel_min
        caramel_spectra['error {}'.format(time)] = np_lines[spectra_index+1]
        caramel_spectra['value {}'.format(time)].meta['name'] = 'CARAMEL'
        caramel_spectra["value {}".format(time)].meta['time'] = time

        # Add this column to create the mean
        caramel_spectra['value'] += np_lines[spectra_index] - caramel_min

    caramel_spectra['value'] /= len(caramel_line)

    # The summed line flux includes the caramel forced minimum for everything
    caramel_line['value'] -= (caramel_min) * len(caramel_spectra)
    return caramel_line, caramel_spectra
