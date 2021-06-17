import sys
import tfpy

import matplotlib.pyplot as plt

import numpy as np
import numpy.random as npr

import astropy as ap
import astropy.constants as apc
import astropy.table as apt
from astropy import units as u
from astropy.units import cds as ucds


def interpolation_across_range(x, y, x_int):
    """
    Simple linear interpolation function

    Args:
        x (numpy.float):    X values
        y (numpy.float):    Y values
        x_int (float):           X to find Y for
    Returns:
        float:  Linear interpolation of Y for x_int
    """

    if x_int >= x[-1]:
        return y[-1]
    elif x_int <= x[0] == 0:
        return y[0]
    else:
        x_max_index = np.searchsorted(x, x_int)
        x_min_index = x_max_index - 1
        x_int_fraction = (x_int - x[x_min_index]) / (x[x_max_index] - x[x_min_index])
        y_diff = y[x_max_index] - y[x_min_index]
        return y[x_min_index] + (y_diff * x_int_fraction)


def generate_spectrum_bounds(spectrum):
    # Take spectrum and produce full list of wavelength bins for it to feed to the TF
    bounds = list(spectrum["wave_min"])
    bounds.append(spectrum["wave_max"][-1])
    return np.array(bounds)


def generate_tf(databases, spectrum, delay_bins, line, wave, name, limit=999999999, dynamic_range=2):
    """
    Generates the response function for a system.

    Arguments
        databases (Dict): Dictionary of 'min', 'mid' and 'max' data, each containing a dictionary
            with 'path' (to the file), 'continuum' (the continuum used in creation) and 'scale' (number of spectral cycles used)
        spectrum (Table): Spectrum to template the wavelength bins off of
        delay_bins (Int): Number of bins to bin delays by
        line (Int): Python line number to select
        wave (Float): Frequency of the line selected (in A)
        name (String): Name of the output files.
        limit (Int): Number of photons to limit the DB query to. Set low for testing.

    Returns:
        TransferFunction: The response-mapped transfer function.
    """
    db_mid = tfpy.open_database(databases['mid']['path'], "root", "password")
    db_min = tfpy.open_database(databases['min']['path'], "root", "password")
    db_max = tfpy.open_database(databases['max']['path'], "root", "password")

    bounds = generate_spectrum_bounds(spectrum)

    tf_mid = tfpy.TransferFunction(db_mid, name, continuum=databases['mid']['continuum'],
                wave_bins=(len(bounds)-1), delay_bins=delay_bins)
    tf_mid.line(line, wave).wavelength_bins(bounds).delay_dynamic_range(dynamic_range).run(
                scaling_factor=databases['mid']['scale'], limit=limit,verbose=True).plot()
    tf_min = tfpy.TransferFunction(db_min, name+'_min', continuum=databases['min']['continuum'], template=tf_mid).run(
                scaling_factor=databases['min']['scale'], limit=limit).plot()
    tf_max = tfpy.TransferFunction(db_max, name+'_max', continuum=databases['max']['continuum'], template=tf_mid).run(
                scaling_factor=databases['max']['scale'], limit=limit).plot()
    tf_mid.response_map_by_tf(tf_min, tf_max).plot(response_map=True, name='resp')
    return tf_mid


def generate_spectra_base(spectrum, spectra_times):
    """
    Generates the base spectra for each timestep.

    Args:
        spectrum (Table): The base, unmodified spectrum used for the output time series
        spectra_times(Table): Times to produce a spectrum for

    Returns:
        Table: With one spectrum per target spectra time, keyed by the times

    """
    spectra = ap.table.Table([spectrum['wave'], spectrum['value'], spectrum['error']])
    spectra.meta['bounds'] = generate_spectrum_bounds(spectrum)
    spectra['wave'].meta['name'] = spectrum['wave'].meta['name']
    spectra['value'].meta['name'] = spectrum['value'].meta['name']
    spectra['error'].meta['name'] = spectrum['error'].meta['name']
    spectra['value_min'] = spectrum["value"].copy(copy_data=True)
    spectra['value_max'] = spectrum["value"].copy(copy_data=True)

    for time in spectra_times['time']:
        # Add a base spectrum to the output spectra file
        spectra["value {}".format(time)] = spectra['value']
        spectra["value {}".format(time)].meta['time'] = time
    return spectra


def generate_times_and_delta_continuum(transfer_function, lightcurve, delay_max):
    """
    Generates the timesteps to evaluate the TF at and the change in continuum at each

    Arguments:
        transfer_function (TransferFunction): The TF
        lightcurve(Table): The lightcurve base used for this.

    Returns:
        times (np.array): Time domain broken down into small steps
    """
    # We need to evaluate at every bin-width's step
    delay_bins = (transfer_function.delay_bins() * u.s).to(lightcurve['time'].unit)

    if delay_max:
        # If we need to rescale this to a given maximum
        delay_bins *= (delay_max / delay_bins[-1])
    bin_width = (delay_bins[1] - delay_bins[0]).value

    # We need continuum in terms of delta from the mean
    times = ap.table.Table([np.arange(lightcurve['time'][0], lightcurve['time'][-1] + bin_width, bin_width)], names=['time'])
    times['time'].unit = lightcurve['time'].unit
    times['C'] = np.zeros(len(times))
    times['C'].unit = lightcurve['value'].unit
    times['dC'] = np.zeros(len(times))
    times['dC'].unit = lightcurve['value'].unit
    times['dC%'] = np.zeros(len(times))
    times['time'].meta['name'] = lightcurve['time'].meta['name']
    times['C'].meta['name'] = lightcurve['value'].meta['name']
    times['dC'].meta['name'] = lightcurve['value'].meta['name']
    times.meta['delay_bins'] = delay_bins

    for step in range(0, len(times)):
        # calculate the delta continuum from the 'current value and the starting value, hence the pulse contribution to later timesteps
        times['C'][step] = interpolation_across_range(lightcurve['time'], lightcurve['value'], times['time'][step])
        times['dC'][step] = (times['C'][step] - lightcurve.meta['mean'].value)
        times['dC%'][step] = (times['dC'][step] / lightcurve.meta['mean'].value)
    return times


def generate_spectra_min_max(times, transfer_function, spectra, spectrum,
                             continuum_fit=None):
    # TODO: Correct this! Unfortunately it's mis-set in the data we sent out
    delay_bins = transfer_function.delay_bins()
    dC_max = np.amax(times['dC'])
    dC_min = np.amin(times['dC'])

    pulses_added = []

    # First we generate 'extreme' spectra
    print("Generating 'extreme' spectra...".format(len(delay_bins)))
    for i in range(0, len(delay_bins)-1):
        response = transfer_function.response(delay_index=i)
        pulses_added.append((dC_max, i))

        for j in range(0, len(spectrum)):
            # Matching the format of spectra.c line:
            # x *= (freq * freq * 1e-8) / (dfreq * dd * C);
            dfreq = spectrum["freq_max"].quantity[j] - spectrum["freq_min"].quantity[j]
            invwave = (spectrum["freq"].quantity[j] / apc.c).to(1/spectrum["wave"].unit)
            spectra["value_min"][j] += (dC_min * response[j] *
                invwave * spectrum["freq"][j] / dfreq).value
            spectra["value_max"][j] += (dC_max * response[j] *
                invwave * spectrum["freq"][j] / dfreq).value

    np.savetxt("pulses_added_maximum.txt", pulses_added)

    if continuum_fit:
        dC_max = np.amax(times['dC%'])
        dC_min = np.amin(times['dC%'])
        for j in range(0, len(spectrum)):
            spectra["value_min"][j] += continuum_fit(spectrum["wave"].quantity[j]) * dC_min
            spectra["value_max"][j] += continuum_fit(spectrum["wave"].quantity[j]) * dC_max
    return


def generate_spectra_details(times, transfer_function, spectra, spectrum,
                             continuum_fit=None,
                             calculate_error=False,
                             error_over_variation=0.01,
                             verbose=True):
    delay_bins = times.meta['delay_bins']

    # Generate prefactors
    prefactor = np.zeros(len(spectrum))
    for j in range(0, len(spectrum)):
        dfreq = spectrum["freq_max"][j] - spectrum["freq_min"][j]
        invwave = (spectrum["freq"].quantity[j] / apc.c).to(1/spectrum["wave"].unit).value
        prefactor[j] = invwave * spectrum['freq'][j] / dfreq

    # For each timestep, we send out a 'pulse' of continuum
    print("Beginning {} time steps to generate {} spectra...".format(len(times), len(spectra.columns)-5))
    for step in range(0, len(times)-1):
        # For each time bin this pulse is smeared out over
        dC_abs = times['dC'][step]
        if verbose:
            print("Step {}: {:.1f}%".format(step+1, 100.0*(step+1)/len(times)))
        for i in range(0, len(delay_bins)-1):
            # Figure out what the bounds are for the region this pulse affects
            time_range = [times['time'][step] + delay_bins[i].value, times['time'][step] + delay_bins[i+1].value]
            response = transfer_function.response(delay_index=i)
            for column in spectra.colnames[5:]:
                # For each spectrum, if it occurs at a timestamp within this bin
                if time_range[0] <= spectra[column].meta['time'] < time_range[1]:
                    # Add this pulse's contribution to it
                    for j in range(0, len(spectrum)):
                        # Matching the format of spectra.c line:
                        # x *= (freq * freq * 1e-8) / (dfreq * dd * C);
                        spectra[column][j] += dC_abs * response[j] * prefactor[j]

    if calculate_error and error_over_variation:
        # If we're assigning errors
        L_t = np.zeros(len(spectra.colnames[5:]))
        for column, step in enumerate(spectra.colnames[5:]):
            L_t[step] = np.sum(column)
        dL = np.amax(L_t) - np.amin(L_t)
        error = dL / (error_over_variation * np.sqrt(len(spectra)))
        spectrum['error'] = error
        spectra['error'] = error

    if continuum_fit:
        # If we're adding a continuum change to this
        for column in spectra.colnames[5:]:
            # For each spectrum,
            dC_rel = interpolation_across_range(x=times['time'], y=times['dC%'],
                                                x_int=spectra[column].meta['time'])
            for j in range(0, len(spectra)):
                # Add the delta continuum to it
                spectra[column][j] += continuum_fit(spectrum["wave"].quantity[j]) * dC_rel

    return


def generate_times_line_emission(spectra, spectra_times, verbose=False):
    line_times = spectra_times.copy(copy_data=True)
    line_times['time'] = line_times['time'].quantity.to(u.s)
    line_times['line'] = np.zeros(len(line_times))

    # Integrated flux error dL = SQRT(dBin1^2 + dBin2^2 + ...)
    # dL = SQRT(N_Bins * dBin^2) = SQRT(N_Bins) * dBin, we divide by SQRT(N_Bins)
    line_times['line_error'] = np.zeros(len(line_times))
    line_times['line_error'] = np.sqrt(len(spectra)) * spectra['error'][0]

    # For each spectrum in the output, sum the total emission
    # This obviously only works for line spectra!
    for step in range(0, len(line_times)):
        line_times['line'][step] = np.sum(spectra[spectra.colnames[5+step]])

    if verbose:
        print("Variation is: {}".format(np.amax(line_times['line'])-np.amin(line_times['line'])))

    #
    # spec_max = np.amax(line_times['line'])
    # spec_min = np.amin(line_times['line'])

    # error = np.array(line_times['line_error']) / (spec_max - spec_min)
    # error = (error * 9)
    # value = (np.array(line_times['line']) - spec_min) / (spec_max - spec_min)
    # value = (value * 9) + 1
    #
    # line_times['line_error'] = error
    # line_times['line'] = value

    return line_times


def generate_spectra_error(spectra,
                           error=0.01,
                           fudge_factor=1.0):
    # We want integrated flux error / continuum variation <= error
    # err_L / var_L = error
    rhs = error
    print("Generating error, aiming for Line Error/Variation = {}".format(rhs))

    # So the error on the line should be err_L * var_L
    # Change in line is calculated from the spectra:
    line_array = np.zeros(len(spectra.colnames[5:]))
    for index, colname in enumerate(spectra.colnames[5:]):
        line_array[index] = np.sum(spectra[colname])
    var_L = np.amax(line_array) - np.amin(line_array)

    # Multiply by var_L to get error_L
    rhs *= var_L * fudge_factor

    # Now, error on the line = sqrt( error on bin 1^2, error on bin 2^2 ... )
    # Which is sqrt(number of bins) * error on bin
    # So divide through by sqrt(number of bins) to convert to single bin
    rhs /= np.sqrt(len(spectra))

    # We now have the actual error, stick on the spectrum
    spectra['error'] = rhs
    return apply_spectra_error(spectra)


def copy_spectra_error(origin, target, rescale=False):
    # Copy across errors and apply them
    if not rescale:
        target['error'] = origin['error'][0]

    else:
        max_origin = np.argmax(origin['value'])
        max_target = np.argmax(target['value'])
        rescaled_error = origin['error'][0] * target['value'][max_target] / origin['value'][max_origin]
        target['error'] = rescaled_error

    return apply_spectra_error(target)


def apply_spectra_error(spectra):
    # Now we have the final spectra, create clean copies then add the experimental errors
    clean_copy = spectra.copy(copy_data=True)

    for column in spectra.colnames[5:]:
        # For each spectrum
        if 'value' in column:
            for j in range(0, len(spectra)):
                # For each wavelength bin in each spectrum, add a random error
                spectra[column][j] += npr.normal(scale=spectra['error'][j])
    return clean_copy
