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
import matplotlib
from astropy import units as u


seconds_per_day = 60*60*24
batch_size = 1e8


# ==============================================================================
def calculate_FWHM(X, Y):
    """Calculate FWHM from arrays"""
    # Taken from http://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
    # Create 'difference' array by subtracting half maximum
    d = Y - (np.amax(Y) / 2)
    # Find the points where the difference is positive
    indexes = np.where(d > 0)[0]
    # The first and last positive points are the edges of the peak
    return abs(X[indexes[-1]] - X[indexes[0]])


def calculate_modal_value(X, Y):
    """Find the modal delay"""
    return X[np.argmax(Y)]


def calculate_midpoints(X):
    X_midp = np.zeros(shape=len(X)-1)
    for i in range(0, len(X)-1):
        X_midp[i] = (X[i] + X[i+1]) / 2
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


def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))


# Add lines for keplerian rotational outflows
def keplerian_plot(ax, keplerian, clr, line, label):
    resolution  = 1000
    r_angle     = np.radians(keplerian["angle"])
    r_mass_bh   = keplerian["mass"] * apc.M_sun.value
    r_rad_grav  = (6 * apc.G.value * r_mass_bh / np.power(apc.c.value, 2))
    ar_wave     = np.zeros(resolution)  # * u.angstrom
    ar_delay    = np.zeros(resolution)  # * u.s
    ar_phase    = np.linspace(0, np.pi*2, resolution)
    ar_rad      = np.linspace(keplerian["radius"][0]*r_rad_grav, keplerian["radius"][1]*r_rad_grav,resolution)
    ar_vel      = np.zeros(resolution)

    # ITERATE OVER BLUE BOUND
    for r_rad, r_delay, r_vel in np.nditer([ar_rad, ar_delay, ar_vel], op_flags=['readwrite']):
        r_rad           = r_rad  # * u.m
        r_vel[...]      = keplerian_velocity( r_mass_bh, r_rad ) * np.sin(r_angle) / (1e3)
        r_delay[...]    = calculate_delay( r_angle, np.pi/2, r_rad, u.day)
    ax.plot(ar_vel, ar_delay/keplerian['rescale'], line, c=clr, label='_nolegend_')

    # ITERATE OVER RED BOUND
    for r_rad, r_delay, r_vel in np.nditer([ar_rad, ar_delay, ar_vel], op_flags=['readwrite']):
        r_rad           = r_rad  # * u.m
        r_vel[...]      = -keplerian_velocity( r_mass_bh, r_rad ) * np.sin(r_angle) / (1e3)
        r_delay[...]    = calculate_delay( r_angle, np.pi/2, r_rad, u.day)
    ax.plot(ar_vel, ar_delay/keplerian['rescale'], line, c=clr, label=label)




kep_sey = {"angle": 40, "mass": 1e7, "radius": [50,5000], "rescale": 2}
kep_sey_rescale = {"angle":40, "mass": 1e7/2, "radius":[50,5000], "rescale": 1}
kep_agn = {"angle": 40, "mass": 1e9, "radius": [50,50000]}


# Set up the multiplot figure and axis
fig, ax = plt.subplots(1, 1, sharey='row')
keplerian_plot(ax, kep_sey, clr='r', line='-', label=r'M_{BH} = 1e7, scale = 1/2')
keplerian_plot(ax, kep_sey_rescale, clr='b', line='--', label=r'M_{BH} = 1e7/2, scale = 1')
ax.legend()
plt.savefig("comparison.eps", bbox_inches='tight')
