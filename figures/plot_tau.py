import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.table import Table
import sys

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')

def interp(original):
    interpolated = np.zeros(len(original)-1)

    for i in range(len(interpolated)):
        interpolated[i] = (original[i] + original[i+1] )/2

    return interpolated


def load_grid(filename):
    x = np.loadtxt(filename+'grid_x.txt')
    z = np.loadtxt(filename+'grid_z.txt')
    x[0] = x[1]
    z[0] = z[1]
    x[-1] = x[-2]
    z[-1] = z[-2]
    return [x, z]


def plot_dat(table, grid_x, grid_z, title, label, volume=True, contour=None):
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.set_xlabel('Log X (cm)')
    ax.set_ylabel('Log Z (cm)')

    size = (len(grid_x)-1, len(grid_z)-1)
    data = np.reshape(table['var'], size)
    if volume:
        for xi in range(size[0]):
            for zi in range(size[1]):
                # We need to correct the per-area emission to a per-volume emission
                area = 2. * 2. * np.pi * (grid_x[xi+1]-grid_x[xi]) * (grid_z[zi+1]-grid_z[zi])
                area += 2. * np.pi * grid_x[xi+1] * (grid_z[zi+1]-grid_z[zi])
                area += 2. * np.pi * grid_x[xi] * (grid_z[zi+1]-grid_z[zi])
                if area > 0:
                    data[xi, zi] = data[xi, zi] / area

    im = ax.pcolormesh(np.log10(grid_x), np.log10(grid_z), np.ma.log10(data.T))
    ax.set_xlim(16.6, 20)
    ax.set_ylim(14.95, 19)
    cbar = fig.colorbar(im, ax=ax).set_label(label)

    if contour:
        ax.contour(np.log10(interp(grid_x)), np.log10(interp(grid_z)), data.T, levels=[1], colors='red')
    return fig


x, z = load_grid('qso_100.dom0.')
table_h1 = Table.read('qso_100.tau.ionH1.dat', format='ascii')

fig = plot_dat(table_h1, x, z, 'H-I', r'Log optical depth $\tau$ at 6562.8$\AA$', volume=False, contour=[1.0])
fig.savefig('qso_100.tau.ionH1.eps')
