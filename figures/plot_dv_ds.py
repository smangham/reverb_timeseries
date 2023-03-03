import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.table import Table
import sys
import vtk

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')


def interp(original):
    interpolated = np.zeros(len(original)-1)

    for i in range(len(interpolated)):
        interpolated[i] = (original[i] + original[i+1]) / 2

    return interpolated


def load_grid(filename):
    x = np.loadtxt(filename+'grid_x.txt')
    z = np.loadtxt(filename+'grid_z.txt')
    x[0] = x[1]
    z[0] = z[1]
    x[-1] = x[-2]
    z[-1] = z[-2]
    return [x, z]


def plot_dat(table, grid_x, grid_z, title, label, volume=True, contour=None, log=True):
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.set_xlabel('Log X (cm)')
    ax.set_ylabel('Log Z (cm)')

    size = (len(grid_x)-1, len(grid_z)-1)
    data = np.reshape(table['var'], size)
    inwind = np.reshape(table['inwind'], size)
    if volume:
        for xi in range(size[0]):
            for zi in range(size[1]):
                # We need to correct the per-area emission to a per-volume emission
                area = 2. * 2. * np.pi * (grid_x[xi+1]-grid_x[xi]) * (grid_z[zi+1]-grid_z[zi])
                area += 2. * np.pi * grid_x[xi+1] * (grid_z[zi+1]-grid_z[zi])
                area += 2. * np.pi * grid_x[xi] * (grid_z[zi+1]-grid_z[zi])
                if area > 0:
                    data[xi, zi] = data[xi, zi] / area

    if log:
        im = ax.pcolormesh(np.log10(grid_x), np.log10(grid_z), np.ma.log10(data.T))
    else:
        max = np.amax(np.abs(data))
        im = ax.pcolormesh(np.log10(grid_x), np.log10(grid_z), np.ma.masked_where(inwind.T < 0, data.T),
                           cmap='RdBu_r', vmax=max, vmin=-max)

    ax.set_xlim(16.6, 20)
    ax.set_ylim(14.95, 19)
    cbar = fig.colorbar(im, ax=ax).set_label(label)

    if contour:
        ax.contour(np.log10(interp(grid_x)), np.log10(interp(grid_z)), data.T, levels=[1], colors='red')
    return fig


x, z = load_grid('qso_100.dom0.')
table_pol = Table.read('qso_100.dv_ds.poloidal.A40.P05.dat', format='ascii')
table_rot = Table.read('qso_100.dv_ds.rotational.A40.P05.dat', format='ascii')
table_real = Table.read('qso_100.dv_ds.real.A40.P05.dat', format='ascii')
#
# fig, (ax1, ax2) = plt.subplots(2)
# fig.savefig('test.eps')
# exit()

fig = plot_dat(table_rot, x, z, 'Phase = 0.5', r'$\dot{v}_{rotational}|_{i=40^\circ}$ (m s$^{-2}$)', log=False, volume=False)
fig.savefig('qso_100.dv_ds.rotational.eps')
plt.close(fig)
fig = plot_dat(table_pol, x, z, 'Phase = 0.5', r'$\dot{v}_{poloidal}|_{i=40^\circ}$ (m s$^{-2}$)', log=False, volume=False)
fig.savefig('qso_100.dv_ds.poloidal.eps')
plt.close(fig)
fig = plot_dat(table_real, x, z, 'Phase = 0.5', r'$\dot{v}|_{i=40^\circ}$ (m s$^{-2}$)', log=False, volume=False)
fig.savefig('qso_100.dv_ds.real.eps')
plt.close(fig)
#
# for phase in np.arange(0.0, 1.0, 0.05):
#     print('File: ', 'qso_100.dv0_ds_A40.0_P{:.2f}.dat'.format(phase))
#     table = Table.read('qso_100.dv0_ds_A40.0_P{:.2f}.dat'.format(phase), format='ascii')
#     fig = plot_dat(table_rot, x, z, 'Phase = {:.2f}'.format(phase),
#                    r'$\dot{v}_{projected}|_{i=40^\circ}$ (m s$^{-1}$)', log=False, volume=False)
#     fig.savefig('dv_ds.projected.phase.{:.2f}.eps'.format(phase))
#     plt.close(fig)

# ==============================================================================


points_legacy = []
cells_legacy = []
data_legacy = []

points_count = 0
voxels_count = 0
phase_offset = 0.05
phases = np.arange(0.0, 1.0, phase_offset)
grid_p = np.linspace(0.0 - phase_offset/2,
                     1.0 - phase_offset/2,
                     1.0/phase_offset + 1, endpoint=True) * 2 * np.pi

print(grid_p)

vals = np.zeros([len(phases), len(x)-1, len(z)-1])

for p_i, phase in enumerate(phases):
    table = Table.read('qso_100.dv0_ds_A87.0_P{:.2f}.dat'.format(phase), format='ascii')
    size = (len(x)-1, len(z)-1)
    data = np.reshape(table['var'], size)
    inwind = np.reshape(table['inwind'], size)
    phase_radians = 2 * np.pi * phase
    phase_radians_next = 2 * np.pi * (phase + phase_offset)
    cos_p = np.cos(phase_radians)
    sin_p = np.sin(phase_radians)
    cos_pn = np.cos(phase_radians_next)
    sin_pn = np.sin(phase_radians_next)

    # print('File: ', 'qso_100.dv0_ds_A40.0_P{:.2f}.dat'.format(phase))
    print('Phase: {:.2f}, Cos: {:.2f}, Sin: {:.2f}'.format(phase_radians, cos_p, sin_p))

    for x_i in range(size[0]):
        for z_i in range(size[1]):
            # Save value to file
            if inwind[x_i, z_i] >= 0:
                vals[p_i, x_i, z_i] = data[x_i, z_i]
            else:
                vals[p_i, x_i, z_i] = np.NAN

            if inwind[x_i, z_i] >= 0:
                cells_legacy.append([points_count + offset for offset in range(0, 8)])
                data_legacy.append(data[x_i, z_i])

                count = 0  # 8 points per voxel numbered 0-8
                for offset_x in [0, 1]:
                    for offset_z in [0, 1]:
                        points_legacy.append((x[x_i+offset_x] * cos_p,
                                              x[x_i+offset_x] * sin_p,
                                              z[z_i+offset_z]))
                        points_legacy.append((x[x_i+offset_x] * cos_pn,
                                              x[x_i+offset_x] * sin_pn,
                                              z[z_i+offset_z]))
                        count += 2
                        points_count += 2
                voxels_count += 1


for z_i in range(0, 100, 10):
    test = inwind[:, z_i]
    x_i = np.argwhere(test == 1)[-1]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    height = (z[z_i]+z[z_i+1])/2
    fig.suptitle('Z = {:.2E} cm'.format(height))
    ax.set_xlabel(r'Phase')
    # ax.set_ylabel('Radius (cm)')
    # ax.set_xlim(0, np.pi*2)
    ax.set_ylim(16.6, np.log10(x[x_i]))
    max = np.nanmax(np.abs(vals[:, :, z_i]))
    im = ax.pcolormesh(grid_p, np.log10(x), vals[:, :, z_i].T,
                       cmap='RdBu_r')  # , vmax=max, vmin=-max)
    cbar = fig.colorbar(im, ax=ax).set_label(r'$\dot{v}|_{i=87^\circ}$ (m s$^{-2}$)')
    ax.plot([np.pi, np.pi], [16.6, np.log10(x[x_i])], c='red', zorder=99)
    fig.savefig('qso_100.dv0_ds_A87.0_Z{}.eps'.format(z_i))
    plt.close(fig)

with open('test.vtk', 'w') as f:
    f.write('# vtk DataFile Version 3.0\n')
    f.write('3D scalar data\n')
    f.write('ASCII\n')
    f.write('DATASET UNSTRUCTURED_GRID\n')
    f.write('POINTS {} float\n'.format(points_count))
    for point in points_legacy:
        f.write('{}\n'.format(' '.join([str(coord) for coord in point])))

    f.write('\nCELLS {} {}\n'.format(voxels_count, voxels_count*9))
    for cell in cells_legacy:
        f.write('8 {}\n'.format(' '.join([str(point) for point in cell])))

    f.write('\nCELL_TYPES {}\n'.format(voxels_count))
    for cell in cells_legacy:
        f.write('12\n')

    f.write('\nCELL_DATA {}\n'.format(voxels_count))
    f.write("SCALARS dvds float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for data in data_legacy:
        f.write('{}\n'.format(data))
