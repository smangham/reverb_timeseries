import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import astropy as ap
import numpy as np
from astropy.table import Table

print("SETTING OFF")

fig, ax = plt.subplots()
print("Plotting base LC")

spectra_times = Table.read('spectra_times.dat', format='ascii')

print("- Spectra times")
for time in spectra_times['HEADER']:
    ax.axvline(x=time, color='lightgrey', linewidth=1.0, linestyle='-', zorder=-10)

lc_orig = Table.read('light_1158.dat', format='ascii')
lc_alt = lc_orig.copy(copy_data=True)

ax.set_xlabel('Time (MJD)')
ax.set_ylabel(r'$10^{−15}$ erg cm$^{−2}$ s${^-1}$ $\AA^{-1}$')

print("- Lightcurve")
ax.errorbar(lc_orig['DAY'], lc_orig['FLUX'], yerr=lc_orig['ERROR'], fmt='none', capsize=2, label='Original lightcurve')
ax.set_ylim((0, 80))

# If we're correcting the continuum range
delta_continuum_range = 0.5
lc_min = np.amin(lc_orig['FLUX'])
lc_max = np.amax(lc_orig['FLUX'])
lc_mean = np.mean(lc_orig['FLUX'])
lc_dc_range = (lc_max - lc_min) / (lc_mean * 2)
lc_alt['FLUX'] -= lc_mean
lc_alt['FLUX'] *= delta_continuum_range / lc_dc_range
lc_alt['ERROR'] *= delta_continuum_range / lc_dc_range
lc_alt['FLUX'] += lc_mean

print("- Rescaled Lightcurve")
ax.errorbar(lc_alt['DAY'], lc_alt['FLUX'], yerr=lc_alt['ERROR'], fmt='none', capsize=2, label='Rescaled lightcurve')
ax.plot([], [], color='lightgrey', linewidth=3.0, linestyle='-', label='Spectra times')

ax.legend()

fig.savefig('lightcurve.eps')


print("Plotting spectra LC")
fig2 = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 3)
ax_lc = fig2.add_subplot(gs[0, :])
ax_s1 = fig2.add_subplot(gs[1, 0])
ax_s2 = fig2.add_subplot(gs[1, 1], sharey=ax_s1)
ax_s3 = fig2.add_subplot(gs[1, 2], sharey=ax_s2)
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])

print("- Continuum")
lc_alt_mean = np.mean(lc_alt['FLUX'])
lc_alt['FLUX'] -= lc_alt_mean
lc_alt['FLUX'] *= 100.0
lc_alt['FLUX'] /= lc_alt_mean
lc_alt['ERROR'] *= 100.0
lc_alt['ERROR'] /= lc_alt_mean

spec_1 = 5
spec_2 = 53
spec_3 = 87

ax_lc.errorbar(lc_alt['DAY'], lc_alt['FLUX'], yerr=lc_alt['ERROR'], fmt='none', capsize=2)
ax_lc.axvline(x=spectra_times['HEADER'][spec_1], color='grey', linewidth=2.0, linestyle='-', zorder=-10)
ax_lc.axvline(x=spectra_times['HEADER'][spec_2], color='grey', linewidth=2.0, linestyle='-', zorder=-10)
ax_lc.axvline(x=spectra_times['HEADER'][spec_3], color='grey', linewidth=2.0, linestyle='-', zorder=-10)
ax_lc.set_xlabel('Time (MJD)')
ax_lc.set_ylabel('ΔC (%)')


print("- Columns")
ax_s1.set_title('MJD {}'.format(spectra_times['HEADER'][spec_1]))
ax_s2.set_title('MJD {}'.format(spectra_times['HEADER'][spec_2]))
ax_s3.set_title('MJD {}'.format(spectra_times['HEADER'][spec_3]))
ax_s1.set_xlabel('Wavelength (Å)')
ax_s2.set_xlabel('Wavelength (Å)')
ax_s3.set_xlabel('Wavelength (Å)')

ax_s1.set_ylabel(r'L(λ)/L$_{max}$')
spectra = ap.io.misc.fnunpickle('pickle_spectra_qso_line.pickle')
spectra_max = np.amax(spectra['value_max'])
print(spectra.columns[5].unit)
ax_s1.plot(spectra['wave'], spectra.columns[5+spec_1]/spectra_max)
ax_s2.plot(spectra['wave'], spectra.columns[5+spec_2]/spectra_max)
ax_s3.plot(spectra['wave'], spectra.columns[5+spec_3]/spectra_max)
fig2.savefig('lightcurve_spectra.eps')

print(spectra.columns)
