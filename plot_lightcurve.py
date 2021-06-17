"""
Plots:
1. lightcurve.eps: Driving light curve & rescaled version
2. lightcurve_spectra.eps: Rescaled & CARAMEL light curves & 3 call-out panels with line spectra on
3. spectra_rms.eps: RMS profile for original & CARAMEL fits
4. spectra_mean_base.eps: Comparison of base spectra & mean CARAMEL
5. spectra_mean_series.eps: Comparison of mean spectra & mean CARAMEL
"""
# Plots the lightcurve plus 3 example panels from the spectra
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import figaspect
import astropy as ap
import numpy as np
from astropy.table import Table
from tss_import import import_caramel
import astropy.units as u
from astropy.units import cds as ucds
import tfpy


# ==============================================================================
# Plot the original and rescaled lightcurves
# ==============================================================================
fig, ax = plt.subplots()
spectra_times = Table.read('spectra_times.dat', format='ascii')

# Import the spectra times, and put a line on the rescaled lightcurves for each
for time in spectra_times['HEADER']:
    ax.axvline(x=time, color='lightgrey', linewidth=1.0, linestyle='-', zorder=-10)

# Import the original continuum lightcurve
lc_orig = Table.read('light_1158.dat', format='ascii')
lc_alt = lc_orig.copy(copy_data=True)

ax.set_xlabel('Time (MJD)')
ax.set_ylabel(r'$10^{−15}$ erg cm$^{−2}$ s$^{-1}$ $\AA^{-1}$')

# Errorbar plot of this original lightcurve
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

# Plot the continuum lightcurve
ax.errorbar(lc_alt['DAY'], lc_alt['FLUX'], yerr=lc_alt['ERROR'], fmt='none', capsize=2, label='Rescaled lightcurve')
ax.plot([], [], color='lightgrey', linewidth=3.0, linestyle='-', label='Spectra times')
ax.legend()
fig.savefig('lightcurve.eps')
plt.close(fig)


# ==============================================================================
# Now we plot the lightcurve plus spectra
# ==============================================================================
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
spec_2 = 59
spec_3 = 91

# Load data files
spectra = ap.io.misc.fnunpickle('pickle_spectra_qso_line.pickle')
line = ap.table.Table.read("out_times_line_qso.dat", format='ascii',
                           names=['time', 'value', 'error'])
line_caramel, spectra_caramel = import_caramel("sim1132_line.txt", "sim1132_spectra.txt")

ax_lc.set_xlabel('Time (MJD)')
ax_lc.set_ylabel('ΔC (%)')

ax_s1.set_title('MJD {}'.format(spectra_times['HEADER'][spec_1]))
ax_s2.set_title('MJD {}'.format(spectra_times['HEADER'][spec_2]))
ax_s3.set_title('MJD {}'.format(spectra_times['HEADER'][spec_3]))
ax_s1.set_xlabel('Wavelength (Å)')
ax_s2.set_xlabel('Wavelength (Å)')
ax_s3.set_xlabel('Wavelength (Å)')

ax_s1.set_ylabel(r'L(λ, t)/$\bar{L}$')

plot_lc = ax_lc.errorbar(lc_alt['DAY'], lc_alt['FLUX'], yerr=lc_alt['ERROR'], fmt='none', capsize=2, zorder=1, color='green')
ax_lc.axvline(x=spectra_times['HEADER'][spec_1], color='grey', linewidth=2.0, linestyle='-', zorder=-10)
ax_lc.axvline(x=spectra_times['HEADER'][spec_2], color='grey', linewidth=2.0, linestyle='-', zorder=-10)
ax_lc.axvline(x=spectra_times['HEADER'][spec_3], color='grey', linewidth=2.0, linestyle='-', zorder=-10)

ax_line = ax_lc.twinx()
ax_line.set_ylabel('ΔL (%)')

plot_line = ax_line.plot((line['time']*u.s).to(ucds.MJD),
           100*(line['value']-np.mean(line['value']))/np.mean(line['value']), zorder=2)
# plot_caramel = ax_line.plot(line_caramel['time'].quantity.to(ucds.MJD),
#            100*(line_caramel['value']-np.mean(line_caramel['value']))/np.mean(line_caramel['value']),
#            zorder=3)

ax_lc.legend(
    [plot_lc]+plot_line, #+plot_caramel,
    # ['Continuum', r'H$\alpha$', r'H$\alpha$ CARAMEL fit'],
    ['Continuum', r'H$\alpha$'],
    loc='upper left'
)

# The first 5 columns are wave/base/error/min/max, then 2-col pairs for val/error
index_1 = 5 + spec_1
index_2 = 5 + spec_2
index_3 = 5 + spec_3
index_1c = 5 + 2*spec_1
index_2c = 5 + 2*spec_2
index_3c = 5 + 2*spec_3


# We want to get the mean, but without a baseline this means the mean of the means
spectra_caramel_mean = []
spectra_mean = []
for i in range(len(line_caramel)):
    spectra_caramel_mean.append(np.mean(spectra_caramel.columns[5+2*i]))
    spectra_mean.append(np.mean(spectra.columns[5+i]))
spectra_caramel_mean = np.mean(spectra_caramel_mean)
spectra_mean = np.mean(spectra_mean)

ax_s1.plot(spectra['wave'], spectra.columns[index_1]/spectra_mean, zorder=1, label='Original')
ax_s2.plot(spectra['wave'], spectra.columns[index_2]/spectra_mean, zorder=1)
ax_s3.plot(spectra['wave'], spectra.columns[index_3]/spectra_mean, zorder=1)

# ax_s1.errorbar(spectra_caramel['wave'], spectra_caramel.columns[index_1c]/spectra_caramel_mean,
#                yerr=spectra_caramel.columns[index_1c+1]/spectra_caramel_mean,
#                fmt='none', capsize=1, zorder=2, label='CARAMEL fit')
# ax_s2.errorbar(spectra_caramel['wave'], spectra_caramel.columns[index_2c]/spectra_caramel_mean,
#                yerr=spectra_caramel.columns[index_2c+1]/spectra_caramel_mean,
#                fmt='none', capsize=1, zorder=2)
# ax_s3.errorbar(spectra_caramel['wave'], spectra_caramel.columns[index_3c]/spectra_caramel_mean,
#                yerr=spectra_caramel.columns[index_3c+1]/spectra_caramel_mean,
#                fmt='none', capsize=1, zorder=2)
fig2.savefig('lightcurve_spectra.eps')
plt.close(fig2)


# ==============================================================================
# Now we plot the RMS residuals
# ------------------------------------------------------------------------------
# We want to normalise the spectra first, so the mean luminosity is set to 1.0
# Then we subtract the mean from everything (i.e. 1) to get the residuals
# Then we square it to get the S residuals
# Then we sum the squares
# Then we divide the sum of the squares by the number of squares to get MS-R
# Then we root it to get the RMS-R
# ==============================================================================
fig, ax = plt.subplots()

rms = np.zeros(len(spectra))
rms_caramel = np.zeros(len(spectra_caramel))

# OK! We've normalised to the mean luminosity = 1
for i in range(len(spectra.columns[1:])):
    spectra.columns[1+i] /= spectra_mean

for i in range(len(spectra_caramel.columns[1:])):
    spectra_caramel.columns[1+i] /= spectra_caramel_mean

# Now we take the residuals and square them
for i in range(len(line)):
    rms += np.power((spectra.columns[5+i] - spectra['value']), 2)
    rms_caramel += np.power((spectra_caramel.columns[5+2*i] - spectra_caramel['value']), 2)

rms /= len(line)
rms = np.sqrt(rms)
rms_caramel /= len(line_caramel)
rms_caramel = np.sqrt(rms_caramel)

ax.plot(spectra['wave'], rms, label=r'H$\alpha$')
ax.plot(spectra_caramel['wave'], rms_caramel, label=r'H$\alpha$ CARAMEL fit')
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel(r'RMS L(λ, t)/$\bar{L}$')
ax.legend()
fig.savefig('spectra_rms.eps')
plt.close(fig)

# ==============================================================================
# Now we plot the mean spectra
# ------------------------------------------------------------------------------
#
# ==============================================================================
fig, ax = plt.subplots()

mean = np.zeros(len(spectra))
mean_caramel = np.zeros(len(spectra_caramel))

# Now we take the residuals and square them
for i in range(len(line)):
    mean += spectra.columns[5+i]
    mean_caramel += spectra_caramel.columns[5+2*i]

# spectra['error'] /= spectra_mean
mean /= len(line)
mean_caramel /= len(line)

ax.errorbar(spectra['wave'], spectra['value'], yerr=spectra['error'], label=r'H$\alpha$ base')
ax.plot(spectra_caramel['wave'], mean_caramel, label=r'H$\alpha$ CARAMEL fit mean')
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel(r'$\bar{L}(λ)/\bar{L}$')
ax.legend()
fig.savefig('spectra_mean_base.eps')
plt.close(fig)

fig, ax = plt.subplots()
ax.errorbar(spectra['wave'], mean, yerr=spectra['error'], label=r'H$\alpha$ mean')
ax.plot(spectra_caramel['wave'], mean_caramel, label=r'H$\alpha$ CARAMEL fit mean')
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel(r'$\bar{L}(λ)/\bar{L}$')
ax.legend()
fig.savefig('spectra_mean_series.eps')


# ==============================================================================
# Now we plot the mean spectra
# ------------------------------------------------------------------------------
#
# ==============================================================================

w, h = figaspect(2/1)
fig, ((ax_spec, ax_none),
      (ax_model, ax_model_cb),
      (ax_fit, ax_fit_cb),
      (ax_residual, ax_residual_cb)
      ) = plt.subplots(4, 2, sharex='col', sharey='row',
      gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 3, 3, 3]}, figsize=(w, h))

ax_none.axis('off')
ax_model_cb.axis('off')
ax_fit_cb.axis('off')
ax_residual_cb.axis('off')

data_model = np.zeros([len(spectra_times), len(spectra)])
data_fit = np.zeros([len(spectra_times), len(spectra)])
error_fit = np.zeros([len(spectra_times), len(spectra)])
error_model = np.zeros([len(spectra_times), len(spectra)])
data_residual = np.zeros([len(spectra_times), len(spectra)])

line_fit, spectra_fit = import_caramel("sim1132_line.txt", "sim1132_spectra.txt")
line_model, spectra_model = import_caramel("sim1132_line.txt", "model_qso_spectra.txt")

index_data = 0
index_colnames = 5
while index_colnames < len(spectra_fit.colnames):
    data_fit[index_data, :] = spectra_fit[spectra_fit.colnames[index_colnames]]
    data_model[index_data, :] = spectra_model[spectra_model.colnames[index_colnames]]
    index_colnames += 1
    error_fit[index_data, :] = spectra_fit[spectra_fit.colnames[index_colnames]]
    error_model[index_data, :] = spectra_model[spectra_model.colnames[index_colnames]]
    index_colnames += 1
    index_data += 1

data_residual = (data_fit - data_model) / error_fit

cb_max = max(np.amax(data_model), np.amax(data_fit))
cb_min = min(np.amin(data_model), np.amin(data_fit))

pcol_model = ax_model.pcolor(spectra.meta["bounds"], np.linspace(0, 100, num=101), data_model)
ax_model.set_xlabel(r'Wavelength ($\AA$)')
ax_model.set_ylabel("Epoch")
ax_model.xaxis.set_tick_params(rotation=0, pad=1)
ax_model.yaxis.set_tick_params(rotation=45, labelsize=8)
ax_model.yaxis.tick_left()
ax_model.set_yticks([20, 40, 60, 80, 100])
ax_model.set_xlim([6300, 6850])

pcol_fit = ax_fit.pcolor(spectra.meta["bounds"], np.linspace(0, 100, num=101), data_fit)
ax_fit.set_xlabel(r'Wavelength ($\AA$)')
ax_fit.set_ylabel("Epoch")
ax_fit.xaxis.set_tick_params(rotation=0, pad=1)
ax_fit.yaxis.set_tick_params(rotation=45, labelsize=8)
ax_fit.yaxis.tick_left()
ax_fit.set_yticks([20, 40, 60, 80, 100])
ax_fit.set_xlim([6300, 6850])

cb_max = np.amax(data_residual)
pcol_residual = ax_residual.pcolor(spectra.meta["bounds"], np.linspace(0, 100, num=101),
                                   data_residual, cmap='RdBu_r',
                                   vmin=-cb_max, vmax=cb_max)
ax_residual.set_xlabel(r'Wavelength ($\AA$)')
ax_residual.set_ylabel("Epoch")
ax_residual.xaxis.set_tick_params(rotation=0, pad=1)
ax_residual.yaxis.set_tick_params(rotation=45, labelsize=8)
ax_residual.yaxis.tick_left()
ax_residual.set_yticks([0, 20, 40, 60, 80, 100])
ax_residual.set_xlim([6300, 6850])

ax_spec.set_xlim([6300, 6850])
ax_spec.set_xlabel("λ (Å)")
ax_spec.set_ylabel(r'$Flux$'+'\n(arbitrary)')
ax_spec.yaxis.set_tick_params(labelsize=8, pad=12)

spec_model = np.zeros([len(spectra)])
spec_fit = np.zeros([len(spectra)])
spec_fit_error = np.zeros([len(spectra)])
for i in range(len(spectra)):
    spec_model += data_model[i]
    spec_fit += data_fit[i]
    spec_fit_error += error_fit[i]
spec_model /= len(spectra)
spec_fit /= len(spectra)
spec_fit_error /= len(spectra)

ax_spec.plot(spectra['wave'], spec_model, label='Mean model')
ax_spec.errorbar(spectra['wave'], spec_fit, yerr=spec_fit_error, capsize=1, fmt='none', label='Mean CARAMEL')
ax_spec.legend(fontsize=6, framealpha=0.0, loc='upper left')

tf_wave = 6562.8
ax_model.axvline(tf_wave, color='red')
ax_fit.axvline(tf_wave, color='red')
ax_residual.axvline(tf_wave, color='red')
ax_spec.axvline(tf_wave, color='red')

ax_vel = ax_spec.twiny()
ax_vel.set_xlim(ax_spec.get_xlim())
ax_vel.set_xticks([
    tfpy.doppler_shift_wave(tf_wave, -1e7),
    tf_wave,
    tfpy.doppler_shift_wave(tf_wave, +1e7)
])
ax_vel.set_xticklabels([
    r"10", r"0", r"10"
])
ax_vel.set_xlabel(r'Velocity (10$^{3}$ km s$^{-1}$)')

cbar_data = fig.colorbar(pcol_model, ax=ax_model_cb, orientation="vertical", fraction=1)
cbar_data.set_label(r'$F_{\rm model}$'+'\n(arbitrary)')
cbar_data.ax.tick_params(labelsize=8)
cbar_fit = fig.colorbar(pcol_fit, ax=ax_fit_cb, orientation="vertical", fraction=1)
cbar_fit.set_label(r'$F_{\rm CARAMEL}$'+'\n(arbitrary)')
cbar_fit.ax.tick_params(labelsize=8)
cbar_residual = fig.colorbar(pcol_residual, ax=ax_residual_cb, orientation="vertical", fraction=1)
cbar_residual.set_label('Standardised\nResiduals')
cbar_residual.ax.tick_params(labelsize=8)

fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("caramel_model_fit.eps", bbox_inches='tight')
plt.close(fig)
