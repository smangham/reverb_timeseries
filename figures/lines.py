
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy as ap
import numpy as np
from astropy.table import Table
import astropy.units as u
from astropy.units import cds as ucds

# ==============================================================================
# Plot the original and rescaled lightcurves
# ==============================================================================
fig, ax = plt.subplots()
qso = Table.read('out_times_line_qso.dat', format='ascii')
sey = Table.read('out_times_line_sey.dat', format='ascii')

ax.set_xlabel('Day (MJD)')
ax.set_ylabel(r'$\Delta L/\bar{L} (\%)$')
# ax.set_xlim([6250., 6850.])
# ax.set_ylim([0.8, 1.25])

line_qso = ax.plot((qso['time']*u.s).to(ucds.MJD), 100*(qso['line']-np.mean(qso['line']))/np.mean(qso['line']), label='QSO')
line_sey = ax.plot((sey['time']*u.s).to(ucds.MJD), 100*(sey['line']-np.mean(sey['line']))/np.mean(sey['line']), label='Seyfert')

ax.legend()
fig.savefig('lines.eps')
