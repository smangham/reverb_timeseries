
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
qso = Table.read('qso_100.spec', format='ascii')
sey = Table.read('sey_100.spec', format='ascii')

ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel(r'$F(\lambda)/F(6300Å)$')
ax.set_xlim([6250., 6850.])
ax.set_ylim([0.8, 1.25])

qso_index = None
for i in range(len(qso['Lambda'])):
    if qso['Lambda'][i] < 6300:
        qso_index = i
        break

sey_index = None
for i in range(len(sey['Lambda'])):
    if sey['Lambda'][i] < 6300:
        sey_index = i
        break

print(qso_index, qso['Lambda'][i], qso['A40P0.50'][i])
print(sey_index, sey['Lambda'][i], sey['A40P0.50'][i])

line_qso = ax.plot(qso['Lambda'], qso['A40P0.50']/qso['A40P0.50'][qso_index], label='QSO')
line_sey = ax.plot(sey['Lambda'], sey['A40P0.50']/sey['A40P0.50'][sey_index], label='Seyfert')

ax.legend()
fig.savefig('spectra.eps')
