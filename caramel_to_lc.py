import numpy as np
import matplotlib.pyplot as plt

times_sey = np.loadtxt('sey/caramel_spectra_times_sey.txt', comments='#')[:, 0]
times_qso = np.loadtxt('qso/caramel_spectra_times_qso.txt', comments='#')[:, 0]


def return_line(filename):
    inputs = np.loadtxt(filename, comments='#')
    spectra_values = inputs[1::2]
    spectra_errors = inputs[2::2]

    line_values = np.zeros(len(spectra_values))
    line_errors = np.zeros(len(spectra_errors))

    for index in range(len(spectra_values)):
        line_values[index] = np.sum(spectra_values[index])

        for error in spectra_errors[index]:
            line_errors[index] += error * error
        line_errors[index] = np.sqrt(line_errors[index])

    return line_values, line_errors


line_values_sey, line_errors_sey = return_line('sey/caramel_spectra_sey.txt')
line_values_qso, line_errors_qso = return_line('qso/caramel_spectra_qso.txt')

fig, ax_sey = plt.subplots()
ax_sey.set_xlabel('Time')
ax_sey.set_ylabel('Value (Sey)')
line_sey = ax_sey.errorbar(x=times_sey, y=line_values_sey, yerr=line_errors_sey, fmt='+b', capsize=5, label='Sey')

ax_qso = ax_sey.twinx()
ax_qso.set_ylabel('Value (QSO)')
line_qso = ax_qso.errorbar(x=times_qso, y=line_values_qso, yerr=line_errors_qso, fmt='+r', capsize=5, label='QSO')

ax_sey.legend([line_sey, line_qso], ['Sey', 'QSO'])

fig.savefig('lines_2.eps')
