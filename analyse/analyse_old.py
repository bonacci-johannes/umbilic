from typing import Tuple

import matplotlib
import numpy
import matplotlib.pyplot as plt
import os
import glob

import scipy

# matplotlib.use('macosx')
# matplotlib.use('QtAgg')

# %%
path = 'TWO_DIR/EXAMPLE_25'

# Assuming systemparameter.txt and Window_Parameter.txt are in the current working directory
syspara = numpy.loadtxt(f'{path}/systemparameter.txt', comments='%')
window = numpy.loadtxt(f'{path}/Window_Parameter.txt', comments='%')

record_files = sorted(glob.glob(f'{path}/Struc_fct_records/*_part_1.txt'))

# %%
rho1 = syspara[3]
rho2 = syspara[4]
gamma = syspara[5]


# %%
class Struct:
    x_points = 300 + 1

    def __init__(self, window, record_files, gamma):
        self.gamma = gamma
        self.window = window
        self.record_files = record_files

        # compute grid points for each time
        self.t_x = {
            window[t_idx, 0]: numpy.arange(start=self.window[t_idx, 3], stop=self.window[t_idx, 4],
                                           step=self.window[t_idx, 6])
            for t_idx in range(1, len(window[:, 0]))}

        # allocate memory for data
        self.t_s_bar = {time: numpy.zeros(shape=(len(x), 2, len(record_files))) for time, x in self.t_x.items()}
        self.t_s_bar_smooth = {time: numpy.zeros(shape=(len(x), 2, len(record_files))) for time, x in self.t_x.items()}
        self.t_s_eigen = {time: numpy.zeros(shape=(self.x_points, 2, 2)) for time, x in self.t_x.items()}

        # load data
        for n, file in enumerate(self.record_files):
            data = numpy.loadtxt(file, comments='%')
            for time in self.t_s_bar.keys():
                # get idx of time in window
                t_idx = numpy.where(self.window[:, 0] == time)[0][0]

                # get data for time and reshape to 2x2 matrix
                y = data[:, t_idx]
                y = y.reshape((int(y.size / 4), 2, 2), order='F')
                y = y[:len(self.t_x[time]), :, :]

                # compute model symmetric part
                s_diag = 0.5 * (y[:, 0, 0] + y[:, 1, 1])
                s_off = 0.5 * (y[:, 0, 1] + y[:, 1, 0])

                # compute mirror symmetric part
                s_diag = 0.5 * (s_diag + s_diag[::-1])
                s_off = 0.5 * (s_off + s_off[::-1])
                self.t_s_bar[time][:, :, n] = numpy.array([s_diag, s_off]).T

        # smooth data
        self.times = numpy.array(list(self.t_x.keys()))
        self.x_grid = numpy.linspace(start=self.window[1:, 3], stop=self.window[1:, 4], num=self.x_points)
        self.s_bar_smooth_single = numpy.zeros(shape=(len(self.times), self.x_points, 2, len(self.record_files)))
        for nt, time in enumerate(self.times):
            std_diag = 0.5 * numpy.std(self.t_s_bar[time][:, 0, :], axis=1)
            std_off = 0.5 * numpy.std(self.t_s_bar[time][:, 1, :], axis=1)
            for n in range(len(record_files)):
                spl_tck_diag = scipy.interpolate.splrep(self.t_x[time], self.t_s_bar[time][:, 0, n], w=1 / std_diag)
                spl_tck_off = scipy.interpolate.splrep(self.t_x[time], self.t_s_bar[time][:, 1, n], w=1 / std_off)
                self.s_bar_smooth_single[nt, :, 0, n] = scipy.interpolate.splev(self.x_grid[:, nt], spl_tck_diag)
                self.s_bar_smooth_single[nt, :, 1, n] = scipy.interpolate.splev(self.x_grid[:, nt], spl_tck_off)

        # compute normed eigen modes
        self.s_smooth_single = numpy.zeros(shape=(len(self.times), self.x_points, 2, len(self.record_files)))
        self.s_smooth_single[:, :, 0, :] = (self.s_bar_smooth_single[:, :, 0, :] - self.s_bar_smooth_single[:, :, 1, :])
        self.s_smooth_single[:, :, 1, :] = (self.s_bar_smooth_single[:, :, 0, :] + self.s_bar_smooth_single[:, :, 1, :])
        self.s_smooth_single[:, :, 0, :] *= (2 + 2 / numpy.sqrt(self.gamma))
        self.s_smooth_single[:, :, 1, :] *= (2 + 2 * numpy.sqrt(self.gamma))

        # compute avg data
        self.s_bar_smooth = numpy.mean(self.s_bar_smooth_single, axis=-1)
        self.s_bar_smooth_std_err = numpy.std(self.s_bar_smooth_single, axis=-1) / numpy.sqrt(len(self.record_files))
        self.s_smooth = numpy.mean(self.s_smooth_single, axis=-1)
        self.s_smooth_std_err = numpy.std(self.s_smooth_single, axis=-1) / numpy.sqrt(len(self.record_files))

        # compute maxima of eigenmodes
        self.s_max = self.s_smooth[:, self.s_smooth.shape[1] // 2, :]
        self.s_max_std = self.s_smooth_std_err[:, self.s_smooth_std_err.shape[1] // 2, :]


struc = Struct(window, record_files, gamma=gamma)

# %%
plt.figure()
plt.plot(numpy.log(struc.times), numpy.log(struc.s_max), '.-', label=['$S_1$', '$S_2$'])
plt.xlabel('$\ln(t)$')
plt.ylabel('$\ln(S_i)$')
plt.legend()
plt.show()
# %%
plt.figure()
for n in range(2):
    c_alpha = 1.644854  # 90% confidence interval
    z = (-numpy.diff(numpy.log(struc.times)) /
         (numpy.log(struc.s_max[1:, n]) - numpy.log(struc.s_max[:-1, n])))

    z1 = (-numpy.diff(numpy.log(struc.times)) /
          (numpy.log((struc.s_max + c_alpha * struc.s_max_std)[1:, n]) - numpy.log(
              (struc.s_max - c_alpha * struc.s_max_std)[:-1, n])))
    z2 = (-numpy.diff(numpy.log(struc.times)) /
          (numpy.log((struc.s_max - c_alpha * struc.s_max_std)[1:, n]) - numpy.log(
              (struc.s_max + c_alpha * struc.s_max_std)[:-1, n])))

    plt.plot(numpy.log(struc.times[1:]),
             z, '.-', label=f'$S_{n + 1}$')
    plt.fill_between(numpy.log(struc.times[1:]), z1, z2, alpha=0.5)
plt.xlabel('$\ln(t)$')
plt.ylabel('$z_i(t)$')
plt.legend()
# include grid on plot
plt.grid()
plt.savefig('figures_temp/z_t_crit')
plt.show()
# %%
colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
b = numpy.sqrt(3)/12
plt.figure()
for n, time in enumerate(struc.times):
    color = colors[n % len(colors)]
    x = struc.x_grid[:, n]
    sbar_smooth = struc.s_bar_smooth
    scale = (b * time) ** (2 / 3)

    plt.plot(x / scale, sbar_smooth[n, :, 0] * scale, linestyle='-', color=color)
    plt.plot(x / scale, sbar_smooth[n, :, 1] * scale, linestyle='--', color=color)
    plt.plot(0, 0, 'o', color=color, label=f't={int(time)}', markersize=5)
    plt.xlabel(r'$x / (bt)^{2/3}$')
    plt.ylabel(r'$(bt)^{2/3} \cdot \bar S_{\alpha\beta}$')
    plt.legend(frameon=False)
plt.savefig('figures_temp/s_bar_gamma_crit')
plt.show()

plt.figure()
for n, time in enumerate(struc.times):
    color = colors[n % len(colors)]
    x = struc.x_grid[:, n]
    s_smooth = struc.s_smooth
    scale = (b * time) ** (2 / 3)

    plt.plot(x / scale, s_smooth[n, :, 0] * scale, linestyle='-', color=color)
    plt.plot(0, 0, 'o', color=color, label=f't={int(time)}', markersize=5)
    plt.xlabel(r'$x / (bt)^{2/3}$')
    plt.ylabel(r'$(bt)^{2/3} \cdot S_{1}$')
    plt.legend(frameon=False)
plt.savefig('figures_temp/s_1_gamma_crit')
plt.show()

plt.figure()
for n, time in enumerate(struc.times):
    color = colors[n % len(colors)]
    x = struc.x_grid[:, n]
    s_smooth = struc.s_smooth
    scale = (b * time) ** (2 / 3)

    plt.plot(x / scale, s_smooth[n, :, 1] * scale, linestyle='-', color=color)
    plt.plot(0, 0, 'o', color=color, label=f't={int(time)}', markersize=5)
    plt.xlabel(r'$x / (bt)^{2/3}$')
    plt.ylabel(r'$(bt)^{2/3} \cdot S_{2}$')
    plt.legend(frameon=False)
plt.savefig('figures_temp/s_2_gamma_crit')
plt.show()
