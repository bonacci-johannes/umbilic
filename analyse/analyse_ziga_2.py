import tqdm
import numpy
import matplotlib.pyplot as plt

from analyse.z_spline_fit import z_func
from analyse.data_loader.burgers_loader import load_burgers_data
from analyse.data_loader.cpp_loader import load_cpp_data

# %% load burgers
gamma = 0.20
corr, time = load_burgers_data('Burgers_data', gamma)
sm_rates = {0.2: 0.01,
            0.25: 0.01, }
smooth_rate = sm_rates[gamma]
nsam = 30

# %% load cpp data
gamma = 0.25
corr, time = load_cpp_data(base_root='cpp_equal_corr', gamma=gamma, length=50000, t_max=10000, num=50)
smooth_rate = 0.01
nsam = 5

# %% evaluate basic statistics
corr_mean = numpy.mean(corr, axis=0)
corr_std = numpy.std(corr, axis=0)
corr_std_err = corr_std / numpy.sqrt(corr.shape[0])

# %% perform smoothing and extract z
nsub = corr.shape[0] // nsam  # the number of grouped sub-samples

sub_corr_mean = numpy.zeros((nsam, len(time), 2))
sub_corr_smooth = numpy.zeros((nsam, len(time), 2))
z_dyn_exp = numpy.zeros((nsam, len(time), 2))
for n in tqdm.tqdm(range(nsam)):
    for m in range(2):
        sub_corr_smooth[n, :, m], z_dyn_exp[n, :, m] = z_func(
            time=time,
            y=numpy.mean(corr[n * nsub:(n + 1) * nsub, m, :], axis=0),
            y_std_err=corr_std[m, :] / numpy.sqrt(nsub),
            smooth_rate=smooth_rate,
            z_exp=1.5)

# %% plot the data
f_x = 3
fig, axs = plt.subplots(num=0, nrows=3, ncols=1, sharex='all', squeeze=True, figsize=(4 * f_x, f_x * 6))

for m, col in zip(range(2), ['tab:blue', 'tab:orange']):
    # plot raw mean and spline fits
    axs[0].plot(time, corr_mean[m, :], '.-', color='black', label='raw', alpha=0.2)
    for n in range(nsam):
        axs[0].plot(time, sub_corr_smooth[n, :, m], '-', color=col, alpha=0.2)
    axs[0].plot(time, numpy.mean(sub_corr_smooth, axis=0)[:, m], '-', color=col, alpha=1)

    # plot dynamical exponent z
    for n in range(nsam):
        axs[1].plot(time, z_dyn_exp[n, :, m], ':', color=col, alpha=0.2)
    axs[1].plot(time, numpy.mean(z_dyn_exp, axis=0)[:, m], '.-', color=col, alpha=1)
    axs[1].fill_between(time,
                        numpy.mean(z_dyn_exp, axis=0)[:, m]
                        - 3 * numpy.std(z_dyn_exp, axis=0)[:, m] / numpy.sqrt(nsam),
                        numpy.mean(z_dyn_exp, axis=0)[:, m]
                        + 3 * numpy.std(z_dyn_exp, axis=0)[:, m] / numpy.sqrt(nsam),
                        color=col, alpha=0.5)
    axs[1].plot([time[0], time[-1]], [1.5, 1.5], '--', color='black', alpha=0.5)

    # plot z-transformed fitted data (z-transform with gaussian error)
    for n in range(nsam):
        axs[2].plot(time, (numpy.mean(corr[:, m, :], axis=0) - sub_corr_smooth[n, :, m]) / corr_std_err[m, :],
                    '.-', color=col, alpha=0.1)

    axs[2].plot(time, (corr_mean[m, :] - numpy.mean(sub_corr_smooth, axis=0)[:, m]) / corr_std_err[m, :],
                '.-', color=col, alpha=1)

    axs[2].plot([time[0], time[-1]], [0, 0], '--', color='black', alpha=0.5)

# adjust axes
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[1].set_xscale('log')
axs[1].set_ylim([1.25, 1.75])
axs[2].set_ylim(3 * numpy.array([-1, 1]))
axs[2].set_xlabel('t')
plt.show()
