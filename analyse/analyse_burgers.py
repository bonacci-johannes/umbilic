from analyse.data_loader.burgers_loader import load_burgers_data

import numpy

import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.use('Qt5Agg')

from analyse.helper.fitting.z_fit import z_func_csaps
import tqdm

# %%
base_root = 'Burgers_data'
gamma = 0.25
gamma_str = f"{int(gamma * 100):03d}"+'_big_dt'

corr_full, time = load_burgers_data(base_root, gamma_str)

samples_full = corr_full.shape[0]

# %%
sub_samples = 100
n_sub = samples_full // sub_samples
corr = numpy.zeros((n_sub, 2, corr_full.shape[2]))
for n in range(n_sub):
    corr[n, :, :] = numpy.mean(corr_full[n * sub_samples:(n + 1) * sub_samples, :, :], axis=0)

corr_std = numpy.std(corr, axis=0)

# %%
smooth=0.1
corr_csaps = numpy.zeros_like(corr)
z_csaps = numpy.zeros_like(corr)

for n in tqdm.tqdm(range(corr.shape[0])):
    for m in range(2):
        corr_csaps[n, m], z_csaps[n, m], _ = z_func_csaps(time=time, y=corr[n, m, :],
                                                          y_std_err=corr_std[m, :],
                                                          sigma_weight=1,
                                                          z_0=3 / 2,
                                                          smooth=smooth)

corr_csaps_mean = numpy.mean(corr_csaps, axis=0)
corr_csaps_std = numpy.std(corr_csaps, axis=0)
z_mean = numpy.mean(z_csaps, axis=0)
z_std = numpy.std(z_csaps, axis=0)


f_x = 1.
plt.figure(1).clf()
fig, axs = plt.subplots(num=1, nrows=2, ncols=1, sharex='all', squeeze=True, figsize=(4 * f_x, f_x * 5.))

axs[-1].set_xlabel('t')

# axs[0].plot(t, yt, color='tab:purple', marker='o', markersize=5)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_ylabel('S(0,t)')
axs[1].set_ylabel('z(t)')

for m, color in zip(range(2), ['tab:blue', 'tab:red']):
    axs[0].plot(time, corr[0, m, :], color=color, alpha=0.25)
    axs[0].plot(time, corr_csaps_mean[m, :], color=color)
    axs[1].fill_between(time,
                        z_mean[m, :] - 3 * z_std[m, :] / numpy.sqrt(n_sub),
                        z_mean[m, :] + 3 * z_std[m, :] / numpy.sqrt(n_sub),
                        color=color, alpha=0.25)
    axs[1].plot(time, z_mean[m, :], color=color)

axs[1].set_ylim([1.4, 1.7])
axs[1].plot([time[0], time[-1]], [1.5, 1.5], '--', color='black', alpha=0.5)
axs[0].set_title(f"burgers data smooth={smooth}")
plt.show()
fig.savefig(f'Burgers_csaps_sm_{gamma_str}_smooth{str(smooth).replace('.', '_')}.png')
