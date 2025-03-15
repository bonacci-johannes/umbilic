import numpy

import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.use('Qt5Agg')

from analyse.data_loader.cpp_loader import load_cpp_data
from analyse.helper.fitting.z_fit import z_func_csaps
import tqdm

# %%
gamma = 0.25
length = 500000
t_max = 100000
num = 50
base_root = 'cpp_data'
base_str = f'gam_{int(gamma * 1000)}_len_{length}_t_{t_max}_num_{num}'
path_res = os.path.join(base_root, base_str)
path_npy = path_res + '.npy'
# %%
if os.path.isfile(path_npy):
    corr_full = numpy.load(path_npy)
else:
    corr_full, _ = load_cpp_data(base_root=base_root, gamma=gamma, length=length, t_max=t_max, num=num)
    numpy.save(path_npy, corr_full)

samples_full = corr_full.shape[0]
time = numpy.arange(1, corr_full.shape[2] + 1)

# %%
sub_samples = 250
n_sub = samples_full // sub_samples
corr = numpy.zeros((n_sub, 2, t_max))
for n in range(n_sub):
    corr[n, :, :] = numpy.mean(corr_full[n * sub_samples:(n + 1) * sub_samples, :, :], axis=0)

corr_std = numpy.std(corr, axis=0)

# %%
smooth = 0.01
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

# %%
idx = numpy.unique(numpy.searchsorted(time, 10**numpy.linspace(0, numpy.log10(time[-1]), 1000)))

f_x = 1.5
plt.rcParams['text.usetex'] = True
plt.figure(1).clf()
fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', squeeze=True, num=1)
fig.set_size_inches(3 * f_x, 2 * f_x)

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_ylabel(r'$S_{\alpha\alpha}(0,t)$')
axs[1].set_ylabel(r'$z_\alpha(t)$')
axs[-1].set_xlabel(r'$t$')

for m, color in zip(range(2), ['tab:blue', 'tab:red']):
    axs[0].plot(time[idx], numpy.mean(corr[:, m, idx], axis=0), color=color, alpha=0.25)
    axs[0].plot(time[idx], corr_csaps_mean[m, idx], color=color, label=r'$\alpha='+f'{m + 1}$')
    axs[1].fill_between(time[idx],
                        z_mean[m, idx] - 3 * z_std[m, idx] / numpy.sqrt(n_sub),
                        z_mean[m, idx] + 3 * z_std[m, idx] / numpy.sqrt(n_sub),
                        color=color, alpha=0.25)
    axs[1].plot(time[idx], z_mean[m, idx], color=color)

axs[0].set_ylim([1e-4, 1e-1])
axs[1].set_ylim([1.4, 1.7])
axs[1].plot([time[0], time[-1]], [1.5, 1.5], '--', color='black', alpha=0.5)
axs[1].set_yticks([1.4, 1.5, 1.6, 1.7])
axs[-1].set_xlim([50, time[-1]])

axs[0].legend(frameon=False)
plt.show()
fig.savefig(path_res + f'_csaps_sm_{str(smooth).replace('.', '_')}.pdf',
            pad_inches=0, bbox_inches='tight')
fig.savefig(path_res + f'Fig_z_time.pdf',
            pad_inches=0, bbox_inches='tight')


