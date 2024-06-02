import glob

import numpy
import matplotlib.pyplot as plt

from analyse.data_loader.cpp_loader import load_cpp_data
from analyse.helper.figures.z_validity_check import z_fit_check_from_dclass
from analyse.data_loader.data_corr_z import DataCorrZ

# %% load old data for comparison
struc_max = numpy.loadtxt('test_python_eval/test_data/gamma_025_old_struc_max.txt')

# %%
# load cpp data
base_root = 'cpp_data'
gamma = 0.25

corr, time = load_cpp_data(base_root=base_root, gamma=gamma, length=500000, t_max=100000, num=50)
# %%
cuttime = 10  # the start time to cut the data
cut_idx = numpy.searchsorted(time, cuttime)
time = time[cut_idx:]
corr = corr[:, :, cut_idx:]

sigma_weight = 0.96
nsam = 10
s_lambda = 0
z_data = DataCorrZ(time=time,
                   corr=corr,
                   nsam=nsam,
                   sigma_weight=sigma_weight,
                   s_lambda=s_lambda)

# %%
fig, axs = z_fit_check_from_dclass(z_data, show=False,
                                   fig_path_name=f'figures_temp/z_fit_check_{int(1000 * gamma)}.png',
                                   title=f'Lattice: gamma = {gamma:0.3f}')

# add old data for comparison

axs[1].set_ylim([1.4, 1.7])
sig = 5
for n in range(2):
    axs[0].plot(struc_max[:, 0], struc_max[:, 1 + n], '.-', label=['$S_1$', '$S_2$'], color='tab:red')
    axs[0].fill_between(struc_max[:, 0], struc_max[:, 1 + n] - sig * struc_max[:, 3 + n],
                        struc_max[:, 1 + n] - sig * struc_max[:, 3 + n], alpha=1, color='tab:red')
plt.show()
