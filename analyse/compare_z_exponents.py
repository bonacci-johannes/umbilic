import glob

import numpy
import matplotlib.pyplot as plt

from analyse.data_loader.cpp_loader import load_cpp_data
from analyse.helper.figures.z_validity_check import z_fit_check_from_dclass
from analyse.data_loader.data_corr_z import DataCorrZ

from analyse.data_loader.cpp_struct_loader import Struct

# %% load old data for comparison
struc = Struct(window=numpy.loadtxt(f'TWO_DIR/EXAMPLE_25/Window_Parameter.txt', comments='%'),
               record_files=sorted(glob.glob(f'TWO_DIR/EXAMPLE_25/Struc_fct_records/*_part_1.txt')),
               gamma=numpy.loadtxt(f'TWO_DIR/EXAMPLE_25/systemparameter.txt', comments='%')[5])

# %%
# load cpp data
base_root = 'cpp_data'
gamma = 0.25



corr, time = load_cpp_data(base_root=base_root, gamma=gamma, length=100000, t_max=10000, num=50)
cuttime = 10  # the start time to cut the data
cut_idx = numpy.searchsorted(time, cuttime)
time = time[cut_idx:]
corr = corr[:, :, cut_idx:]

sigma_weight = 1
nsam = 4
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

#% add old data for comparison
sig=5
for n in range(2):
    axs[0].plot(struc.times, struc.s_max[:, n], '.-', label=['$S_1$', '$S_2$'], color='tab:red')
    axs[0].fill_between(struc.times, struc.s_max[:, n] - sig*struc.s_max_std[:, n],
                        struc.s_max[:, n] + sig*struc.s_max_std[:, n], alpha=1, color='tab:red')
plt.show()
