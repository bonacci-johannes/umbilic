import numpy
import matplotlib.pyplot as plt

from analyse.data_loader.burgers_loader import load_burgers_data
from analyse.data_loader.cpp_loader import load_cpp_data
from analyse.helper.figures.z_validity_check import z_fit_check_from_dclass
from analyse.data_loader.data_corr_z import DataCorrZ

# %% load burgers
gamma = 0.25
corr, time = load_burgers_data('Burgers_data', gamma)
nsam = 10

burgers_data = DataCorrZ(time=time, corr=corr, nsam=nsam, sigma_weight=1, s_lambda=0)

# %%
z_fit_check_from_dclass(burgers_data, show=True,
                        fig_path_name=f'figures_temp/burgers_z_fit_check_{gamma}.png',
                        title=f'Burgers: gamma = {gamma}')

# %%
# load cpp data
gamma = 0.25
length = 500000
t_max = 100000
num = 50
base_root = 'cpp_data'
cuttime = 10  # the start time to cut the data


nsam = 10


z_data = {}
#for gamma in numpy.arange(start=0.05, stop=1.0, step=0.05):

corr, time = load_cpp_data(base_root=base_root, gamma=gamma, length=length, t_max=t_max, num=num)
cut_idx = numpy.searchsorted(time, cuttime)

# %%
smooth_rate = 1.0
s_lambda = 0
time = time[cut_idx:]
corr = corr[:, :, cut_idx:]
z_data[gamma] = DataCorrZ(time=time, corr=corr,
                          nsam=nsam, sigma_weight=smooth_rate, s_lambda=s_lambda)
z_fit_check_from_dclass(z_data[gamma], show=True,
                        fig_path_name=f'figures_temp/gam_series/z_fit_check_{int(1000*gamma)}.png',
                        title=f'Lattice: gamma = {gamma:0.3f}')
