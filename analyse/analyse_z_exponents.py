import numpy
import matplotlib.pyplot as plt

from analyse.data_loader.burgers_loader import load_burgers_data
from analyse.data_loader.cpp_loader import load_cpp_data
from analyse.helper.figures.z_validity_check import z_fit_check_from_dclass
from data_loader.data_corr_z import DataCorrZ

# %% load burgers
gamma = 0.25
corr, time = load_burgers_data('Burgers_data', gamma)
sm_rates = {0.2: 0.01,
            0.25: 0.01, }
smooth_rate = sm_rates[gamma]
s_lambda = 100 / smooth_rate
nsam = 30

burgers_data = DataCorrZ(time=time, corr=corr, nsam=nsam, smooth_rate=smooth_rate, s_lambda=s_lambda)
z_fit_check_from_dclass(burgers_data, show=True,
                        fig_path_name=f'figures_temp/burgers_z_fit_check_{gamma}.png',
                        title=f'Burgers: gamma = {gamma}')

# %%
# load cpp data
length = 100000
t_max = 10000
num = 50
base_root = 'cpp_equal_corr/gam_series'
cuttime = 10  # the start time to cut the data

smooth_rate = 0.1
nsam = 8
s_lambda = 100 / smooth_rate

z_data = {}
for gamma in numpy.arange(start=0.75, stop=0.85, step=0.05):
    corr, time = load_cpp_data(base_root=base_root, gamma=gamma, length=length, t_max=t_max, num=num)
    cut_idx = numpy.searchsorted(time, cuttime)
    time = time[cut_idx:]
    corr = corr[:, :, cut_idx:]
    z_data[gamma] = DataCorrZ(time=time, corr=corr,
                              nsam=nsam, smooth_rate=smooth_rate, s_lambda=s_lambda)
    z_fit_check_from_dclass(z_data[gamma], show=False,
                            fig_path_name=f'figures_temp/gam_series/z_fit_check_{int(1000*gamma)}.png',
                            title=f'Lattice: gamma = {gamma:0.3f}')