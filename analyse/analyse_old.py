from typing import Tuple

import matplotlib
import numpy
import matplotlib.pyplot as plt
import os
import glob

import scipy

# matplotlib.use('macosx')
# matplotlib.use('QtAgg')

from analyse.data_loader.cpp_struct_loader import Struct

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
b = numpy.sqrt(3) / 12
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
