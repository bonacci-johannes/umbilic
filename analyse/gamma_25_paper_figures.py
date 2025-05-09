import matplotlib
import numpy
import matplotlib.pyplot as plt
import glob

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
n_plot = [2,6,10,14]

f_x = 1.5
plt.rcParams['text.usetex'] = True
colors = list(matplotlib.colors.TABLEAU_COLORS.keys())

#%%

fig=plt.figure()
fig.set_size_inches(3 * f_x, 2 * f_x)
for n, np in enumerate(n_plot):
    time = struc.times[np]
    color = colors[n % len(colors)]
    x = struc.x_grid[:, np]
    s_smooth = struc.s_bar_smooth
    z_plot = 1.5
    scale = time ** (1/z_plot)
    plt.plot(x / scale, s_smooth[np, :, 0] * scale, linestyle='-', color=color)
    plt.plot(x / scale, s_smooth[np, :, 1] * scale, linestyle='-', color=color)
    plt.plot(-100, 0, 'o', color=color, label=f'$t={int(time)}$', markersize=5)
#plt.xlabel(r'$t^{-2/3} \cdot x$')
plt.ylabel(r'$t^{2/3} \cdot \tilde{S}_{1\alpha}$')
plt.xlabel(r'$t^{-2/3} \cdot x$')
plt.legend(frameon=False)

plt.xlim(-2.5,2.5)
plt.ylim(-0.0025,0.14)
plt.savefig('figures_temp/sbar_gamma_025_zKPZ.pdf',pad_inches=0.05, bbox_inches='tight')
plt.show()

# %%

fig=plt.figure()
fig.set_size_inches(3 * f_x, 2 * f_x)
for n, np in enumerate(n_plot):
    time = struc.times[np]
    color = colors[n % len(colors)]
    x = struc.x_grid[:, np]
    s_smooth = struc.s_smooth
    scale = time ** (2 / 3)

    plt.plot(x / scale, s_smooth[np, :, 0] * scale, linestyle='-', color=color)
    plt.plot(-100, 0, 'o', color=color, label=f'$t={int(time)}$', markersize=5)
plt.xlabel(r'$t^{-2/3} \cdot x$')
plt.ylabel(r'$t^{2/3} \cdot S_{22}$')
plt.legend(frameon=False)
plt.xlim(-2.5,2.5)
plt.ylim(-.0075,0.75)
plt.savefig('figures_temp/s_2_gamma_025_zKPZ.pdf',pad_inches=0, bbox_inches='tight')
plt.show()

# %%

fig=plt.figure()
fig.set_size_inches(3 * f_x, 2 * f_x)
for n, np in enumerate(n_plot):
    time = struc.times[np]
    color = colors[n % len(colors)]
    x = struc.x_grid[:, np]
    s_smooth = struc.s_smooth
    scale = time ** (2 / 3)

    plt.plot(x / scale, s_smooth[np, :, 0] * scale, linestyle='-', color=color)
    plt.plot(-100, 0, 'o', color=color, label=f'$t={int(time)}$', markersize=5)
plt.xlabel(r'$t^{-2/3} \cdot x$')
plt.ylabel(r'$t^{2/3} \cdot S_{22}$')
plt.legend(frameon=False)
plt.xlim(-2.5,2.5)
plt.ylim(-.0075,0.75)
plt.savefig('figures_temp/s_2_gamma_025_zKPZ.pdf',pad_inches=0, bbox_inches='tight')
plt.show()
# %%
fig=plt.figure()
fig.set_size_inches(3 * f_x, 2 * f_x)
for n, np in enumerate(n_plot):
    time = struc.times[np]
    color = colors[n % len(colors)]
    x = struc.x_grid[:, np]
    s_smooth = struc.s_smooth
    scale = time ** (1 / 1.456)

    plt.plot(x / scale, s_smooth[np, :, 1] * scale, linestyle='-', color=color)
    plt.plot(-100, 0, 'o', color=color, label=f'$t={int(time)}$', markersize=5)
#plt.xlabel(r'$t^{-1/z_1} \cdot x$')
plt.ylabel(r'$t^{1/z_1} \cdot S_{11}$')
plt.xlim(-2.5,2.5)
plt.ylim(-0.0075,0.6)
plt.legend(frameon=False)
plt.savefig('figures_temp/s_1_gamma_025_zFIT.pdf',pad_inches=0, bbox_inches='tight')
plt.show()

fig=plt.figure()
fig.set_size_inches(3 * f_x, 2 * f_x)
for n, np in enumerate(n_plot):
    time = struc.times[np]
    color = colors[n % len(colors)]
    x = struc.x_grid[:, np]
    s_smooth = struc.s_smooth
    scale = time ** (1 / 1.587)

    plt.plot(x / scale, s_smooth[np, :, 0] * scale, linestyle='-', color=color)
    plt.plot(-100, 0, 'o', color=color, label=f'$t={int(time)}$', markersize=5)
plt.xlabel(r'$t^{-1/z_\alpha} \cdot x$')
plt.ylabel(r'$t^{1/z_2} \cdot S_{22}$')
plt.legend(frameon=False)
plt.xlim(-2.5,2.5)
plt.ylim(-.0075,0.5)
plt.savefig('figures_temp/s_2_gamma_025_zFIT.pdf',pad_inches=0, bbox_inches='tight')
plt.show()

