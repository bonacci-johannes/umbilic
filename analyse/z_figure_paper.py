import scipy
import numpy
import matplotlib
import matplotlib.pyplot as plt

# %%
data = numpy.array([
    [0.05,     1.46,   1.54],
    [0.10,     1.453,   1.55],
    [0.18666,  1.455,  1.585],
    [0.25,     1.456,  1.587],
    [0.35,     1.475,  1.56],
    [0.45,     1.4825, 1.535],
    [0.55,     1.49,   1.52],
    [0.65,     1.495,  1.51],
    [1, 1.5, 1.5],
])

#%%
f_x = 1.5
plt.rcParams['text.usetex'] = True
fig=plt.figure(1)
fig.set_size_inches(3 * f_x, 2 * f_x)
plt.plot([0, 1], [1.5, 1.5], linestyle='--', color='tab:gray')
plt.plot([0.25, 0.25], [1.45, 1.6], linestyle='--', color='tab:gray')
plt.plot(data[:,0], data[:,2], linestyle='-', color='tab:red',marker='o', markersize=3, label=r'$\alpha=2$')
plt.plot(data[:,0], data[:,1], linestyle='-', color='tab:blue',marker='o', markersize=3, label=r'$\alpha=1$')
plt.xlim(0, 1)
plt.ylim(1.45, 1.6)
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$z_\alpha$')
plt.legend(frameon=False)
plt.yticks([1.45, 1.5, 1.55, 1.6])
plt.xticks([0, 0.25, 0.5, 0.75, 1])
plt.savefig('figures_temp/z_gamma.pdf',pad_inches=0, bbox_inches='tight')
plt.show()