import numpy
import scipy

import matplotlib.pyplot as plt

# %%

t_max = 1e6
eps = 1e-5

time = numpy.arange(start=1, stop=t_max + 1, step=1)
time_log = numpy.log(time)
numpy.random.seed(0)
xi = numpy.random.normal(loc=0, scale=eps, size=time.size)

z_ana = -2 / 3
y = numpy.power(time, z_ana)

# %%
sig = 0.925

spl_default = scipy.interpolate.splrep(
    x=time_log,
    y=y + xi,
    w=numpy.sqrt((time[1] - time[0]) / time) / (sig*eps),
    s=time_log[-1] - time_log[0])

y_default = scipy.interpolate.splev(time_log, spl_default)
z_default = scipy.interpolate.splev(time_log, spl_default, der=1) / y_default
# refit with old knots
spl_knots = scipy.interpolate.splrep(
    x=time_log,
    y=y + xi,
    w=numpy.sqrt((time[1] - time[0]) / time) / (sig*eps),
    s=(time_log[-1] - time_log[0]),
    t=spl_default[0][spl_default[2] + 1:-spl_default[2] - 1]  # take inner knots
)
y_knots = scipy.interpolate.splev(time_log, spl_knots)
z_knots = scipy.interpolate.splev(time_log, spl_knots, der=1) / y_knots

# %%

f_x = 1.5
fig, axs = plt.subplots(nrows=3, ncols=1, sharex='all', squeeze=True, figsize=(4 * f_x, f_x * 5.))
axs[0].plot(time, y, color='black', alpha=1)
axs[0].plot(time, y + xi, color='tab:blue', alpha=0.2)
axs[0].plot(time, y_default, color='tab:green')
axs[0].plot(time, y_knots, color='tab:orange')
axs[0].set_xscale('log')
axs[0].set_yscale('log')

axs[1].plot(time, z_default, color='tab:green')
axs[1].plot(time, z_knots, color='tab:orange')
axs[1].plot([time[0], time[-1]], [z_ana, z_ana], '--', color='black', alpha=0.5)

axs[1].plot(numpy.exp(spl_default[0]), numpy.ones_like(spl_default[0]) * z_ana * 0.99, '.', color='tab:green',
            alpha=0.5)

axs[2].plot(time, xi / eps)

axs[2].plot([time[0], time[-1]], [0, 0], '--', color='black', alpha=0.5)

axs[2].set_ylim([-6, 6])

plt.show()
