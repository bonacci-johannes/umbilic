import numpy
import scipy
import matplotlib.pyplot as plt

from analyse.helper.fitting.log_chunker import log_equal_chunker
from analyse.helper.fitting.z_finite_difference import z_finite_difference
from analyse.helper.fitting.z_fit import z_fit_linear

# %% seed random number generator
numpy.random.seed(1)
# %%
a = 3.25
b = 3
t_min = 1e0
t_max = 1.e5
eps = 1e-3

time = numpy.arange(start=t_min, stop=t_max + 1, step=1)
time_log = numpy.log(time)

s = 1 / numpy.sqrt(a * time + b * time ** (4 / 3))
z_ana = 2 * (a * time + b * time ** (4 / 3)) / (a * time + (4 / 3) * b * time ** (4 / 3))

xi = numpy.random.normal(loc=0, scale=eps, size=time.size)
s_raw = s + xi


# %%
def z_func(time,
           y,
           y_std_err,
           sigma_weight=1.,
           z_0=3 / 2,
           knots=None, ):
    # transform data
    time_log = numpy.log(time)
    y_rescale = numpy.power(time, 1 / z_0)

    # calculate log-log smoothed spline
    spl_tck_diag = scipy.interpolate.splrep(
        x=time_log,
        y=y * y_rescale,
        w=numpy.sqrt((time[1] - time[0]) / time) / (sigma_weight * y_std_err * y_rescale),
        s=time_log[-1] - time_log[0],
        t=knots
    )

    # calculate smoothed result
    corr_smooth = scipy.interpolate.splev(time_log, spl_tck_diag)
    # calculate derivative and extract z
    z = 1 / (1 / z_0 - scipy.interpolate.splev(time_log, spl_tck_diag, der=1) / corr_smooth)

    return corr_smooth / y_rescale, z, spl_tck_diag


# %%
sigma_weight = 1
corr_smooth0, z0, spl0 = z_func(time=time, y=s_raw, y_std_err=eps, sigma_weight=sigma_weight, z_0=3 / 2,)

# %% determine indizes by window width
rec_max = 10
eps_break = 1e-5
log_window_width = 0.5
dx_chunk = log_window_width / 10

chunk_idx = log_equal_chunker(time, dx_chunk=dx_chunk, x_chunk_width=log_window_width)
t_chunks = time[chunk_idx[:, 0]]
s_smooth, z_smooth = z_fit_linear(chunk_idx=chunk_idx, time=time, y=s_raw, y_err=eps, rec_max=10)
t_fd, z_fd = z_finite_difference(t_chunks, s_smooth, dx=dx_chunk, ndx=10, n=1)

# %% plot results


f_x = 1.5
fig, axs = plt.subplots(nrows=4, ncols=1, sharex='all', squeeze=True, figsize=(5 * f_x, f_x * 5.))
axs[0].plot(time, s, color='black', alpha=1)
axs[0].plot(time, s_raw, color='tab:blue', alpha=0.2)
axs[0].plot(time, corr_smooth0, color='tab:green')
axs[0].plot(t_chunks, s_smooth, color='tab:orange')
axs[0].set_xscale('log')
axs[0].set_yscale('log')

axs[0].set_ylabel('$S(t)$')

axs[1].plot(time, z0, color='tab:green')
axs[1].plot(t_chunks, z_smooth, '-', color='tab:orange')
axs[1].plot([time[0], time[-1]], [1.5, 1.5], '--', color='black', alpha=0.5)
axs[1].plot(time, z_ana, color='black', linewidth=1)
axs[1].plot(t_fd, z_fd, color='tab:red')

axs[1].set_ylim([1.45, 1.75])

axs[1].set_ylabel('$z(t)$')

axs[2].plot([time[0], time[-1]], [0, 0], '--', color='black', alpha=0.5)

axs[2].plot(t_chunks, (s_smooth - s[chunk_idx[:, 0]]) / s[chunk_idx[:, 0]],
            '.-', alpha=0.5, color='tab:orange')
axs[2].plot(time, (corr_smooth0 - s) / s,
            '.-', alpha=0.5, color='tab:green')

axs[2].set_ylabel('$S_{fit} / S - 1$')

axs[3].plot([time[0], numpy.log(time[0]) + numpy.exp(log_window_width)], [0, 0], '-', color='black', alpha=1,
            linewidth=3)

axs[3].plot(t_chunks, (z_smooth - z_ana[chunk_idx[:, 0]]) / z_ana[chunk_idx[:, 0]], color='tab:orange')
axs[3].plot(t_fd, z_fd / numpy.interp(t_fd, time, z_ana) - 1, color='tab:red')
axs[3].plot(time, z0 / z_ana - 1, color='tab:green')
axs[3].set_ylim([-0.01, 0.01])
axs[3].plot([time[0], time[-1]], [0, 0], '--', color='black', alpha=0.5)
axs[3].set_ylabel('$z_{fit} / z-1$')

axs[-1].set_xlabel('t')
plt.show()
