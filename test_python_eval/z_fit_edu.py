import numpy
import scipy

import matplotlib.pyplot as plt
import csaps

# %%
a = 2.25
b = 3
t_min = 1e2
t_max = 1.e3
eps = 2e-4

time = numpy.arange(start=t_min, stop=t_max + 1, step=1)
time_log = numpy.log(time)

s = 1 / numpy.sqrt(a * time + b * time ** (4 / 3))

z_ana = 2 * (a * time + b * time ** (4 / 3)) / (a * time + (4 / 3) * b * time ** (4 / 3))

numpy.random.seed(0)
xi = numpy.random.normal(loc=0, scale=eps, size=time.size)


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


def z_func_csaps(time,
                 y,
                 y_std_err,
                 sigma_weight=1.,
                 z_0=3 / 2,
                 smooth=0.5):
    # transform data
    time_log = numpy.log(time)
    y_rescale = numpy.power(time, 1 / z_0)

    # calculate log-log smoothed spline

    spl_csaps = csaps.csaps(xdata=time_log,
                            ydata=y * y_rescale,
                            smooth=smooth,
                            weights=((time[1] - time[0]) / time) / numpy.square(sigma_weight * y_std_err * y_rescale),
                            normalizedsmooth=True,
                            ).spline

    # calculate smoothed result
    corr_smooth = spl_csaps(time_log)
    # calculate derivative and extract z
    z = 1 / (1 / z_0 - spl_csaps.derivative(nu=1)(time_log) / corr_smooth)

    return corr_smooth / y_rescale, z, spl_csaps


# %%
sigma_weight = 0.95

corr_smooth0, z0, spl0 = z_func(time=time,
                                y=s + xi,
                                y_std_err=eps,
                                sigma_weight=sigma_weight,
                                z_0=3 / 2)

knots_diff = time_log[numpy.searchsorted(time_log, spl0[0])] - spl0[0]

corr_smooth_r, z_r, spl_r = z_func(time=time,
                                   y=s + xi,
                                   y_std_err=eps,
                                   sigma_weight=sigma_weight,
                                   knots=spl0[0][4:-4],
                                   z_0=3 / 2)

# %%
knots_x = numpy.linspace(time_log[0], time_log[-1], 3)[1:-1]
knots_idx = numpy.unique(numpy.searchsorted(time_log, knots_x))
knots_x = time_log[knots_idx]

corr_smooth_x, z_x, spl_x = z_func(time=time,
                                   y=s + xi,
                                   y_std_err=eps,
                                   sigma_weight=sigma_weight,
                                   knots=knots_x,
                                   z_0=3 / 2)

# %% csaps

corr_smooth_csaps, z_csaps, spl_csaps = z_func_csaps(time=time, y=s + xi,
                                                     y_std_err=eps,
                                                     sigma_weight=1,
                                                     z_0=3 / 2,
                                                     smooth=1e-5)

# %%

import statsmodels.api as sm

window = int((t_max - t_min) / 5)
window=100
t = int((t_max + t_min) / 2)
t_idx = t - int(t_min)
idx = t_idx + numpy.array([-1, 1]) * window // 2

res = sm.WLS(endog=(s[idx[0]:idx[1]] + xi[idx[0]:idx[1]]) * time[idx[0]:idx[1]] ** (2 / 3),
             exog=sm.add_constant(time_log[idx[0]:idx[1]]),
             weights=1 / (eps * time[idx[0]:idx[1]] ** (2 / 3)) ** 2).fit()
# print(res.summary())
yt = (res.params[0] + res.params[1] * time_log[t_idx]) * time[t_idx] ** (-2 / 3)
print(res.params)
# %%


f_x = 1.5
fig, axs = plt.subplots(nrows=4, ncols=1, sharex='all', squeeze=True, figsize=(4 * f_x, f_x * 5.))
axs[0].plot(time, s, color='black', alpha=1)
axs[0].plot(time, s + xi, color='tab:blue', alpha=0.2)
axs[0].plot(time, corr_smooth0, color='tab:green')
axs[0].plot(time, corr_smooth_x, color='tab:orange')
axs[0].plot(time, corr_smooth_csaps, color='tab:red')
axs[0].plot(t, yt, color='tab:purple', marker='o', markersize=5)
axs[0].set_xscale('log')
axs[0].set_yscale('log')

axs[1].plot(time, z_csaps, color='tab:red')
axs[1].plot(time, z_ana, color='black', linewidth=1)
axs[1].plot(time, z0, color='tab:green')
axs[1].plot(time, z_x, color='tab:orange')
axs[1].plot(time[:-1]+(time[1]-time[0])/2, numpy.log(time[:-1]/time[1:])/numpy.log(corr_smooth_csaps[1:]/corr_smooth_csaps[:-1]),
            color='tab:blue', linestyle='--')
axs[1].plot(0.5 * (time[1:] + time[:-1]),
            -numpy.diff(time_log) / numpy.diff(numpy.log(corr_smooth_x)), color='tab:orange', linestyle='--')

axs[1].plot([time[0], time[-1]], [1.5, 1.5], '--', color='black', alpha=0.5)

axs[1].plot(numpy.exp(spl0[0]), numpy.ones_like(spl0[0]) * 1.55, '.', color='tab:green', alpha=0.5)
axs[1].plot(numpy.exp(spl_x[0]), numpy.ones_like(spl_x[0]) * 1.5375, '.', color='tab:orange', alpha=0.5)

axs[1].set_ylim([1.4, 1.6])

axs[2].plot(time, (xi) / eps,
            '.-', alpha=0.5, color='tab:blue')
axs[2].plot(time, (s + xi - corr_smooth0) / eps,
            '.-', alpha=0.5, color='tab:green')
axs[2].plot(time, (s + xi - corr_smooth_x) / eps,
            '.-', alpha=0.5, color='tab:orange')

axs[2].plot([time[0], time[-1]], [0, 0], '--', color='black', alpha=0.5)

axs[3].plot(time, (s - corr_smooth_csaps) / s,
            '.-', alpha=0.5, color='tab:red')

axs[3].plot(time, (s - corr_smooth0) / s,
            '.-', alpha=0.5, color='tab:green')

axs[3].plot(time, (s - corr_smooth_x) / s,
            '.-', alpha=0.5, color='tab:orange')

axs[3].plot(t, (s[t_idx] - yt) / s[t_idx], color='tab:purple', marker='o', markersize=5)

axs[3].plot([time[0], time[-1]], [0, 0], '--', color='black', alpha=0.5)

axs[2].set_ylim([-6, 6])

plt.show()
