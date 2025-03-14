import warnings

import csaps
import numpy
import scipy
import statsmodels.api as sm


def alpha_fit_linear(t_0, time, y, y_err, alpha_0=2 / 3):
    """
    Fit the data with a linear model in log-log space.
    :param t_0:
    :param time:
    :param y:
    :param y_err:
    :param alpha_0:
    :return:
    """
    rescale = numpy.power(time, alpha_0)
    rescale_0 = numpy.power(t_0, alpha_0)
    time_log = numpy.log(time)
    fit = sm.WLS(endog=y * rescale,
                 exog=sm.add_constant(time_log),
                 weights=((time[1] - time[0]) / time) / (y_err * rescale) ** 2
                 ).fit()

    y_smooth = (fit.params[0] + fit.params[1] * numpy.log(t_0)) / rescale_0
    alpha = alpha_0 - fit.params[1] / (y_smooth * rescale_0)

    return y_smooth, alpha, fit.params


def alpha_fit_spline(t_0, time, y, y_err, alpha_0=2 / 3):
    rescale = numpy.power(time, alpha_0)
    rescale_0 = numpy.power(t_0, alpha_0)
    time_log = numpy.log(time)
    spl_tck_diag = scipy.interpolate.splrep(
        x=time_log,
        y=y * rescale,
        w=numpy.sqrt((time[1] - time[0]) / time) / (y_err * rescale),
        s=time_log[-1] - time_log[0],
    )

    # calculate smoothed result
    corr_smooth = scipy.interpolate.splev(numpy.log(t_0), spl_tck_diag)
    # calculate derivative and extract z
    alpha = (alpha_0 - scipy.interpolate.splev(numpy.log(t_0), spl_tck_diag, der=1) / corr_smooth)

    return corr_smooth / rescale_0, alpha, spl_tck_diag


def z_fit_linear(chunk_idx, time, y, y_err, eps_break=1e-5, rec_max=10, alpha_0=2 / 3,
                 fit_method=alpha_fit_linear):
    """
    alpha = 1/z
    :param chunk_idx:
    :param time:
    :param y:
    :param y_err:
    :param eps_break:
    :param rec_max:
    :param alpha_0:
    :return:
    """
    s_smooth = numpy.zeros((chunk_idx.shape[0]), dtype=float)
    z_smooth = numpy.zeros_like(s_smooth)

    for n in range(chunk_idx.shape[0]):
        alpha_fit = 1 / z_smooth[n - 1] if n > 0 else alpha_0
        for m in range(rec_max):
            alpha_mem = alpha_fit
            s_fit, alpha_fit, _ = fit_method(t_0=time[chunk_idx[n, 0]],
                                             time=time[chunk_idx[n, 1]:chunk_idx[n, 2] + 1],
                                             y=y[chunk_idx[n, 1]:chunk_idx[n, 2] + 1],
                                             y_err=y_err,
                                             alpha_0=alpha_mem)
            if numpy.abs(alpha_fit - alpha_mem) < eps_break:
                break

        if m == rec_max - 1:
            warnings.warn(f"Max recursion reached at time {time[chunk_idx[n, 0]]}!")

        s_smooth[n] = s_fit
        z_smooth[n] = 1 / alpha_fit

    return s_smooth, z_smooth


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
                            ).spline

    # calculate smoothed result
    corr_smooth = spl_csaps(time_log)
    # calculate derivative and extract z
    z = 1 / (1 / z_0 - spl_csaps.derivative(nu=1)(time_log) / corr_smooth)

    return corr_smooth / y_rescale, z, spl_csaps