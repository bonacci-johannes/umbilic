import numpy
import scipy


def z_func(time, y, y_std_err, smooth_rate=1 / 2, z_exp=3 / 2):
    time_log = numpy.log(time)
    time_scale = numpy.power(time, 1 / z_exp)

    # calculate log-log smoothed spline
    corr_smooth = scipy.interpolate.make_smoothing_spline(
        x=time_log,
        y=y * time_scale,
        lam=1.5 / smooth_rate,
        w=smooth_rate / numpy.square(y_std_err * time_scale))(time_log) / time_scale

    # calculate derivative and extract z
    spl_tck_diag = scipy.interpolate.splrep(time_log, corr_smooth)
    z = - corr_smooth / scipy.interpolate.splev(time_log, spl_tck_diag, der=1)

    return corr_smooth, z
