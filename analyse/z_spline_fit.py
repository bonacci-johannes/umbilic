import numpy
import scipy
from tqdm import tqdm


def z_func(time,
           y,
           y_std_err,
           smooth_rate=1 / 2,
           s_lambda=1,
           z_exp=3 / 2):
    time_log = numpy.log(time)
    time_scale = numpy.power(time, 1 / z_exp)

    # calculate log-log smoothed spline
    corr_smooth = scipy.interpolate.make_smoothing_spline(
        x=time_log,
        y=y * time_scale,
        lam=s_lambda,
        w=(1 / time) * numpy.square(1 / (smooth_rate * y_std_err * time_scale))
    )(time_log)
    corr_smooth /= time_scale

    # calculate derivative and extract z
    spl_tck_diag = scipy.interpolate.splrep(time_log, corr_smooth)
    z = - corr_smooth / scipy.interpolate.splev(time_log, spl_tck_diag, der=1)

    return corr_smooth, z


def z_func_series(time,
                  corr,
                  smooth_rate=1,
                  s_lambda=1,
                  nsam=8):
    # perform smoothing and extract z
    nsub = corr.shape[0] // nsam  # the number of grouped sub-samples
    sub_corr_smooth = numpy.zeros((nsam, len(time), 2))
    z_dyn_exp = numpy.zeros_like(sub_corr_smooth)

    # smooth each sub-sample
    corr_std = numpy.std(corr, axis=0)
    for n in tqdm(range(nsam)):
        for m in range(2):
            sub_corr_smooth[n, :, m], z_dyn_exp[n, :, m] = z_func(
                time=time,
                y=numpy.mean(corr[n * nsub:(n + 1) * nsub, m, :], axis=0),
                y_std_err=corr_std[m, :] / numpy.sqrt(nsub),
                smooth_rate=smooth_rate,
                s_lambda=s_lambda,
                z_exp=1.5)

    return sub_corr_smooth, z_dyn_exp
