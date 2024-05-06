import numpy


class DataCorr:
    time: numpy.ndarray
    corr: numpy.ndarray
    smooth_rate: float
    nsam: int
    s_lambda: float
    sub_corr_smooth: numpy.ndarray
    z_dyn_exp: numpy.ndarray
