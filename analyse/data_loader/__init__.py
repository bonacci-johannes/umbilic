import numpy


class DataCorr:
    time: numpy.ndarray
    corr: numpy.ndarray
    sigma_weight: float
    nsam: int
    s_lambda: float
    sub_corr_smooth: numpy.ndarray
    z_dyn_exp: numpy.ndarray
