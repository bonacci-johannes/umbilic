from data_loader import DataCorr
from z_spline_fit import z_func_series


class DataCorrZ(DataCorr):
    def __init__(self, time, corr, nsam, smooth_rate, s_lambda):
        self.time = time
        self.corr = corr
        self.smooth_rate = smooth_rate
        self.nsam = nsam
        self.s_lambda = s_lambda

        self.sub_corr_smooth, self.z_dyn_exp = z_func_series(time=self.time, corr=self.corr, nsam=self.nsam,
                                                             smooth_rate=self.smooth_rate, s_lambda=s_lambda)
