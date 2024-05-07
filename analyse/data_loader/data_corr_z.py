from analyse.data_loader import DataCorr
from analyse.z_spline_fit import z_func_series


class DataCorrZ(DataCorr):
    def __init__(self, time, corr, nsam, sigma_weight=1, s_lambda=0):
        self.time = time
        self.corr = corr
        self.sigma_weight = sigma_weight
        self.nsam = nsam
        self.s_lambda = s_lambda

        self.sub_corr_smooth, self.z_dyn_exp = z_func_series(time=self.time, corr=self.corr, nsam=self.nsam,
                                                             sigma_weight=self.sigma_weight, s_lambda=s_lambda)
