# %%
import numpy
import scipy


class Struct:
    x_points = 300 + 1

    def __init__(self, window, record_files, gamma):
        self.gamma = gamma
        self.window = window
        self.record_files = record_files

        # compute grid points for each time
        self.t_x = {
            window[t_idx, 0]: numpy.arange(start=self.window[t_idx, 3], stop=self.window[t_idx, 4],
                                           step=self.window[t_idx, 6])
            for t_idx in range(1, len(window[:, 0]))}

        # allocate memory for data
        self.t_s_bar = {time: numpy.zeros(shape=(len(x), 2, len(record_files))) for time, x in self.t_x.items()}
        self.t_s_bar_smooth = {time: numpy.zeros(shape=(len(x), 2, len(record_files))) for time, x in self.t_x.items()}
        self.t_s_eigen = {time: numpy.zeros(shape=(self.x_points, 2, 2)) for time, x in self.t_x.items()}

        # load data
        for n, file in enumerate(self.record_files):
            data = numpy.loadtxt(file, comments='%')
            for time in self.t_s_bar.keys():
                # get idx of time in window
                t_idx = numpy.where(self.window[:, 0] == time)[0][0]

                # get data for time and reshape to 2x2 matrix
                y = data[:, t_idx]
                y = y.reshape((int(y.size / 4), 2, 2), order='F')
                y = y[:len(self.t_x[time]), :, :]

                # compute model symmetric part
                s_diag = 0.5 * (y[:, 0, 0] + y[:, 1, 1])
                s_off = 0.5 * (y[:, 0, 1] + y[:, 1, 0])

                # compute mirror symmetric part
                s_diag = 0.5 * (s_diag + s_diag[::-1])
                s_off = 0.5 * (s_off + s_off[::-1])
                self.t_s_bar[time][:, :, n] = numpy.array([s_diag, s_off]).T

        # smooth data
        self.times = numpy.array(list(self.t_x.keys()))
        self.x_grid = numpy.linspace(start=self.window[1:, 3], stop=self.window[1:, 4], num=self.x_points)
        self.s_bar_smooth_single = numpy.zeros(shape=(len(self.times), self.x_points, 2, len(self.record_files)))
        for nt, time in enumerate(self.times):
            std_diag = 0.5 * numpy.std(self.t_s_bar[time][:, 0, :], axis=1)
            std_off = 0.5 * numpy.std(self.t_s_bar[time][:, 1, :], axis=1)
            for n in range(len(record_files)):
                spl_tck_diag = scipy.interpolate.splrep(self.t_x[time], self.t_s_bar[time][:, 0, n], w=1 / std_diag)
                spl_tck_off = scipy.interpolate.splrep(self.t_x[time], self.t_s_bar[time][:, 1, n], w=1 / std_off)
                self.s_bar_smooth_single[nt, :, 0, n] = scipy.interpolate.splev(self.x_grid[:, nt], spl_tck_diag)
                self.s_bar_smooth_single[nt, :, 1, n] = scipy.interpolate.splev(self.x_grid[:, nt], spl_tck_off)

        # compute normed eigen modes
        self.s_smooth_single = numpy.zeros(shape=(len(self.times), self.x_points, 2, len(self.record_files)))
        self.s_smooth_single[:, :, 0, :] = (self.s_bar_smooth_single[:, :, 0, :] - self.s_bar_smooth_single[:, :, 1, :])
        self.s_smooth_single[:, :, 1, :] = (self.s_bar_smooth_single[:, :, 0, :] + self.s_bar_smooth_single[:, :, 1, :])
        self.s_smooth_single[:, :, 0, :] *= (2 + 2 / numpy.sqrt(self.gamma))
        self.s_smooth_single[:, :, 1, :] *= (2 + 2 * numpy.sqrt(self.gamma))

        # compute avg data
        self.s_bar_smooth = numpy.mean(self.s_bar_smooth_single, axis=-1)
        self.s_bar_smooth_std_err = numpy.std(self.s_bar_smooth_single, axis=-1) / numpy.sqrt(len(self.record_files))
        self.s_smooth = numpy.mean(self.s_smooth_single, axis=-1)
        self.s_smooth_std_err = numpy.std(self.s_smooth_single, axis=-1) / numpy.sqrt(len(self.record_files))

        # compute maxima of eigenmodes
        self.s_max = self.s_smooth[:, self.s_smooth.shape[1] // 2, :]
        self.s_max_std_err = self.s_smooth_std_err[:, self.s_smooth_std_err.shape[1] // 2, :]