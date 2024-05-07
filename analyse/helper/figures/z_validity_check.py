import numpy
import matplotlib.pyplot as plt

from analyse.data_loader import DataCorr


# plot the data
def z_fit_check_figure(time, corr, corr_smooth, z_dyn_exp,
                       fig_path_name=None,
                       show=True,
                       title=None):
    corr_mean = numpy.mean(corr, axis=0)
    corr_std_err = numpy.std(corr, axis=0) / numpy.sqrt(corr.shape[0])

    nsam = corr_smooth.shape[0]

    f_x = 1.5
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex='all', squeeze=True, figsize=(4 * f_x, f_x * 5.))

    for m, col in zip(range(2), ['tab:blue', 'tab:orange']):
        # plot raw mean and spline fits
        axs[0].plot(time, corr_mean[m, :], '.-', color='black', label='raw', alpha=0.2)
        for n in range(nsam):
            axs[0].plot(time, corr_smooth[n, :, m], '-', color=col, alpha=0.2)
        axs[0].plot(time, numpy.mean(corr_smooth, axis=0)[:, m], '-', color=col, alpha=1)

        # plot dynamical exponent z
        for n in range(nsam):
            axs[1].plot(time, z_dyn_exp[n, :, m], ':', color=col, alpha=0.2)
        axs[1].plot(time, numpy.mean(z_dyn_exp, axis=0)[:, m], '.-', color=col, alpha=1)
        axs[1].fill_between(time,
                            numpy.mean(z_dyn_exp, axis=0)[:, m]
                            - 3 * numpy.std(z_dyn_exp, axis=0)[:, m] / numpy.sqrt(nsam),
                            numpy.mean(z_dyn_exp, axis=0)[:, m]
                            + 3 * numpy.std(z_dyn_exp, axis=0)[:, m] / numpy.sqrt(nsam),
                            color=col, alpha=0.5)
        axs[1].plot([time[0], time[-1]], [1.5, 1.5], '--', color='black', alpha=0.5)

        # plot z-transformed fitted data (z-transform with gaussian error)
        for n in range(nsam):
            axs[2].plot(time, (corr_mean[m, :] - corr_smooth[n, :, m]) / corr_std_err[m, :],
                        '.-', color=col, alpha=0.1)

        axs[2].plot(time, (corr_mean[m, :] - numpy.mean(corr_smooth, axis=0)[:, m]) / corr_std_err[m, :],
                    '.-', color=col, alpha=1)

        axs[2].plot([time[0], time[-1]], [0, 0], '--', color='black', alpha=0.5)

    # adjust axes
    if title is not None:
        axs[0].set_title(title)
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[1].set_ylim([1.25, 1.75])
    axs[2].set_ylim(5 * numpy.array([-1, 1]))
    axs[2].set_xlabel('t')

    axs[0].set_ylabel('$S_{i}$')
    axs[1].set_ylabel('$z_i(t)$')
    axs[2].set_ylabel('$(S_{\\mathrm{raw}} - S_{\\mathrm{smooth}}) / \\sigma$')
    if fig_path_name is not None:
        plt.savefig(fig_path_name)
    if show:
        plt.show()

    return fig, axs


def z_fit_check_from_dclass(data: DataCorr, fig_path_name=None, show=True, title=None):
    return z_fit_check_figure(time=data.time, corr=data.corr,
                              corr_smooth=data.sub_corr_smooth, z_dyn_exp=data.z_dyn_exp,
                              fig_path_name=fig_path_name, show=show, title=title)
