import tqdm
import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt

# %% load numpy npy file


def read_data_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        # Skip the first 4 rows
        for _ in range(4):
            next(file)

        # Read each subsequent row
        for line in file:
            # Split the line into two integers
            values = line.strip().split()
            if len(values) == 2:
                try:
                    num1 = int(values[0])
                    num2 = int(values[1])
                    data.append((num1, num2))
                except ValueError:
                    print("Error: Skipping line with invalid data format:", line.strip())
            else:
                print("Error: Skipping line with invalid data format:", line.strip())

    return data

# load data from a csv where cols are seperated bz a empty space and skip the first 4 rows
corr = numpy.zeros((32,2,10000))
for n in range(32):
    corr[n,:,:] = numpy.array(read_data_file(
        f'cpp_equal_corr/gam_250_len_50000_t_10000_num_50/{n+1}_gam_250_len_50000_t_10000_num_50.txt')).T

times = numpy.arange(1,10001)

step = 1

corr = corr[:, :, ::step]
times = times[::step]

# %% data smoothing level 1
def z_func(times, corrs, smooth_rate=1 / 2, z_exp=2):
    samples = corrs.shape[0]
    f_exp = 1 / z_exp
    time_scale = numpy.power(times, f_exp)
    corr_mean = numpy.mean(corrs, axis=0) * time_scale
    corr_std = numpy.std(corrs, axis=0) * time_scale
    times_log = numpy.log(times)
    spl_tck_diag = scipy.interpolate.splrep(times_log, corr_mean,
                                            w=smooth_rate * numpy.sqrt(samples) / corr_std)

    corr_smooth = scipy.interpolate.splev(times_log, spl_tck_diag)
    z = - corr_smooth * time_scale / (scipy.interpolate.splev(times_log, spl_tck_diag, der=1) + f_exp)
    z = -f_exp + scipy.interpolate.splev(times_log, spl_tck_diag, der=1) / corr_smooth
    z = -1 / z

    return corr_mean / time_scale, corr_smooth / time_scale, z


# %%

samples = corr.shape[0]
nsub = 5
nsam = samples // nsub
#nsam = 20

s_corr = numpy.zeros((nsam, len(times), 2))
s_corr_smooth = numpy.zeros((nsam, len(times), 2))
s_dcorr_smooth = numpy.zeros((nsam, len(times), 2))

for n in tqdm.tqdm(range(nsam)):
    for m in range(2):
        s_corr[n, :, m], s_corr_smooth[n, :, m], s_dcorr_smooth[n, :, m] = z_func(
            times, corr[n * nsub:(n + 1) * nsub, m, :],
            smooth_rate=1,
            z_exp=3/2)

# plot the data
f_x = 3
fig, axs = plt.subplots(num=0, nrows=2, ncols=1, sharex='all', squeeze=True, figsize=(4 * f_x, f_x * 6))

for m, col in zip(range(2), ['tab:blue', 'tab:orange']):
    axs[0].plot(times, numpy.mean(corr[:, m, :], axis=0), '.-', color='black', label='raw', alpha=0.2)
    for n in range(nsam):
        axs[0].plot(times, s_corr_smooth[n, :, m], '-', color=col, alpha=0.2)
    axs[0].plot(times, numpy.mean(s_corr_smooth, axis=0)[:, m], '-', color=col, alpha=1)

    for n in range(nsam):
        axs[1].plot(times, s_dcorr_smooth[n, :, m], '-', color=col, alpha=0.2)
    axs[1].plot(times, numpy.mean(s_dcorr_smooth, axis=0)[:, m], '.-', color=col, alpha=1)

    axs[1].fill_between(times, numpy.mean(s_dcorr_smooth, axis=0)[:, m] - 3 * numpy.std(s_dcorr_smooth, axis=0)[:,
                                                                              m] / numpy.sqrt(nsam),
                        numpy.mean(s_dcorr_smooth, axis=0)[:, m] + 3 * numpy.std(s_dcorr_smooth, axis=0)[:,
                                                                       m] / numpy.sqrt(nsam), color=col, alpha=0.5)

axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[1].set_xscale('log')
# plt.xlim([1e1, 5*1e2])
#axs[0].set_ylim([3 * 1.e-4, 1.e-1])
axs[1].set_ylim([1.25, 1.75])
# plt.yscale('log')
plt.show()
