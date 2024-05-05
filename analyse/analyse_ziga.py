import tqdm
import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt

# %% load numpy npy file
corr = numpy.load(
    'Burgers_data/gamma025/StochasticBurgersSystem_L32768_b3.00_lam0.00_nsteps409600_dt0.0050_samplefreq800_equal_space_correlator_nsample3996.npy')
times = numpy.load(
    'Burgers_data/gamma025/StochasticBurgersSystem_L32768_b3.00_lam0.00_nsteps409600_dt0.0050_samplefreq800_ts_nsample3996.npy')
# drop t=0
corr = corr[:, :, 1:]
times = times[1:]

# %% data smoothing level 1
times_log = numpy.linspace(start=numpy.log(times[0]), stop=numpy.log(times[-1]), num=300)
corr_mean = numpy.mean(corr, axis=0)
corr_std = numpy.std(corr, axis=0)

smooth_rate = 1
corr_l1 = numpy.zeros((corr.shape[0], 2, len(times_log)))
for n in tqdm.tqdm(range(corr.shape[0])):
    for m in range(2):
        corr_l1[n, m, :] = scipy.interpolate.make_smoothing_spline(
            numpy.log(times), corr[n, m, :],
            lam=1, w=smooth_rate / numpy.square(corr_std[m, :]))(times_log)

# %% data smoothing level 2
smooth_rate2 = 3
corr_l2 = numpy.zeros_like(corr_l1)
dcorr_l2 = numpy.zeros_like(corr_l1)

corr_l1_mean = numpy.mean(corr_l1, axis=0)
corr_l1_std = numpy.std(corr_l1, axis=0)
for n in tqdm.tqdm(range(corr.shape[0])):
    for m in range(2):
        spl_tck_diag = scipy.interpolate.splrep(times_log,
                                                corr_l1[n, m, :],
                                                w=smooth_rate2 / corr_l1_std[m, :])
        corr_l2[n, m, :] = scipy.interpolate.splev(times_log, spl_tck_diag)
        dcorr_l2[n, m, :] = scipy.interpolate.splev(times_log, spl_tck_diag, der=1)

corr_l2_mean = numpy.mean(corr_l2, axis=0)
dcorr_l2_mean = numpy.mean(dcorr_l2, axis=0)



# %% plot the data
plt.figure()
plt.plot(times, corr[0, 0, :], '.-', color='black', label='raw', alpha=0.2)
plt.plot(times, numpy.mean(corr, axis=0)[0, :], '.-', color='black', label='raw mean')
plt.plot(numpy.exp(times_log), corr_l1[0, 0, :], '.-', color='cyan', label='raw smooth level 1')
plt.plot(numpy.exp(times_log), numpy.mean(corr_l1, axis=0)[0, :], '-', color='cyan', label='smooth level 1')
plt.plot(numpy.exp(times_log), numpy.mean(corr_l2, axis=0)[0, :], '-', color='magenta', label='smooth level 2')
for n in range(1,2):
    plt.plot(numpy.exp(times_log), corr_l2[n, 0, :], '-', color='magenta', label='raw smooth level 2')
plt.plot(numpy.exp(times_log), corr_l2[0, 0, :], '-', color='tab:blue', label='raw smooth level 2')

plt.yscale('log')
plt.xscale('log')
plt.ylim([1.e-7, 1.e-1])
plt.show()

# %%
plt.figure()

plt.plot(numpy.exp(times_log),  - corr_l2_mean[0, :] / dcorr_l2_mean[0, :] , '-', color='magenta', label='raw smooth level 2')
plt.plot(numpy.exp(times_log),  - corr_l2_mean[1, :] / dcorr_l2_mean[1, :] , '-', color='cyan', label='raw smooth level 2')
#plt.plot(numpy.exp(times_log),  numpy.mean(dcorr_l2[:, 0, :] ,axis=0) , '-', color='tab:blue', label='raw smooth level 2')
#plt.plot(numpy.exp(times_log), numpy.mean(dcorr_l2, axis=0)[0, :], '-', color='cyan', label='raw smooth level 2')
plt.xscale('log')
#plt.ylim([-0.5, 3])
plt.show()