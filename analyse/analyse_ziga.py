import tqdm
import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt
#%%
matplotlib.use('TkAgg')


#%%

plt.figure()
plt.plot([0, 1],[0, 1])
plt.show()

# %% load numpy npy file
corr = numpy.load(
    'Burgers_data/gamma025/StochasticBurgersSystem_L32768_b3.00_lam0.00_nsteps409600_dt0.0050_samplefreq800_equal_space_correlator_nsample3996.npy')
times = numpy.load(
    'Burgers_data/gamma025/StochasticBurgersSystem_L32768_b3.00_lam0.00_nsteps409600_dt0.0050_samplefreq800_ts_nsample3996.npy')
# drop t=0
corr = corr[:, :, 1:]
times = times[1:]

# %% data smoothing
times_log = numpy.linspace(start=numpy.log(times[0]), stop=numpy.log(times[-1]), num=300)
corr_mean = numpy.mean(corr, axis=0)
corr_std = numpy.std(corr, axis=0)

smooth_rate = 1
corr_l1 = numpy.zeros((corr.shape[0], 2, len(times_log)))
for n in tqdm.tqdm(range(corr.shape[0])):
    corr_l1[n, 0, :] = scipy.interpolate.make_smoothing_spline(
        numpy.log(times), corr[n, 0, :],
        lam=1, w=smooth_rate / numpy.square(corr_std[0, :]))(times_log)

# %% plot the data
plt.figure()
plt.plot(times, corr[0, 0, :], '.-')
plt.plot(numpy.exp(times_log), corr_l1[0, 0, :], '.-')
plt.plot(times, numpy.mean(corr, axis=0)[0, :], '.-')
plt.plot(numpy.exp(times_log), numpy.mean(corr_l1, axis=0)[0, :], '-')

plt.yscale('log')
plt.xscale('log')
plt.ylim([1.e-4, 1.e-1])
plt.show()
