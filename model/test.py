import numpy
import scipy
import tqdm
import matplotlib.pyplot as plt

from model.initial_state import create_state
from model.update import update_state_sync
import glob

# %% load old data for comparison
from analyse.data_loader.cpp_struct_loader import Struct

struc = Struct(window=numpy.loadtxt(f'TWO_DIR/EXAMPLE_25/Window_Parameter.txt', comments='%'),
               record_files=sorted(glob.glob(f'TWO_DIR/EXAMPLE_25/Struc_fct_records/*_part_1.txt')),
               gamma=numpy.loadtxt(f'TWO_DIR/EXAMPLE_25/systemparameter.txt', comments='%')[5])
sx = struc.times
sy = [
    struc.s_bar_smooth[:, struc.x_points // 2, 0],
    struc.s_bar_smooth[:, struc.x_points // 2, 1],
    struc.s_smooth[:, struc.x_points // 2, 0],
    struc.s_smooth[:, struc.x_points // 2, 1]
]
sye = [
    struc.s_bar_smooth_std_err[:, struc.x_points // 2, 0],
    struc.s_bar_smooth_std_err[:, struc.x_points // 2, 1],
    struc.s_smooth_std_err[:, struc.x_points // 2, 0],
    struc.s_smooth_std_err[:, struc.x_points // 2, 1]
]
# %% run MC simulation
msys = 5
num = 1000
gamma = 0.25
length = 5000
t_max = 500
corr = numpy.zeros((msys, 4, t_max))

for m in range(msys):
    state = create_state(length, gamma, m, num=num)
    statem1 = state - 1
    state_t = state.copy()
    for t in tqdm.tqdm(range(t_max)):
        state_t = update_state_sync(1, gamma, state_t)
        state_tm1 = state_t - 1
        corr[m, 0, t] = (numpy.mean(state * state_t) + numpy.mean(statem1 * state_tm1)) / 2
        corr[m, 1, t] = (numpy.mean(state * state_t[:, :, [1, 0]]) + numpy.mean(statem1 * state_tm1[:, :, [1, 0]])) / 2

corr[:, [0, 1], :] -= 0.25
corr[:, 2, :] = (corr[:, 0, :] + corr[:, 1, :]) * (2 + 2 * numpy.sqrt(gamma))
corr[:, 3, :] = (corr[:, 0, :] - corr[:, 1, :]) * (2 + 2 / numpy.sqrt(gamma))

# %% spline fit
z_exp = 2
times = numpy.arange(start=1, stop=t_max + 1)
times_log = numpy.linspace(start=numpy.log(times[0]), stop=numpy.log(times[-1]), num=1000)

corr_mean = numpy.mean(corr, axis=0)
corr_std = numpy.std(corr, axis=0)
corr_std_tt = corr_std * numpy.power(times, 1 / z_exp)

corr_log = numpy.zeros((msys, 4, len(times_log)))
smooth_rate = 1
for m in range(msys):
    for n in range(4):
        corr_log[m, n, :] = scipy.interpolate.make_smoothing_spline(
            numpy.log(times), corr[m, n, :] * numpy.power(times, 1 / z_exp),
            lam=1, w=smooth_rate / numpy.square(corr_std_tt[n, :]))(times_log)
        corr_log[m, n, :] *= numpy.power(numpy.exp(times_log), -1 / z_exp)

corr_log_mean = numpy.mean(corr_log, axis=0)
corr_log_std = numpy.std(corr_log, axis=0)  # / numpy.sqrt(msys)
corr_log_std_tp = corr_log_std * numpy.power(numpy.exp(times_log), 1 / z_exp)

smooth_rate2 = 1
corr_log_smooth = numpy.zeros_like(corr_log)
corr_z_log_smooth = numpy.zeros_like(corr_log)
for m in range(msys):
    for n in range(4):
        spl_tck_diag = scipy.interpolate.splrep(times_log,
                                                corr_log[m, n, :] * numpy.power(numpy.exp(times_log), 1 / z_exp),
                                                w=smooth_rate2 / corr_log_std_tp[n, :])
        corr_log_smooth[m, n, :] = scipy.interpolate.splev(times_log, spl_tck_diag)
        corr_log_smooth[m, n, :] *= numpy.power(numpy.exp(times_log), -1 / z_exp)

corr_log_std = corr_log_std / numpy.sqrt(msys)
corr_log_smooth_mean = numpy.mean(corr_log_smooth, axis=0)

# %% plot scaled maximum
b = 3
plt.figure()
for n in range(4):
    plt.plot(times, numpy.power(b * times, 2 / 3) * corr_mean[n, :], label='raw mean' if n == 0 else None,
             color='black', alpha=0.2)
    plt.plot(numpy.exp(times_log), numpy.power(b * numpy.exp(times_log), 2 / 3) * corr_log_mean[n, :],
            color='cyan', label='smooth level 1' if n == 0 else None)
    #for m in range(msys):
    #   plt.plot(numpy.exp(times_log),
    #            numpy.power(b * numpy.exp(times_log), 2 / 3) * (corr_log_smooth)[m, n, :],
    #            color='tab:blue', label='smooth level 2' if n == 0 and m == 0 else None, alpha=1)
    plt.plot(numpy.exp(times_log),
            numpy.power(b * numpy.exp(times_log), 2 / 3) * (corr_log_smooth_mean)[n, :],
            color='black', label='smooth level 2' if n == 0 else None, alpha=1)
    plt.fill_between(numpy.exp(times_log),
                    numpy.power(b * numpy.exp(times_log), 2 / 3) * (corr_log_mean - 3 * corr_log_std)[n, :],
                    numpy.power(b * numpy.exp(times_log), 2 / 3) * (corr_log_mean + 3 * corr_log_std)[n, :],
                    color='tab:cyan', alpha=0.5)

for n in range(4):
    plt.fill_between(struc.times,
                     numpy.power(b * struc.times, 2 / 3) * (sy[n] - 3 * sye[n]),
                     numpy.power(b * struc.times, 2 / 3) * (sy[n] + 3 * sye[n]),
                     alpha=0.5, color='magenta',
                     zorder=100)
    plt.plot(struc.times,
             numpy.power(b * struc.times, 2 / 3) * (sy[n]),
             alpha=0.9, color='red',
             marker='.', markersize=1, linestyle='',
             label='old data 99% conf' if n == 0 else None,
             zorder=100)

plt.xscale('log')
plt.xlabel('t')
plt.ylabel('S*t^(2/3)')
plt.legend()
plt.savefig('figures_temp/s_max_smoothing')
plt.show()

# %%
plt.figure()
for n in range(4):
    plt.plot(numpy.exp(times_log), corr_z_log_smooth[n, :], label='log smooth spline')
plt.xscale('log')

plt.xlim([1, t_max])
plt.ylim([0, 4])
plt.legend()
plt.show()

# %%
nsub = 2
plt.figure()
for n in range(1):
    # n = 3
    n = nsub
    for m in range(1):
        plt.plot(times, corr[m, n, :], color='tab:blue', alpha=0.3, label=('raws' if m == 0 and n == 0 else None))
        # plt.fill_between(times, corr[m, n, :] - corr_std[n, :], corr[m, n, :] + corr_std[n, :],
        #                 color='tab:green', alpha=0.1)
        # plt.plot(times_log, corr_log[m, n, :], color='tab:red', alpha=0.9)

    plt.plot(times, corr_mean[n, :], label=('raws mean' if n == 0 else None), alpha=0.5, color='black')

    for m in range(1):
        plt.plot(numpy.exp(times_log), corr_log[m, n, :], color='tab:red', alpha=0.3,
                 label=('log smooth' if m == 0 and n == 0 else None))
for n in range(1):
    n = nsub
    plt.plot(times, corr_mean[n, :], label=('raws mean' if n == 0 else None), alpha=0.5, color='black')

    plt.plot(numpy.exp(times_log), corr_log_mean[n, :], label=('log smooth mean' if n == 0 else None),
             color='tab:orange')
    plt.plot(numpy.exp(times_log), corr_log_smooth[n, :], label=('log smooth spline' if n == 0 else None),
             color='black')

# set x and y axis on log scale
plt.xscale('log')
plt.yscale('log')

plt.xlim([10, t_max])
# plt.ylim([0.003, 0.03])
plt.legend()
plt.show()

# %%
# state_t should be an array of shape (num, length, 2)
# Compute differences using broadcasting
diffs_0 = (state[:, numpy.newaxis, :, :] - state[numpy.newaxis, :, :, :])
diffs_t = (state_t[:, numpy.newaxis, :, :] - state_t[numpy.newaxis, :, :, :])

diags = numpy.sum(diffs_0 * diffs_t) / (length * 2 * num * (num - 1))

# %%

rng = numpy.random.default_rng()
# clone rng to produce same random numbers
rng2 = numpy.random.default_rng()
rng2.bit_generator.state = rng.bit_generator.state

a = rng.random(3) - rng2.random(3)

# %%
p = numpy.mean(state_t, axis=0)
p = p * (1 - p)
