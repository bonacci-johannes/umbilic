import glob
import numpy


# %%
def load_burgers_data(base_root, gamma_str):
    time = numpy.load(glob.glob(f'{base_root}/gamma{gamma_str}/*_ts*.npy')[0])
    corr = numpy.load(glob.glob(f'{base_root}/gamma{gamma_str}/*_equal_space_*.npy')[0])
    if time[0] == 0:
        time = time[1:]
        corr = corr[:, :, 1:]
    return corr, time
