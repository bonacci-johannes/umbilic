import numpy


# %%
def create_state(length, gamma, seed, num=1):
    p_xx = (-1 + numpy.sqrt(gamma)) / (2 * gamma - 2)
    n10 = int(length * (0.5 - p_xx))

    numpy.random.seed(seed)

    if num == 1:
        state = numpy.zeros((length, 2), dtype=numpy.int8)
        x = numpy.random.choice(length, length // 2 + n10, replace=False)
        state[x[:n10], 0] = 1
        state[x[n10:2 * n10], 1] = 1
        state[x[2 * n10:], :] = 1
    else:
        state = numpy.zeros((num, length, 2), dtype=numpy.int8)
        for n in range(num):
            x = numpy.random.choice(length, length // 2 + n10, replace=False)
            state[n, x[:n10], 0] = 1
            state[n, x[n10:2 * n10], 1] = 1
            state[n, x[2 * n10:], :] = 1

    return state

# %%
# possible mapping
# 00 -> 0, 11 -> 1, 01 -> 2, 10 -> 3
