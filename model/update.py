import numpy


def update_state(del_t, gamma, state):
    # Configure random number generator
    length = state.shape[0]
    rng = numpy.random.default_rng()  # New recommended way to use random in NumPy

    for k in range(length * 2 * del_t):
        sysindex = rng.integers(0, 2)  # Randomly 0 or 1
        pos1 = rng.integers(0, length)  # Random position within the system length

        # Only update position if it's occupied
        if state[pos1, sysindex] == 1:
            # Calculating pos2, consider moving direction
            pos2 = (length + pos1 + 1 - 2 * sysindex) % length
            # Check if hopping is possible
            if state[pos2, sysindex] == 0:
                sysindex2 = (sysindex + 1) % 2
                if state[pos1, sysindex2] * (1 - state[pos2, sysindex2]) == 1:
                    if rng.random() <= gamma:
                        # Perform the hopping
                        state[pos1, sysindex] = 0
                        state[pos2, sysindex] = 1
                else:
                    # Perform the hopping regardless of gamma
                    state[pos1, sysindex] = 0
                    state[pos2, sysindex] = 1

    return state


def update_state_sync(del_t, gamma, state):
    # Configure random number generator
    length = state.shape[1]
    rng = numpy.random.default_rng()  # New recommended way to use random in NumPy

    for k in range(length * 2 * del_t):
        pos1 = rng.integers(0, length * 2)
        sysindex1 = pos1 // length
        sysindex2 = (sysindex1 + 1) % 2
        pos1 = pos1 % length
        pos2 = (length + pos1 + 1 - 2 * sysindex1) % length

        # Only update position if it's occupied
        jumps_all = state[:, pos1, sysindex1] * (1 - state[:, pos2, sysindex1]) == 1
        if rng.random() <= gamma:
            jumps = jumps_all
        else:
            jumps = numpy.logical_and(jumps_all, state[:, pos1, sysindex2] * (1 - state[:, pos2, sysindex2]) == 0)

        state[jumps, pos1, sysindex1] = 0
        state[jumps, pos2, sysindex1] = 1
    return state
