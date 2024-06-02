import numpy


def z_finite_difference(time, corr, dx=0.01, ndx=10, n=1):
    """
    Calculate the finite difference of the correlation function
    Note: n = 1 works best
    :param time:
    :param corr:
    :param dx:
    :param ndx:
    :param n: n=1 best working order as default
    :return:
    """
    coeffs = {
        2: [1 / 2],
        4: [2 / 3, -1 / 12],
        6: [3 / 4, -3 / 20, 1 / 60],
        8: [4 / 5, -1 / 5, 4 / 105, -1 / 280],
    }

    scoeffs = {1: [-1, 1],
               2: [-3 / 2, 2, -1 / 2],
               3: [-11 / 6, 3, -3 / 2, 1 / 3],
               4: [-25 / 12, 4, -3, 4 / 3, -1 / 4],
               5: [-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5],
               6: [-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6]}

    coeff = numpy.array(coeffs[2 * n])
    scoeff = numpy.array(scoeffs[2 * n])

    x = numpy.log(time)
    y = numpy.log(corr)

    x_grid = numpy.linspace(start=x[0], stop=x[-1], num=int((x[-1] - x[0]) / dx + 1))
    y_grid = numpy.interp(x_grid, x, y)
    z_grid = numpy.zeros_like(y_grid)

    grid_bound = n * ndx

    # left/right boundary
    for i in range(len(scoeff)):
        z_grid[:grid_bound] += y_grid[i * ndx:i * ndx + grid_bound] * scoeff[i]
        z_grid[-grid_bound:] -= y_grid[-i * ndx - grid_bound:-i * ndx if i > 0 else None] * scoeff[i]

    # bulk
    for i in range(n):
        for sign in [-1, 1]:
            redge = -grid_bound + sign * (i + 1) * ndx
            redge = None if redge == 0 else redge
            z_grid[grid_bound:-grid_bound] += (
                    coeff[i] * sign *
                    y_grid[grid_bound + sign * (i + 1) * ndx:redge])

    z_grid = -ndx * dx / z_grid

    return numpy.exp(x_grid), z_grid
