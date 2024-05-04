import numpy


def jacobi(u: float, v: float, q: float) -> numpy.ndarray:
    t1 = q * (u + v - 1)
    s1 = numpy.sqrt(1 + q * (4 * u * v - 2 * (u + v - 1) + q * ((u + v - 1) ** 2)))
    s2 = 2 * numpy.sqrt(4 * q * u * v + (t1 - 1) ** 2)

    j11 = 1 + s1 - t1 - 2 * v + 2 * s1 * (v - 2 * u)
    j22 = -1 - s1 + t1 + 2 * u - 2 * s1 * (u - 2 * v)
    j12 = 1 - s1 - t1 - 2 * u + 2 * s1 * u
    j21 = -1 + s1 + t1 + 2 * v - 2 * s1 * v

    j = numpy.array([[j11, j12], [j21, j22]])
    j = j / s2
    return j


def comp(u, v, q):
    t1 = q * (u + v - 1)
    o11 = (t1 - 1 + numpy.sqrt((t1 - 1) ** 2 + 4 * q * u * v)) / (2 * q)
    c = [[u * (1 - u), o11 - u * v], [o11 - u * v, v * (1 - v)]]
    return c


def gmat_two_dir(u, v, q):
    j = jacobi(u, v, q)
    e, ev = numpy.linalg.eig(j)  # Eigenvalue decomposition, returns eigenvalues and eigenvectors
    c = comp(u, v, q)
    r = numpy.linalg.inv(ev)
    norm = r @ c @ r.T  # Matrix multiplication
    r[0, :] = r[0, :] / numpy.sqrt(norm[0, 0])
    r[1, :] = r[1, :] / numpy.sqrt(norm[1, 1])
    r_inv = numpy.linalg.inv(r)

    # Assuming you need to return R and Rinv based on your MATLAB code structure
    return r, r_inv