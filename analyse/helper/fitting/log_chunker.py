import numpy


def log_equal_chunker(time, dx_chunk=0.1, x_chunk_width=1):
    # transform time to log space
    x = numpy.log(time)

    # determine chunk centers. Note that times have to be integers
    t_center = numpy.exp(numpy.linspace(start=x[0], stop=x[-1], num=int((x[-1] - x[0]) / dx_chunk) + 1)).astype(int)
    t_center = numpy.unique(t_center)

    # determine chunk indices for center and left/right boundaries
    chunk_idx = numpy.zeros(shape=(t_center.size, 3), dtype=int)
    chunk_idx[:, 0] = numpy.searchsorted(time, t_center)
    chunk_idx[:, 1] = numpy.searchsorted(x, x[chunk_idx[:, 0]] - x_chunk_width / 2., side='left')
    chunk_idx[:, 2] = numpy.searchsorted(x, x[chunk_idx[:, 0]] + x_chunk_width / 2., side='right') - 1
    chunk_idx[chunk_idx[:, 1] == chunk_idx[:, 2], 2] += 1

    return chunk_idx
