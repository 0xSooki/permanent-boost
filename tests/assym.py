import numpy as np


def assym_reduce(array, row_reduce_on, col_reduce_on):
    particles = np.sum(row_reduce_on)

    proper_row_index = np.zeros(particles, dtype=int)
    proper_col_index = np.zeros(particles, dtype=int)

    row_stride = 0
    col_stride = 0

    for index in range(len(row_reduce_on)):
        row_multiplier = row_reduce_on[index]
        proper_row_index[row_stride: row_stride + row_multiplier] = index
        row_stride += row_multiplier

    for index in range(len(col_reduce_on)):
        col_multiplier = col_reduce_on[index]
        proper_col_index[col_stride: col_stride + col_multiplier] = index
        col_stride += col_multiplier

    return array[np.ix_(proper_row_index, proper_col_index)]
