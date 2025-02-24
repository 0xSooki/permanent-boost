from perm.permanent import perm
import numpy as np
import jax


def test_permanent_trivial_case():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.complex128)

    rows = cols = np.ones(3, dtype=np.uint64)

    output = jax.grad(perm, holomorphic=True)(matrix, rows, cols)

    assert np.array_equal(output, np.array([[93.+0.j, 78.+0.j, 67.+0.j], [42.+0.j, 30. +
                                                                          0.j, 22.+0.j], [27.+0.j, 18.+0.j, 13.+0.j]]))
