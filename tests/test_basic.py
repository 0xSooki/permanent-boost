from __future__ import annotations

from sooki import permanent
import numpy as np


def test_permanent():
    complex_matrix = np.array(
        [[1.1 + 2.2j, 2 + 1j], [3 + 2j, 4 + 4j]], dtype=np.complex128)
    row_mult = np.array([1, 1], dtype=np.int64, copy=False)
    col_mult = np.array([1, 1], dtype=np.int64, copy=False)

    assert permanent(complex_matrix, row_mult, col_mult) == 10.1+9.2j
