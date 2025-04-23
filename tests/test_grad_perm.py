from permanent import perm
import numpy as np
import jax


def test_permanent_trivial_case():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.complex128)
    rows = cols = np.ones(3, dtype=np.uint64)
    output = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.array_equal(output, np.array([[93.+0.j, 78.+0.j, 67.+0.j], [42.+0.j, 30. +
                                                                          0.j, 22.+0.j], [27.+0.j, 18.+0.j, 13.+0.j]]))

def test_grad_perm_identity():
    matrix = np.eye(3, dtype=np.complex128)
    rows = cols = np.ones(3, dtype=np.uint64)
    grad = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.allclose(grad, np.eye(3, dtype=np.complex128))

def test_grad_perm_single_entry():
    matrix = np.array([[2.0]], dtype=np.complex128)
    rows = cols = np.ones(1, dtype=np.uint64)
    grad = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.allclose(grad, np.array([[1.0+0j]]))

def test_grad_perm_zero_matrix():
    matrix = np.zeros((2, 2), dtype=np.complex128)
    rows = cols = np.ones(2, dtype=np.uint64)
    grad = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.allclose(grad, 0)

def test_grad_perm_all_ones():
    matrix = np.ones((2, 2), dtype=np.complex128)
    rows = cols = np.ones(2, dtype=np.uint64)
    grad = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.allclose(grad, np.ones((2, 2), dtype=np.complex128))

def test_grad_perm_zero_input_output():
    matrix = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    rows = cols = np.zeros(3, dtype=np.uint64)
    grad = jax.grad(perm, holomorphic=True)(matrix, rows, cols)
    assert np.allclose(grad, 0)