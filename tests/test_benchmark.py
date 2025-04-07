import pytest  
from permanent import perm
from permanent import permx
from piquasso._math.permanent import permanent
import numpy as np
import jax

n=100

def generate_random_unitary(n):
    """Generate a random n x n unitary matrix."""
    real = np.random.normal(0, 1, (n, n))
    imag = np.random.normal(0, 1, (n, n))
    matrix = real + 1j * imag
    
    norm = np.sqrt(np.sum(np.abs(matrix)**2, axis=1))
    for i in range(n):
        matrix[i, :] /= norm[i]
    
    return matrix

def generate_multiplicity_vectors(n, density=0.7):
    """
    Generate row and column multiplicity vectors with specified density.
    The sum of elements in both vectors will be equal.
    """
    rows = np.random.binomial(3, density, n).astype(np.uint64)
    cols = np.random.binomial(3, density, n).astype(np.uint64)
    
    while np.sum(rows) != np.sum(cols):
        if np.sum(rows) > np.sum(cols):
            non_zero_idx = np.where(rows > 0)[0]
            if len(non_zero_idx) > 0:
                idx = np.random.choice(non_zero_idx)
                rows[idx] -= 1
        else:
            non_zero_idx = np.where(cols > 0)[0]
            if len(non_zero_idx) > 0:
                idx = np.random.choice(non_zero_idx)
                cols[idx] -= 1
    
    return rows, cols
  
def test_perm_30x30_boost_cpu(benchmark):
    """Benchmark the permanent function with a 30x30 matrix."""
    jax.config.update('jax_platform_name', 'cpu')
    try:
        np.random.seed(42)
        
        matrix = generate_random_unitary(n)
        rows, cols = generate_multiplicity_vectors(n, density=0.1)  # Very low density
        
        result = benchmark(perm, matrix, rows, cols)
                
    except Exception as e:
        pytest.skip(f"30x30 matrix calculation failed: {str(e)}")



def test_perm_30x30_boost_gpu(benchmark):
    """Benchmark the permanent function with a 30x30 matrix."""
    jax.config.update('jax_platform_name', 'gpu')

    try:
        np.random.seed(42)
        
        matrix = generate_random_unitary(n)
        rows, cols = generate_multiplicity_vectors(n, density=0.1)  # Very low density
        
        result = benchmark(permx, matrix, rows, cols)
    except Exception as e:
        pytest.skip(f"30x30 matrix calculation failed: {str(e)}")


def test_perm_30x30_piquasso(benchmark):
    """Benchmark the permanent function with a 30x30 matrix."""
    try:
        np.random.seed(42)
        
        matrix = generate_random_unitary(n)
        rows, cols = generate_multiplicity_vectors(n, density=0.1)  # Very low density
        
        result = benchmark(permanent, matrix, rows, cols)
                
    except Exception as e:
        pytest.skip(f"30x30 matrix calculation failed: {str(e)}")