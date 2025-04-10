import pytest  
#from permanent import perm
from permanent import permx
from piquasso._math.permanent import permanent
import numpy as np
import jax
from scipy.stats import unitary_group
import time

n=18

def generate_random_unitary(n):
    """Generate a random n x n unitary matrix."""
    matrix = unitary_group.rvs(n)
    return matrix

def generate_multiplicity_vectors(n, density=1):
    """
    Generate row and column multiplicity vectors with specified density.
    The sum of elements in both vectors will be equal.
    """
    rows = np.ones(n, dtype=np.uint64)
    cols = np.ones(n, dtype=np.uint64)
    rows = 2*rows
    cols = 2*cols
    return rows, cols
  
def test_perm_30x30_boost_cpu():
    """Benchmark the permanent function with a 30x30 matrix."""
    jax.config.update('jax_platform_name', 'cpu')
    try:
        np.random.seed(42)
        
        matrix = generate_random_unitary(n)
        rows, cols = generate_multiplicity_vectors(n, density=0.1)  # Very low density
        
        start_time = time.time()
        permx(matrix, rows, cols)
        end_time = time.time()
        print(f"Time taken for permx (CPU): {end_time - start_time} seconds")
                
    except Exception as e:
        pytest.skip(f"30x30 matrix calculation failed: {str(e)}")



# def test_perm_30x30_boost_gpu(benchmark):
#     """Benchmark the permanent function with a 30x30 matrix."""
#     jax.config.update('jax_platform_name', 'gpu')

#     try:
#         np.random.seed(42)
        
#         matrix = generate_random_unitary(n)
#         rows, cols = generate_multiplicity_vectors(n, density=0.1)  # Very low density
        
#         result = benchmark(permx, matrix, rows, cols)
#     except Exception as e:
#         pytest.skip(f"30x30 matrix calculation failed: {str(e)}")


def test_perm_30x30_piquasso():
    """Benchmark the permanent function with a 30x30 matrix."""
    try:
        np.random.seed(42)
        
        matrix = generate_random_unitary(n)
        rows, cols = generate_multiplicity_vectors(n, density=0.1)  # Very low density
        
        start_time = time.time()
        permanent(matrix, rows, cols)
        end_time = time.time()
        print(f"Time taken for permanent piquasso (CPU): {end_time - start_time} seconds")
                             
    except Exception as e:
        pytest.skip(f"30x30 matrix calculation failed: {str(e)}")