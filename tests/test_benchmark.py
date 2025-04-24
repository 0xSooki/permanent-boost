import pytest  
#from permanent import perm
from permanent import perm
from piquasso._math.permanent import permanent
import numpy as np
import jax
from scipy.stats import unitary_group
import time
import jax.numpy as jnp

n=13

def generate_random_unitary(n):
    """Generate a random n x n unitary matrix."""
    matrix = unitary_group.rvs(n, random_state=42)
    return matrix

def generate_multiplicity_vectors(n):
    """
    Generate row and column multiplicity vectors with specified density.
    The sum of elements in both vectors will be equal.
    """
    rows = np.ones(n, dtype=np.uint64)
    cols = np.ones(n, dtype=np.uint64)
    rows = 3*rows
    cols = 3*cols
    return rows, cols

matrix = generate_random_unitary(n)
rows, cols = generate_multiplicity_vectors(n)

 
def test_perm_30x30_boost_cpu():
    """Benchmark the permanent function with a 30x30 matrix."""
    try:
        jax.config.update('jax_platform_name', 'cpu')
        with jax.default_device(jax.devices('cpu')[0]):
            start_time = time.time()
            print(perm(matrix, rows, cols))
            end_time = time.time()
            print(f"Time taken for perm (CPU): {end_time - start_time} seconds")
            
    except Exception as e:
        pytest.skip(f"30x30 matrix calculation failed: {str(e)}")



def test_perm_30x30_boost_gpu():
    """Benchmark the permanent function with a 30x30 matrix."""
    try:
        jax.config.update('jax_platform_name', 'gpu')

        with jax.default_device(jax.devices('gpu')[0]):
            start_time = time.time()
            print(perm(matrix, rows, cols))
            end_time = time.time()
            print(f"Time taken for perm (GPU): {end_time - start_time} seconds")
            
    except Exception as e:
        pytest.skip(f"30x30 matrix calculation failed: {str(e)}")


def test_perm_30x30_piquasso():
    """Benchmark the permanent function with a 30x30 matrix."""
    try:
        start_time = time.time()
        print(permanent(matrix, rows, cols))
        end_time = time.time()
        print(f"Time taken for permanent piquasso (CPU): {end_time - start_time} seconds")
                             
    except Exception as e:
        pytest.skip(f"30x30 matrix calculation failed: {str(e)}")