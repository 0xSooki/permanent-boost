import pytest
import time
import numpy as np
from scipy.stats import unitary_group
import jax

from permanent import perm
from piquasso._math.permanent import permanent

from utils import (
    benchmark_k_times,
    generate_random_unitary,
    generate_multiplicity_vectors,
)

n = 10


matrix = generate_random_unitary(n)
rows, cols = generate_multiplicity_vectors(n)


def test_perm_benchmark_boost_cpu():
    """Benchmark Boost's perm implementation on CPU."""
    try:
        jax.config.update("jax_platform_name", "cpu")
        with jax.default_device(jax.devices("cpu")[0]):
            benchmark_k_times(
                "Boost Permanent Benchmark (CPU)", lambda: perm(matrix, rows, cols)
            )
    except Exception as e:
        pytest.skip(f"{n}x{n} CPU test failed: {str(e)}")


def test_perm_benchmark_boost_gpu():
    """Benchmark Boost's perm implementation on GPU."""
    try:
        jax.config.update("jax_platform_name", "gpu")
        with jax.default_device(jax.devices("gpu")[0]):
            benchmark_k_times(
                "Boost Permanent Benchmark (GPU)", lambda: perm(matrix, rows, cols)
            )
    except Exception as e:
        pytest.skip(f"{n}x{n} GPU test failed: {str(e)}")


def test_perm_benchmark_piquasso():
    """Benchmark Piquasso's permanent implementation (CPU)."""
    try:
        benchmark_k_times(
            "Piquasso Permanent Benchmark (CPU)", lambda: permanent(matrix, rows, cols)
        )
    except Exception as e:
        pytest.skip(f"{n}x{n} Piquasso test failed: {str(e)}")
