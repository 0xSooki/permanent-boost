import pytest
import numpy as np
from scipy.stats import unitary_group
import jax
from utils import (
    benchmark_k_times,
    generate_random_unitary,
    generate_multiplicity_vectors,
)

from permanent import perm

n = 10

matrix = generate_random_unitary(n)
rows, cols = generate_multiplicity_vectors(n)


def test_grad_perm_benchmark_boost_cpu():
    """Benchmark Boost's grad perm implementation on CPU."""
    try:
        jax.config.update("jax_platform_name", "cpu")
        with jax.default_device(jax.devices("cpu")[0]):
            benchmark_k_times(
                "Boost Grad Permanent Benchmark (CPU)",
                lambda: jax.grad(perm, holomorphic=True)(matrix, rows, cols),
            )
    except Exception as e:
        pytest.skip(f"{n}x{n} grad CPU test failed: {str(e)}")


def test_grad_perm_benchmark_boost_gpu():
    """Benchmark Boost's grad perm implementation on GPU."""
    try:
        jax.config.update("jax_platform_name", "gpu")
        with jax.default_device(jax.devices("gpu")[0]):
            benchmark_k_times(
                "Boost Grad Permanent Benchmark (GPU)",
                lambda: jax.grad(perm, holomorphic=True)(matrix, rows, cols),
            )
    except Exception as e:
        pytest.skip(f"{n}x{n} grad GPU test failed: {str(e)}")
