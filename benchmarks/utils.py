import pytest
import time
import numpy as np
from scipy.stats import unitary_group
import jax


def generate_random_unitary(n):
    """Generate a random n x n unitary matrix."""
    return unitary_group.rvs(n, random_state=42)


def generate_multiplicity_vectors(n, factor=3):
    """Generate row and column multiplicity vectors scaled by a given factor."""
    return factor * np.ones(n, dtype=np.uint64), factor * np.ones(n, dtype=np.uint64)


def print_header(title):
    print("=" * 60)
    print(f"{title}".center(60))
    print("=" * 60)


def benchmark_k_times(label, func, k=5):
    print_header(f"{label} (mean of {k} runs)")
    results = []
    times = []
    for _ in range(k):
        start_time = time.time()
        result = func()
        elapsed = time.time() - start_time
        results.append(result)
        times.append(elapsed)
    mean_time = np.mean(times)
    print(f"Mean Elapsed Time: {mean_time:.6f} seconds")
    print(f"All times: {times}")
    return results, times
