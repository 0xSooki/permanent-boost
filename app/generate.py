import numpy as np
from scipy.stats import unitary_group


def generate_random_unitary(n):
    """Generate a random n x n unitary matrix."""
    return unitary_group.rvs(n, random_state=42)


def generate_multiplicity_vectors(n, max_val=5):
    """Generate row and column multiplicity vectors with random values from 0 to max_val,
    ensuring their sums are equal."""
    rng = np.random.default_rng(seed=42)
    total = rng.integers(n, n * max_val + 1)
    rows = np.zeros(n, dtype=np.uint64)
    cols = np.zeros(n, dtype=np.uint64)
    while True:
        rows = rng.multinomial(total, np.ones(n) / n)
        cols = rng.multinomial(total, np.ones(n) / n)
        if np.all(rows <= max_val) and np.all(cols <= max_val):
            break
    return rows, cols


n = 10
matrix = generate_random_unitary(n)
rows, cols = generate_multiplicity_vectors(n)

data = {
    "matrix": matrix,
    "rows": rows,
    "cols": cols,
}

np.save("example_data.npy", data)

loaded = np.load("example_data.npy", allow_pickle=True).item()
print("Matrix:\n", loaded["matrix"])
print("Rows:", loaded["rows"])
print("Cols:", loaded["cols"])
