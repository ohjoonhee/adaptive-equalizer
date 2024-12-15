import os.path as osp
import numpy as np


def random_walk(x):
    y = 0
    result = []
    for _ in x:
        result.append(y)
        y += np.random.normal(scale=1)
    return np.array(result)


def running_mean(x, N):
    return np.convolve(x, np.ones((N,)) / N)[(N - 1) :]


def rescale(x, min, max):
    return (x - np.min(x)) / (np.max(x) - np.min(x)) * (max - min) + min


def make_random_eq(shape, ma_window=15, min_db=-20, max_db=2):
    """
    Generates a random db-scaled equalizer filter.

    This function creates a random equalizer filter by performing a random walk,
    applying a running mean, and rescaling the values.

    Args:
        shape (int): The length of the equalizer filter.

    Returns:
        numpy.ndarray: The generated equalizer filter.
    """
    x = np.linspace(0, shape, shape)
    filter = random_walk(x)
    filter = running_mean(filter, ma_window)
    filter = rescale(filter, min_db, max_db)
    return filter


if __name__ == "__main__":
    # eq_path = "data/random_walk_eqs_db"
    eq_path = "/mnt/ssd/datasets/fma_medium/random_walk_eqs_db"

    for i in range(6000):
        eq = make_random_eq(1025)
        np.save(osp.join(eq_path, f"{i:06d}.npy"), eq)
