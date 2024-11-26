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


def db2mag(x):
    return np.power(10, x / 20)


def make_random_eq(shape):
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
    filter = running_mean(filter, 15)
    filter = rescale(filter, -20, 2)
    return filter


if __name__ == "__main__":
    eq_path = "data/random_walk_eqs_db"

    for i in range(1010):
        eq = make_random_eq(1025)
        np.save(osp.join(eq_path, f"{i:05d}.npy"), eq)
