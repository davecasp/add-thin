"""
Metrics taken from TriTPP implementation:

https://github.com/shchur/triangular-tpp/

For further details checkout Appendix E.2 of the TriTPP paper.
"""

from typing import List

import numpy as np
from scipy.stats import wasserstein_distance


def forecast_wasserstein(
    X: List,
    Y: List,
    t_max: float,
):
    # Do some shape matching
    X, Y = match_shapes(X, Y, t_max)
    X = X / t_max
    Y = Y / t_max
    t_max = 1

    distance = []
    for x, y in zip(X, Y):
        distance.append(counting_distance(x, y.reshape(1, -1), t_max=t_max))
    return np.concatenate(distance).mean()


def counting_distance(x: np.ndarray, Y: np.ndarray, t_max: float):
    """Computes the distance between the counting process of x and y.
    This computation is batched and expects a 1D array for x and a 2D array for Y.
    From: https://arxiv.org/abs/1705.08051

    Args:
        x (np.ndarray): Event times for x
        Y (np.ndarray): List of sequences
        t_max (float): max time

    Returns:
        np.ndarray: distance
    """
    x, Y = x.copy(), Y.copy()
    x = x[None].repeat(Y.shape[0], 0)
    x_len = (x < t_max).sum(-1)
    y_len = (Y < t_max).sum(-1)
    to_swap = x_len > y_len
    # Fix implementation error in TriTPP code that does not take effect as match_shapes() takes tmax as filling value.
    x[to_swap], Y[to_swap] = Y[to_swap].copy(), x[to_swap].copy()
    mask_x = x < t_max
    mask_y = Y < t_max
    result = (np.abs(x - Y) * mask_x).sum(-1)
    result += ((t_max - Y) * (~mask_x & mask_y)).sum(-1)
    return result


def gaussian_kernel(x: np.ndarray, sigma2: float = 1):
    return np.exp(-x / (2 * sigma2))


def match_shapes(X: List, Y: List, t_max: float):
    """Match shapes between two lists of np.ndarray. Returns two np.ndarray with the same length in the second dim.

    Args:
        X (List): List of sequences
        Y (List): List of sequences
        t_max (float): max time
    """
    max_x = max([(x < t_max).sum() for x in X])
    max_y = max([(y < t_max).sum() for y in Y])
    max_size = max(max_x, max_y)
    new_X = np.ones((len(X), max_size)) * t_max
    new_Y = np.ones((len(Y), max_size)) * t_max
    for i, x in enumerate(X):
        x = x[x < t_max]
        new_X[i, : len(x)] = x
    for i, y in enumerate(Y):
        y = y[y < t_max]
        new_Y[i, : len(y)] = y
    return new_X, new_Y


def MMD(
    X: List, Y: List, t_max: float, sample_size: int = None, sigma: float = None
):
    """Computes the maximum mean discrepency between the samples X and samples Y.
    MMD is defined as E[k(x, x)] - 2*E[k(x, y)] + E[k(y, y)]. We use a Gaussian kernel
    with the counting distance. k(x, y) = exp(-d(x, y)/(2*sigma2)) where d is the counting distance
    and sigma is either given or estimated as the median distance between all pairs.

    Args:
        X (List): List of sequences
        Y (List): List of sequences
        t_max (float): max time
        sample_size (int, optional): If given MMD is only computed for subsets of X and Y.
            This improves performance at the cost of performance. Defaults to None.
        sigma (float, optional): Sigma for the Gaussian kernel. If not given it is estimated. Defaults to None.

    Returns:
        float: MMD(X, Y)
    """
    # Do some shape matching
    X, Y = match_shapes(X, Y, t_max)

    # Sample from both distributions
    if sample_size is not None:
        X = [X[i] for i in np.random.choice(len(X), sample_size)]
        Y = [Y[i] for i in np.random.choice(len(Y), sample_size)]
    # Normalize the time
    X = X / t_max
    Y = Y / t_max
    t_max = 1

    x_x_d = []
    for i, x1 in enumerate(X):
        x_x_d.append(counting_distance(x1, X, t_max=t_max))
    x_x_d = np.concatenate(x_x_d)

    x_y_d = []
    for x in X:
        x_y_d.append(counting_distance(x, Y, t_max=t_max))
    x_y_d = np.concatenate(x_y_d)

    y_y_d = []
    for i, y1 in enumerate(Y):
        y_y_d.append(counting_distance(y1, Y, t_max=t_max))
    y_y_d = np.concatenate(y_y_d)

    if sigma is None:
        sigma = np.median(np.concatenate([x_x_d, x_y_d, y_y_d]))
    sigma2 = sigma**2
    E_x_x = np.mean(gaussian_kernel(x_x_d, sigma2))
    E_x_y = np.mean(gaussian_kernel(x_y_d, sigma2))
    E_y_y = np.mean(gaussian_kernel(y_y_d, sigma2))

    return np.sqrt(E_x_x - 2 * E_x_y + E_y_y), sigma


def lengths_distribution_wasserstein_distance(
    X: List, Y: List, t_max: float, mean_number_items: float
):
    """
    Returns the Wasserstein between the distribution of sequence lengths between X and Y.

    Args:
        X (List): List of sequences
        Y (List): List of sequences
        t_max (float): max time
        mean_number_items (float): Mean number of events from the dataset. This is used for normalization

    Returns:
        float: Wasserstein distance
    """
    X_lengths = np.array([(s < t_max).sum().item() for s in X])
    Y_lengths = np.array([(s < t_max).sum().item() for s in Y])
    return wasserstein_distance(
        X_lengths / mean_number_items, Y_lengths / mean_number_items
    )
