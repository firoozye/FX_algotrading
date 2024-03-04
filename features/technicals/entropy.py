import numpy as np
from math import factorial
from utils.numpy_ext import rolling_apply


def embed(x, order, delay):
    """
    Time-delay embedding
    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay
    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series
    """
    n = len(x)
    y = np.empty((order, n - (order - 1) * delay))
    for i in range(order):
        y[i] = x[i * delay:i * delay + y.shape[1]]
    return y.T


def permutation_entropy(time_series, order, delay, normalize=False):
    """
    Calculates permutation entropy
    Parameters
    ----------
    time_series : list or np.array
        Time series
    order : int
        Order of permutation entropy
    delay : int
        Delay
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1
    Returns
    -------
    pe : float
        Permutation entropy
    """
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    # Embed x and sort the order of permutations
    sorted_idx = embed(x, order=order, delay=delay).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


def rolling_pe(time_series, win, order, delay, normalize=False):
    """
    Calculates permutation entropy in a rolling window fashion
    Parameters
    ----------
    time_series : list or np.array
        Time series
    win : int
        Rolling window size
    order : int
        Order of permutation entropy
    delay : int
        Delay
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1
    Returns
    -------
    pe_rolling : np.ndarray
        Permutation entropy
    """
    pe_rolling = rolling_apply(permutation_entropy, win, time_series, order=order, delay=delay, normalize=normalize)
    return pe_rolling


def weighted_permutation_entropy(time_series, order, delay, normalize=False):
    """
    Calculates weighted permutation entropy
    Parameters
    ----------
    time_series : list or np.array
        Time series
    order : int
        Order of permutation entropy
    delay : int
        Delay
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1
    Returns
    -------
    wpe : float
        Weighted permutation entropy
    """
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    embedded = embed(x, order=order, delay=delay)
    sorted_idx = embedded.argsort(kind='quicksort')
    weights = np.var(embed(x, order, delay), 1)
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    mapping = {}
    for i in np.unique(hashval):
        mapping[i] = np.where(hashval == i)[0]
    weighted_counts = dict.fromkeys(mapping)
    for k, v in mapping.items():
        weighted_count = 0
        for i in v:
            weighted_count += weights[i]
        weighted_counts[k] = weighted_count
    weighted_counts_array = np.array(list(weighted_counts.values()))
    p = np.true_divide(weighted_counts_array, weighted_counts_array.sum())
    wpe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        wpe /= np.log2(factorial(order))
    return wpe


def rolling_wpe(time_series, win, order, delay, normalize=False):
    """
    Calculates weighted permutation entropy in a rolling window fashion
    Parameters
    ----------
    time_series : list or np.array
        Time series
    win : int
        Rolling window size
    order : int
        Order of permutation entropy
    delay : int
        Delay
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1
    Returns
    -------
    pe_rolling : np.ndarray
        Weighted permutation entropy
    """
    wpe_rolling = rolling_apply(weighted_permutation_entropy, win, time_series, order=order, delay=delay,
                                normalize=normalize)
    return wpe_rolling
