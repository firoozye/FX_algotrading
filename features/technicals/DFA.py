import nolds
import numpy as np
from utils.numpy_ext import rolling_apply


def get_dfa(time_series, order=2, overlap=True):
    """
    Performs detrended fluctuation analysis, computes Hurst parameter similar to the Hurst exponent.
    Can be used for non-stationary TS.
    Parameters
    ----------
    time_series : pd.Series/np.ndarray/list
        Experimental time series (aka dynamical system trajectory)
    overlap : bool
        Whether to use window overlapping or not
    order : int
        Polynomial's order to fit the data
    Returns
    -------
    hurst_param : float
        Hurst parameter
    """
    hurst_param = nolds.dfa(data=time_series, overlap=overlap, order=order)
    return hurst_param


def get_rolling_dfa(time_series, win, order=2, overlap=True):
    """
    Performs detrended fluctuation analysis in a rolling window fashion,
    computes Hurst parameter similar to the Hurst exponent.
    Can be used for non-stationary TS.
    Parameters
    ----------
    time_series : pd.Series/np.ndarray/list
        Experimental time series (aka dynamical system trajectory)
    win: int
        Window for model estimation
    overlap : bool
        Whether to use window overlapping or not
    order : int
        Polynomial's order to fit the data
    Returns
    -------
    rolling_hurst_param : np.ndarray
        Rollign Hurst parameter
    """
    rolling_hurst = rolling_apply(get_dfa, win, time_series, order=order, overlap=overlap)
    return rolling_hurst
