import nolds
import numpy as np
from utils.numpy_ext import rolling_apply


def get_lyapunov_exponent(returns):
    """
    Computes 1st Lyapunov exponent given dynamical system's trajectory (returns). Positive value -> chaotic system,
    negative value -> stable system.
    Parameters
    ----------
    returns : pd.Series/np.ndarray/list
        Experimental time series (aka dynamical system trajectory)
    Returns
    -------
    exponent : Float
        1st exponent from the Lyapunov spectrum
    """
    returns = np.array(returns)
    returns = np.nan_to_num(returns, 0.0)
    exponent = nolds.lyap_r(returns)
    return exponent


def get_rolling_lyapunov_exponent(returns, win):
    """
    Computes 1st Lyapunov exponent given dynamical system's trajectory (returns) in a rolling fashion.
    Positive value -> chaotic system, negative value -> stable system.
    Parameters
    ----------
    returns : pd.Series/np.ndarray/list
        Experimental time series (aka dynamical system trajectory)
    win: int
        Moving window size
    Returns
    -------
    rolling_exponent : np.ndarray
        Rolling estimation of 1st exponent from the Lyapunov spectrum
    """
    rolling_exponent = rolling_apply(get_lyapunov_exponent, win, returns)
    return rolling_exponent
