import pandas as pd
from utils.numpy_ext import rolling_apply


def get_return(prices, period):
    """
    Computes backward-looking returns
    Parameters
    ----------
    prices : pd.Series
        Historical price data
    period : int
        Time period
    Returns
    -------
    result : pd.Series
        pd.Series with backward-looking returns
    """
    result = pd.Series(prices).pct_change(period)
    return result


def cum_ret(returns, offset=None):
    if offset:
        return (returns[:-offset] + 1).prod() - 1
    else:
        return (returns + 1).prod() - 1


def get_cum_return(returns, window, offset=None):
    """
    Computes backward-looking cumulative returns
    Parameters
    ----------
    returns : pd.Series
        Historical (daily) return data
    window : int
        Moving window size
    offset : int
        Offset parameter
    Returns
    -------
    result : pd.Series
        pd.Series with backward-looking cumulative returns
    """
    result = rolling_apply(cum_ret, window, returns, offset=offset)
    result = pd.Series(result)
    return result
