from scipy.stats import skew, kurtosis, moment
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.numpy_ext import rolling_apply


def get_skew(returns):
    """
    Computes total skewness of returns' distribution
    Parameters
    ----------
    returns : list/ndarray/pd.Series
        Historical returns
    Returns
    -------
    skewness_ : Float
        Distribution skewness
    """
    skewness_ = skew(returns)
    return skewness_


def get_rolling_skew(returns, win):
    """
    Computes skewness of returns' distribution in a rolling window
    Parameters
    ----------
    returns : list/ndarray/pd.Series
        Historical returns
    win : int
        Rolling window size
    Returns
    -------
    rolling_skew : np.ndarray
        Rolling skewness
    """
    rolling_skew = rolling_apply(get_skew, win, returns)
    return rolling_skew


def get_kurtosis(returns):
    """
    Computes total kurtosis of returns' distribution
    Parameters
    ----------
    returns : list/ndarray/pd.Series
        Historical returns
    Returns
    -------
    kurtosis_ : Float
        Distribution kurtosis
    """
    kurtosis_ = kurtosis(returns)
    return kurtosis_


def get_rolling_kurtosis(returns, win):
    """
    Computes kurtosis of returns' distribution in a rolling window
    Parameters
    ----------
    returns : list/ndarray/pd.Series
        Historical returns
    win : int
        Rolling window size
    Returns
    -------
    rolling_kurt : np.ndarray
        Rolling kurtosis
    """
    rolling_kurt = rolling_apply(get_skew, win, returns)
    return rolling_kurt


def get_nth_moment(returns, n):
    """
    Computes n-th moment for set of returns
    Parameters
    ----------
    returns : list/ndarray/pd.Series
        Historical returns
    n : int
        Ordinal number of a moment to compute
    Returns
    -------
    moment_ : Float
        n-th moment
    """
    moment_ = moment(a=returns, moment=n)
    return moment_


def get_rolling_moment(returns, win, n):
    """
    Computes n-th moment of returns in a rolling window
    Parameters
    ----------
    returns : list/ndarray/pd.Series
        Historical returns
    win : int
        Rolling window size
    n : int
        Ordinal number of a moment to compute
    Returns
    -------
    rolling_mom : np.ndarray
        Rolling n-th moment
    """
    rolling_mom = rolling_apply(get_nth_moment, win, returns, n=n)
    return rolling_mom


def get_iskew_sskew(returns, index, gcurve):
    """
    Computes idiosyncratic and systematic skewness
    Parameters
    ----------
    returns : pd.Series
        Historical returns
    index : pd.Series
        Historical index returns
    gcurve : pd.Series
        Historical (3-months) treasuries' returns
    Returns
    -------
    res : list [iskew, sskew]
        List with two elements: idiosyncratic and systematic skewness
    """
    gcurve /= 252
    adj_index = index - gcurve
    adj_returns = returns - gcurve
    x = pd.concat([adj_returns, adj_index, adj_index ** 2], axis=1)
    x.columns = ['target', 'ind', 'ind2']
    clf = LinearRegression()
    clf.fit(x[['ind', 'ind2']], x['target'])
    iskew = (clf.predict(x[['ind', 'ind2']]) - x['target']).skew()
    sskew = clf.coef_[1]
    res = [iskew, sskew]
    return res


def get_skurtosis(returns, index, gcurve):
    """
    Computes systematic kurtosis
    Parameters
    ----------
    returns : pd.Series
        historical returns
    index : pd.Series
        historical index returns
    gcurve : pd.Series
        historical (3-months) treasuries' returns
    Returns
    -------
    skurtosis : float
        systematic kurtosis
    """
    gcurve /= 252
    adj_index = index - gcurve
    adj_returns = returns - gcurve

    x = pd.concat([adj_returns, adj_index, adj_index ** 2, adj_index ** 3], axis=1)
    x.columns = ['target', 'ind', 'ind2', 'ind3']
    clf = LinearRegression()
    clf.fit(x[['ind', 'ind2', 'ind3']], x['target'])
    skurtosis = clf.coef_[2]

    return skurtosis
