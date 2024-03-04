import pandas as pd
import numpy as np
from scipy import stats
import itertools
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator


def get_sma(series, use_custom_window=False, wind_min=3, wind_max=100, window=126):
    """
    Computes Simple moving averages
    Parameters
    ----------
    series : pd.Series (T, 1)
        historical data
    use_custom_window : bool
        Use user-specified window for SMA or not
    window : int
        User-specified window
    wind_min : int
        minimum moving window size
    wind_max : int
        maximum moving window size
    Returns
    -------
    sma_df : pd.DataFrame
        dataframe (T, wind_max - wind_min) with sma values in columns, if use_custom_window==False
    sma : pd.Series
        SMA values, in use_custom_window==True
    """
    if not use_custom_window:
        sma_df = pd.DataFrame()
        for i in range(wind_min, wind_max + 1):
            colname = 'sma_' + str(i)
            sma_df[colname] = series.rolling(i).mean()
        return sma_df
    else:
        sma = series.rolling(window).mean()
        return sma


def get_ema(series, wind_min=3, wind_max=100, use_custom_window=False, window=126):
    """
    Computes Exponential moving averages
    Parameters
    ----------
    series : pd.Series (T, 1)
        historical data
    wind_min : int
        minimum moving window size
    wind_max : int
        maximum moving window size
    use_custom_window : bool
        Use user-specified window for SMA or not
    window : int
        User-specified window
    Returns
    -------
    ema_df : pd.DataFrame
        dataframe (T, wind_max - wind_min) with ema values in columns, if use_custom_window==False
    ema : pd.Series
        pd.Series win ema values, if use_custom_window==True
    """
    if not use_custom_window:
        ema_df = pd.DataFrame()
        for i in range(wind_min, wind_max + 1):
            colname = 'ema_' + str(i)
            ema_df[colname] = series.ewm(span=i, adjust=False).mean()
        return ema_df
    else:
        ema = series.ewm(span=window, adjust=False).mean()
        return ema


def get_macd(series, period1=12, period2=26):
    """
    Computes Moving average convergence divergence (MACD)
    Parameters
    ----------
    series : pd.Series (T, 1)
        historical data
    period1 : int
        period for EMA 1
    period2 : int
        period for EMA 2
    Returns
    -------
    macd : pd.Series
        Series with MACD values
    """
    ema1 = series.ewm(span=period1, adjust=False).mean()
    ema2 = series.ewm(span=period2, adjust=False).mean()
    macd = ema1 - ema2

    return macd


def get_oscillator(close, low, high, use_custom_window=False, window=21, wind_min=3, wind_max=100):
    """
    Computes stochastic oscillator
    Parameters
    ----------
    high : pd.Series (T, 1)
        historical prices - high
    low : pd.Series (T, 1)
        historical prices - low
    close : pd.Series (T, 1)
        historical prices - close
    use_custom_window : bool
        Use custom window or not
    window : int
        Custom window to use
    wind_min : int
        minimum moving window size
    wind_max : int
        maximum moving window size
    Returns
    -------
    osc_df : pd.DataFrame
        dataframe (T, wind_max - wind_min) with stochastic oscillator values in columns if use_custom_window==False
    osc : pd.Series
        pd.Series with stochastic oscillator values if use_custom_window==True
    """
    if not use_custom_window:
        osc_df = pd.DataFrame()
        for i in range(wind_min, wind_max + 1):
            colname = 'oscillator_' + str(i)
            osc_df[colname] = 100*((close - low.rolling(window = i).min()) / (high.rolling(window = i).min() - low.rolling(window = i).min()))
        return osc_df
    else:
        osc = 100*((close - low.rolling(window=window).min()) / (high.rolling(window=window).min() - low.rolling(window=window).min()))
        return osc


def get_bollinger_bands(series_close, use_custom_window=False, window=21, wind_min=5, wind_max=100):
    """
    Computes Bollinger bands
    Parameters
    ----------
    series_close : pd.Series (T, 1)
        historical close prices
    use_custom_window : bool
        Whether to use custom window or not
    window : int
        Custom, user-specified window
    wind_min : int
        minimum moving window size
    wind_max : int
        maximum moving window size
    Returns
    -------
    bollinger_df : pd.DataFrame
        dataframe (T, 2 * (wind_max - wind_min)) with BB values in columns, if use_custom_window==False
    bollinger_h, bollinger_l : pd.Series, pd.Series
        pd.Series wint BB values, if use_custom_window==True
    """
    if not use_custom_window:
        bollinger_df = pd.DataFrame()
        for i in range(wind_min, wind_max + 1):
            colname_h = 'bollinger_h_' + str(i)
            colname_l = 'bollinger_l_' + str(i)
            bollinger_df[colname_h] = series_close.rolling(i).mean() + series_close.rolling(i).std()
            bollinger_df[colname_l] = series_close.rolling(i).mean() - series_close.rolling(i).std()
        return bollinger_df
    else:
        bollinger_h = series_close.rolling(window).mean() + series_close.rolling(window).std()
        bollinger_l = series_close.rolling(window).mean() - series_close.rolling(window).std()
        return bollinger_h, bollinger_l


def get_rsi(series_close, use_custom_window=False, window=21, wind_min=5, wind_max=100):
    """
    Computes Relative strength index (RSI)
    Parameters
    ----------
    series_close : pd.Series (T, 1)
        historical close prices
    use_custom_window : bool
        Whether to use custom window or not
    window : int
        Custom, user-specified window
    wind_min : int
        minimum moving window size
    wind_max : int
        maximum moving window size
    Returns
    -------
    rsi_df : pd.DataFrame
        dataframe (T, wind_max - wind_min) with RSI values in columns, if use_custom_window==False
    rsi : pd.Series
        Series with RSI values, if use_custom_window==True
    """
    if not use_custom_window:
        rsi_df = pd.DataFrame()
        for i in range(wind_min, wind_max + 1):
            colname = 'rsi_' + str(i)
            rsi_df[colname] = RSIIndicator(close=series_close, n=i).rsi()
        return rsi_df
    else:
        rsi = RSIIndicator(close=series_close, window=window).rsi()
        return rsi


def get_adx(high, low, close, use_custom_window=False, window=21, wind_min=5, wind_max=100):
    """
    Computes Average directional index
    Parameters
    ----------
    high : pd.Series (T, 1)
        historical prices - high
    low : pd.Series (T, 1)
        historical prices - low
    close : pd.Series (T, 1)
        historical prices - close
    use_custom_window : bool
        Whether to use custom window or not
    window : int
        Custom, user-specified window
    wind_min : int
        minimum moving window size
    wind_max : int
        maximum moving window size
    Returns
    -------
    adx_df : pd.DataFrame
        dataframe (T, wind_max - wind_min) with ADX values in columns, if use_custom_window==False
    adx : array-like
        Array with ADX values, if use_custom_window==True
    """
    if not use_custom_window:
        adx_df = pd.DataFrame()
        for i in range(wind_min, wind_max + 1):
            colname = 'adx_' + str(i)
            adxI = ADXIndicator(high=high, low=low, close=close, window=i)
            adx_df[colname] = adxI.adx()
        return adx_df
    else:
        adxI = ADXIndicator(high=high, low=low, close=close, window=window)
        adx = adxI.adx()
        return adx
