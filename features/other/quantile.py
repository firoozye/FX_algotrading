import numpy as np
import pandas as pd


def quantile(array, q):
    return np.quantile(a=array, q=q)


def get_rolling_quantile(series, window, q):
    return pd.Series(series).rolling(window).quantile(q)
