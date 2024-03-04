import numpy as np
from utils.numpy_ext import rolling_apply


def trend(price: np.ndarray, win: int, deg: int, step: int, is_return: bool) -> float:
    """Predict price by trend

    Parameters
    ----------
    price: np.ndarray
        close price
    win: int
        window for model estimation
    step: int
        step from the last estimation point that we want to predict
    is_return: bool
        return price forecast or return from the last point
    deg: int
        Polynomial degree

    Returns
    -------
    prediction: float
        trend forecast on step
    """
    x = np.arange(win)
    y = price[-win:]
    coeff = np.polyfit(x, y, deg=deg)
    model = np.poly1d(coeff)
    if is_return:
        return (model(step+win-1)-y[-1])/y[-1]
    else:
        return model(step+win-1)


def get_rolling_trend(price_array: np.ndarray, win: int, deg: int, step: int, is_return: bool):
    """Predict price by trend - rolling window method

    Parameters
    ----------
    price_array: np.ndarray
        close price historical data
    win: int
        window for model estimation
    step: int
        step from the last estimation point that we want to predict
    is_return: bool
        return price forecast or return from the last point
    deg: int
        Polynomial degree

    Returns
    -------
    rolling_prediction: np.ndarray
        trend forecast on step - rolling window method
    """
    rolling_prediction = rolling_apply(trend, win, price_array, win=win, deg=deg, step=step, is_return=is_return)
    return rolling_prediction
