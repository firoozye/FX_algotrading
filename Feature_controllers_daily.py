import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator


def get_AR_MA_features(df_old, columns, window_sizes, ar=True):

    df = pd.DataFrame()
    df.index = df_old.index
    """
    Adds financial features to the DataFrame for specified columns and multiple window sizes.

    Parameters:
    df (DataFrame): The original DataFrame.
    columns (list): List of column names to calculate features for.
    window_sizes (list): List of window sizes for calculating SMA, EMA, and rolling std.

    Returns:
    DataFrame: The original DataFrame with added financial features.
    """
    for col in columns:
        # Calculate returns
        df[f'{col}_returns'] = df_old[col].pct_change()
        for window_size in window_sizes:
            # SMA of returns
            # df[f'{col}_sma_{window_size}'] = df[f'{col}_returns'].rolling(window=window_size).mean()

            if not ar:
                # EMA of returns
                df[f'{col}_ema_{window_size}'] = (df[f'{col}_returns'].
                                                  ewm(span=window_size, adjust=False).mean())
            else:
                # Lagged Returns
                df[f'{col}_AR_{window_size}'] = (df[f'{col}_returns'].
                                                 shift(window_size))

            if window_size > 2:
                # Rolling Standard Deviation
                df[f'{col}_rolling_std_{window_size}'] = (df[f'{col}_returns'].
                                                          rolling(window=window_size).std())

    return df


def calc_surprise(df):

    eco_index_cols = [col for col in df.columns if '_ACTUAL_RELEASE' in col]
    prefixes = [col.split('_ACTUAL_RELEASE')[0] for col in eco_index_cols]




    for prefix in prefixes:
        act_col = f"{prefix}_ACTUAL_RELEASE"
        est_col = f"{prefix}_SURVEY_AVERAGE"
        std_col = f"{prefix}_FORECAST_STANDARD_DEVIATION"
        high_col = f"{prefix}_SURVEY_HIGH"
        low_col = f"{prefix}_SURVEY_LOW"
        surp_col = f"{prefix}_SURP"


        df[surp_col] = (df[act_col] - df[est_col])/np.where(df[std_col]==0,(df[high_col]-df[low_col])/4,df[std_col])
        df[surp_col].replace([np.inf, -np.inf], 0, inplace=True)
        df[surp_col].fillna(0,inplace=True)
    return df


def get_technicals(df, window_sizes):

    technicals_df = pd.DataFrame()
    technicals_df.index = df.index
    close_prices = df['GBPUSD_PX_LAST']
    high_prices = df['GBPUSD_PX_HIGH']
    low_prices = df['GBPUSD_PX_LOW']

    for i in window_sizes:
        colname = 'RSI_' + str(i)
        technicals_df[colname] = RSIIndicator(close=close_prices, window =i).rsi()

    for i in window_sizes:
        colname = 'oscillator_' + str(i)
        technicals_df[colname] = 100*((close_prices - low_prices.rolling(window = i).min()) / (high_prices.rolling(window = i).min() - low_prices.rolling(window = i).min()))

    for i in window_sizes:
        colname = 'adx_' + str(i)
        adxI = ADXIndicator(high=high_prices, low=low_prices, close=close_prices, window=i)
        technicals_df[colname] = adxI.adx()
    return technicals_df
