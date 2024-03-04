import pandas as pd
import numpy as np

from utils.settings import get_settings



def create_all_price_based_features(prices, command_dict):

    """
    Adds financial features to the DataFrame for specified columns and multiple window sizes.

    Parameters:
    df (DataFrame): The original DataFrame.
    columns (list): List of column names to calculate features for.
    window_sizes (list): List of window sizes for calculating SMA,  and rolling std.

    Returns:
    DataFrame: The original DataFrame with added financial features.
    """
    df = pd.DataFrame()
    df.index = prices.index
    default_dict ={
        'return_type': 'return', # return, log-return or price_change
        'horizon': 1, # one period ahead
        "sma_max_lag": 30,
        "sma_ret_max_lag":0,
        "ewma_max_span": 0,
        "ewma_ret_max_span": 0,
        "ar_max_lag": 0,
        "ar_ret_max_lag":0
    }

    local_prices = prices.copy()
    local_prices  = local_prices.rename(
        columns={
            x : x.split(' ')[0]
            for x in prices.columns
        }
    )

    columns = local_prices.columns
    df_dict = dict()
    for col in columns:

        df = pd.DataFrame()
        cleaning_spec_dict = get_settings(ticker=col, settings_type='features',
                                          command_dict=command_dict,
                                          default_dict=default_dict)


        # Calculate returns = target variables
        df[col] = local_prices[col]
        horizon = -1 * cleaning_spec_dict['horizon']

        if cleaning_spec_dict['return_type'] == 'returns':
            _ret = (lambda x: x.pct_change())
        elif cleaning_spec_dict['return_type'] == 'log returns':
            _ret = (lambda x: x.map(np.log).diff())
        else:
            _ret = (lambda x: x.diff() )
        # Warning- this is Label data (i.e., lookahead if you don't remove)
        df[f'ret_{str(-1 * horizon)}'] = local_prices[col].pipe(_ret).shift(horizon)


        for window_size in range(2, cleaning_spec_dict['sma_max_lag']+1):
            # SMA of levels
            df[f'sma_{window_size}'] = df[col].rolling(window=window_size,
                                                       min_periods=1).mean().ffill().bfill()

        for window_size in range(2, cleaning_spec_dict['sma_ret_max_lag'] + 1):
            # SMA of ret doesn't make sense to have SMA(1)
            df[f'sma_ret_{window_size}'] = df[col].pipe(_ret).rolling(window=window_size,
                                                                     min_periods=1).mean().ffill().bfill()

        for window_size in range(1, cleaning_spec_dict['ewma_max_span'] + 1):
            # EWMA of levels
            df[f'ewma_{window_size}'] = df[col].ewm(span=window_size).mean().ffill().bfill()

        for window_size in range(1, cleaning_spec_dict['ewma_ret_max_span'] + 1):
            # EWMA of levels
            df[f'ewma_ret_{window_size}'] = df[col].pipe(_ret).ewm(span=window_size).mean().ffill().bfill()

        # Rolling Standard Deviation
        # df[f'{col}_rolling_std_{window_size}'] = df[f'{col}_returns'].rolling(window=window_size).std()

        for lag in range(1, cleaning_spec_dict['ar_max_lag']):
            # Autoregression features
            df[f'ar_{lag}'] = df[col].shift(lag).ffill().bfill() # save the first few obs

        for lag in range(1, cleaning_spec_dict['ar_ret_max_lag']):
            df[f'ar_ret_{lag}'] = df[col].pipe(_ret).shift(lag).ffill().bfill()

        df = df.rename(columns = {col: 'price'})
        df_dict[col] = df
    df_tot = pd.concat(df_dict, axis=1) # multi-index columns!
    df_tot = df_tot.sort_index(axis=1)

    return df_tot



def get_technicals(df, window_sizes):
    technicals_df = pd.DataFrame()
    technicals_df.index = df.index
    close_prices = df['GBPUSD_PX_MID']
    high_prices = df['GBPUSD_PX_HIGH']
    low_prices = df['GBPUSD_PX_LOW']

    for i in window_sizes:
        colname = 'RSI_' + str(i)
        technicals_df[colname] = RSIIndicator(close=close_prices, window=i).rsi()

    for i in window_sizes:
        colname = 'oscillator_' + str(i)
        technicals_df[colname] = 100 * ((close_prices - low_prices.rolling(window=i).min()) / (
                    high_prices.rolling(window=i).min() - low_prices.rolling(window=i).min()))

    for i in window_sizes:
        colname = 'adx_' + str(i)
        adxI = ADXIndicator(high=high_prices, low=low_prices, close=close_prices, window=i)
        technicals_df[colname] = adxI.adx()
    return technicals_df



def __main__():
    df_ARMA = get_AR_MA_features(df,AR_MA_columns, window_sizes)
    df_technicals = get_technicals(df,window_sizes)
    df_feature = pd.merge(df_technicals, df_ARMA, left_index=True, right_index=True)

    eco_index_cols = [col for col in df.columns if '_ACTUAL_RELEASE' in col]
    prefixes = [col.split('_ACTUAL_RELEASE')[0] for col in eco_index_cols]
    macro_columns = [col for col in df.columns if any(s in col for s in prefixes)]

    df_macro = pd.DataFrame()
    df_macro[macro_columns] = df[macro_columns]

    df_surprise = calc_surprise(df[macro_columns])

    df_feature = pd.merge(df_feature, df_macro, left_index=True, right_index=True)
    df_feature = pd.merge(df_feature, df_surprise, left_index=True, right_index=True)

    df.rename(columns={'GBPUSD_PX_MID': 'close'}, inplace= True)

    df_feature['close'] = df['close']

    df_feature['spread_close'] = df['GBPUSD_SPREAD']

    df_feature[feature_columns] = df[feature_columns]

    df_feature.replace([np.inf, -np.inf], np.nan, inplace=True)

    df_feature = df_feature.ffill().bfill()

    df_feature.to_parquet('~/Dropbox/FX/features_2.pqt')