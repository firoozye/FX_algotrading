import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator

# features = pd.read_excel('~/Dropbox/FX/GBPUSD_df_daily.xlsx')
# features.set_index('Date',inplace = True)

df = pd.read_parquet('~/Dropbox/FX/GBPUSD_df_daily.pqt')
df['GBPUSD_SPREAD'] = df['GBPUSD_PX_ASK'] - df['GBPUSD_PX_BID']

AR_MA_columns = ['SPX_PX_MID','UKX_PX_MID','GBPUSD_PX_MID',
                 'GBPUSD_PX_LOW','GBPUSD_PX_HIGH',
                 'GBPUSD_FRD_1W','GBPUSD_FRD_1M','GBPUSD_FRD_3M']

feature_columns = ['GBPUSD_SPREAD','GBPUSD_BASIS_1W','GBPUSD_BASIS_1M',
           'GBPUSD_BASIS_3M','USD_OIS_1W','USD_OIS_1M','USD_OIS_3M',
           'GBP_OIS_1W','GBP_OIS_1M','GBP_OIS_3M',
           'USD_BOND_3M','GBP_BOND_1Y','GBPUSD_VOLA_1W',
           'GBPUSD_VOLA_1M','GBPUSD_VOLA_3M','GBPUSD_SKEW_1W',
           'GBPUSD_SKEW_1M','GBPUSD_SKEW_3M','GBPUSD_KURT_1W',
           'GBPUSD_KURT_1M','GBPUSD_KURT_3M']

### shifting SP and US bonds 1 day due to time differences
df['SPX_PX_MID'] = df['SPX_PX_MID'].shift(1)
USD_features = [x for x in df.columns if 'USD' in x and 'GBP' not in x]
df[USD_features]=df[USD_features].shift(1)

window_sizes = [5,10,30]


def get_AR_MA_features(df_old, columns, window_sizes):
    df = pd.DataFrame()
    df.index = df_old.index
    """
    Adds financial features to the DataFrame for specified columns and multiple window sizes.

    Parameters:
    features (DataFrame): The original DataFrame.
    columns (list): List of column names to calculate features for.
    window_sizes (list): List of window sizes for calculating SMA,  and rolling std.

    Returns:
    DataFrame: The original DataFrame with added financial features.
    """
    for col in columns:
        # Calculate returns
        df[col] = df_old[col]
        df[f'{col}_returns'] = df_old[col].pct_change()
        for window_size in window_sizes:
            # SMA of returns
            df[f'{col}_sma_{window_size}'] = df[f'{col}'].rolling(window=window_size).mean()

            df[f'{col}_sma_ret_{window_size}'] = df[f'{col}_returns'].rolling(window=window_size).mean()

            # Rolling Standard Deviation
            df[f'{col}_rolling_std_{window_size}'] = df[f'{col}_returns'].rolling(window=window_size).std()

            # Mean Reversion = Current Value - SMA
            df[f'{col}_mean_rev_{window_size}'] = df[f'{col}'] - df[f'{col}_sma_{window_size}']
            df[f'{col}_mean_rev_ret_{window_size}'] = df[f'{col}_returns'] - df[f'{col}_sma_ret_{window_size}']

        for lag in [1, 2, 3]:
            # Autoregression features
            df[f'{col}_AR_{lag}'] = df[f'{col}'].shift(lag)
            df[f'{col}_AR_ret_{lag}'] = df[f'{col}_returns'].shift(lag)

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

        df[surp_col] = (df[act_col] - df[est_col]) / np.where(df[std_col] == 0, (df[high_col] - df[low_col]) / 4,
                                                              df[std_col])
        df[surp_col].replace([np.inf, -np.inf], 0, inplace=True)
        df[surp_col].fillna(0, inplace=True)
    return df


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