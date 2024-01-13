import pandas as pd
import numpy as np


def get_calendar_info(datetime_series):
    """
    Calendar information retrieval
    Parameters
    ----------
    datetime_series : DateTime pd.Series
        pd.Series in DateTime format
    Returns
    -------
    res_df : pd.DataFrame
        DataFrame with columns: 'year', 'month', 'day', 'day_of_year', 'week_of_year',
        'day_of_week', 'quarter', 'days_in_month', 'is_month_start', 'is_month_end',
        'is_quarter_start', 'is_quarter_end'.
    """
    datetime_series = pd.Series(datetime_series)
    year = np.array(datetime_series.dt.year)
    month = np.array(datetime_series.dt.month)
    day = np.array(datetime_series.dt.day)
    day_of_year = np.array(datetime_series.dt.dayofyear)
    week_of_year = np.array(datetime_series.dt.isocalendar().week)
    day_of_week = np.array(datetime_series.dt.dayofweek)
    quarter = np.array(datetime_series.dt.quarter)
    days_in_month = np.array(datetime_series.dt.days_in_month)
    is_month_start = np.array(datetime_series.dt.is_month_start)
    is_month_end = np.array(datetime_series.dt.is_month_end)
    is_quarter_start = np.array(datetime_series.dt.is_quarter_start)
    is_quarter_end = np.array(datetime_series.dt.is_quarter_end)

    res_df = pd.DataFrame()
    res_df['year'] = year
    res_df['month'] = month
    res_df['day'] = day
    res_df['day_of_year'] = day_of_year
    res_df['week_of_year'] = week_of_year
    res_df['day_of_week'] = day_of_week
    res_df['quarter'] = quarter
    res_df['days_in_month'] = days_in_month
    res_df['is_month_start'] = is_month_start
    res_df['is_month_end'] = is_month_end
    res_df['is_quarter_start'] = is_quarter_start
    res_df['is_quarter_end'] = is_quarter_end

    return res_df
