import pandas as pd


def get_sample_statistics(data, window_length):
    """
     Computes statistics of a time series
     Parameters
     ----------
     data : array_like
         Time series
     window_length : int
        Moving window length for statistics' computation
     Returns
     -------
     stats_list : list [mean, median, variance, std]
         List with statistics of a time series (np.ndarrays)
     """
    mean = pd.Series(data).rolling(window_length).mean()
    median = pd.Series(data).rolling(window_length).median()
    variance = pd.Series(data).rolling(window_length).var()
    std = pd.Series(data).rolling(window_length).std()
    min = pd.Series(data).rolling(window_length).min()
    max = pd.Series(data).rolling(window_length).max()
    dev_mean_min = mean / min
    dev_mean_max = mean / max
    dev_min_max = min / max
    dev_mean_std = mean / std
    diff_mean_median = mean - median
    diff_max_min = max - min
    diff_max_mean = max - min
    diff_min_mean = min - mean
    dev_max_std = max / std
    dev_min_std = min / std

    stats_list = [mean,
                  median,
                  variance,
                  std,
                  min,
                  max,
                  dev_mean_max,
                  dev_mean_min,
                  dev_min_max,
                  dev_mean_std,
                  diff_mean_median,
                  diff_max_min,
                  diff_max_mean,
                  diff_min_mean,
                  dev_max_std,
                  dev_min_std]

    return stats_list
