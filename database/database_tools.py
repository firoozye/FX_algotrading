import pandas as pd

from controllers.backtest_abo_strat import filter_multi
from utils import settings as settings
import regex as re

STORAGE_FILE = 'All_forecasts.pqt'

def filter_multi(df, index_level, regex, axis=0):
    def f(x):
        return matcher.search(str(x)) is not None

    matcher = re.compile(regex)
    values = df.axes[axis].get_level_values(index_level).map(f)
    return df.loc(axis=axis)[values]


def append_and_save_forecasts(forex_forecast_storage, results_df, cross, meta_data):
    append_series = pd.DataFrame(results_df['mean'])
    append_series.columns = pd.MultiIndex.from_tuples([(cross,
                                                        'forecast',
                                                        meta_data['no_rff'],
                                                        meta_data['forgetting_factor'],
                                                        meta_data['roll_size'])])
    append_series.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'roll_size']

    forex_forecast_storage = pd.merge(forex_forecast_storage, append_series, left_index=True, right_index=True,
                                      how='left')
    forex_forecast_storage = remove_duplicate_columns(forex_forecast_storage)

    forex_forecast_storage.to_parquet(settings.OUTPUT_FILES + STORAGE_FILE)
    print('Appended results to Storage Parquet')
    return forex_forecast_storage


def read_initialize_forecast_storage(forex_price_features, crosses):
    try:
        forex_forecast_storage = pd.read_parquet(settings.OUTPUT_FILES + STORAGE_FILE)
        all_subcols = [x for x in forex_price_features.columns if x[0] in crosses]
        all_labels = filter_multi(forex_price_features[all_subcols], index_level=1, regex='^ret', axis=1)
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 1, 1))
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 3, 1))
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 2, 1))
        all_labels.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'roll_size']

        forex_forecast_storage = pd.merge(forex_forecast_storage, all_labels, how='left', left_axis=True,
                                          right_axis=True)

        forex_forecast_storage = remove_duplicate_columns(forex_forecast_storage)

        #todo: make them regex "_x$"


        # read in what we have done so far
    except FileNotFoundError:

        all_subcols = [x for x in forex_price_features.columns if x[0] in crosses]
        all_labels = filter_multi(forex_price_features[all_subcols], index_level=1, regex='^ret', axis=1)
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 1, 1))
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 3, 1))
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 2, 1))
        all_labels.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'roll_size']

        # add column metadata AUDCAD | ret_1 | labels   so we can add more metadata to other columns
        all_labels.to_parquet(settings.OUTPUT_FILES + STORAGE_FILE)
        forex_forecast_storage = all_labels.copy()
    return forex_forecast_storage


def remove_duplicate_columns(forex_forecast_storage):
    forex_forecast_storage = forex_forecast_storage.loc[:, forex_forecast_storage.T.duplicated(keep='last')]
    # remove duplication
    # and if there was a merge, we get _x and _y, remove those
    matcher = re.compile("_x$|_y$")
    mismarked = [x for x in forex_forecast_storage.columns if matcher.search(x) is not None]
    remark = {x: x[:-2] for x in mismarked}
    forex_forecast_storage = forex_forecast_storage.rename(columns=remark)
    return forex_forecast_storage
