import pandas as pd
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

        # now see if we added more crosses, and if so tack them on
        all_subcols = [x for x in forex_price_features.columns if x[0] in crosses]
        all_labels = filter_multi(forex_price_features[all_subcols], index_level=1, regex='^ret', axis=1)
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 1, 1))
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 3, 1))
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 2, 1))
        all_labels.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'roll_size']

        overlap = list({x for x in forex_forecast_storage.columns if x in all_labels.columns})
        # overlap matches the entire metadata.
        forex_forecast_storage = pd.merge(forex_forecast_storage, all_labels.drop(columns=overlap),
                                          how='left', left_index=True,
                                          right_index=True)

        forex_forecast_storage = remove_duplicate_columns(forex_forecast_storage)

        # read in what we have done so far
    except FileNotFoundError or AttributeError:
        # it's not there or we f**ed it up so have to restart!
        # Start by adding just the returns data
        all_subcols = [x for x in forex_price_features.columns if x[0] in crosses]
        all_labels = filter_multi(forex_price_features[all_subcols], index_level=1, regex='^ret', axis=1)
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 1, 1))
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 3, 1))
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 2, 1))
        all_labels.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'roll_size']

        # add column metadata AUDCAD | ret_1 | labels   so we can add more metadata to other columns
        all_labels.to_parquet(settings.OUTPUT_FILES + STORAGE_FILE)
        forex_forecast_storage = all_labels.copy()
        overlap = []
    return forex_forecast_storage, overlap


def remove_duplicate_columns(forex_forecast_storage):

    forex_forecast_storage = forex_forecast_storage.loc[:, forex_forecast_storage.T.duplicated(keep='last')]
    # remove duplication
    # and if there was a merge, we get _x and _y, remove those
    matcher = re.compile("_x$|_y$")
    mismarked = [x for x in forex_forecast_storage.columns if (matcher.search(str(x[0])) is not None)]
    remark = {x: (x[0][:-2],x[1],x[2],x[3],x[4]) for x in mismarked}
    # must be a better way!!!
    forex_forecast_storage = forex_forecast_storage.rename(columns=remark)
    return forex_forecast_storage
