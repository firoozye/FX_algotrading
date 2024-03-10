import pandas as pd
import numpy as np
from utils import settings as settings
import regex as re

STORAGE_FILE = 'All_forecasts.pqt'



#TODO: Duh, create a class


class ForecastStorage(object):
    def __init__(self, forex_price_features:pd.DataFrame, crosses:list):
        self.storage = None
        # now see if we added more crosses, and if so tack them on
        all_subcols = [x for x in forex_price_features.columns if x[0] in crosses]
        all_labels = filter_multi(forex_price_features[all_subcols], index_level=1, regex='^ret', axis=1)
        all_labels.index = pd.to_datetime(all_labels.index)
        all_labels.index.name = 'date'
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 1, 1))
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 3, 1))
        all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 2, 1))
        all_labels.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'roll_size']
        all_labels = all_labels.unstack().reset_index()  # flatten it
        # consistent column schema
        all_labels = all_labels.rename(columns=
                                       {'level_5': 'date',
                                        0: 'val'}).applymap(
            lambda x:
            x if not (x == '') else np.nan
        )
        all_labels['date'] = pd.to_datetime(all_labels['date'])  # try again!

        try:
            forex_forecast_storage = pd.read_parquet(settings.OUTPUT_FILES + STORAGE_FILE)

            forex_forecast_storage = pd.concat([forex_forecast_storage, all_labels], axis=0)
            forex_forecast_storage = forex_forecast_storage.drop_duplicates()

            # read in what we have done so far
        except FileNotFoundError or AttributeError:
            # it's not there or we f**ed it up so have to restart!
            # Start by adding just the returns data

            # add column metadata AUDCAD | ret_1 | labels   so we can add more metadata to other columns
            all_labels.to_parquet(settings.OUTPUT_FILES + STORAGE_FILE)
            forex_forecast_storage = all_labels.copy()
            # overlap = []
        self.storage = forex_forecast_storage
        self.initiated_crosses = crosses
        listed_runs = list_stored_forecasts()
        self.crosses = listed_runs['cross']

    def unique_runs(self):
        pass
    def list_stored_forecasts(self):
        listed_runs = {x: list(self.storage[x].unique())
                       for x in self.storage.columns
                       if x not in ['date', 'val']}
        return listed_runs


def filter_multi(df, index_level, regex, axis=0):
    def f(x):
        return matcher.search(str(x)) is not None

    matcher = re.compile(regex)
    values = df.axes[axis].get_level_values(index_level).map(f)
    return df.loc(axis=axis)[values]


def append_and_save_forecasts(forex_forecast_storage, results_df, cross, meta_data, test=True):
    append_series = pd.DataFrame(results_df['mean'])
    append_series.columns = pd.MultiIndex.from_tuples([(cross,
                                                        'forecast',
                                                        meta_data['no_rff'],
                                                        meta_data['forgetting_factor'],
                                                        meta_data['roll_size'])])
    append_series.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'roll_size']

    append_series = append_series.unstack().reset_index()  # flatten it
    # consistent column schema
    append_series = append_series.rename(columns=
                                   {'level_5': 'date',
                                    0: 'val'}).applymap(
        lambda x:
        x if not (x == '') else np.nan
    )
    append_series['date'] = pd.to_datetime(append_series['date'])  # try again!


    forex_forecast_storage = pd.concat([forex_forecast_storage, append_series], axis=0)
    forex_forecast_storage = forex_forecast_storage.drop_duplicates()
    # forex_forecast_storage = pd.merge(forex_forecast_storage, append_series, left_index=True, right_index=True,
    #                                   how='left')
    # forex_forecast_storage = remove_duplicate_columns(forex_forecast_storage)

    # Try not to fuck up the parquet file
    forex_forecast_storage.to_parquet(settings.OUTPUT_FILES + STORAGE_FILE)
    print('Appended results to Storage Parquet')
    if test:
        try:
            test_read = pd.read_parquet(settings.OUTPUT_FILES + STORAGE_FILE)
        except:
            forex_forecast_storage.to_csv(settings.OUTPUT_FILES + 'forex_storage_backup.csv')
            print('Could not read parquet file. Wrote Backup CSV just in case')
    return forex_forecast_storage


def read_initialize_forecast_storage(forex_price_features, crosses):
    # now see if we added more crosses, and if so tack them on
    all_subcols = [x for x in forex_price_features.columns if x[0] in crosses]
    all_labels = filter_multi(forex_price_features[all_subcols], index_level=1, regex='^ret', axis=1)
    all_labels.index = pd.to_datetime(all_labels.index)
    all_labels.index.name = 'date'
    all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 1, 1))
    all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 3, 1))
    all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 2, 1))
    all_labels.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'roll_size']
    all_labels = all_labels.unstack().reset_index()  # flatten it
    # consistent column schema
    all_labels = all_labels.rename(columns=
                                   {'level_5': 'date',
                                    0: 'val'}).applymap(
        lambda x:
        x if not (x == '') else np.nan
    )
    all_labels['date'] = pd.to_datetime(all_labels['date'])  # try again!

    try:
        forex_forecast_storage = pd.read_parquet(settings.OUTPUT_FILES + STORAGE_FILE)

        forex_forecast_storage = pd.concat([forex_forecast_storage,all_labels], axis=0)
        forex_forecast_storage = forex_forecast_storage.drop_duplicates()
        # overlap = list({x for x in forex_forecast_storage.columns if x in all_labels.columns})
        # overlap matches the entire metadata.
        # forex_forecast_storage = pd.merge(forex_forecast_storage, all_labels.drop(columns=overlap),
        #                                   how='left', left_index=True,
        #                                   right_index=True)

        # forex_forecast_storage = remove_duplicate_columns(forex_forecast_storage)

        # read in what we have done so far
    except FileNotFoundError or AttributeError:
        # it's not there or we f**ed it up so have to restart!
        # Start by adding just the returns data

        # add column metadata AUDCAD | ret_1 | labels   so we can add more metadata to other columns
        all_labels.to_parquet(settings.OUTPUT_FILES + STORAGE_FILE)
        forex_forecast_storage = all_labels.copy()
        # overlap = []
    return forex_forecast_storage


def revert_multi_column(forex_forecast_storage):
    forex_forecast_storage = forex_forecast_storage.set_index([
        'cross',
        'type',
        'no_rff',
        'forgetting_factor',
        'roll_size',
        'date'
    ])
    # create multiindex columns
    forex_forecast_storage = forex_forecast_storage.unstack(
        0).unstack(0).unstack(0).unstack(0).unstack(0)
    forex_forecast_storage.columns = forex_forecast_storage.columns.droplevel(0)
    # made our data 'balanced' by exploding size and increasing nan columns
    # now remove them
    not_all_missing = forex_forecast_storage.isna().all(axis=0).map(lambda x: not(x))
    forex_forecast_storage = forex_forecast_storage.loc[:,not_all_missing]
    # final shape = Obs x (number of forecasts saved + 1 actual data)
    return forex_forecast_storage

def flatten_forecasts_for_storage(forex_forecast_storage):
    forex_forecast_storage = forex_forecast_storage.unstack().reset_index()
    val_header = forex_forecast_storage['level_0'].unique()[0]
    forex_forecast_storage = forex_forecast_storage.drop(columns=['level_0'])
    # the column name 'val' got stuck over there
    forex_forecast_storage = forex_forecast_storage.rename(columns={0:val_header})
    # reverts except (meaningless) index is now reset and unique
    # it is now also balanced (same number of date obs for each)
    return forex_forecast_storage


def list_stored_forecasts(forex_forecast_storage):
    listed_runs = {x: list(forex_forecast_storage[x].unique())
                   for x in forex_forecast_storage.columns
                   if x not in ['date', 'val']}
    return listed_runs


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



'''
Changed the PQT since it wasn't readable with a multiindex.
Now it is 

Current Schema

  "type" : "record",
  "name" : "schema",
  "fields" : [ {
    "name" : "cross",
    "type" : [ "null", "string" ],
    "default" : null
  }, {
    "name" : "type",
    "type" : [ "null", "string" ],
    "default" : null
  }, {
    "name" : "no_rff",
    "type" : [ "null", "double" ],
    "default" : null
  }, {
    "name" : "forgetting_factor",
    "type" : [ "null", "double" ],
    "default" : null
  }, {
    "name" : "roll_size",
    "type" : [ "null", "double" ],
    "default" : null
  }, {
    "name" : "date",
    "type" : [ "null", {
      "type" : "long",
      "logicalType" : "timestamp-micros"
    } ],
    "default" : null
  }, {
    "name" : "val",
    "type" : [ "null", "double" ],
    "default" : null
  }, {
    "name" : "__index_level_0__",
    "type" : [ "null", "long" ],
    "default" : null
  } ]
}

'''