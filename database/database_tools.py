import pandas as pd
import numpy as np
from utils import settings as settings
import regex as re
from typing import List, Dict

feat_set = 'price'

STORAGE_FILE = 'All_forecasts_' + feat_set + '.pqt'


#TODO: Duh, create a class


class ForecastStorage(object):
    def __init__(self,
                 forex_price_features:pd.DataFrame,
                 crosses: List[str] | None = None,
                 storage_file=STORAGE_FILE):
        self.storage_file = storage_file

        self.forex_price_feaures = forex_price_features
        # self.storage = forex_forecast_storage
        self.initiated_crosses = crosses
        listed_runs = list_stored_forecasts()
        self.crosses = listed_runs['cross']

    def initialize_forecast_storage(self):
        # now see if we added more crosses, and if so tack them on need an append function
        all_labels = prepare_ret_data(
            self.forex_price_features,
            self.crosses)
        all_labels.to_parquet(settings.OUTPUT_FILES +
                              self.storage_file,
                              engine='fastparquet')
        return all_labels

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


def append_and_save_forecasts(forex_forecast_storage:pd.DataFrame|None =None,
                              results_df:pd.DataFrame|None =None,
                              cross:str='',
                              meta_data:dict={},
                              storage_file:str=STORAGE_FILE,
                              test:bool=False):
    append_series = pd.DataFrame(results_df['mean'])
    append_series.columns = pd.MultiIndex.from_tuples([(cross,
                                                        'forecast',
                                                        meta_data['no_rff'],
                                                        meta_data['forgetting_factor'],
                                                        meta_data['sigma'],
                                                        meta_data['roll_size'])])
    append_series.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'sigma','roll_size']

    append_series = append_series.unstack().reset_index()  # flatten it
    # consistent column schema
    append_series = append_series.rename(columns=
                                   {'level_6': 'date',
                                    0: 'val'}).applymap(
        lambda x:
        x if not (x == '') else np.nan
    )
    append_series.index.name = 'index'  # consistent with saved data
    append_series['date'] = pd.to_datetime(append_series['date'])  # try again!
    append_series['date'] = append_series['date'].map(lambda x: x.isoformat())
    append_series['date'] = append_series['date'].astype('str') # needed?
    append_series = append_series.astype({'cross':'object',
                                          'type':'object',
                                          'no_rff':'float64',
                                          'forgetting_factor':'float64',
                                          'roll_size':'float64',
                                          'sigma':'float64',
                                          'date': 'object',
                                          'val': 'float64'
                                          })
    append_series.index.name = 'index'
    if forex_forecast_storage is not None:
        forex_forecast_storage = pd.concat([forex_forecast_storage, append_series], axis=0)
        forex_forecast_storage = forex_forecast_storage.drop_duplicates(keep='first')
        forex_forecast_storage.to_parquet(settings.OUTPUT_FILES + storage_file,
                                          engine='fastparquet') #, append=False)
    else:
        try:
            append_series.to_parquet(settings.OUTPUT_FILES + storage_file,
                                     engine='fastparquet', append=True)
        except OSError:  # parquet format f*ed up
            stored_data = pd.read_parquet(settings.OUTPUT_FILES + storage_file,
                                          engine='fastparquet')
            stored_data = pd.concat([stored_data, append_series], axis=0)
            stored_data.to_parquet(settings.OUTPUT_FILES + storage_files,
                                   engine='fastparquet')

    # forex_forecast_storage = pd.merge(forex_forecast_storage, append_series, left_index=True, right_index=True,
    #                                   how='left')
    # forex_forecast_storage = remove_duplicate_columns(forex_forecast_storage)

    # Try not to fuck up the parquet file
    print('Appended results to Storage Parquet')
    if test:
        try:
            test_read = pd.read_parquet(settings.OUTPUT_FILES + storage_file,
                                        engine='fastparquet')
        except:
            forex_forecast_storage.to_csv(settings.OUTPUT_FILES + 'forex_storage_backup.csv')
            print('Could not read parquet file. Wrote Backup CSV just in case')
    return forex_forecast_storage

def fix_corrupted_data():
    total = pd.read_parquet(settings.OUTPUT_FILES + 'All_Forecasts_nosigma.pqt')
    total['sigma']=np.nan
    total.loc[:,'date'] = total['date'].map(lambda x: x.isoformat())
    total['date'] = total['date'].astype('str')
    total.to_parquet(settings.OUTPUT_FILES +'All_forecasts.pqt', engine='fastparquet')
    print('restored from earlier backup and nans')

def prepare_ret_data(forex_price_features:pd.DataFrame,
                     crosses:List[str]|None=None):
    if crosses is None:
        all_subcols =[x for x in forex_price_features.columns]
    else:
        all_subcols = [x for x in forex_price_features.columns if x[0] in crosses]

    all_labels = filter_multi(forex_price_features[all_subcols], index_level=1, regex='^ret', axis=1)
    all_labels.index = pd.to_datetime(all_labels.index)
    all_labels.index.name = 'date'
    all_labels.index = all_labels.index.map(lambda x: x.isoformat())
    # type string or object for storage
    # all_labels.loc[:, 'date'] = all_labels['date'].map(lambda x: x.isoformat())
    # all_labels['date'] = all_labels['date'].astype('str')  # force it
    all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 1, 1))
    all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 3, 1))
    all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 2, 1))
    all_labels = (pd.concat([all_labels], axis=1, keys=['']).swaplevel(0, 1, 1).
                  swaplevel(1,2,1))
    all_labels.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'sigma', 'roll_size']
    # TODO: Missing 'simga'!!!
    all_labels = all_labels.unstack().reset_index()  # flatten it
    # consistent column schema
    all_labels = all_labels.rename(columns=
                                   {0: 'val'}).applymap(
        lambda x:
        x if not (x == '') else np.nan
    )
    # all_labels['date'] = pd.to_datetime(all_labels['date'])  # try again!
    all_labels = all_labels.reset_index(drop=True) # make numbering nice!
    all_labels.index.name = 'index'   # being consistent
    return all_labels


def initialize_forecast_storage(forex_price_features:pd.DataFrame,
                                crosses:List[str]|None =None,
                                storage_file=STORAGE_FILE):

    # now see if we added more crosses, and if so tack them on need an append function
    all_labels = prepare_ret_data(forex_price_features, crosses)
    all_labels.to_parquet(settings.OUTPUT_FILES + storage_file, engine='fastparquet')
    return all_labels



def read_initialize_forecast_storage(forex_price_features=None,
                                     crosses: List[str]|None=None,
                                     drop_duplicates:bool=False,
                                     storage_file:str = STORAGE_FILE):

    try:
        forex_forecast_storage = pd.read_parquet(settings.OUTPUT_FILES + storage_file,
                                                 engine='fastparquet')

        # read in what we have done so far
    except FileNotFoundError or AttributeError:
        # it's not there or we f**ed it up so have to restart!
        if forex_price_features is not None:
            forex_forecast_storage = initialize_forecast_storage(forex_price_features, crosses,
                                                                 storage_file=storage_file)
        else:
            raise('File Not Found, and Initializing data not supplied')

    if drop_duplicates:
        total_duplicate_categories = forex_forecast_storage.duplicated(['cross','type','no_rff',
                                                              'forgetting_factor','sigma','roll_size','date']).sum()
        total_duplicate_values =  forex_forecast_storage.duplicated().sum()
        print(f'Keeping first of each of {total_duplicate_categories}. Note there are {total_duplicate_values}'
              f' duplicate values')
        forex_forecast_storage = forex_forecast_storage.drop_duplicates(subset=[
            'cross','type','no_rff','forgetting_factor','sigma','roll_size','date'
        ], keep='first')
    return forex_forecast_storage


def revert_multi_column(forex_forecast_storage):
    forex_forecast_storage = forex_forecast_storage.set_index([
        'cross',
        'type',
        'no_rff',
        'forgetting_factor',
        'sigma',
        'roll_size',
        'date'
    ])
    # create multiindex columns
    forex_forecast_storage = forex_forecast_storage.unstack(
        0).unstack(0).unstack(0).unstack(0).unstack(0).unstack(0)
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

Current Schema / before we put sigma in it

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
    "name" : "index",
    "type" : [ "null", "long" ],
    "default" : null
  } ]
}

'''


def select_relevant_cols(forex_forecast_storage, cross, horizon=1):
    actual_column = (cross, f'ret_{horizon}', np.nan, np.nan, np.nan, np.nan)
    relevant_columns = [x for x in forex_forecast_storage.columns if x[0] == cross]
    relevant_data = forex_forecast_storage.loc[:, relevant_columns]
    actual_forward_returns = relevant_data.loc[:, actual_column].rename('actual')
    base_forecasts = relevant_data.drop(columns=[actual_column])
    return actual_forward_returns, base_forecasts


def main():
    # show that it works for one version
    forex_price_features = pd.read_parquet(settings.FEATURE_DIR +
                                           'features_280224_macd_curncy_spot.pqt',
                                           engine='pyarrow')
    forex_price_features = forex_price_features.sort_index(axis=1, level=0)
    forex_price_features = forex_price_features.sort_index(axis=0)

    forex_storage = initialize_forecast_storage(forex_price_features=forex_price_features, crosses=['GBPUSD'],
                                                storage_file='test.pqt')

    append_series = prepare_ret_data(forex_price_features, crosses=['JPYUSD'])
    append_series.to_parquet(settings.OUTPUT_FILES + 'test.pqt',
                             engine='fastparquet', append=True)

    test_again = pd.read_parquet(settings.OUTPUT_FILES + 'test.pqt',
                                 engine='fastparquet')


    # append_series.columns = pd.MultiIndex.from_tuples([(cross,
    #                                                     'forecast',
    #                                                     meta_data['no_rff'],
    #                                                     meta_data['forgetting_factor'],
    #                                                     meta_data['sigma'],
    #                                                     meta_data['roll_size'])])
    # append_series.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'sigma','roll_size']
    #
    # append_series = append_series.unstack().reset_index()  # flatten it
    # # consistent column schema
    # append_series = append_series.rename(columns=
    #                                {'level_6': 'date',
    #                                 0: 'val'}).applymap(
    #     lambda x:
    #     x if not (x == '') else np.nan
    # )
    # append_series.index.name = 'index'  # consistent with saved data
    # append_series.columns = pd.MultiIndex.from_tuples([('JPYUSD',
    #                                                     'ret_1',
    #                                                     np.nan,
    #                                                     np.nan,
    #                                                     np.nan,
    #                                                     np.nan)])
    # append_series.columns.names = ['cross', 'type', 'no_rff', 'forgetting_factor', 'sigma','roll_size']
    # append_series['date'] = pd.to_datetime(append_series['date'])  # try again!
    # append_series['date'] = append_series['date'].map(lambda x: x.isoformat())
    # append_series['date'] = append_series['date'].astype('str')
    #, append=False)
    # if forex_forecast_storage is not None:
    #     forex_forecast_storage = pd.concat([forex_forecast_storage, append_series], axis=0)
    #     forex_forecast_storage = forex_forecast_storage.drop_duplicates(keep='first')
    # else:
    #     append_series.to_parquet(settings.OUTPUT_FILES + STORAGE_FILE, engine='fastparquet', append=True)



if __name__=='__main__':
    main()