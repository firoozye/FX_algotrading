#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
# HOME_DIRECTORY = '~/PycharmProjects/FX_algostrategy/'
# sys.path.append(HOME_DIRECTORY)

import pandas as pd

from features.forex_cleaning import clean_and_define_columns
# from features.futures_roll_routines import create_continuous_roll
import utils.settings as settings
from features.feature_master import create_all_price_based_features
from performance.analysis import plot_lines
import json

def main():
    check_cleaning = True
    feature_set = 'macd' #'macd'  #rice' # or 'macd' or 'carry'

    if feature_set in ['price', 'macd']:
        json_filename = './utils/data_clean_settings_' + feature_set + '.json'
    elif feature_set == 'carry':
        raise NotImplementedError
    else:
        raise NotImplementedError

    with open(json_filename) as params_file:
        control_dict = json.load(params_file)

    if check_cleaning:
        forex_spot= pd.read_parquet(settings.RAW_DATA_DIR + 'combo_FOREX_280224_curncy_spot.pqt')
        # 'combo_FOREX_280224_curncy_tn.pqt' besides
        forex_spot, cleaning_progress = clean_and_define_columns(forex_spot, control_dict=control_dict)
        cleaning_progress.to_csv(settings.CLEANED_DATA_DIR + 'cleaning_done_per_detector.csv')
        # NOTE: pyarrow allows MultiIndex Columsn, fastpqarquet does not (but allows append!)
        forex_spot.to_parquet(settings.CLEANED_DATA_DIR + 'cleaned_280224_curncy_spot.pqt',
                              engine='pyarrow')
        # save it so we can get started quicker next time
    else:
        forex_spot = pd.read_parquet(settings.CLEANED_DATA_DIR +
                                     'cleaned_280224_curncy.pqt', engine='pyarrow')

    forex_spot.columns = forex_spot.columns.swaplevel(0,1)
    forex_spot_prices = forex_spot.loc[:,'PX_MID']


    forex_spot_spreads = forex_spot.loc[:,'BID_ASK_SPREAD']
    forex_spot_spreads = forex_spot_spreads.ffill().bfill()

    forex_spot_spreads.columns = [(x.split(' ')[0],'spread') for x in forex_spot_spreads.columns]
    forex_price_features = create_all_price_based_features(forex_spot_prices, command_dict=control_dict)
    forex_price_features = pd.concat([forex_price_features, forex_spot_spreads], axis=1)
    output_file = 'features_280224_' + feature_set + '_curncy_spot.pqt'

    forex_price_features.to_parquet(settings.FEATURE_DIR + output_file, engine='pyarrow')

    print(f'Finished features of shape {forex_price_features.shape}')

    test = pd.read_parquet(settings.FEATURE_DIR + output_file, engine='pyarrow')
    if test.equals(forex_price_features):
        print('Saved and verified. Parquet file read correctly.')
    else:
        print(f'Read in new dataset with shape {test.shape}, '
              f'not the same as old with shape {forex_price_features.shape}')
    forex_spot_prices = forex_spot_prices.rename(columns = {x:x.split(' ')[0] for x in forex_spot_prices.columns})
    plot_lines(forex_spot_prices, None,  column_names='', value_names='price',
               filename_prefix='cleaned_forex_')


if __name__ == '__main__':
    main()



