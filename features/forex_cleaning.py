import datetime

import numpy as np
import pandas as pd
from utils.utilities import get_settings


def correct_pricing_post_fill(forex_row, price_order='PX_MID'):
    '''
    Define fallback pricing in case of missing data to get PX_MID
    Note we persist BID_ASK_SPREAD
    we fall back on the next the previous is pd.isna()
    1.PX_MIO
    2.PX_ASK - BID_ASK_SPREAD/2
    3.PX_BID + BID_ASK_SPREAD/2
    4.PX_LAST
    We only expect to use PX_LAST at beginning of sample where no bid/ask exists
    PS: This is slow since .apply() by row. Try to replace by DataFrame Op!
    '''
    if price_order =='PX_MID':
        if pd.isna(forex_row['PX_MID']) and not pd.isna(forex_row['PX_ASK']):
            forex_row.loc['PX_MID'] = forex_row['PX_ASK'] - forex_row['BID_ASK_SPREAD']/2
        if pd.isna(forex_row['PX_MID']) and not pd.isna(forex_row['PX_BID']):
            forex_row.loc['PX_MID'] = forex_row['PX_BID'] + forex_row['BID_ASK_SPREAD'] / 2
        if pd.isna(forex_row['PX_MID']):
            forex_row.loc['PX_MID'] = forex_row['PX_LAST']
    else:
        forex_row.loc['PX_MID'] = forex_row['PX_LAST'] #screw it, use last price
    return forex_row

def correct_pricing_pre_fill(forex_data):
    pd.set_option('mode.chained_assignment', None)
    forex_data.loc[:, 'PX_MID'] = forex_data['PX_MID'].fillna((forex_data['PX_BID'] + forex_data['PX_ASK'])/2)
    forex_data.loc[:, 'BID_ASK_SPREAD'] = (forex_data['PX_ASK'] - forex_data['PX_BID'])
    # will groupby ffill this later when operating on dataframe rather than just row
    pd.set_option('mode.chained_assignment', 'warn')
    return forex_data




def clean_pricing_data(forex_data, cleaning_dict):
    '''
    Generic Futures Cleaning routines
    '''

    default_dict =  {
        "start_date": 20100101,
        "zscore_diff_cutoff": 3,
        "zscore_level_cutoff": 3.5,
        "zscore_diff_min_periods": 30,
        "zscore_diff_window": 50,
        "zscore_level_min_periods": 30,
        "zscore_level_window": 100,
        "price_order": "PX_MID"
    }

    bbg_codes = list({x[0] for x in forex_data.columns})
    bbg_codes.sort()
    reassembled_data = dict()
    cleaning_progress_total_dict = dict()
    for ticker in bbg_codes:

        ticker_prefix = ticker.split(' ')[0][:-1] # prefix from genrics, RX1, RX2->RX

        cleaning_spec_dict = get_settings(ticker=ticker_prefix, settings_type='cleaning',
                                          command_dict=cleaning_dict,
                                          default_dict=default_dict)


        pd.set_option('mode.chained_assignment', None)
        temp_all = forex_data.loc[:, ticker]
        # zeroed data
        prices = ['PX_MID', 'PX_ASK', 'PX_BID', 'PX_LAST']
        cleaning_progress_dict = {}
        for px in prices:  # replace zero prices with nans
            if px not in temp_all.columns: # print(f'Warning {tick} is missing {px} pricing column')
                if px =='PX_MID':
                    temp_all.loc[:, px] = temp_all.loc[:,'PX_LAST']  # just give up! it's not there!
                else:
                    temp_all.loc[:, px] = pd.Series([np.nan] * temp_all.shape[0], name=px)

        for px in prices:  # replace zero prices with nans
            temp_all.loc[:, px] = temp_all.loc[:, px].map(lambda x: x if pd.isna(x) or (x > 0) else np.nan)
            # adjust for quotes in PJ and CA contracts (CADEUR and GBPJPY) which were quoted as 15555
            # instead of 155.55
        cleaning_progress_dict['pre-cleaning'] = temp_all[prices].isna().sum()
        # print(f'Pre-cleaning - {ticker_prefix} nans at {temp_all[prices].isna().sum()}')

        for px in prices:  # replace zero prices with nans

            # zscore on diff with mean 0, expanding window
            z_score_diff = (temp_all.loc[:,px].diff().abs()/
                            temp_all.loc[:,px].diff().rolling(window=cleaning_spec_dict['zscore_diff_window'],
                                                              min_periods=cleaning_spec_dict['zscore_diff_min_periods']
                                                              ).std())
            # allow first 50 obs no matter what
            z_score_diff = z_score_diff.apply(lambda x: x if not np.isinf(x) else np.nan)

            temp_all.loc[z_score_diff > cleaning_spec_dict['zscore_diff_cutoff'], px] = np.nan

        cleaning_progress_dict['zscore_diff'] = temp_all[prices].isna().sum()
        # print(f'Zscore diff - {ticker_prefix} nans at {temp_all[prices].isna().sum()}')

        for px in prices:
            # zscore on levels rolling 50 day window
            z_score_level = ((temp_all.loc[:, px] -
                              temp_all.loc[:,px].rolling(window=cleaning_spec_dict['zscore_level_window'],
                                                         min_periods=cleaning_spec_dict['zscore_level_min_periods']
                                                         ).mean()).abs() /
                             temp_all.loc[:, px].rolling(window=cleaning_spec_dict['zscore_level_window'],
                                                            min_periods=cleaning_spec_dict['zscore_level_min_periods']
                                                            ).std())
            # allow first 50 obs no matter what
            z_score_level = z_score_level.apply(lambda x: x if not np.isinf(x) else np.nan)
            temp_all.loc[z_score_level > cleaning_spec_dict['zscore_level_cutoff'], px] = np.nan

        cleaning_progress_dict['zscore_level'] = temp_all[prices].isna().sum()
        # print(f'Zscore diff + level - {ticker_prefix} nans at {temp_all[prices].isna().sum()}')
        pd.set_option('mode.chained_assignment', 'warn')

        cleaning_progress = pd.DataFrame(cleaning_progress_dict).T
        cleaning_progress_total_dict[ticker_prefix] = cleaning_progress.copy()
        # print(f'Cleaning progress {ticker_prefix} - {cleaning_progress}')

        temp_all.loc[:,'PX_LAST'] = temp_all['PX_LAST'].ffill()
        temp_all = correct_pricing_pre_fill(temp_all)
        temp_all.loc[:, 'BID_ASK_SPREAD'] = temp_all['BID_ASK_SPREAD'].ffill()
        #TODO: Replace this super-slow apply by a dataframe op
        post_fill = (lambda x: correct_pricing_post_fill(x, price_order=cleaning_spec_dict.get('price_order',
                                                                                               'PX_MID')))
        temp_all = temp_all.apply(post_fill, axis=1)

        fill_fields = ['PX_MID']
        for x in fill_fields:
            temp_all.loc[:,x] = temp_all[x].ffill()

        cutoff = datetime.date.fromisoformat(str(cleaning_spec_dict.get('start_date',20000101)))
        temp_all = temp_all.loc[cutoff:, :]

        if cleaning_spec_dict.get('delete', False):
            # delete timeseries as spec in jsonfile
            continue # no entry for that ticker
        else:
            reassembled_data[ticker] = temp_all
    forex_data = pd.concat(reassembled_data, axis=1)
    cleaning_progress_total = pd.concat(cleaning_progress_total_dict, axis=0)
    return forex_data, cleaning_progress_total


def col_group_transform(input_data, input_cols, output_cols,
                        func, replace=True, ffill=True):
    '''
    This transformation logic gets reused.
    Take input_data[input_cols] apply func to them and then concatenate
    or replace them
    This crap is nec since we are dealing with multiindex col format
    '''
    temp = input_data.copy()
    temp = temp.loc[:, input_cols].applymap(func)
    if ffill:
        temp = temp.ffill()
    if replace:
        output_cols = input_cols
    temp = pd.concat({output_cols: temp}, axis=1) # prepend multiindex level
    if replace:
        input_data = input_data.drop(columns=[input_cols],level=0)
    # insure there are no duplicate columns
    temp = temp.drop(columns=[x for x in temp.columns if x in
                              input_data.columns])
    # then tack it on
    input_data = pd.concat([input_data, temp], axis=1)
    return input_data


def balance_data(forex_data):
    # if unbalanced data, just create missing columns fill with nans
    minimal_cols = ['PX_BID', 'PX_ASK', 'PX_MID', 'PX_LAST', 'PX_HIGH', 'PX_LOW']
    # OPEN AND CLOSE?
    tickers = list({x[0] for x in forex_data.columns})
    tickers.sort()
    min_col_list = [(x, y) for x in tickers for y in minimal_cols]
    min_col_list.sort()

    for z in min_col_list:
        if z not in forex_data.columns:
            # print(z)
            forex_data.loc[:, z] = np.nan
    return forex_data


def clean_and_define_columns(forex_data, control_dict={}):
    '''
    Balance Data, add and fill date columns if they do not exist and clean data
    '''
    forex_data = forex_data.sort_index(axis=0).sort_index(axis=1, level=0)
    forex_data = balance_data(forex_data)
    forex_data.columns = forex_data.columns.swaplevel(0, 1)
    # do all group level transforms

    # can we do multiple at once?



    # forex_data = col_group_transform(forex_data,
    #                                    input_cols='FUT_CUR_GEN_TICKER',
    #                                    output_cols='LAST_DELIVERY_DATE',
    #                                    func=(lambda x: delivery_date(x)),
    #                                    replace=False)


    forex_data.columns = forex_data.columns.swaplevel(0, 1)

    forex_data = forex_data.sort_index(axis=1, level=0)
    forex_data = forex_data.sort_index(axis=0)
    forex_data, cleaning_progress = clean_pricing_data(forex_data, cleaning_dict=control_dict)
    forex_data = forex_data.sort_index(axis=0)  #WTF is it always unsorting?
    return forex_data, cleaning_progress
