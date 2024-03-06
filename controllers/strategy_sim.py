#!/usr/bin/env python

# # Optimal Signal Combination Experiments (Equities)


# import scipy
# import seaborn as sns
import sys

# from allocate_simulate.backtest_routines import simulate_backtest

# import cov cleaning methods
# from cca_lib.cvc import

# Import the necessaries libraries
# import plotly.offline as pyo
import pandas as pd
import numpy as np
import datetime
import random
import warnings
import json
import statsmodels.api as sm
import regex as re

from database.database_tools import append_and_save_forecasts, read_initialize_forecast_storage

warnings.filterwarnings('ignore')


HOME_DIRECTORY = '~/PycharmProjects/abo_fx/'
sys.path.append(HOME_DIRECTORY)

# from performance.performance import *

from performance.analysis import plot_lines
# from features.features import signal_combo, reversion
import utils.settings as settings
# from performance.performance import build_perf_anal_table, build_pos_anal_table
#

from forecasts.AdaptiveBenignOverfitting import GaussianRFF
from forecasts.forecast_utils import (normalize_data, process_initial_bag,
                                      process_updated_bag, sample_features)
from allocators.trade_allocator import TradeAllocator

# from backtest_master import return_args, return_output_dir



random.seed(12)


def filter_multi(df, index_level, regex, axis=0):
    def f(x):
        return matcher.search(str(x)) is not None

    matcher = re.compile(regex)
    values = df.axes[axis].get_level_values(index_level).map(f)
    return df.loc(axis=axis)[values]


def main():

    with open('../utils/data_clean_settings.json') as params_file:
        control_dict = json.load(params_file)


    # Show multiple outputs
    # plt.style.use('seaborn-ticks')
    # plt.rcParams['font.size'] = 14
    # plt.rcParams['axes.titlesize'] = 14
    # plt.rcParams["figure.figsize"] = [6.4,4.8]
    # plt.rc("figure", figsize=(30,10))
    # plt.rc("font", size=25)
    # plt.rc("lines", linewidth=1)
    # plt.rcParams['mathtext.fontset'] = 'custom'
    switch = 'combo'

    corr_dict = {}
    forex_price_features = pd.read_parquet(settings.FEATURE_DIR + 'features_280224_curncy_spot.pqt')
    forex_price_features = forex_price_features.sort_index(axis=1, level=0)
    forex_price_features = forex_price_features.sort_index(axis=0)
    # I trust nothing! Nothing!

    default_dict= {
        'RFF': {'tests': False, 'no_rff': 3000, 'sigma': 1},
        'ABO': {'forgetting_factor': 1, 'l': 0, 'roll_size': 120},
        'Bagged_ABO': {'n_bags': 1, 'feature_num': 3000}
        # missing optimizer params - risk-aversion,
    }

    crosses = ['GBPUSD']  #, 'CADGBP', 'AUDCAD', 'GBPJPY', 'CADUSD', 'JPYUSD', "SEKNZD"]  # audcad gbpjpy

    (forex_forecast_storage, overlap) = read_initialize_forecast_storage(forex_price_features, crosses)


    # now run strategies!
    # Refresh - start again with strategies


    # TODO: How to not run if you've run it before?
    crosses_copy = crosses.copy()

    for cross in crosses_copy:
        specific_dict = control_dict.get(cross, default_dict.copy())
        # make sure the dict has all entries
        specific_full_dict = default_dict.copy()
        specific_full_dict.update(specific_dict)

        (meta_data, _, _, _, _, _, _, _, _) = extract_params(specific_full_dict)

        forecast_column = (cross,
                           'forecast',
                           meta_data['no_rff'],
                           meta_data['forgetting_factor'],
                           meta_data['roll_size'])

        actual_column = (cross, f'ret_{meta_data["horizon"]}', '', '', '')
        relevant_columns = [x for x in forex_forecast_storage.columns if x[0] == cross]
        relevant_data = forex_forecast_storage.loc[:, relevant_columns]
        results_df = pd.concat([relevant_data.loc[:, actual_column].rename('actual'),
                                relevant_data.loc[:, forecast_column].rename('mean')], axis=1)

        # results_df =
        # TODO: Vol scaling and stacking
        # now just sign

        roll_win = 100
        betas = pd.DataFrame(np.ones((len(results_df), 1)), index=results_df.index, columns=['betas'])
        roll_beta_vol = pd.DataFrame(np.ones((len(results_df), 1)), index=results_df.index, columns=['betas'])
        for t in range(results_df.shape[0] - 1):
            start_samp = max(0, t - roll_win)
            subsamp = results_df.iloc[start_samp:t, :].dropna()
            if len(subsamp) < 10:
                pass
            else:
                simple_stacking_model = sm.OLS(subsamp[['mean']], subsamp[['actual']])
                simple_stacking_results = simple_stacking_model.fit()
                betas.iloc[t + 1, 0] = simple_stacking_results.params[0]

        beta_forecasts = betas['betas'] * results_df['mean']
        # beta has from 0:t in it as does mean, both to forecast t+1

        for t in range(results_df.shape[0] - 1):
            start_samp = max(0, t - roll_win)
            subsamp = results_df.iloc[start_samp:t, :].dropna()
            if len(subsamp) < 10:
                pass
            else:
                roll_beta_vol.iloc[t + 1, 0] = beta_forecasts.iloc[start_samp + 1:t + 1].std()  # a series
        vol_scale = subsamp['actual'].std().shift() / roll_beta_vol
        vol_betas = betas * vol_scale

        costs = (forex_price_features.loc[:, cross].loc[:, ['spread']] / 2).rename(columns={'spread': 'tcosts'})
        alpha_orig = results_df[['mean']].rename(columns={'mean': 'alpha'})
        alpha = pd.DataFrame(alpha_orig['alpha'] * vol_betas['betas'], columns=['alpha'])
        vol = results_df[['actual']].shift().rolling(30).var().rename(columns={'actual': 'risk'})
        # todo: vol should be a feature inthe featurestore
        target = results_df[['actual']].rename(columns={'actual': 'realized_gain'})
        # TODO: create a vol features

        signals = pd.concat([alpha, vol, costs, target], axis=1)

        multiplier_dict = {'alpha_multiplier': 1,
                           'risk_aversion': 0.007,
                           'funding_multiplier': 1.0,
                           'gross_limit': 1E+6,
                           'scale_limits': 1.0}

        ta = TradeAllocator(init_allocation=0.0, business_days=list(results_df.index),
                            multiplier_dict=multiplier_dict)
        ta.block_update(signals, dataframe_update=True)
        returns = ta.returns_frame
        returns.to_csv(settings.OUTPUT_REPORTS + f'returns_{cross}_{meta_data["roll_size"]}.csv')

        meta_data.update(multiplier_dict)
        pnl = returns[['total_pnl']]  # should want to append to storage


        plot_lines(pnl, None, column_names='',
                   value_names='cum_pnl', filename_prefix=f'net_pnl_{cross}_{roll_size}')

        print(f'Finished PnL for {cross}_{roll_size}')

    print('Finished PnL')

    # save our results and append them to the storage

    #     # Parallel(n_jobs=-1)(delayed(process_updated_bag)
    #     for p in
    #     range(0, n_bags))
    # new obeservation will be appended to train set in the next iteration


    # record results
    # Parallel execution of the second loop


    # backward shift since actual is fwd looking
    # fill in first 2 days

    #     results_df = pd.merge(results_df, (results_df['actual'].
    #                                        shift().ewm(span=30).
    #                                        std().bfill().rename('risk')),
    #                           left_index=True, right_index=True, how='left')
    #     results_df = pd.merge(results_df, df['spread_close'].div(2).rename('tcosts'),
    #                           left_index=True, right_index=True, how='left')
    #
    #     df_perf, p = fx_backtest(10000, results_df, df, hold_enabled=hold_enabled,
    #                              n=pctile_roll_size, p=pctile_trigger)
    #
    #     store_results(df_perf, no_rff, forgetting_factor, roll_size, n_bags, feature_num, p)
    #
    # pass
