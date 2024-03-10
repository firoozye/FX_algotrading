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

from database.database_tools import append_and_save_forecasts, read_initialize_forecast_storage, revert_multi_column

from utils.utilities import extract_params
warnings.filterwarnings('ignore')


HOME_DIRECTORY = '~/PycharmProjects/abo_fx/'
sys.path.append(HOME_DIRECTORY)

# from performance.performance import *

from performance.analysis import plot_lines
from reporting.performance_report import PerformanceReport
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
    forex_price_features.index = pd.to_datetime(forex_price_features.index)
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

    multiplier_dict = {'alpha_multiplier': 1,
                       'risk_aversion': 0.01,
                       'funding_multiplier': 1.0,
                       'gross_limit': 1E+7,
                       'scale_limits': 1.0}

    forex_forecast_storage = read_initialize_forecast_storage(forex_price_features, crosses)
    original_storage_shape = forex_forecast_storage.shape
    forex_forecast_storage = revert_multi_column(forex_forecast_storage)

    # forex_forecast_storage.to_csv(settings.OUTPUT_REPORTS + 'all_forecasts.csv')
    # this was really not necesssary - we can lok at the forecasts
    new_storage_shape = forex_forecast_storage.shape
    print(f'Read in storage {original_storage_shape} and reshaped {new_storage_shape}')
    # now run strategies!
    # Refresh - start again with strategies


    # TODO: How to not run if you've run it before?
    crosses_copy = crosses.copy()

    cur_returns = {}
    formatted_report ={}
    reports = {}

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


        actual_column = (cross, f'ret_{meta_data["horizon"]}', np.nan, np.nan ,np.nan)
        relevant_columns = [x for x in forex_forecast_storage.columns if x[0] == cross]
        relevant_data = forex_forecast_storage.loc[:, relevant_columns]

        actual_forward_returns = relevant_data.loc[:, actual_column].rename('actual')
        base_forecasts = relevant_data.drop(columns=[actual_column])



        forecast_forward_returns = relevant_data.loc[:, forecast_column].rename('mean')
        results_df = pd.concat([actual_forward_returns,
                                forecast_forward_returns], axis=1)
        results_df.index = pd.to_datetime(results_df.index)
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

        for t in range(roll_win, results_df.shape[0] - 1):
            start_samp = max(0, t - roll_win)
            roll_beta_vol.iloc[t + 1, 0] = beta_forecasts.iloc[start_samp :t ].std()  # a series
        vol_scale = (results_df['actual'].shift().rolling(roll_win).std() / roll_beta_vol['betas']).fillna(1.0)
        # remove final entry just in case lookahead
        vol_betas = betas['betas'] * vol_scale

        alpha_orig = results_df[['mean']].rename(columns={'mean': 'alpha'})
        alpha = pd.DataFrame(alpha_orig['alpha'] * vol_betas, columns=['alpha'])
        # percentage costs
        costs = pd.DataFrame(
            (forex_price_features.loc[:, cross].loc[:, 'spread'] /
             (2 * results_df['actual'].shift())
             ),
            columns=['tcosts'])
        vol = results_df[['actual']].shift().rolling(30).var().rename(columns={'actual': 'risk'})
        # todo: vol should be a feature inthe featurestore
        target = results_df[['actual']].rename(columns={'actual': 'realized_gain'})
        # TODO: create a vol features

        signals = pd.concat([alpha, vol, costs, target], axis=1)


        ta = TradeAllocator(init_allocation=0.0, business_days=list(results_df.index),
                            multiplier_dict=multiplier_dict)
        ta.block_update(signals, dataframe_update=True)
        returns = ta.returns_frame
        print(f'Percentage of positive vol_betas {(vol_betas>0).mean()}, negative {(vol_betas<0).mean()},'
              f'Over 1 in magnitude {(vol_betas.abs()>1).mean()}')

        print(f'Fraction of time not trading - {(returns["realized_usd_tcosts"]==0).mean()}')
        returns.to_csv(settings.OUTPUT_REPORTS + f'returns_{cross}_{meta_data["roll_size"]}.csv')

        meta_data.update(multiplier_dict)
        pnl = returns[['total_pnl']]  # should want to append to storage


        plot_lines(pnl, None, column_names='',
                   value_names='cum_pnl', filename_prefix=f'net_pnl_{cross}_{meta_data["roll_size"]}')

        print(f'Finished PnL for {cross}_{meta_data["roll_size"]}')

        cur_returns[cross] = returns.copy()
        reports[cross] = PerformanceReport()
        reports[cross].run_analysis(returns_series=returns['realized_pnl_gain'])
        reports[cross].run_report()
        reports[cross].format_csv()
        report_meta = meta_data.copy()
        report_meta.update(reports[cross].formatted_report_dict)
        formatted_report[cross] = pd.DataFrame(index=report_meta.keys(),data=report_meta.values(), columns=[cross]).T

    formatted_combo_report = pd.concat(formatted_report, axis=0)
    formatted_combo_report.to_csv(settings.OUTPUT_REPORTS + f'Combo_output_report_{meta_data["roll_size"]}_.csv')
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


if __name__ == '__main__':
    main()
