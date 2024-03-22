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
import random
import warnings
import json
import regex as re

from database.database_tools import read_initialize_forecast_storage, revert_multi_column, \
    select_relevant_cols

from utils.utilities import extract_params
warnings.filterwarnings('ignore')
import statsmodels.api as sm

# HOME_DIRECTORY = '~/PycharmProjects/abo_fx/'
# sys.path.append(HOME_DIRECTORY)

# from performance.performance import *

from performance.analysis import plot_lines
from reporting.performance_report import PerformanceReport
# from features.features import signal_combo, reversion
import utils.settings as settings
# from performance.performance import build_perf_anal_table, build_pos_anal_table
#

from allocators.trade_allocator import TradeAllocator

# from backtest_master import return_args, return_output_dir



random.seed(12)


def filter_multi(df, index_level, regex, axis=0):
    def f(x):
        return matcher.search(str(x)) is not None

    matcher = re.compile(regex)
    values = df.axes[axis].get_level_values(index_level).map(f)
    return df.loc(axis=axis)[values]

def remove_collinear(corr_mat, cut_off=0.98, verbose=False):
    # eliminate collinear first
    '''
    Remove any feature with correl>cut_off to another feature.
    Order matters (if you carre)
    '''
    N = corr_mat.shape[1]
    super_corr = ((corr_mat - np.eye(N)) >= cut_off)
    # focus on upper tri
    regressors = list(super_corr.index)
    remaining_regressors = regressors.copy()
    # alter the copy - don't mess with regressors - no guarantee enumerate works in loop
    for index_loc, reg in enumerate(regressors):
        poss = super_corr.loc[reg]
        # index_loc = np.where(poss.index == reg)[0][0]
        if index_loc < N:
            eliminate_poss = np.where(poss.iloc[index_loc + 1:])[0]
            if len(eliminate_poss) > 0:
                # must eliminate it from our list
                if verbose:
                    print(f'to remove {len(eliminate_poss)} indices')
                indices_to_be_removed = [poss.index[index_loc + 1 + x] for x in eliminate_poss]
                # all upper triangular indices with corr=1
                indices_to_be_removed = [x for x in indices_to_be_removed if x in remaining_regressors]
                # make sure not already removed
                print(indices_to_be_removed)
                _ = [remaining_regressors.remove(x) for x in indices_to_be_removed]
    return remaining_regressors


def main():

    with open('./utils/data_clean_settings.json') as params_file:
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


    forex_forecast_storage = read_initialize_forecast_storage(forex_price_features, crosses, drop_duplicates=True)
    original_storage_shape = forex_forecast_storage.shape
    print(f'Total unchanged storage size {original_storage_shape}')
    cat_values = {x: forex_forecast_storage[x].unique() for x in
                  ['cross', 'type', 'no_rff', 'forgetting_factor', 'sigma', 'roll_size']}
    # print(f'Category values {cat_values}')
    forex_forecast_storage = revert_multi_column(forex_forecast_storage)


    stored_curr = list({x[0] for x in forex_forecast_storage.columns})

    report_dataframe = pd.DataFrame()
    for cur in stored_curr:
        print(cur)
        actual_forward_returns, base_forecast = select_relevant_cols(forex_forecast_storage, cur, horizon=1)

        corr_mat = base_forecast.corr()
        regressors = remove_collinear(corr_mat)
        sub_corr = corr_mat.loc[regressors,regressors]
        base_features = base_forecast.loc[:,regressors]
        X = base_features.ffill().bfill()
        X.columns = [x for x in range(X.shape[1])]

        Y = actual_forward_returns.ffill()

        sum_data = {}
        for i in range(X.shape[1]):
            strat = Y*X.iloc[:,i]
            strat_sr = strat.dropna().mean()/strat.dropna().std()*np.sqrt(252)
            # print(f'{i} full_strat SR = {strat_sr}')
            stdev_sign = strat.rolling(60).mean().std()
            strat_sign = strat.rolling(60).mean().map(lambda x: x if  np.abs(x)>0.01*stdev_sign else 0).map(np.sign)
            new_strat = Y * X.iloc[:,i] * strat_sign.shift()
            new_strat_sr = new_strat.dropna().mean() / new_strat.dropna().std()*np.sqrt(252)
            # print(f'{i} dyn_strat SR = {new_strat_sr}')
            mse = (Y-X.iloc[:,i]).pow(2).mean()
            sum_data[regressors[i]] = pd.Series({'sr':strat_sr,'new_strat_sr':new_strat_sr,'mse':mse})

        report_dataframe = pd.concat([report_dataframe, pd.concat(sum_data, axis=0)], axis=0)

    report_dataframe.to_csv(settings.OUTPUT_REPORTS + 'Analyze_all_runs.csv')

        # Y = Y.map(lambda x: 0 if np.abs(x-Y.mean()) < Y.std() else x).map(np.sign)
        # .map(np.sign))
        # Y.columns = ['returns']
        # model = sm.OLS(Y,X)
        # results = model.fit()
        # print(results.rsquared)   #summary())


    #
    # sub_corr = corr_mat.loc[regressors, regressors]
    #                 # for i in range(N):
    #
    #
    #
    #
    # possible_column = super_corr.columns[i]
    # # for j in range(i+1,N):
    #        # super_corr.iloc[i,i+1:]
    # # forex_forecast_storage.to_csv(settings.OUTPUT_REPORTS + 'all_forecasts.csv')
    # # this was really not necesssary - we can lok at the forecasts
    # new_storage_shape = forex_forecast_storage.shape
    # print(f'Read in storage {original_storage_shape} and reshaped {new_storage_shape}')
    # # now run strategies!
    # # Refresh - start again with strategies
    #
    #
    # # TODO: How to not run if you've run it before?
    # crosses_copy = crosses.copy()
    #
    # cur_returns = {}
    # formatted_report ={}
    # reports = {}
    #
    # for cross in crosses_copy:
    #     specific_dict = control_dict.get(cross, default_dict.copy())
    #     # make sure the dict has all entries
    #     specific_full_dict = default_dict.copy()
    #     specific_full_dict.update(specific_dict)
    #
    #     (meta_data, _, _, _, _, _, _, _, _) = extract_params(specific_full_dict)
    #
    #     forecast_column = (cross,
    #                        'forecast',
    #                        meta_data['no_rff'],
    #                        meta_data['forgetting_factor'],
    #                        meta_data['roll_size'])
    #
    #
    #     actual_column = (cross, f'ret_{meta_data["horizon"]}', np.nan, np.nan ,np.nan)
    #     relevant_columns = [x for x in forex_forecast_storage.columns if x[0] == cross]
    #     relevant_data = forex_forecast_storage.loc[:, relevant_columns]
    #
    #     actual_forward_returns = relevant_data.loc[:, actual_column].rename('actual')
    #     base_forecasts = relevant_data.drop(columns=[actual_column])
    #
    #
    #
    #     forecast_forward_returns = relevant_data.loc[:, forecast_column].rename('mean')
    #     results_df = pd.concat([actual_forward_returns,
    #                             forecast_forward_returns], axis=1)
    #     results_df.index = pd.to_datetime(results_df.index)
    #     # results_df =
    #     # TODO: Vol scaling and stacking
    #     # now just sign
    #
    #     roll_win = 100
    #     betas = pd.DataFrame(np.ones((len(results_df), 1)), index=results_df.index, columns=['betas'])
    #     roll_beta_vol = pd.DataFrame(np.ones((len(results_df), 1)), index=results_df.index, columns=['betas'])
    #     for t in range(results_df.shape[0] - 1):
    #         start_samp = max(0, t - roll_win)
    #         subsamp = results_df.iloc[start_samp:t, :].dropna()
    #         if len(subsamp) < 10:
    #             pass
    #         else:
    #             simple_stacking_model = sm.OLS(subsamp[['mean']], subsamp[['actual']])
    #             simple_stacking_results = simple_stacking_model.fit()
    #             betas.iloc[t + 1, 0] = simple_stacking_results.params[0]
    #
    #     beta_forecasts = betas['betas'] * results_df['mean']
    #     # beta has from 0:t in it as does mean, both to forecast t+1
    #
    #     for t in range(roll_win, results_df.shape[0] - 1):
    #         start_samp = max(0, t - roll_win)
    #         roll_beta_vol.iloc[t + 1, 0] = beta_forecasts.iloc[start_samp :t ].std()  # a series
    #     vol_scale = (results_df['actual'].shift().rolling(roll_win).std() / roll_beta_vol['betas']).fillna(1.0)
    #     # remove final entry just in case lookahead
    #     vol_betas = betas['betas'] * vol_scale
    #
    #     alpha_orig = results_df[['mean']].rename(columns={'mean': 'alpha'})
    #     alpha = pd.DataFrame(alpha_orig['alpha'] * vol_betas, columns=['alpha'])
    #     # percentage costs
    #     costs = pd.DataFrame(
    #         (forex_price_features.loc[:, cross].loc[:, 'spread'] /
    #          (2 * results_df['actual'].shift())
    #          ),
    #         columns=['tcosts'])
    #     vol = results_df[['actual']].shift().rolling(30).var().rename(columns={'actual': 'risk'})
    #     # todo: vol should be a feature inthe featurestore
    #     target = results_df[['actual']].rename(columns={'actual': 'realized_gain'})
    #     # TODO: create a vol features
    #
    #     signals = pd.concat([alpha, vol, costs, target], axis=1)
    #
    #
    #     ta = TradeAllocator(init_allocation=0.0, business_days=list(results_df.index),
    #                         multiplier_dict=multiplier_dict)
    #     ta.block_update(signals, dataframe_update=True)
    #     returns = ta.returns_frame
    #     print(f'Percentage of positive vol_betas {(vol_betas>0).mean()}, negative {(vol_betas<0).mean()},'
    #           f'Over 1 in magnitude {(vol_betas.abs()>1).mean()}')
    #
    #     print(f'Fraction of time not trading - {(returns["realized_usd_tcosts"]==0).mean()}')
    #     returns.to_csv(settings.OUTPUT_REPORTS + f'returns_{cross}_{meta_data["roll_size"]}.csv')
    #
    #     meta_data.update(multiplier_dict)
    #     pnl = returns[['total_pnl']]  # should want to append to storage
    #
    #
    #     plot_lines(pnl, None, column_names='',
    #                value_names='cum_pnl', filename_prefix=f'net_pnl_{cross}_{meta_data["roll_size"]}')
    #
    #     print(f'Finished PnL for {cross}_{meta_data["roll_size"]}')
    #
    #     cur_returns[cross] = returns.copy()
    #     reports[cross] = PerformanceReport()
    #     reports[cross].run_analysis(returns_series=returns['realized_pnl_gain'])
    #     reports[cross].run_report()
    #     reports[cross].format_csv()
    #     report_meta = meta_data.copy()
    #     report_meta.update(reports[cross].formatted_report_dict)
    #     formatted_report[cross] = pd.DataFrame(index=report_meta.keys(),data=report_meta.values(), columns=[cross]).T
    #
    # formatted_combo_report = pd.concat(formatted_report, axis=0)
    # formatted_combo_report.to_csv(settings.OUTPUT_REPORTS + f'Combo_output_report_{meta_data["roll_size"]}_.csv')
    # print('Finished PnL')
    #
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
