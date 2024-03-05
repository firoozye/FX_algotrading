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

HOME_DIRECTORY = '~/PycharmProjects/abo_fx/'
sys.path.append(HOME_DIRECTORY)

# from performance.performance import *

from allocate_simulate.strategy import TradingStrategy
from performance.analysis import plot_lines
# from features.features import signal_combo, reversion
import utils.settings as settings
# from performance.performance import build_perf_anal_table, build_pos_anal_table
#
from joblib import Parallel, delayed
import random
import warnings
import json
warnings.filterwarnings('ignore')

from forecasts.AdaptiveBenignOverfitting import GaussianRFF
from allocators.backtesting_utils import fx_backtest, store_results
from forecasts.forecast_utils import (normalize_data, process_initial_bag,
                                      process_updated_bag, sample_features)

from tqdm import tqdm
# from backtest_master import return_args, return_output_dir



random.seed(12)



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
                'ABO': {'forgetting_factor': 1, 'l': 0, 'roll_size': 10},
                'Bagged_ABO': {'n_bags': 1, 'feature_num': 3000}
    # missing optimizer params - risk-aversion,
    }


    crosses = ['GBPUSD', 'GBPCAD','AUDCAD','GBPJPY','CADUSD','JPYUSD', "SEKNZD"] # audcad gbpjpy
    
    for cross in crosses:
        print(f'Starting run for {cross}')
        specific_dict = control_dict.get(cross, default_dict.copy())
        # make sure the dict has all entries
        specific_full_dict = default_dict.copy()
        specific_full_dict.update(specific_dict)

        # RFF params
        tests = specific_full_dict['RFF']['tests']  # test ABO every 20 points
        no_rff = specific_full_dict['RFF']['no_rff']
        sigma = specific_full_dict['RFF']['sigma']

        # ABO params
        forgetting_factor = specific_full_dict['ABO']['forgetting_factor']
        # untested for forgetting_factor<1 in new version
        l = specific_full_dict['ABO']['l']  # unused regularisation
        roll_size = specific_full_dict['ABO']['roll_size']

        # Bagged ABO params
        n_bags = specific_full_dict['Bagged_ABO']['n_bags']
        feature_num = specific_full_dict['Bagged_ABO']['feature_num']

        subcols = [x for x in forex_price_features.columns if x[0] == cross ]

        # pick our cross, and drop all na's

        features = forex_price_features[subcols].dropna(axis=0)

        # spec_data = spec_data.rename(columns = {x: x[1] for x in spec_data.columns})
        features.columns = features.columns.droplevel(0)

        labels = features.filter(regex='^ret')
        features = features.drop(columns=labels.columns)
        time_steps = labels.shape[0]
        # prep the rff
        features_dim = features.shape[1]
        rff = GaussianRFF(features_dim=features_dim, no_rff=no_rff, kernel_var=sigma)

        # prep the bag_dict
        # Sampling features in each bag
        results_df = pd.DataFrame(np.nan * np.ones((time_steps, 2)),
                                  columns=['mean','actual'],
                                  index=labels.index)

        features_bag_index = sample_features(no_rff, n_bags, feature_num)


        # subset the data
        ind = 0
        labels_roll = labels.iloc[ind: ind+roll_size,:]
        features_roll = features.iloc[ind:ind+roll_size,:]
        features_final = features.iloc[[ind+roll_size+1],:]
        # scale the data
        (
            labels_norm,
            features_norm,
            features_final_norm,
            labels_scaler,
            features_scaler
        ) = normalize_data(labels_roll, features_roll, features_final)

        # perform RFF transformation

        # TODO: Get rid of this retarded shaping!
        features_rff = rff.transform(features_norm)
        features_final_rff = rff.transform(features_final_norm)
        all_bags_preds = None
        bag_dict={}
        bag_dict_final ={}
        model_dict={}
        for bagno  in range(n_bags):
            bag_dict[bagno] = features_rff[:, features_bag_index[bagno]]
            bag_dict_final[bagno] = features_final_rff[:, features_bag_index[bagno]]


            # Parallel execution of the first loop. Model initialization
            # Parallel(n_jobs=-1)(delayed(yada  for bag_no in tqdm(range(0, n_bags))
            (pred_abo, mod_abo) = process_initial_bag(bag_no=bagno,
                                          bag_dict=bag_dict,
                                          bag_dict_final = bag_dict_final,
                                          labels=labels_norm,
                                          forgetting_factor=forgetting_factor,
                                          l=l,
                                          feature_num=features_dim,
                                          roll_size=roll_size,
                                          tests=tests)
            rescaled_pred_abo = labels_scaler.inverse_transform(pred_abo)
            if all_bags_preds is None:
                all_bags_preds = rescaled_pred_abo
            else:
                all_bags_preds = np.r_[all_bags_preds, rescaled_pred_abo]
            model_dict[bagno] = mod_abo

        results_df['mean'].iloc[ind+roll_size+1] = np.mean(all_bags_preds)
        results_df['actual'].iloc[ind+roll_size+1] =labels.iloc[roll_size+1]


        # iterate until len(df_future)-1 max

        for ind in range(1,  time_steps - roll_size - 1):

            # removed tqdm
            labels_roll = labels.iloc[ind: ind + roll_size, :]
            features_roll = features.iloc[ind:ind + roll_size, :]
            features_final = features.iloc[[ind + roll_size + 1], :]

            (
                labels_norm,
                features_norm,
                features_final_norm,
                labels_scaler,
                features_scaler
            ) = normalize_data(labels_roll, features_roll, features_final)

            features_rff = rff.transform(features_norm)
            update_features_rff = features_rff[[-1],:]
            update_labels = labels_norm[[-1]]
            features_final_rff = rff.transform(features_final_norm)

            update_bag_dict = {}
            update_bag_dict_final = {}

            all_bags_preds = None
            for bagno in range(n_bags):
                update_bag_dict[bagno] = update_features_rff[:,features_bag_index[bagno]]
                update_bag_dict_final[bagno] = features_final_rff[:,features_bag_index[bagno]]


                (pred_abo, _ ) = process_updated_bag(bag_no=bagno,
                                                     update_bag_dict=update_bag_dict,
                                                     update_bag_dict_final=update_bag_dict_final,
                                                     update_label=update_labels,
                                                     mod_ABO=model_dict[bagno],
                                                     )


                rescaled_pred_abo = labels_scaler.inverse_transform(pred_abo)
                if all_bags_preds is None:
                    all_bags_preds = rescaled_pred_abo
                else:
                    all_bags_preds =np.r_[all_bags_preds, rescaled_pred_abo]
            # print(ind)
            results_df['mean'].iloc[ind + roll_size + 1] = np.mean(all_bags_preds)
            results_df['actual'].iloc[ind + roll_size + 1] = labels.iloc[ind + roll_size + 1]
            reporting_iter = 200

            if (ind % reporting_iter == 0) & (ind > 0):
                mean_pred = results_df['mean'].iloc[ind - reporting_iter+1 : ind]
                actuals = results_df['actual'].iloc[ind - reporting_iter+1 : ind]
                percentage_same_sign = (((mean_pred > 0) & (actuals > 0)).mean()
                                        + ((mean_pred < 0) & (actuals < 0)).mean())
                running_correl = np.corrcoef(mean_pred, actuals)[0, 1]
                print(f"Rolling accuracy on iteration {ind}: {percentage_same_sign * 100:.2f}%"
                      f" Correl: {running_correl * 100 :.2f}%")
                # cumulative is sensitive to nans
                sub_results = results_df.dropna(axis=0)
                mean_pred = sub_results['mean'].iloc[: ind]
                actuals = sub_results['actual'].iloc[ : ind]
                percentage_same_sign = (((mean_pred > 0) & (actuals > 0)).mean()
                                        + ((mean_pred < 0) & (actuals < 0)).mean())
                running_correl = np.corrcoef(mean_pred, actuals)[0, 1]
                print(f"Cumulative accuracy on iteration {ind}: {percentage_same_sign * 100:.2f}%"
                      f" Correl: {running_correl * 100 :.2f}%")

        # basic reporting
        corr_dict[cross]={}
        for yr in range(2010,2024):
            range = [x for x in results_df.index if x>=datetime.date(yr,1,1) and x< datetime.date(yr+1,1,1)]
            if len(range)>0:
                corr = results_df.loc[range].corr().iloc[0,1]
            else:
                corr = np.nan
            corr_dict[cross][yr] = corr
        corr_dict[cross]['all']= results_df.corr().iloc[0,1]

        print(f'Final Fit {cross}  of {results_df.corr().iloc[0,1]:.2f}')
        results_df.to_csv(settings.OUTPUT_FILES + f'results_{cross}_{roll_size}.csv')

        results_prod = results_df['mean'] * results_df['actual']
        results_prod = results_prod.dropna()
        cum_results = pd.DataFrame((results_prod + 1).cumprod(),columns=['cum_pnl'])

        plot_lines(cum_results,None, column_names= '',
                   value_names='cum_pnl',filename_prefix=f'pnl_{cross}_{roll_size}')

    pd.DataFrame(corr_dict).to_csv(settings.OUTPUT_REPORTS + f'corrs_for_{roll_size}.csv')

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