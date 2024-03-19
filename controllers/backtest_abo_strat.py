#!/usr/bin/env python

# # Optimal Signal Combination Experiments (Equities)


# import scipy
# import seaborn as sns
import sys

# from allocate_simulate.backtest_routines import simulate_backtest

# import cov cleaning methods
# from cca_lib.cvc import




import pandas as pd
import numpy as np
import datetime
import random
import warnings
import json
import itertools
from database.database_tools import append_and_save_forecasts, read_initialize_forecast_storage
from utils.utilities import extract_params

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

# from backtest_master import return_args, return_output_dir



random.seed(12)


def main():

    run_all_iterations()


    default_dict = {
        'RFF': {'tests': False, 'no_rff': 3000, 'sigma': 1},
        'ABO': {'forgetting_factor': 1.0, 'l': 0, 'roll_size': 60},
        'Bagged_ABO': {'n_bags': 1, 'feature_num': 3000}
    # missing optimizer params - risk-aversion,
    }
    # run_forecasts_settings(default_dict=default_dict)

def run_all_iterations():
    sigma_list = [2,2.5,3]   #[1,1.25,0.8,1.5,0.60] # should run 1.5 on 60,80,100,120. Running 0.75 on 30
    rollsize_list = [100,30,60,10,80,125,250]
    no_rff_list = [3000] #,5000]  # did loads of 1000
    forgetting_factor_list = [1, 0.95]  #[0.7, 0.8,0.9,0.95,0.98,1] #0.99 seemed to give more high correl forecasts
    forgetting_factor_list.sort(reverse=True)
    # rollsize_list.sort(reverse=True)
    # sigma_list.sort(reverse=True)
    all_iters = [x for x in itertools.product(sigma_list, no_rff_list, rollsize_list, forgetting_factor_list)]

    iter = 0
    for combo in all_iters:
        sigma, no_rff,rollsize, forgetting_factor = combo



        default_dict = {
            'RFF': {'tests': False, 'no_rff': no_rff, 'sigma': sigma},
            'ABO': {'forgetting_factor': forgetting_factor, 'l': 0, 'roll_size': rollsize},
            'Bagged_ABO': {'n_bags': 1, 'feature_num': no_rff}
            # missing optimizer params - risk-aversion,
        }
        if forgetting_factor**rollsize <0.01:
            continue  # skip it, decays too quickly
        # elif rollsize_list == 250 and forgetting_factor != 1:
        #     continue # only do
        elif iter> 20:
            continue # done too much!
        else:
            iter = iter+1
            print(f'ANOTHER RUN: sigma, rff, rollsize, ff = {combo}')
            run_forecasts_settings(default_dict=default_dict)


def run_forecasts_settings(default_dict: dict|None=None):
    with open('./utils/data_clean_settings.json') as params_file:
        control_dict = json.load(params_file)
    #TODO: Give json a full pathname! (not relative)
    switch = 'combo'

    corr_dict = {}
    forex_price_features = pd.read_parquet(settings.FEATURE_DIR + 'features_280224_curncy_spot.pqt')
    forex_price_features = forex_price_features.sort_index(axis=1, level=0)
    forex_price_features = forex_price_features.sort_index(axis=0)
    # I trust nothing! Nothing!

    if default_dict is None:
        default_dict= {
            'RFF': {'tests': False, 'no_rff': 3000, 'sigma': 1.0},
            'ABO': {'forgetting_factor': 1, 'l': 0, 'roll_size': 120},
            'Bagged_ABO': {'n_bags': 1, 'feature_num': 3000}
            # missing optimizer params - risk-aversion,
        }

    crosses = ['CADGBP', 'AUDCAD', 'GBPJPY', 'CADUSD', 'JPYUSD', "SEKNZD", 'GBPUSD']  # audcad gbpjpy 'GBPUSD'


    crosses_copy = crosses.copy()
    # if len(overlap) > 0:
    #     # overlap is based on all metadata. If overlap then remove that cross
    #     overlap_crosses = [x[0] for x in overlap if not ('ret' in x[1])]
    #
    #     _ = [crosses_copy.remove(x) for x in overlap_crosses]

    for cross in crosses_copy:
        print(f'Starting run for {cross}')

        subcols = [x for x in forex_price_features.columns if x[0] == cross ]
        # pick our cross, and drop all na's
        features = forex_price_features[subcols].dropna(axis=0)
        # spec_data = spec_data.rename(columns = {x: x[1] for x in spec_data.columns})
        features.columns = features.columns.droplevel(0)


        specific_dict = control_dict.get(cross, default_dict.copy())
        # make sure the dict has all entries
        specific_full_dict = default_dict.copy()
        specific_full_dict.update(specific_dict)

        (meta_data,
         feature_num,
         forgetting_factor,
         l,
         n_bags,
         no_rff,
         roll_size,
         sigma,
         tests
         ) = extract_params(specific_full_dict)

        (results_df, meta_data) = bagged_abo_forecast(features, specific_full_dict)

        append_and_save_forecasts(forex_forecast_storage=None, results_df=results_df,
                                                           cross=cross, meta_data=meta_data)
        # dot read it - just append to it
        # results_df =
        # TODO: Vol scaling and stacking
        # now just sign
        # corr = results_df.corr().iloc[0,-1]


        results_df.to_csv(settings.OUTPUT_FILES +
                          f'results_{cross}_{meta_data["roll_size"]}_{meta_data["no_rff"]}_{meta_data["sigma"]}.csv')

        # # basic reporting
        # corr_dict[cross]={}
        # for yr in range(2010,2024):
        #     range_dates = [x for x in results_df.index if x>=datetime.date(yr,1,1)
        #                    and x< datetime.date(yr+1,1,1)]
        #     if len(range_dates)>0:
        #         corr = results_df.loc[range_dates].corr().iloc[0,1]
        #     else:
        #         corr = np.nan
        #     corr_dict[cross][yr] = corr
        # corr_dict[cross]['all']= results_df.corr().iloc[0,1]
        #
        # print(f'Final Fit {cross}  of {results_df.corr().iloc[0,1]:.2f}')
        #


        # results_prod = results_df['mean'] * results_df['actual']
        # results_prod = results_prod.dropna()
        # roll_size = meta_data["roll_size"]
        # cum_results = pd.DataFrame((results_prod + 1).cumprod(),columns=['cum_pnl'])
        #
        # plot_lines(cum_results,None, column_names= '',
        #            value_names='cum_pnl',filename_prefix=f'pnl_{cross}_{roll_size}')
        #
        # print(f'Forecasts for {cross} with roll_sz ={roll_size}')
        print(f'finished forecasts for {cross} for {[(x,y) for (x,y) in meta_data.items()]}')

    # pd.DataFrame(corr_dict).to_csv(settings.OUTPUT_REPORTS + f'corrs_for_{roll_size}.csv')
    print('End forecasts')
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


def bagged_abo_forecast(features, specific_full_dict):

    (meta_data,
     feature_num,
     forgetting_factor,
     l,
     n_bags,
     no_rff,
     roll_size,
     sigma,
     tests
     ) = extract_params(specific_full_dict)

    labels = features.filter(regex='^ret')
    features = features.drop(columns=list(labels.columns)+['price','spread'])
    time_steps = labels.shape[0]
    # prep the rff
    features_dim = features.shape[1]
    rff = GaussianRFF(features_dim=features_dim, no_rff=no_rff, kernel_var=sigma)
    # prep the bag_dict
    # Sampling features in each bag
    results_df = pd.DataFrame(np.nan * np.ones((time_steps, 2)),
                              columns=['mean', 'actual'],
                              index=labels.index)
    features_bag_index = sample_features(no_rff, n_bags, feature_num)
    # subset the data
    ind = 0
    labels_roll = labels.iloc[: ind + roll_size, :]
    features_roll = features.iloc[:ind + roll_size, :]
    # Expanding window for normalization
    features_final = features.iloc[[ind + roll_size + 1], :]
    # scale the data
    (
        labels_norm,
        features_norm,
        features_final_norm,
        labels_scaler,
        features_scaler
    ) = normalize_data(labels_roll, features_roll, features_final)
    # perform RFF transformation

    features_rff = rff.transform(features_norm)
    features_rff = features_rff[ind:, :]
    # only now turn into a rolling sample
    features_final_rff = rff.transform(features_final_norm)
    all_bags_preds = None
    bag_dict = {}
    bag_dict_final = {}
    model_dict = {}
    for bagno in range(n_bags):
        bag_dict[bagno] = features_rff[:, features_bag_index[bagno]]
        bag_dict_final[bagno] = features_final_rff[:, features_bag_index[bagno]]

        # Parallel execution of the first loop. Model initialization
        # Parallel(n_jobs=-1)(delayed(yada  for bag_no in tqdm(range(0, n_bags))
        (pred_abo, mod_abo) = process_initial_bag(bag_no=bagno,
                                                  bag_dict=bag_dict,
                                                  bag_dict_final=bag_dict_final,
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
    results_df['mean'].iloc[ind + roll_size + 1] = np.mean(all_bags_preds)
    results_df['actual'].iloc[ind + roll_size + 1] = labels.iloc[roll_size + 1]
    # iterate until len(df_future)-1 max
    for ind in range(1, time_steps - roll_size - 1):

        # removed tqdm
        labels_roll = labels.iloc[: ind + roll_size, :]
        features_roll = features.iloc[:ind + roll_size, :]
        features_final = features.iloc[[ind + roll_size + 1], :]

        (
            labels_norm,
            features_norm,
            features_final_norm,
            labels_scaler,
            features_scaler
        ) = normalize_data(labels_roll, features_roll, features_final)

        features_rff = rff.transform(features_norm)
        update_features_rff = features_rff[[-1], :]
        update_labels = labels_norm[[-1]]
        features_final_rff = rff.transform(features_final_norm)

        update_bag_dict = {}
        update_bag_dict_final = {}

        all_bags_preds = None
        for bagno in range(n_bags):
            update_bag_dict[bagno] = update_features_rff[:, features_bag_index[bagno]]
            update_bag_dict_final[bagno] = features_final_rff[:, features_bag_index[bagno]]

            (pred_abo, _) = process_updated_bag(bag_no=bagno,
                                                update_bag_dict=update_bag_dict,
                                                update_bag_dict_final=update_bag_dict_final,
                                                update_label=update_labels,
                                                mod_ABO=model_dict[bagno],
                                                )

            rescaled_pred_abo = labels_scaler.inverse_transform(pred_abo)
            if all_bags_preds is None:
                all_bags_preds = rescaled_pred_abo
            else:
                all_bags_preds = np.r_[all_bags_preds, rescaled_pred_abo]
        # print(ind)
        results_df['mean'].iloc[ind + roll_size + 1] = np.mean(all_bags_preds)
        results_df['actual'].iloc[ind + roll_size + 1] = labels.iloc[ind + roll_size + 1]
        reporting_iter = 1000   # just don't print

        if (ind % reporting_iter == 0) & (ind > 0):
            mean_pred = results_df['mean'].iloc[ind - reporting_iter + 1: ind]
            actuals = results_df['actual'].iloc[ind - reporting_iter + 1: ind]
            percentage_same_sign = (((mean_pred > 0) & (actuals > 0)).mean()
                                    + ((mean_pred < 0) & (actuals < 0)).mean())
            running_correl = np.corrcoef(mean_pred, actuals)[0, 1]
            print(f"Rolling accuracy on iteration {ind}: {percentage_same_sign * 100:.2f}%"
                  f" Correl: {running_correl * 100 :.2f}%")
            # cumulative is sensitive to nans
            sub_results = results_df.dropna(axis=0)
            mean_pred = sub_results['mean'].iloc[: ind]
            actuals = sub_results['actual'].iloc[: ind]
            percentage_same_sign = (((mean_pred > 0) & (actuals > 0)).mean()
                                    + ((mean_pred < 0) & (actuals < 0)).mean())
            running_correl = np.corrcoef(mean_pred, actuals)[0, 1]
            print(f"Cumulative accuracy on iteration {ind}: {percentage_same_sign * 100:.2f}%"
                  f" Correl: {running_correl * 100 :.2f}%")



    return results_df, meta_data


if __name__ == '__main__':
    main()