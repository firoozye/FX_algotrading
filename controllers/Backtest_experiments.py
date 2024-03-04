#!/usr/bin/env python
# coding: utf-8
from joblib import Parallel, delayed
import random
import warnings

warnings.filterwarnings('ignore')

from forecasts.AdaptiveBenignOverfitting import GaussianRFF
from allocators.backtesting_utils import fx_backtest, store_results
from forecasts.forecast_utils import (prepare_data, process_initial_bag,
                                      process_updated_bag, sample_features)
import pandas as pd
import numpy as np
from tqdm import tqdm
from backtest_master import return_args, return_output_dir
random.seed(12)
# hourly data
# features = pd.read_parquet('~/Dropbox/FX/df_ohlc_all_features.pqt')

# daily
df = pd.read_parquet('~/Dropbox/FX/Daily_features_1.pqt')
df['spread_close'] = df['GBPUSD_SPREAD']
# In[6]:
hyperparams = return_args()
print('Parsed Hyperparameter Json file')
output_path = return_output_dir(unch=False)
cutoff=hyperparams['backtest']['cutoff']
df_past = df.iloc[:cutoff, :]
df_future = df.iloc[cutoff:, :]



# RFF params
tests = hyperparams['RFF']['tests']  # test ABO every 20 points
D = hyperparams['RFF']['D']
sigma = hyperparams['RFF']['sigma']

# ABO params
ff = hyperparams['ABO']['ff']  # untested for ff<1 in new version
l = hyperparams['ABO']['l']  # unused regularisation
roll_size = hyperparams['ABO']['roll_size']

# Bagged ABO params
n_bags = hyperparams['Bagged_ABO']['n_bags']
feature_num = hyperparams['Bagged_ABO']['feature_num']

# strategy params
pctile_trigger = hyperparams['strategy']['pctile_trigger']
hold_enabled = hyperparams['strategy']['hold_enabled']
pctile_roll_size = hyperparams['strategy']['pctile_roll_size']

# backtest stuff
final_iter =  hyperparams['backtest']['final_iter'] #len(df_future)-1
reporting_iter = hyperparams['backtest']['reporting_iter']
# how long to update progress

# In[ ]:
last_index_df_past = df_past.index[-1:]
indices_df_future = df_future.index[:-1]
combined_indices = last_index_df_past.append(indices_df_future)
results_df = pd.DataFrame(index=combined_indices)
results_df['actual'] = np.nan
results_df['mean'] = np.nan

# Initialize the neccesary lists
models = []
bags = []
# Select the most recent data from the available dataframe
df_model = df_past[-(roll_size + 1):]  # size is roll_size + 1, because we need 1 more point to make prediction
# for that point we don't know the target variable yet

# calculate targets and scale the data
Y, X, scaler_Y, scaler_X = prepare_data(df_model)

# perform RFF transformation
feature_dim = X.shape[0]
rff = GaussianRFF(feature_dim, D, sigma)
X_trans = rff.transform(X.reshape(feature_dim, roll_size + 1)).T

# Sampling features in each bag
features_array = sample_features(D, n_bags, feature_num)

for p in range(n_bags):
    bags.append(X_trans[:, features_array[p]])

# Parallel execution of the first loop. Model initialization
results = Parallel(n_jobs=-1)(delayed(process_initial_bag)(p,
                                                           bags,
                                                           Y,
                                                           scaler_Y,
                                                           ff,
                                                           l,
                                                           feature_num,
                                                           roll_size,
                                                           tests=tests)
                              for p in tqdm(range(0, n_bags)))
all_bags_preds = np.array([result[0] for result in results])
models = [result[1] for result in results]

# Add results in a results dataframe for comparison
results_df['actual'].iloc[0] = df_future['close'][0] / df_past['close'][-1] - 1  # actual target
results_df['mean'].iloc[0] = np.mean(all_bags_preds)

# Continue performing forecasts by updating QR_RLS model
df_temp = df_model

# we need the last row of RFF dataset to append it to train set on next iteration
X_old = X_trans[-1, :].T

# iterate until len(df_future)-1 max

for i in tqdm(range(0, min(final_iter, len(df_future) - 1))):

    # Delete old data and append data, that just became available
    df_temp = df_temp.iloc[1:]
    # axis=0 messes up col names
    df_temp = pd.concat([df_temp.T, df_future.iloc[i]], axis=1).T
    Y, X, scaler_Y, scaler_X = prepare_data(df_temp)
    X_new = rff.transform(X[:, -1:].reshape(feature_dim, 1))
    # Parallel execution of the second loop
    all_bags_preds = Parallel(n_jobs=-1)(
        delayed(process_updated_bag)(p, X_old, X_new, models, scaler_Y, Y, features_array, feature_num) for p in
        range(0, n_bags))
    # new obeservation will be appended to train set in the next iteration
    X_old = X_new

    # record results
    results_df['actual'].iloc[i + 1] = df_future['close'][i + 1] / df_temp['close'][-1] - 1  # actual target
    results_df['mean'].iloc[i + 1] = np.mean(all_bags_preds)

    if (i % reporting_iter == 0) & (i > 0):
        mean_pred = results_df.loc[:results_df.index[i], 'mean']
        actuals = results_df.loc[:results_df.index[i], 'actual']
        percentage_same_sign = (((mean_pred > 0) & (actuals > 0)).mean()
                                + ((mean_pred < 0) & (actuals < 0)).mean())
        running_correl = np.corrcoef(mean_pred, actuals)[0, 1]
        print(f"Accuracy on iteration {i}: {percentage_same_sign * 100:.2f}%"
              f" Correl: {running_correl * 100 :.2f}%")

# backward shift since actual is fwd looking
# fill in first 2 days

results_df = pd.merge(results_df, (results_df['actual'].
                                   shift().ewm(span=30).
                                   std().bfill().rename('risk')),
                      left_index=True, right_index=True, how='left')
results_df = pd.merge(results_df, df['spread_close'].div(2).rename('tcosts'),
                      left_index=True, right_index=True, how='left')

df_perf, p = fx_backtest(10000, results_df, df, hold_enabled=hold_enabled,
                         n=pctile_roll_size, p=pctile_trigger)

store_results(df_perf, D, ff, roll_size, n_bags, feature_num, p)





