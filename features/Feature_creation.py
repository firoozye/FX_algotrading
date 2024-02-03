#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings

import pandas as pd

from Feature_controllers_daily import get_AR_MA_features, calc_surprise, get_technicals

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import random
from joblib import Parallel, delayed
# import xgboost as xgb


# In[3]:


from allocators.backtesting_utils import *
from features.feature_module import *


# In[4]:


random.seed(12)


# In[5]:


# df = pd.read_excel('~/Dropbox/FX/GBPUSD_df_daily.xlsx')
# df.set_index('Date',inplace = True)
# df = pd.read_parquet('~/Dropbox/FX/GBPUSD_daily_data.pqt')


# df_midas = pd.read_excel('~/Dropbox/FX/GBPUSD_MIDAS.xlsx')
df_midas.set_index('Date', inplace=True)
df = df_midas
df['GBPUSD_SPREAD'] = df['GBPUSD_PX_ASK'] - df['GBPUSD_PX_BID']
# In[6]:
# In[7]:
# In[8]:
# In[9]:

columns = ['SPX_PX_MID','UKX_PX_MID','GBPUSD_PX_LAST',
           'GBPUSD_PX_LOW','GBPUSD_PX_HIGH',
           'GBPUSD_SPREAD','GBPUSD_BASIS_1W','GBPUSD_BASIS_1M',
           'GBPUSD_FRD_1W','GBPUSD_FRD_1M',
           'USD_BOND_3M', 'USD_BOND_1Y', 'GBP_BOND_1Y', 'GBP_BOND_2Y']  # no GBP_BOND_10Y !
# In[10]:
window_sizes = [5,10,30] # for MA terms
lag_lengths = [1, 2, 3] # for AR terms
# In[11]:
df_MA = get_AR_MA_features(df, columns, window_sizes, ar=False)
df_AR = get_AR_MA_features(df, columns, lag_lengths, ar=True)
# In[12]:
overlap = [x for x in df_MA.columns if x in df_AR.columns]
df_ARMA = pd.merge(df_AR, df_MA.drop(columns=overlap), left_index=True, right_index=True)
df_technicals = get_technicals(df, window_sizes)
# In[13]:
df_feature = pd.merge(df_technicals, df_ARMA, left_index=True, right_index=True)

# should lag USD data
df_levels = df[columns]

df_feature = pd.merge(df_feature, df_levels, left_index=True, right_index=True)
# In[14]:
eco_index_cols = [col for col in df.columns if '_ACTUAL_RELEASE' in col]
prefixes = [col.split('_ACTUAL_RELEASE')[0] for col in eco_index_cols]

# In[15]:
macro_columns = [col for col in df.columns if any(s in col for s in prefixes)]

# In[16]:
df_macro = pd.DataFrame()
df_macro[macro_columns] = df[macro_columns]
# In[17]:
df_macro = calc_surprise(df_macro)
# In[18]:
df_feature = pd.merge(df_feature, df_macro, left_index=True, right_index=True)

# df_feature = pd.merge(df_feature.drop(columns=macro_columns), df_midas[macro_columns],
#                       left_index=True, right_index=True)
# In[19]:
df_feature['close'] = df['GBPUSD_PX_MID']
# In[20]:

deriv_cols = ['GBPUSD_VOLA_1W','GBPUSD_VOLA_1M','GBPUSD_SKEW_1W','GBPUSD_SKEW_1M',
             'GBPUSD_KURT_1W','GBPUSD_KURT_1M']
df_feature[deriv_cols] = df[deriv_cols]
# In[21]:
df_feature.ffill(inplace=True)

# In[22]:

df_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
# In[23]:
df_feature = df_feature.ffill().bfill()
# KEEP IT ALL!
df_feature.dropna(inplace=True)

df_feature.to_parquet('~/Dropbox/FX/GBPUSD_DailyFeatures_all2.pqt')
# In[ ]:
#
#
#
#
#
#
#
# # backtest = df_perf['binary']*df_perf['target']
# # In[ ]:
# # In[ ]:
# # array_2 = backtest.cumsum()
#
# # In[ ]:
#
#
# #array_eps = (df_perf['signal_large']*df_perf['target']).cumsum()
#
#
# # In[ ]:
#
#
# # plt.plot((1+array)*100000, label = 'Expanding window + ff')
# # plt.plot((1+array_2)*100000, label = 'Rolling window')
# # plt.xticks(rotation = 45)
# # plt.legend()
# # plt.show()
# #
# #
# # # In[ ]:
# #
# #
# # plt.plot(df_perf['Close'])
# # plt.xticks(rotation = 45)
# # plt.show()
# #
# #
# # # In[ ]:
# #
# #
# # df_perf
#
#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
# last_index_df_past = df_past.index[-1:]
# indices_df_future = df_future.index[:-1]
# combined_indices = last_index_df_past.append(indices_df_future)
# results_df = pd.DataFrame(index=combined_indices)
# results_df['actual'] = np.nan
# results_df['mean'] = np.nan
#
# # Initialize the neccesary lists
# models = []
# bags = []
# all_bags_array = []
# betas_array = []
# # Select the most recent data from the available dataframe
# df_model = df_past[-(roll_size+1):] # size is roll_size + 1, because we need 1 more point to make prediction
#                                     # for that point we don't know the target variable yet
#
# # calculate targets and scale the data
# Y, X, scaler_Y, scaler_X = prepare_data(df_model)
#
# # perform RFF transformation
# lags = X.shape[0]
# rff = GaussianRFF(lags, D, sigma)
# X_trans = rff.transform(X.reshape(lags, roll_size+1)).T
#
# #Sampling features in each bag
# features_array = sample_features(D,n_bags,feature_num)
#
# for p in range(n_bags):
#     bags.append(X_trans[:,features_array[p]])
#
# # Parallel execution of the first loop. Model initialization
# results = Parallel(n_jobs=-1)(delayed(process_initial_bag)(p, bags, Y, scaler_Y, ff, l, feature_num, roll_size, exp_window) for p in tqdm(range(0, n_bags)))
# all_bags_preds = np.array([result[0] for result in results])
# models = [result[1] for result in results]
# betas = np.array([result[2] for result in results])
# betas_array.append(betas)
# all_bags_array.append(np.array(all_bags_preds).T)
# #Add results in a results dataframe for comparison
# results_df['actual'].iloc[0] = df_future['close'][0]/df_past['close'][-1]-1 #actual target
# results_df['mean'].iloc[0] = np.mean(all_bags_preds)
#
# #Continue performing forecasts by updating QR_RLS model
# df_temp = df_model
#
# # we need the last row of RFF dataset to append it to train set on next iteration
# X_old = X_trans[-1,:].T
#
# for i in tqdm(range(0, 300)):
#
#     #Delete old data and append data, that just became available
#     df_temp = df_temp.iloc[1:]
#     df_temp = df_temp.append(df_future.iloc[i])
#
#     Y, X, scaler_Y, scaler_X = prepare_data(df_temp)
#
#     ## We need to perform RFF expansion on the new observation row. For which we don't have target
#     ## And which will be used for forecasting
#     X_new = rff.transform(X[:, -1:].reshape(lags, 1))
#
#     # Parallel execution of the second loop
#     results = Parallel(n_jobs=-1)(delayed(process_updated_bag)(p, X_old, X_new, models, scaler_Y, Y, features_array, feature_num) for p in range(0, n_bags))
#     all_bags_preds = np.array([result[0] for result in results])
#     betas = [result[1] for result in results]
#     betas_array.append(betas)
#     all_bags_array.append(np.array(all_bags_preds).T)
#     #new obeservation will be appended to train set in the next iteration
#     X_old = X_new
#
#     # record results
#     results_df['actual'].iloc[i+1] = df_future['close'][i+1]/df_temp['close'][-1]-1 #actual target
#     results_df['mean'].iloc[i+1] = np.mean(all_bags_preds)
#
#     ##CHANGE-POINT test here
#
#
#
#     if i % 10 == 0 and i > 0:
#
#         same_sign_count = ((results_df['mean'][:i] > 0) & (results_df['actual'][:i] > 0)).sum() + ((results_df['mean'][:i] < 0) & (results_df['actual'][:i] < 0)).sum()
#
#         # Calculate the percentage
#         total_rows = len(results_df['mean'][:i])
#         percentage_same_sign = (same_sign_count / total_rows) * 100
#
#         print(f"Accuracy on iteration {i}: {percentage_same_sign:.2f}%")
#
#         p = results_df['mean'][:i].corr(results_df['actual'][:i])
#
#
#
#         # Calculate rolling Sharpe ratio
#         if not np.isnan(p):  # Check if p is not NaN
#             rolling_sharpe_ratio = (p / np.sqrt(p**2 + 1)) * np.sqrt(252)
#             print(f"Accuracy on iteration {i}: {percentage_same_sign:.2f}%, Rolling Sharpe Ratio: {rolling_sharpe_ratio:.2f}")
#         else:
#             print(f"Accuracy on iteration {i}: {percentage_same_sign:.2f}%, Rolling Sharpe Ratio: Cannot be calculated (NaN)")
#
