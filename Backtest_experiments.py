#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing
import time
import os
import random
import multiprocessing
import warnings
warnings.filterwarnings('ignore')
import datetime
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt

import kaleido
import pandas as pd
import numpy as np

from copy import deepcopy
from scipy.linalg import pinv
import random
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from joblib import Parallel, delayed
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate


# In[2]:


from AdaptiveBenignOverfitting import *
from forecast_utils import *
from backtesting_utils import *


# In[3]:


fsize = 15
tsize = 16
tdir = 'in'
major = 9.0
minor = 5.0
lwidth = 0.8
lhandle = 2.0
plt.style.use('default')
#plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300  # for preview, doesn't affect savefig
plt.rcParams['savefig.dpi'] = 800
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['axes.linewidth'] = lwidth
plt.rcParams['legend.handlelength'] = 2


# In[4]:


random.seed(12)


# In[5]:

# hourly data
# df = pd.read_parquet('~/Dropbox/FX/df_ohlc_all_features.pqt')

# daily
df = pd.read_parquet('~/Dropbox/FX/GBPUSD_DailyFeatures_all2.pqt')

# In[6]:


df_past = df.iloc[:500,:]
df_future = df.iloc[500:,:]


# In[9]:


D = 1200
sigma = 1
ff = 0.8
l = 0
roll_size = 200
n_bags = 1
feature_num = D


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
df_model = df_past[-(roll_size+1):] # size is roll_size + 1, because we need 1 more point to make prediction
                                    # for that point we don't know the target variable yet

# calculate targets and scale the data
Y, X, scaler_Y, scaler_X = prepare_data(df_model) 

# perform RFF transformation
feature_dim = X.shape[0]
rff = GaussianRFF(feature_dim, D, sigma)
X_trans = rff.transform(X.reshape(feature_dim, roll_size + 1)).T

#Sampling features in each bag
features_array = sample_features(D,n_bags,feature_num)

for p in range(n_bags):
    bags.append(X_trans[:,features_array[p]])

# Parallel execution of the first loop. Model initialization
results = Parallel(n_jobs=-1)(delayed(process_initial_bag)(p, bags, Y, scaler_Y, ff, l, feature_num, roll_size) for p in tqdm(range(0, n_bags)))
all_bags_preds = np.array([result[0] for result in results])
models = [result[1] for result in results]

#Add results in a results dataframe for comparison
results_df['actual'].iloc[0] = df_future['close'][0]/df_past['close'][-1]-1 #actual target
results_df['mean'].iloc[0] = np.mean(all_bags_preds)

#Continue performing forecasts by updating QR_RLS model
df_temp = df_model

# we need the last row of RFF dataset to append it to train set on next iteration
X_old = X_trans[-1,:].T  

# iterate until len(df_future)-1
for i in tqdm(range(0, 200)):
    
    #Delete old data and append data, that just became available
    df_temp = df_temp.iloc[1:]

    # axis=0 messes up col names
    df_temp = pd.concat([df_temp.T, df_future.iloc[i]], axis=1).T

    Y, X, scaler_Y, scaler_X = prepare_data(df_temp)

    X_new = rff.transform(X[:, -1:].reshape(feature_dim, 1))
    
    # Parallel execution of the second loop
    all_bags_preds = Parallel(n_jobs=-1)(delayed(process_updated_bag)(p, X_old, X_new, models, scaler_Y, Y, features_array, feature_num) for p in range(0, n_bags))
    
    #new obeservation will be appended to train set in the next iteration
    X_old = X_new 
    
    # record results
    results_df['actual'].iloc[i+1] = df_future['close'][i+1]/df_temp['close'][-1]-1 #actual target
    results_df['mean'].iloc[i+1] = np.mean(all_bags_preds)
    
    if i % 10 == 0:

        mean_pred = results_df.loc[:results_df.index[i],'mean']
        actuals = results_df.loc[:results_df.index[i],'actual']
        percentage_same_sign = (((mean_pred > 0) & (actuals > 0)).mean()
                               + ((mean_pred < 0) & (actuals < 0)).mean())

        running_correl = np.corrcoef(mean_pred, actuals)[0,1]

        print(f"Accuracy on iteration {i}: {percentage_same_sign * 100:.2f}%"
              f" Correl: {running_correl * 100 :.2f}%")

# In[ ]:

# backward shift since actual is fwd looking
# fill in first 2 days

results_df = pd.merge(results_df, (results_df['actual'].
                                   shift().ewm(span=30).
                                   std().bfill().rename('risk')),
                      left_index=True, right_index=True, how='left')
results_df = pd.merge(results_df, df['spread_close'].div(2).rename('tcosts'),
                      left_index=True, right_index=True, how='left')

df_perf,p = fx_backtest(10000,results_df, df, hold_enabled=True, n=roll_size, p=10)


# In[ ]:


plt.plot(df_perf['portfolio_value'])
plt.xticks(rotation=45)
plt.show


# In[ ]:


store_results(df_perf,D,ff,roll_size,n_bags,feature_num,p)


# In[ ]:




