from AdaptiveBenignOverfitting import GaussianRFF, QrRlS

d#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import matplotlib.pyplot as plt

import pandas as pd

import random
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed





from features.feature_module import *


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





random.seed(12)

df_ohlc = pd.read_excel('GBPUSD_1h.xlsx')

df_ohlc.drop(columns='Unnamed: 0', inplace=True)
df_ohlc.set_index('Datetime', inplace=True)
df_ohlc.index = pd.to_datetime(df_ohlc.index, utc = True)
df_ohlc.index = df_ohlc.index.tz_convert('UTC')





def get_all_features(df_ohlc):
    list_of_windows = [63, 126]
    momentum_periods = [2, 4, 8, 16, 32]
    quantiles = [0.01, 0.05, 0.95, 0.99]
    predict_periods = [1, 2, 4, 6]
    offsets = [1, 2, 4, 6]

    frame = Features(df=df_ohlc,
                     list_of_windows=list_of_windows,
                     momentum_periods=momentum_periods,
                     quantiles=quantiles,
                     predict_periods=predict_periods,
                     offsets=offsets).get_features()
    
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.fillna(method='ffill',inplace = True)
    frame.dropna(inplace=True)
    
    return frame





# df_ohlc.head()




# Extremely slow!
# df = get_all_features(df_ohlc)


# df.to_parquet('df_ohlc_all_features.pqt')
df = pd.read_parquet('~/Dropbox/FX/df_ohlc_all_features.pqt')

## Initial dataframe is divided into past and future in order to prevent look ahead basis

df_past = df[:7000] ## arbitrary division 
df_future = df[7000:]


## generates features ##


#


def prepare_data_new(df):
    
    """
    Performs scaling and calculates targets for training from initial dataframe
    Args:
    df (Pandas dataframe): Initial dataframe of size rolling_window + 1


    Returns:
    scaled_target_var (np.array): scaled target variable 
    feature_df (np.array): scaled feature matrix
    scaler_target,scaler_feature: scalers for target variable and feature matrix respectively 
    """
    scaler_target = StandardScaler()
    scaler_feature = StandardScaler()
    
    #df = get_all_features(df_past)
    #df = df[-(roll_size+1):]
    target_var = df['close'].shift(-1) / df['close'] -1
    target_var.dropna(inplace=True)
    target_var = pd.DataFrame(target_var)
    scaled_target_var = scaler_target.fit_transform(target_var)
    
    feature_matrix = df.drop(columns = ["close"])
    feature_df = scaler_feature.fit_transform(feature_matrix)

    scaled_target_var = scaled_target_var.T
    feature_df = feature_df.T
    
    #return target_var, df
    return scaled_target_var,feature_df,scaler_target,scaler_feature;



def sample_features(D, n_bags, feat_num):
    """
    Randomly samples the feature space and allocates features per bag. Samples with replacement if the total 
    feature space is smaller than n_bags*feat_num, otherwise without replacement.

    Args:
    features_array (np.array): Array of features.
    n_bags (int): Number of bags.
    feat_num (int): Number of features in each bag.

    Returns:
    list: A list of arrays, where each array contains the indices of the features in each bag.
    """
    total_features = D

    # Determine if sampling should be with or without replacement
    replace = total_features < n_bags * feat_num

    # Initialize an empty list to store the feature indices for each bag
    features_array = []

    # Randomly sample feature indices for each bag
    for _ in range(n_bags):
        bag_indices = np.random.choice(total_features, size=feat_num, replace=replace)
        features_array.append(bag_indices)

    return features_array




# Define the function that will process each bag in the first loop
def process_initial_bag(p, bags, Y, scaler_Y, ff, l, feature_num, roll_size):
    
    #initialize and train models
    mod_QRRLS = QrRlS(bags[p].T[:, :roll_size], Y.T[:roll_size], roll_size, ff, l)
    
    #make prediction
    pred_QRRLS = scaler_Y.inverse_transform(np.array(mod_QRRLS.pred(bags[p][roll_size].reshape(feature_num, 1))).reshape(-1,1))
    return pred_QRRLS, mod_QRRLS

# Define the function that will process each bag in the second loop
def process_updated_bag(p, X_trans, X_new, models, scaler_Y, Y, features_array, feature_num):
    
    u = X_trans[features_array[p]].reshape(feature_num, 1) #record features for update training 
    d = Y.T[-1].reshape(-1, 1) #record targets for update training
    
    #update models
    models[p].update(u, d)
    
    #make prediction 
    pred_QRRLS = scaler_Y.inverse_transform(np.array(models[p].pred(X_new[features_array[p]].reshape(feature_num, 1))).reshape(-1,1))
    
    return pred_QRRLS



D = 600
sigma = 1
ff = 1
l = 0
roll_size = 400
n_bags = 1
feature_num = D 
overlap = False

#initialize dataframe for results recording
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
Y, X, scaler_Y, scaler_X = prepare_data_new(df_model) 

# perform RFF transformation
lags = X.shape[0]
rff = GaussianRFF(lags, D, sigma)
X_trans = rff.transform(X.reshape(lags, roll_size+1)).T

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

for i in tqdm(range(0, 300)):
    
    #Delete old data and append data, that just became available
    df_temp = df_temp.iloc[1:]
    df_temp = df_temp.append(df_future.iloc[i])
    
    Y, X, scaler_Y, scaler_X = prepare_data_new(df_temp)
    
    ## We need to perform RFF expansion on the new observation row. For which we don't have target
    ## And which will be used for forecasting
    X_new = rff.transform(X[:, -1:].reshape(lags, 1))
    
    # Parallel execution of the second loop
    all_bags_preds = Parallel(n_jobs=-1)(delayed(process_updated_bag)(p, X_old, X_new, models, scaler_Y, Y, features_array, feature_num) for p in range(0, n_bags))
    
    #new obeservation will be appended to train set in the next iteration
    X_old = X_new 
    
    # record results
    results_df['actual'].iloc[i+1] = df_future['close'][i+1]/df_temp['close'][-1]-1 #actual target
    results_df['mean'].iloc[i+1] = np.mean(all_bags_preds)





results_df.dropna(inplace=True)



same_sign_count = ((results_df['mean'] > 0) & (results_df['actual'] > 0)).sum() + ((results_df['mean'] < 0) & (results_df['actual'] < 0)).sum()

# Calculate the percentage
total_rows = len(results_df)
percentage_same_sign = (same_sign_count / total_rows) * 100

print(f"Count of rows with columns of the same sign: {same_sign_count}")
print(f"Percentage of total: {percentage_same_sign:.2f}%")



plt.figure(figsize=(12, 4))
plt.plot(results_df.index, results_df['actual'], label = 'Actual value')
plt.plot(results_df.index, results_df['mean'], label = 'Predicted value')
plt.xlabel('Date')
plt.ylabel('GBP/USD  return')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# ### Algostrategy backtesting


results_df['Close'] = df['close']


signals = results_df['mean'].apply(lambda x: "BUY" if x > 0 else 'SELL')
df_performance = pd.DataFrame()
df_performance['signal'] = signals
df_performance.index = signals.index
df_performance['Close'] = df["close"]
df_performance['spread'] = df['spread_close']




def run_backtest(df):
    initial_cash = 10000
    cash = initial_cash
    shares = 0
    portfolio_value = initial_cash
    position = None  # 'BUY' for long, 'SELL' for short, None for no position
    entry_price = 0

    # Add columns for portfolio value and cumulative return to the dataframe
    df['Portfolio Value'] = initial_cash
    df['Cumulative Return'] = 0.0

    # Backtesting algorithm
    for i in range(len(df)):
        current_signal = df.iloc[i]['signal']
        spread = df.iloc[i]['spread']
        half_spread = spread / 2

        # Adjusted buy and sell prices considering the spread
        adjusted_buy_price = df.iloc[i]['Close'] + half_spread
        adjusted_sell_price = df.iloc[i]['Close'] - half_spread

        # If it's the first iteration or if there is a signal change, we need to act
        if i == 0 or current_signal != df.iloc[i-1]['signal']:
            if position is not None:
                # Close the current position
                if position == 'BUY':
                    # Sell at the adjusted sell price
                    cash += shares * adjusted_sell_price
                elif position == 'SELL':
                    # Close the short at the adjusted buy price
                    cash += shares * (2 * entry_price - adjusted_buy_price)

            shares = 0  # Reset the shares after closing the position

            # Open new position based on the signal
            if current_signal == 'BUY':
                position = 'BUY'
                shares = cash // adjusted_buy_price
                cash -= shares * adjusted_buy_price
                entry_price = adjusted_buy_price
            elif current_signal == 'SELL':
                position = 'SELL'
                shares = cash // adjusted_sell_price
                cash -= shares * adjusted_sell_price
                entry_price = adjusted_sell_price

        # If the signal is the same as the previous, we need to update the portfolio value
        if position == 'BUY':
            # Update portfolio value for a long position
            portfolio_value = cash + shares * df.iloc[i]['Close']
        elif position == 'SELL':
            # Update portfolio value for a short position
            portfolio_value = cash + shares * (2 * entry_price - df.iloc[i]['Close'])
        else:
            # Update portfolio value if there is no position
            portfolio_value = cash

        # Update the portfolio value and cumulative return for the current step
        df.at[df.index[i], 'Portfolio Value'] = portfolio_value
        df.at[df.index[i], 'Cumulative Return'] = (portfolio_value - initial_cash) / initial_cash

    # Display the dataframe
    return df

    



perf = run_backtest(df_performance)



plt.plot(perf['Portfolio Value'])
plt.xticks(rotation=45)
plt.show


# ## Hyperparameter optimization. To be continued...

# In[ ]:


from skopt.space import Integer
from skopt.utils import use_named_args
from skopt.callbacks import VerboseCallback


# In[ ]:


space = [
    Integer(450, 4000, name='D'),
    Integer(50, 200, name='roll_size'),
    Integer(1, 50, name='n_bags'),
    Integer(50, 200, name='feature_num'),
    # Add more hyperparameters here if needed
]


# In[ ]:


@use_named_args(space)
def model_evaluation(D, roll_size, n_bags, feature_num):
    
    sigma = 1
    ff = 1
    l = 0
    # Initialize the neccesary lists
    models = []
    bags = []

    # Select the most recent data from the available dataframe
    df_model = df_past[-(roll_size+1):] # size is roll_size + 1, because we need 1 more point to make prediction
                                        # for that point we don't know the target variable yet

    # calculate targets and scale the data
    Y, X, scaler_Y, scaler_X = prepare_data_new(df_model) 

    # perform RFF transformation
    lags = X.shape[0]
    rff = GaussianRFF(lags, D, sigma)
    X_trans = rff.transform(X.reshape(lags, roll_size+1)).T

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

    for i in tqdm(range(0, 300)):

        #Delete old data and append data, that just became available
        df_temp = df_temp.iloc[1:]
        df_temp = df_temp.append(df_future.iloc[i])

        Y, X, scaler_Y, scaler_X = prepare_data_new(df_temp)

        ## We need to perform RFF expansion on the new observation row. For which we don't have target
        ## And which will be used for forecasting
        X_new = rff.transform(X[:, -1:].reshape(lags, 1))

        # Parallel execution of the second loop
        all_bags_preds = Parallel(n_jobs=-1)(delayed(process_updated_bag)(p, X_old, X_new, models, scaler_Y, Y, features_array, feature_num) for p in range(0, n_bags))

        #new obeservation will be appended to train set in the next iteration
        X_old = X_new 

        # record results
        results_df['actual'].iloc[i+1] = df_future['close'][i+1]/df_temp['close'][-1]-1 #actual target
        results_df['mean'].iloc[i+1] = np.mean(all_bags_preds)

    # Calculate the performance metric
    results_df.dropna(inplace = True)
    same_sign_count = ((results_df['mean'] > 0) & (results_df['actual'] > 0)).sum() + ((results_df['mean'] < 0) & (results_df['actual'] < 0)).sum()
    total_rows = len(results_df)
    percentage_different_sign = (1-(same_sign_count / total_rows)) * 100

    return percentage_different_sign


# In[ ]:


class CustomCallback(VerboseCallback):
    def __init__(self, n_total_calls):
        super().__init__(n_total_calls)

    def __call__(self, res):
        super().__call__(res)
        # Print the current iteration number and best score
        print(f"Iteration {len(res.x_iters)} completed.")
        print(f"Current best score: {res.fun}")

        # Print the best parameters found so far
        print("Current best parameters:", res.x)

