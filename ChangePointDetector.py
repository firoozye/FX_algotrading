import warnings
warnings.filterwarnings('ignore')

from forecasts.AdaptiveBenignOverfitting import *
from forecasts.forecast_utils import *
from features.feature_module import *


df_feature = pd.read_parquet('~/Dropbox/FX/GBPUSD_DailyFeatures_all.pqt')

df_past = df_feature[:500]
df_future = df_feature[500:]

D = 1000
sigma = 1
ff = 0.3
l = 0
roll_size = 400
exp_window = 0
n_bags = 1
feature_num = D

K = 2.5


def detect_change_point(betas_array, K, p, current_run, last_reset_point):
    current_run = current_run + 1
    if current_run == 0 or current_run == last_reset_point:
        return False

    dbetas = [np.sum(np.square(betas_array[i][p] - betas_array[i - 1][p])) for i in
              range(last_reset_point + 1, current_run + 1)]
    dbeta = dbetas[-1]
    avg_dbeta = np.mean(dbetas)
    stdev_dbeta = np.std(dbetas)

    return (dbeta - avg_dbeta) / stdev_dbeta > K


last_index_df_past = df_past.index[-1:]
indices_df_future = df_future.index[:-1]
combined_indices = last_index_df_past.append(indices_df_future)
results_df = pd.DataFrame(index=combined_indices)
results_df['actual'] = np.nan
results_df['mean'] = np.nan

# Initialize the neccesary lists
models = []
bags = []
all_bags_array = []
betas_array = []
# Select the most recent data from the available dataframe
df_model = df_past[-(roll_size + 1):]  # size is roll_size + 1, because we need 1 more point to make prediction
# for that point we don't know the target variable yet

# calculate targets and scale the data
Y, X, scaler_Y, scaler_X = prepare_data(df_model)

# perform RFF transformation
lags = X.shape[0]
rff = GaussianRFF(lags, D, sigma)
X_trans = rff.transform(X.reshape(lags, roll_size + 1)).T

# Sampling features in each bag
features_array = sample_features(D, n_bags, feature_num)

for p in range(n_bags):
    bags.append(X_trans[:, features_array[p]])

# Parallel execution of the first loop. Model initialization
results = Parallel(n_jobs=-1)(
    delayed(process_initial_bag)(p, bags, Y, scaler_Y, ff, l, feature_num, roll_size, exp_window) for p in
    tqdm(range(0, n_bags)))
all_bags_preds = np.array([result[0] for result in results])
models = [result[1] for result in results]
betas = np.array([result[2] for result in results])
betas_array.append(betas)
all_bags_array.append(np.array(all_bags_preds).T)
# Add results in a results dataframe for comparison
results_df['actual'].iloc[0] = df_future['close'][0] / df_past['close'][-1] - 1  # actual target
results_df['mean'].iloc[0] = np.mean(all_bags_preds)

# Continue performing forecasts by updating QR_RLS model
df_temp = df_model

# we need the last row of RFF dataset to append it to train set on next iteration
X_old = X_trans[-1, :].T

for i in tqdm(range(0, 100)):

    # Delete old data and append data, that just became available
    # df_temp = df_temp.iloc[1:]
    df_temp = pd.concat([df_temp.T, df_future.iloc[i]], axis=1).T

    Y, X, scaler_Y, scaler_X = prepare_data(df_temp)

    ## We need to perform RFF expansion on the new observation row. For which we don't have target
    ## And which will be used for forecasting
    X_new = rff.transform(X[:, -1:].reshape(lags, 1))

    # Parallel execution of the second loop
    results = Parallel(n_jobs=-1)(
        delayed(process_updated_bag)(p, X_old, X_new, models, scaler_Y, Y, features_array, feature_num) for p in
        range(0, n_bags))
    all_bags_preds = np.array([result[0] for result in results])
    betas = [result[1] for result in results]
    betas_array.append(betas)
    all_bags_array.append(np.array(all_bags_preds).T)
    # new obeservation will be appended to train set in the next iteration
    X_old = X_new

    last_reset_points = [0] * n_bags

    for p in range(n_bags):
        if detect_change_point(betas_array, K, p, i, last_reset_points[p]):
            print(f"Change point detected. Bag {p} is being reinitialized at iteration {i}.")
            # Reinitialize mod_QRRLS for this bag
            df_temp = df_temp[-(roll_size + 1):]

            # calculate targets and scale the data
            Y, X, scaler_Y, scaler_X = prepare_data(df_temp)

            # perform RFF transformation
            lags = X.shape[0]
            rff = GaussianRFF(lags, D, sigma)
            X_trans = rff.transform(X.reshape(lags, roll_size + 1)).T

            # Sampling features in each bag
            features_array = sample_features(D, n_bags, feature_num)

            for p in range(n_bags):
                bags[p] = (X_trans[:, features_array[p]])

            result = process_initial_bag(p, bags, Y, scaler_Y, ff, l, feature_num, roll_size, exp_window)
            models[p] = result[1]  # Replace the model in the models list
            betas_array[i + 1][p] = result[2]
            pred_QRRLS, _, _ = result
            all_bags_array[i + 1][0, 0, p] = pred_QRRLS.reshape(-1)
            last_reset_points[p] = i + 1

        # record results
    results_df['actual'].iloc[i + 1] = df_future['close'][i + 1] / df_temp['close'][-1] - 1  # actual target
    results_df['mean'].iloc[i + 1] = np.mean(all_bags_preds)

    if i % 10 == 0 and i > 0:

        same_sign_count = ((results_df['mean'][:i] > 0) & (results_df['actual'][:i] > 0)).sum() + (
                    (results_df['mean'][:i] < 0) & (results_df['actual'][:i] < 0)).sum()

        # Calculate the percentage
        total_rows = len(results_df['mean'][:i])
        percentage_same_sign = (same_sign_count / total_rows) * 100

        print(f"Accuracy on iteration {i}: {percentage_same_sign:.2f}%")

        p = results_df['mean'][:i].corr(results_df['actual'][:i])

        # Calculate rolling Sharpe ratio
        if not np.isnan(p):  # Check if p is not NaN
            rolling_sharpe_ratio = (p / np.sqrt(p ** 2 + 1)) * np.sqrt(252)
            print(
                f"Accuracy on iteration {i}: {percentage_same_sign:.2f}%, Rolling Sharpe Ratio: {rolling_sharpe_ratio:.2f}")
        else:
            print(
                f"Accuracy on iteration {i}: {percentage_same_sign:.2f}%, Rolling Sharpe Ratio: Cannot be calculated (NaN)")