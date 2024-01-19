import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from AdaptiveBenignOverfitting import *

def calculate_percentiles(df, n, p):
    """
    Calculate the lower and upper pth percentiles based on the last n points for each row in a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name for which to calculate the percentiles.
    n (int): The number of points to consider for each calculation.
    p (float): The percentile value (between 0 and 100).

    Returns:
    pd.DataFrame: A new DataFrame with the same index as df and two new columns for lower and upper percentiles.
    """
    # Initialize new DataFrame with the same index as df
    df_percentiles = pd.DataFrame(index=df.index)
    df_percentiles['target'] = df['close'].shift(-1) / df['close'] -1

    # Create columns for lower and upper percentiles
    lower_col = f'lower_{p}_percentile'
    upper_col = f'upper_{p}_percentile'
    df_percentiles[lower_col] = np.nan
    df_percentiles[upper_col] = np.nan

    # Loop over the DataFrame
    for i in tqdm(range(n-1, len(df_percentiles))):
        # Select last n points
        window = df_percentiles['target'][i-n+1:i+1]

        # Calculate the lower and upper percentiles
        lower_percentile = np.percentile(window, p)
        upper_percentile = np.percentile(window, 100-p)

        # Assign the percentile values to the new DataFrame
        df_percentiles[lower_col].iloc[i] = lower_percentile
        df_percentiles[upper_col].iloc[i] = upper_percentile

    return df_percentiles


def prepare_data(df):
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

    # df = get_all_features(df_past)
    # df = df[-(roll_size+1):]
    target_var = df['close'].shift(-1) / df['close'] - 1
    target_var.dropna(inplace=True)
    target_var = pd.DataFrame(target_var)
    scaled_target_var = scaler_target.fit_transform(target_var)

    feature_matrix = df.drop(columns=["close"])
    feature_df = scaler_feature.fit_transform(feature_matrix)

    scaled_target_var = scaled_target_var.T
    feature_df = feature_df.T

    # return target_var, df
    return scaled_target_var, feature_df, scaler_target, scaler_feature;

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
    # initialize and train models
    mod_QRRLS = QR_RLS(bags[p].T[:, :roll_size], Y.T[:roll_size], roll_size, ff, l)

    # make prediction
    pred_QRRLS = scaler_Y.inverse_transform(
        np.array(mod_QRRLS.pred(bags[p][roll_size].reshape(feature_num, 1))).reshape(-1, 1))
    return pred_QRRLS, mod_QRRLS


# Define the function that will process each bag in the second loop
def process_updated_bag(p, X_trans, X_new, models, scaler_Y, Y, features_array, feature_num):
    u = X_trans[features_array[p]].reshape(feature_num, 1)  # record features for update training
    d = Y.T[-1].reshape(-1, 1)  # record targets for update training

    # update models
    models[p].update(u, d)

    # make prediction
    pred_QRRLS = scaler_Y.inverse_transform(
        np.array(models[p].pred(X_new[features_array[p]].reshape(feature_num, 1))).reshape(-1, 1))

    return pred_QRRLS