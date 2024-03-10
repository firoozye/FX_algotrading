import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from forecasts.AdaptiveBenignOverfitting import ABO
import numpy as np

def calculate_percentiles(df, n, p):
    """
    Calculate the lower and upper pth percentiles based on the last n points for each row in a DataFrame.

    Args:
    features (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name for which to calculate the percentiles.
    n (int): The number of points to consider for each calculation.
    p (float): The percentile value (between 0 and 100).

    Returns:
    pd.DataFrame: A new DataFrame with the same index as features and two new columns for lower and upper percentiles.
    """
    # Initialize new DataFrame with the same index as features
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


def normalize_data(labels, features,features_final):
    """
    Performs scaling and calculates targets for training from initial dataframe
    Args:
    features (Pandas dataframe): Initial dataframe of size rolling_window + 1


    Returns:
    scaled_target_var (np.array): scaled target variable
    feature_df (np.array): scaled feature matrix
    scaler_target,scaler_feature: scalers for target variable and feature matrix respectively
    """
    scaler_labels = StandardScaler()
    scaler_features = StandardScaler()
    # scalers create N(0,1) data as numpy.array

    labels = pd.DataFrame(labels)
    scaled_labels = scaler_labels.fit_transform(labels)

    overlap= [x for x in labels.columns if x in features.columns]
    if len(overlap)>0:
        features_drop = features.drop(columns=overlap)
        features_final_drop = features.drop(columns=overlap)
    else:
        features_drop = features.copy()
        features_final_drop = features_final.copy()

    features_df = scaler_features.fit_transform(features_drop)
    features_final_df = scaler_features.transform(features_final_drop)

    # return target_var, features
    return scaled_labels, features_df, features_final_df, scaler_labels, scaler_features;

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
def process_initial_bag(bag_no,
                        bag_dict,
                        bag_dict_final,
                        labels,
                        forgetting_factor,
                        l,
                        feature_num,
                        roll_size,
                        tests=False):
    # initialize and train models

    mod_ABO = ABO(bag_dict[bag_no], labels, roll_size, forgetting_factor, l, tests=tests)

    # do saler on [0:roll_size, then do scaler.transform(babs[roll_size})
    # make prediction
    pred_ABO = np.array(mod_ABO.pred(bag_dict_final[bag_no]))

    if tests:
        mod_ABO.in_sample_tests()

        print(f'ABO in-sample resids (should be 0) {mod_ABO.in_sample_resids}')


    return pred_ABO, mod_ABO


# Define the function that will process each bag in the second loop
def process_updated_bag(bag_no,
                        update_bag_dict,
                        update_bag_dict_final,
                        update_label,
                        mod_ABO  ):
    # record features for update trainin
    # record targets for update training

    # update models
    mod_ABO.process_new_data(update_bag_dict[bag_no], update_label)
    # make prediction
    pred_ABO =  np.array(mod_ABO.pred(update_bag_dict_final[bag_no]))

    return pred_ABO, mod_ABO