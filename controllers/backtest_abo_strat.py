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
from performance.analysis import plot_lines
from allocate_simulate.strategy import TradingStrategy
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



    forex_price_features = pd.read_parquet(settings.FEATURE_DIR + 'features_280224_curncy_spot.pqt')
    forex_price_features = forex_price_features.sort_index(axis=1, level=0)
    forex_price_features = forex_price_features.sort_index(axis=0)
    # I trust nothing! Nothing!

    default_dict= {
                'RFF': {'tests': False, 'D': 3000, 'sigma': 1},
                'ABO': {'ff': 1, 'l': 0, 'roll_size': 60},
                'Bagged_ABO': {'n_bags': 1, 'feature_num': 3000}
    # missing optimizer params - risk-aversion,
    }


    crosses = ['GBPUSD']
    
    for cross in crosses:
        specific_dict = control_dict.get(cross, default_dict.copy())
        # make sure the dict has all entries
        specific_full_dict = default_dict.copy()
        specific_full_dict.update(specific_dict)

        # RFF params
        tests = specific_full_dict['RFF']['tests']  # test ABO every 20 points
        D = specific_full_dict['RFF']['D']
        sigma = specific_full_dict['RFF']['sigma']

        # ABO params
        ff = specific_full_dict['ABO']['ff']  # untested for ff<1 in new version
        l = specific_full_dict['ABO']['l']  # unused regularisation
        roll_size = specific_full_dict['ABO']['roll_size']

        # Bagged ABO params
        n_bags = specific_full_dict['Bagged_ABO']['n_bags']
        feature_num = specific_full_dict['Bagged_ABO']['feature_num']

        subcols = [x for x in forex_price_features.columns if x[0] == cross ]
        features = forex_price_features[subcols]

        # spec_data = spec_data.rename(columns = {x: x[1] for x in spec_data.columns})
        features.columns = features.columns.droplevel(0)

        labels = features.filter(regex='^ret')
        features = features.drop(columns=labels.columns)
        # prep the rff
        features_dim = features.shape[1]
        rff = GaussianRFF(features_dim, D, sigma)

        # prep the bags
        # Sampling features in each bag
        features_bag_index = sample_features(D, n_bags, feature_num)
        bags={}
        models=[]

        # subset the data
        # QUESTION: Do we have lookahead bias here?
        labels_roll = labels.iloc[:roll_size+1,:]
        features_roll = features.iloc[:roll_size+1,:]

        # scale the data
        (
            labels_norm,
            features_norm,
            labels_scaler,
            features_scaler
        ) = normalize_data(labels_roll, features_roll)

        # perform RFF transformation

        # TODO: Get rid of this retarded shaping!
        features_rff = rff.transform(features_norm.T).T

        for p in range(n_bags):
            bags[p] = features_rff[:, features_bag_index[p]]


        # Parallel execution of the first loop. Model initialization
        results = Parallel(n_jobs=-1)(delayed(process_initial_bag)(p,
                                                                   bags,
                                                                   features_scaler,
                                                                   labels_norm,
                                                                   labels_scaler,
                                                                   ff,
                                                                   l,
                                                                   features_dim,
                                                                   roll_size,
                                                                   tests=tests)
                                      for p in tqdm(range(0, n_bags)))
        all_bags_preds = np.array([result[0] for result in results])
        models = [result[1] for result in results]

        results_df['mean'].iloc[0] = np.mean(all_bags_preds)
        results_df['actual'].iloc[0] =labels[roll_size+1]

        # with open('../utils/data_clean_settings.json') as params_file:
        #     cleaning_dict = json.load(params_file)

        # spot_data.columns = spot_data.columns.swaplevel(0,1)


    pass

if __name__ == '__main__':
    main()