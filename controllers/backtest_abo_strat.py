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
from utils.utilities import extract_params, parse_cmd_line_args

warnings.filterwarnings('ignore')


# HOME_DIRECTORY = '~/PycharmProjects/abo_fx/'
# sys.path.append(HOME_DIRECTORY)

# from performance.performance import *

from performance.analysis import plot_lines
# from features.features import signal_combo, reversion
import utils.settings as settings
# from performance.performance import build_perf_anal_table, build_pos_anal_table
#

from forecasts.AdaptiveBenignOverfitting import GaussianRFF
from forecasts.forecast_utils import (normalize_data, process_initial_bag,
                                      process_updated_bag, sample_features)

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space.space import (Integer, Real, Categorical)

# from backtest_master import return_args, return_output_dir



random.seed(12)


def main():

    command_line_args = parse_cmd_line_args()
    params_blob_file = command_line_args['params']
    cross = command_line_args['cross']
    feat_set = command_line_args['feat_set'] #MACD, Price, or Carry
    obj = command_line_args['obj']
    # help = 'Objective: mse, rms, mape, sr',

    # with open(params_blob_file) as params_file:
    #     control_dict = json.load(params_file)


    # run_all_iterations()


    default_dict = {
        'RFF': {'tests': False, 'no_rff': 3000, 'sigma': 1},
        'ABO': {'forgetting_factor': 1.0, 'l': 0, 'roll_size': 60},
        'Bagged_ABO': {'n_bags': 1, 'feature_num': 3000}
    # missing optimizer params - risk-aversion,
    }
    # run_forecasts_settings(default_dict=default_dict)

    # no_rff in [1000,3000,5000]
    #         no_rff, sigma, ff, roll_size, cross, feat_set, obj = list_of_params
    dim1 = Categorical(name='no_rff', categories=(500, 1000,2000,3000,5000,6000))
    dim2 = Real(name='sigma', low=0.25, high=10.0)
    dim3 = Real(name='ff', low=0.5, high=1.0)
    dim4 = Categorical(name='roll_size', categories=(10, 30, 60, 90, 150, 180, 210, 250))
    # dim5 = Categorical(name='cross',categories =(cross))
    # dim6= Categorical(name='feat_set', categories=(feat_set))
    # dim7= Categorical(name='obj', categories=(obj))
    dimensions = [dim1, dim2, dim3, dim4]   # dim5, dim6, dim7]


    @use_named_args(dimensions=dimensions)
    def objective_function(no_rff: int = 3000, sigma: float = 1.0,
                           ff: float = 1.0, roll_size: int = 60):
                           # cross: str = 'GBPUSD', feat_set: str = 'macd',
                           # obj: str = 'mse'):

        default_dict = {
            'feature': {'horizon': 1},
            'RFF': {'tests': False, 'no_rff': no_rff, 'sigma': sigma},
            'ABO': {'forgetting_factor': ff, 'l': 0, 'roll_size': roll_size},
            'Bagged_ABO': {'n_bags': 1, 'feature_num': no_rff}
            # missing optimizer params - risk-aversion,
        }

        abo = ABOModelClass(default_dict, cross=cross)
        abo.load_data(cut_off="2018", feat_set=feat_set)
        abo.run_forecasts()
        abo.forecast_error()
        abo.save_forecasts()  # tack it onto our database
        abo.forecast_error()
        if obj == 'mse':
            obj_val = abo.mse['all']
        elif obj == 'rms':
            obj_val = abo.rms['all']
        elif obj == 'mape':
            obj_val = abo.mape['all']
        elif obj == 'neg_sr':
            obj_val = -1 * abo.sr['all']
        else:
            raise NotImplementedError
        # help = 'Objective: mse, rms, mape, sr',

        # let's be noisy
        print(f' Evaluated at {abo.meta_data}, with value {obj_val}')
        return obj_val

    res = gp_minimize(func=objective_function, dimensions=dimensions,
                      acq_func="LCB",  # "PI" - prob improve "EI" - exp improv, the acquisition function
                      n_calls=15,  # the number of evaluations of f
                      n_initial_points=5,  # the number of random initialization points
                      noise=0.1 ** 2,  # the noise level (optional)
                      random_state=1234  # the random seed
                      )

    # res=gp_minimize(objective_function, dimensions=[Categorical((1000,3000,5000)),
    #                                                 Real(0.5,5),
    #                                                 Real(0.6,1.0),
    #                                                 Categorical((10, 30, 60, 90, 150, 180, 250)),
    #                                                 Categorical((cross)),  # pass params to optimizer
    #                                                 Categorical((feat_set)),
    #                                                 Categorical((obj))
    #                                                 ],
    #                 acq_func="LCB",  # "PI" - prob improve "EI" - exp improv, the acquisition function
    #                 n_calls=15,  # the number of evaluations of f
    #                 n_initial_points=5,  # the number of random initialization points
    #                 noise=0.1 ** 2,  # the noise level (optional)
    #                 random_state=1234  # the random seed
    #                 )
    print(f'x ={res.x}, f(x) = {res.fun}')


# def objective_function(list_of_params: list|None=None):
#     if list_of_params is None:
#         no_rff=3000
#         sigma=1
#         ff=1.0
#         roll_size=60
#         cross='GBPUSD'
#         feat_set = 'macd'
#         obj ='mse'
#     else:
#         no_rff, sigma, ff, roll_size, cross, feat_set, obj = list_of_params



class ABOModelClass(object):
    default_dict= {
            'features':{'horizon': 1 },
            'RFF': {'tests': False, 'no_rff': 3000, 'sigma': 1.0},
            'ABO': {'forgetting_factor': 1.0, 'l': 0, 'roll_size': 100},
            'Bagged_ABO': {'n_bags': 1, 'feature_num': 3000}
            # missing optimizer params - risk-aversion,
        }

    def __init__(self,
                 model_dict:dict|None=None,
                 cross:str='CADUSD'):

        self.crosses = ['CADGBP', 'AUDCAD', 'GBPJPY', 'CADUSD', 'JPYUSD', "SEKNZD", 'GBPUSD']

        if model_dict is None:
            model_dict = ABOModelClass.default_dict
        self.model_dict = ABOModelClass.default_dict.copy() # take as base
        self.model_dict.update(model_dict)
        #  update by passed params if they exist

        (self.meta_data,
         self.feature_num,
         self.forgetting_factor,
         self.l,
         self.n_bags,
         self.no_rff,
         self.roll_size,
         self.sigma,
         self.tests
         ) = self.extract_params(self.model_dict)
        if self.forgetting_factor ** self.roll_size < 0.01:
            print('forgetting_factor too small for rollsize, will reset')
            self.forgetting_factor = (0.01)**(1/self.roll_size)
            self.model_dict['ABO']['forgetting_factor'] = self.forgetting_factor



        self.cross = cross
        if cross not in self.crosses:
            print(f'Cross {cross} not in dataset yet')
            raise NotImplementedError

        self.features = None
        self.results = None

    def load_data(self, cut_off:datetime.datetime|str|None =None, feat_set:str|None=None):
        if feat_set is None:
            feat_set = 'macd'   # dont do anything yet


        feature_file = 'features_280224_' + feat_set + '_curncy_spot.pqt'
        forex_price_features = pd.read_parquet(settings.FEATURE_DIR + feature_file,
                                               engine='pyarrow')
        forex_price_features.index = pd.to_datetime(forex_price_features.index)
        if cut_off is not None:
            forex_price_features = forex_price_features.loc[:cut_off,:]
        forex_price_features = forex_price_features.sort_index(axis=1, level=0)
        forex_price_features = forex_price_features.sort_index(axis=0)
        subcols = [x for x in forex_price_features.columns if x[0] == self.cross]
        # pick our cross, and drop all na's
        features = forex_price_features[subcols].dropna(axis=0)
        # spec_data = spec_data.rename(columns = {x: x[1] for x in spec_data.columns})
        features.columns = features.columns.droplevel(0)
        self.features = features

    def read_json_dicts(self, default_dict):
        # run_forecasts_settings(default_dict=default_dict)

        with open('./utils/data_clean_settings.json') as params_file:
            control_dict = json.load(params_file)
        # TODO: Give json a full pathname! (not relative)

        if self.features is None:
            self.load_data()
        cross = self.cross
        print(f'Starting run for {cross}')

        specific_dict = control_dict.get(cross, default_dict.copy())
        # make sure the dict has all entries
        specific_full_dict = ABOModelClass.default_dict.copy()
        specific_full_dict.update(specific_dict)

    def run_forecasts(self):
        if self.features is None:
            self.load_data()

        if self.tests:
            max_steps = 300
        else:
            max_steps = None
        (results_df, meta_data) = bagged_abo_forecast(self.features, self.model_dict, max_steps=max_steps)
        self.results = results_df
        print(f'finished forecasts for {self.cross} for {[(x, y) for (x, y) in meta_data.items()]}')

        # pd.DataFrame(corr_dict).to_csv(settings.OUTPUT_REPORTS + f'corrs_for_{roll_size}.csv')
        print('End forecasts')

    def save_forecasts(self):
        append_and_save_forecasts(forex_forecast_storage=None,
                                  results_df=self.results,
                                  cross=self.cross,
                                  meta_data=self.meta_data,
                                  # storage_file='AllPrice_forecast'
                                  )


    def forecast_error(self):
        res = self.results.dropna(axis=0)
        if res.index.dtype == object:
            res.index = pd.to_datetime(res.index)
        # this may not be needed! Can we check if res.index.type == object
        X = res['mean']
        Y = res['actual']

        years = list(X.index.year.unique())
        years.sort()
        years = [str(x) for x in years if len(Y.loc[str(x)]) > 22]
        # every year that we have more than a month of data

        def localizer(func):
            all_entry = {'all': func(Y.index)}
            yrly_entry = {yr:func(yr) for yr in years}
            yrly_entry.update(all_entry)
            return yrly_entry

        mse_func = (lambda yr: (Y.loc[yr] - X.loc[yr]).pow(2).mean())
        hit_ratio_func = (lambda yr: ((Y.loc[yr].map(np.sign) * X.loc[yr].map(np.sign)).mean() + 1)/2)
        # prob of same direction
        mae_func = (lambda yr: (Y.loc[yr]-X.loc[yr]).abs().mean())
        corr_func = (lambda yr: np.corrcoef(Y.loc[yr],X.loc[yr])[0,1])
        mape_func = lambda yr: ((Y.loc[yr].loc[Y.loc[yr]!=0] - X.loc[yr].loc[Y.loc[yr]!=0]) /
                                Y.loc[yr].loc[Y.loc[yr]!=0]).abs().mean()
        # remove 0s in Y. probably market close. Convoluted def to
        # accommodate 'all' as Y.index
        rms_func = (lambda yr: np.sqrt((Y.loc[yr] - X.loc[yr]).pow(2).mean()))
        strat = Y * X
        sr_func = lambda yr: strat.loc[yr].dropna().mean() / strat.loc[yr].dropna().std() * np.sqrt(252)
        y_last = lambda yr: Y.loc[yr].iloc[-1]
        y_std = lambda yr: Y.loc[yr].std()  # for order of magnitude

        self.mse = pd.Series(localizer(mse_func), name='mse')
        self.hit_ratio = pd.Series(localizer(hit_ratio_func), name='hit_ratio')
        self.mae = pd.Series(localizer(mae_func), name='mae')
        self.corr = pd.Series(localizer(corr_func), name='corr')
        self.mape = pd.Series(localizer(mape_func),name='mape')
        self.rms = pd.Series(localizer(rms_func), name='rms')
        self.sr = pd.Series(localizer(sr_func), name='sr')
        self.y_last = pd.Series(localizer(y_last),name='actual_last')
        self.y_std = pd.Series(localizer(y_std), name='actual_std' )
        self.all_output = pd.concat([self.mse, self.rms,
                                     self.mae, self.corr,
                                     self.sr, self.hit_ratio,
                                     self.mape, self.y_last,
                                     self.y_std], axis=1)

    @staticmethod
    def extract_params(specific_full_dict):
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
        horizon = specific_full_dict['features']['horizon']
        meta_data = {'no_rff': no_rff, 'forgetting_factor': forgetting_factor, 'roll_size': roll_size,
                     'sigma': sigma,
                     'horizon': horizon}
        return (meta_data, feature_num, forgetting_factor, l, n_bags, no_rff, roll_size, sigma, tests)


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


def run_forecasts_settings(default_dict: dict|None=None, feat_set:str|None=None):
    with open('./utils/data_clean_settings.json') as params_file:
        control_dict = json.load(params_file)
    #TODO: Give json a full pathname! (not relative)

    switch = 'combo'
    if feat_set is None:
        feat_set = 'macd'

    if feat_set not in ['macd','price']: # carry missing
        raise NotImplementedError

    feature_file = 'features_280224_' + feat_set +'_curncy_spot.pqt'
    corr_dict = {}
    forex_price_features = pd.read_parquet(settings.FEATURE_DIR + 'features_280224_curncy_spot.pqt',
                                           engine='pyarrow')
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

        @staticmethod

        def bagged_abo_forecast(features, specific_full_dict, max_steps=None):

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
        if max_steps is not None:
            iter_time_steps = min(time_steps, max_steps)
        else:
            iter_time_steps = time_steps
            # meant just for testing purposes!
        # prep the rff
        features_dim = features.shape[1]
        rff = GaussianRFF(features_dim=features_dim, no_rff=no_rff, kernel_var=sigma)
        # prep the bag_dict
        # Sampling features in each bag
        results_df = pd.DataFrame(np.nan * np.ones((iter_time_steps, 2)),
                                  columns=['mean', 'actual'],
                                  index=labels.index[:iter_time_steps])
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
        for ind in range(1, iter_time_steps - roll_size - 1):

            # removed tqdm
            labels_roll = labels.iloc[:ind + roll_size, :]
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