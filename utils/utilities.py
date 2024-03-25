import regex as re
import argparse
import json
import os
import re
from typing import Dict

import numpy as np

from utils.settings import OUTPUT_FILES


def parse_cmd_line_args(pgm:str='cv') -> Dict[str,str]:
    # help_msg_dict =
    help_msg = "hyper-param Json file"
    parser = argparse.ArgumentParser(description=help_msg,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('--params', '-p',
                        help='Json Param File',
                        default='params.json',
                        type=str)

    parser.add_argument('--cross', '-c',
                         help='Cross to fit',
                         default='GPUSD',
                         type=str)

    parser.add_argument('--feat_set', '-f',
                         help='Feature_set: macd, price, macd_carry',
                         default='macd',
                         type=str)

    parser.add_argument('--obj', '-o',
                        help='Objective: mse, rms, mape, sr',
                        default='mse',
                        type=str)

    args = vars(parser.parse_args())
    params_filename = args.pop('params')
    cross = args.pop('cross')
    feat_set = args.pop('feat_set')

    obj = args.pop('obj')
    return {'params':params_filename,
            'cross': cross,
            'feat_set':feat_set,
            'obj':obj}


def create_next_output_file(file_name_template, unch=False):
    listdir = os.listdir(OUTPUT_FILES)
    rextx = "^{}.*".format(file_name_template)
    txtfiles = [x for x in listdir if re.search(rextx, x)]
    if len(txtfiles)>0:
        nums = [int(re.findall('\d+', x)[0]) for x in txtfiles]
        if unch:
            nextnum = np.max(np.array_str(nums))
        else: # increment the max
            nextnum = np.max(np.array_str(nums)) + 1
    else:
        nextnum = 1

    output_str = file_name_template + str(nextnum)
    return output_str, nextnum


def return_output_dir(unch=False) ->str:
    output_dirname = create_next_output_file(
        file_name_template='Output_', unch=unch)

    full_path = OUTPUT_DIR + '/' + output_dirname
    if os.path.isdir(full_path):
        os.mkdir(full_path)
    return full_path


def return_args():

    command_line_args = parse_cmd_line_args()
    params_blob_file = command_line_args['params']
    cross = command_line_args['cross']
    feat_set = command_line_args['feat_set'] #MACD, Price, or Carry
    obj = command_line_args['obj']
    # help = 'Objective: mse, rms, mape, sr',

    with open(params_blob_file) as params_file:
        control_dict = json.load(params_file)

    return control_dict


def get_settings(ticker, settings_type, command_dict, default_dict):
    ticker_prefix = ticker.split(' ')[0] # just get currency pair

    cleaning_spec_dict = default_dict.copy()
    total_spec_dict = command_dict.get(ticker_prefix, {})
    if len(total_spec_dict) > 0:
        specific_settings = total_spec_dict.get(settings_type, {})
        cleaning_spec_dict.update(specific_settings) # overwrite if specific, otherwise use default
    return cleaning_spec_dict


def filter_multi(df, index_level, regex, axis=0):
    def f(x):
        return matcher.search(str(x)) is not None

    matcher = re.compile(regex)
    values = df.axes[axis].get_level_values(index_level).map(f)
    return df.loc(axis=axis)[values]


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
    return meta_data, feature_num, forgetting_factor, l, n_bags, no_rff, roll_size, sigma, tests
