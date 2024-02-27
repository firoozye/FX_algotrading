import argparse
import json
import re
from typing import Dict
import copy
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd


def parse_cmd_line_args(pgm:str='cv') -> Dict[str,str]:
    # help_msg_dict =
    help_msg = "hyper-param Json file"
    parser = argparse.ArgumentParser(description=help_msg,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('--params', '-p',
                        help='Json Param File',
                        default='params.json'
                        type=str)

    args = vars(parser.parse_args())
    params_filename = args.pop('params')

    return {'params':params_filename}

OUTPUT_DIR = '~/Dropbox/FX'

def create_next_output_file(file_name_template, unch=False):
    listdir = os.listdir(OUTPUT_DIR)
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


    with open(params_blob_file) as params_file:
        control_dict = json.load(params_file)

    return control_dict


if __name__ == "__main__":
    main()