from os.path import expanduser
import os

# note only Plotly seems to be sensitive to home and needs it
# explicitly. We use it in OUTPUT_FILES as a consequence

#TODO: Maintain a list of node names for different directories

home = expanduser("~")  # cross platform home directory
if os.uname()[1] in ['edwin-jaynes.cs.ucl.ac.uk']:
    # base_dir = '~/Dropbox/FX_alt/'
    # home = /home/firoozye  or same as ~
    new_base = '/Dropbox/FX_alt/'

else:  # on the cluster
    new_base = '/abo_research/abo_data/FX_alt/'

RAW_DATA_DIR = home + new_base + 'forex_data_raw/'
CLEANED_DATA_DIR = home + new_base + 'forex_data_cleaned/'
FEATURE_DIR = home + new_base + 'forex_features/'
OUTPUT_FILES = home + new_base + 'output/'
OUTPUT_FIGS = home + new_base + 'output/figures/'
OUTPUT_REPORTS = home + new_base + 'output/reports/'
experiment_path = home + new_base + "output/experiments/"
# TODO: NOTE - do we want output files by content type or by which run generated them? (i.e., set up dirs for
#  each experiment?) Note I have code to create output_22/ if output_21/ already exists....



