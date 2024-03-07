from os.path import expanduser

home = expanduser("~")  # cross platform home directory
# note only Plotly seems to be sensitive to home and needs it
# explicitly. We use it in OUTPUT_FILES as a consequence

RAW_DATA_DIR = '~/Dropbox/FX_alt/forex_data_raw/'
CLEANED_DATA_DIR = '~/Dropbox/FX_alt/forex_data_cleaned/'
FEATURE_DIR = '~/Dropbox/FX_alt/forex_features/'

# TODO: NOTE - do we want output files by content type or by which run generated them? (i.e., set up dirs for
#  each experiment?) Note I have code to create output_22/ if output_21/ already exists....

OUTPUT_FILES = home + '/Dropbox/FX_alt/output/'
OUTPUT_FIGS = home + '/Dropbox/FX_alt/output/figures/'
OUTPUT_REPORTS = home + '/Dropbox/FX_alt/output/reports/'



