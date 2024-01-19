import multiprocessing
import time
import os
import random
import multiprocessing
import warnings
warnings.filterwarnings('ignore')
import datetime
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from copy import deepcopy
from scipy.linalg import pinv
import random
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from joblib import Parallel, delayed

random.seed(12)

df = pd.read_parquet('~/Dropbox/FX/df_ohlc_all_features.pqt')

print(df)
