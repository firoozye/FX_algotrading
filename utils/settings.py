import datetime
import logging.handlers

import pandas as pd
import os
from pathlib import Path

"""

LOGGER Setup

"""

LOGGER = logging.getLogger(__name__)

# TODO: Threading and multi-processing. Which run wrote that entry?
# TODO: consider using logging.conf file!
# .basicConfig is neater than current setup
ERROR_DIRECTORY = './logs/'
ERROR_DIR = './logs/'
# TODO: Make log file save to output_dir

# TODO: Test different start dates or leave out!
START_PERFORMANCE_MSMT = datetime.date(2000, 1, 1)

date = datetime.date.today().strftime('%Y%m%d')

LOG_FILENAME = ERROR_DIRECTORY + f'run_{date}.log'

# logging.basicConfig(
#     level=logging.INFO,
#     format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
#     datefmt="%m-%d %H:%M:%S",
#     filename=LOG_FILENAME,
#     # filemode='w'  # overwrites
# )
# Add the log message handler to the logger
formatter = logging.Formatter(
    fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%m-%d %H:%M:%S")
# can set up overwritten files in handler or backup ones
handler = logging.handlers.RotatingFileHandler(LOG_FILENAME, backupCount=5)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.info('STARTING UP LOGGER')
