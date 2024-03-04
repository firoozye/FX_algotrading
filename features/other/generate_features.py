import multiprocessing
import logging
from logging.handlers import QueueHandler, QueueListener

from features.other.feature_module import *


def get_feature_for_ticker(ticker_name):
    logging.info(f"starting {ticker_name}")
    list_of_windows = [126, 252]
    momentum_periods = [21, 42, 63, 126, 252]
    quantiles = [0.01, 0.05, 0.95, 0.99]
    predict_periods = [1, 5, 10, 21]
    offsets = [1, 5, 10, 21]

    feature_df_path = 'folder_name/'
    print('starting {}'.format(ticker_name))
    frame = Features(ticker=ticker_name,
                     startdate='2020-01-01',
                     interval='1d',
                     list_of_windows=list_of_windows,
                     momentum_periods=momentum_periods,
                     quantiles=quantiles,
                     predict_periods=predict_periods,
                     offsets=offsets).get_features()
    frame.to_csv(feature_df_path + '{}.csv'.format(ticker_name))
    print('done {}'.format(ticker_name))
    return ticker_name


def worker_init(q):
    # all records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)


def logger_init():
    q = multiprocessing.Queue()
    # this is the handler for all log records
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s"))

    # ql gets records from the queue and sends them to the handler
    ql = QueueListener(q, handler)
    ql.start()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # add the handler to the logger so records from this process are handled
    logger.addHandler(handler)

    return ql, q


def main():
    tickers = np.array(pd.read_csv('~/Desktop/sp100_tickers.csv').Symbol)

    q_listener, q = logger_init()

    logging.info('hello from main thread')
    pool = multiprocessing.Pool(4, worker_init, [q])
    for result in pool.map(get_feature_for_ticker, tickers):
        pass
    pool.close()
    pool.join()
    q_listener.stop()


if __name__ == '__main__':
    main()
