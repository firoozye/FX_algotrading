import time

import pandas as pd

from cca_lib.canonical_port import CCAStrategy


def simulate_backtest(cov_models,
                      returns,
                      signals,
                      rolling_cov=1000,
                      rebalance='1B',
                      cca_option='unconstrain',
                      cross_corr_option='new'):
    """Run a backtest on return data with specified rebalance periods and model parameters

    :param cov_models: List of covariance models
    :param xscov_models: String of cross-correlation model
    :param returns: Dataframe of returns where the time component is row-wise
    :param signals: Dataframe of signals where the time component is row-wise
    :param n_lag: Number of day to lag
    :param rolling_cov: Time length used to compute the covariance and cross-correlation matrices
    :param rebalance: Pandas resample frequency
    :return: Optimal portfolio weights
    """
    returns.index = pd.to_datetime(returns.index)
    signals.index = pd.to_datetime(signals.index)

    print('OPTIMIZING FOR MODELS: {}'.format('-'.join(cov_models)))
    all_dates = returns.index.tolist()
    T_tot = returns.shape[0]

    # use pandas resample method to obtain dates of rebalance   - WHY BOTHER?
    if rebalance not in ['1B', '1D']:
        rebalance_dates = returns.resample(rebalance).asfreq().index
    else:
        rebalance_dates = all_dates.copy()
    # dictionary of dataframes to store the calculated portfolio weights
    portfolio_weights = {m: pd.DataFrame(columns=returns.columns) for m in cov_models}

    cca_models = dict()
    for cov_method in cov_models:
        cca_models[cov_method] = CCAStrategy(cov_method=cov_method,
                                             cca_option=cca_option,
                                             cross_corr_option=cross_corr_option)


    # begin only when a covariance matrix can be computed on specified interval + lag
    start_time = time.time()
    for t in range(rolling_cov + 1, T_tot):
        # only rebalance on the specified dates
        if all_dates[t] in rebalance_dates:
            start, end = t - rolling_cov - 1, t - 1
            date_start, date_end, date_rebal = all_dates[start], all_dates[end], all_dates[t]

            print('Rebalance: {}, Covariance Matrix: start {} -  end {}'.format(
                date_rebal.strftime('%b %d, %Y'), date_start.strftime('%b %d, %Y'), date_end.strftime('%b %d, %Y')))

            # signal cca_option
            R = returns.values[start:end]
            X = signals.values[start:end]
            X_t = signals.values[t]


            # save the weights for each model
            for cov_method in cov_models:
                cca_models[cov_method].cca_weights(R, X)
                cca_models[cov_method].forecast(X_t)
                portfolio_weights[cov_method].loc[date_rebal] = cca_models[cov_method].w_optimal

    print("Time: %s seconds" % str(round(time.time() - start_time)))
    return portfolio_weights


