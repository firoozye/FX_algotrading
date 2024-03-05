# copied from Matteo
# from datetime import time

# from typing import Tuple


# reportlab only for fancy PDFs

# from reportlab.lib.units import inch

from forecasts.forecast_utils import calculate_percentiles
from reporting.reporting_utils import PerformanceReport
import pandas as pd


# Image,
# from smart_open import open

def fx_backtest(initial_amount, results_df, df, hold_enabled=False, n=None, p=None):
    results_df.dropna(inplace=True)

    if hold_enabled == False:
        signals = results_df['mean'].apply(lambda x: "BUY" if x > 0 else 'SELL')

    else:
        df_percentiles = calculate_percentiles(df, n, p)
        results_df['lower'] = df_percentiles[f'lower_{p}_percentile']
        results_df['upper'] = df_percentiles[f'upper_{p}_percentile']
        signals = results_df.apply(
            lambda row: "BUY" if row['mean'] > row['upper'] else ('SELL' if row['mean'] < row['lower'] else 'HOLD'),
            axis=1)

    df_performance = pd.DataFrame()
    df_performance.index = signals.index
    df_performance['signal'] = signals
    df_performance['Close'] = df['close']
    df_performance['forecast'] = results_df['mean']
    df_performance['target'] = results_df['actual']
    df_performance['spread'] = df['spread_close']
    dollars = initial_amount
    pounds = 0.0
    last_action = None

    # Initialize portfolio_value column with the first value as initial_dollars
    df_performance['portfolio_value'] = initial_amount

    for index, row in df_performance.iterrows():
        signal = row['signal']
        current_price = row['Close']
        spread = row['spread']

        # Wait for the first 'BUY' signal to start trading
        if last_action is None:
            if signal == 'BUY':
                pounds = dollars / (current_price + spread / 2)
                dollars = 0
                last_action = 'BUY'
            df_performance.loc[index, 'portfolio_value'] = initial_amount
            continue

        # BUY signal and no consecutive BUY
        if signal == 'BUY' and last_action != 'BUY':
            pounds = dollars / (current_price + spread / 2)
            dollars = 0
            last_action = 'BUY'

        # SELL signal and no consecutive SELL
        elif signal == 'SELL' and last_action != 'SELL':
            dollars = pounds * (current_price - spread / 2)
            pounds = 0
            last_action = 'SELL'

        # HOLD signal
        elif hold_enabled and signal == 'HOLD':
            pass

        # Update the portfolio value in the DataFrame
        df_performance.loc[index, 'portfolio_value'] = dollars + (pounds * current_price)

    df_performance['return'] = df_performance['portfolio_value'].shift(-1) / df_performance['portfolio_value'] - 1
    df_performance.dropna(inplace=True)

    return df_performance, p

def store_results(df_perf, D, ff, roll_size, n_bags, feature_num, p):
    hyperparameters_dict = {
        "no_rff": D,
        "ff": ff,
        "roll_size": roll_size,
        "n_bags": n_bags,
        "feature_num": feature_num,
        "threshold_percentile": p
    }

    returns = pd.Series(df_perf['return'])

    # Adding start and end dates to the hyperparameters dictionary
    start_date = returns.index[0].strftime('%Y-%m-%d')
    end_date = returns.index[-1].strftime('%Y-%m-%d')
    hyperparameters_dict["start_date"] = start_date
    hyperparameters_dict["end_date"] = end_date

    # Create an instance of the PerformanceReport class
    report = PerformanceReport()

    # Run analysis on the returns series
    report.run_analysis(returns)

    # Generate the report
    report.run_report()

    # Accessing the dictionary with all metrics
    performance_metrics = report.report_dict

    variables_df = pd.DataFrame(list(hyperparameters_dict.items()), columns=['Metric', 'Value'])

    # Filtering out metrics with NaN values from the performance metrics
    filtered_metrics = {metric: value for metric, value in performance_metrics.items() if pd.notna(value)}

    # Creating a DataFrame from the filtered metrics
    metrics_df = pd.DataFrame(list(filtered_metrics.items()), columns=['Metric', 'Value'])

    # Concatenating the two DataFrames
    final_df = pd.concat([variables_df, metrics_df]).reset_index(drop=True)

    directory_path = "~/Dropbox/FX/output_experiments/"

    # Constructing the filename based on the variable values and dates
    filename_stat = f"new_{start_date}_to_{end_date}_{D}_{roll_size}_{n_bags}_{feature_num}_statistics.csv"
    filename_perf = f"new_{start_date}_to_{end_date}_{D}_{roll_size}_{n_bags}_{feature_num}_performance.csv"

    # Full paths for the CSV files
    full_path_stat = directory_path + filename_stat
    full_path_perf = directory_path + filename_perf

    # Saving the DataFrames as CSV files
    final_df.to_csv(full_path_stat, index=False)
    df_perf.to_csv(full_path_perf, index=False)

    print(f"Statistics CSV file saved at: {full_path_stat}")
    print(f"Performance CSV file saved at: {full_path_perf}")


