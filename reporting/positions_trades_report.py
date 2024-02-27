from typing import Any, Callable

import numpy as np
import pandas as pd

from constants import TERM_ORDERING_DICT
from trade_allocation.trade_allocator import discrete_trade_holdings_sizes


def create_risk_report(total_holdings, optimizer_parameters, control):
    # total_holdings.loc()
    discretize = lambda y: 'discrete_' + y if y.startswith('net') else y
    level_1_cols_old = ['term', 'net_bond_posn_usd', 'net_bond_margin_usd', 'future_ticker', 'net_futures_posn_usd',
                        'net_futures_margin_usd', 'dv01', 'fut_dv01', 'yield', 'bid', 'ask', 'fut_bid', 'fut_ask']
    level_1_cols = [discretize(x) for x in level_1_cols_old]
    old_names = {
        'net_bond_posn_usd': 'Bond Position (USD)', 'net_bond_margin_usd': 'Bond Margin (USD)',
        'term': 'Term', 'future_ticker': 'Future Ticker', 'net_futures_posn_usd': 'Futures Position (USD)',
        'net_futures_margin_usd': 'Futures Margin (USD)'}
    new_names = {discretize(x[0]): x[1] for x in old_names.items()}
    term_map_dict = {x: x[:-5] + 'y' for x in control.SELECT_TERMS}

    def fut_map(tickers):
        if isinstance(tickers, str):
            # tickers are normalized - remove 2 digit date + month
            return tickers[:-3] + '1'
        else:
            return ''

    #  US1, FV1, TY1, WN1, TU1, etc

    def term_map(terms):
        if isinstance(terms, str):
            return term_map_dict[terms]
        else:
            return ''

    temp_report = total_holdings.loc[:, pd.IndexSlice[:, level_1_cols]]
    temp_report = temp_report.rename(columns=new_names, level=1)
    temp_report.loc[:, pd.IndexSlice[:, 'Future Ticker']] = temp_report.loc[:,
                                                            pd.IndexSlice[:, 'Future Ticker']].applymap(fut_map)
    temp_report.loc[:, pd.IndexSlice[:, 'Term']] = temp_report.loc[:,
                                                   pd.IndexSlice[:, 'Term']].applymap(term_map)

    temp_report.loc[:, pd.IndexSlice['Total', 'Futures Margin (USD)']] = temp_report.loc[:,
                                                                         pd.IndexSlice[:,
                                                                         'Futures Margin (USD)']].sum(axis=1)
    temp_report.loc[:, pd.IndexSlice['Total', 'Bond Margin (USD)']] = temp_report.loc[:,
                                                                      pd.IndexSlice[:,
                                                                      'Bond Margin (USD)']].sum(axis=1)

    return temp_report


def create_tear_sheet(total_holdings, as_of_date, optimizer_parameters, control):
    futures_contract_sizes = optimizer_parameters['ALL']['FUTURES_CONTRACT_SIZES']
    bond_size = optimizer_parameters['ALL']['BOND_MINIMUM_SIZE']
    SELECT_TERMS = control.SELECT_TERMS
    day_before_index = list(total_holdings.index).index(as_of_date) - 1
    day_before = list(total_holdings.index)[day_before_index]

    as_of_date_data = total_holdings.loc[as_of_date, :]
    as_of_date_data.index = as_of_date_data.index.swaplevel(0, 1)
    as_of_date_data = as_of_date_data.sort_index()

    day_before_data = total_holdings.loc[day_before, :]
    day_before_data.index = day_before_data.index.swaplevel(0, 1)
    day_before_data = day_before_data.sort_index()

    # term_data = []
    # for tm in SELECT_TERMS:
    #     term_data.append(tm)
    #     term_data.append(' ')
    # term_data.append('Totals (Net)')
    # term_data.append('Totals (Gross)')
    # term_column = pd.Series(term_data)
    # term_column.name = 'TERM'
    # TODO: Put in reporting module ? Move one way or another
    def extract_and_stack(multi_series: pd.DataFrame,
                          bond_column: str,
                          futures_column: str,
                          col_name: str) -> pd.Series:

        if bond_column is not None:
            bond_data = multi_series.loc[bond_column]
            bond_data.name = 'bonds'
        else:
            temp = multi_series.loc['net_bond_posn_usd']
            bond_data = pd.Series([''] * len(temp), index=temp.index, name='bonds')

        bond_data.index = bond_data.index.map(lambda x: TERM_ORDERING_DICT[x])

        if futures_column is not None:
            fut_data = multi_series.loc[futures_column]
            fut_data.name = 'futures'
        else:
            temp = multi_series.loc['net_futures_posn_usd']
            fut_data = pd.Series([''] * len(temp), index=temp.index, name='futures')
        fut_data.index = fut_data.index.map(lambda x: TERM_ORDERING_DICT[x])
        all_data = pd.concat([bond_data, fut_data], axis=1).unstack()
        all_data.index = all_data.index.swaplevel(0, 1)
        all_data = all_data.sort_index()
        all_data.name = col_name
        return all_data

    ignore_str: Callable[[Any], Any] = lambda x: np.nan if isinstance(x, str) else x
    int_ignore_str: Callable[[Any], str | int] = lambda x: '' if np.isnan(x) else round(x)
    zero_to_str = lambda x: '' if x == 0 else x

    yesterday_notional_col = extract_and_stack(day_before_data, 'net_bond_posn_usd',
                                               'net_futures_posn_usd', 'Notional (t-1)')
    yesterday_notional_col = yesterday_notional_col.map(ignore_str).map(int_ignore_str).map(zero_to_str)
    yesterday_discrete_notional_col = extract_and_stack(day_before_data, 'discrete_net_bond_posn_usd',
                                                        'discrete_net_futures_posn_usd', 'Discrete Notional (t-1)')
    todays_notional_col = extract_and_stack(as_of_date_data, 'net_bond_posn_usd',
                                            'net_futures_posn_usd', 'Notional(t)')
    todays_notional_col = todays_notional_col.map(ignore_str).map(int_ignore_str).map(zero_to_str)
    todays_discrete_notional_col = extract_and_stack(as_of_date_data, 'discrete_net_bond_posn_usd',
                                                     'discrete_net_futures_posn_usd', 'Discrete Notional(t)')
    ticker_col = extract_and_stack(as_of_date_data, 'cusip', 'future_ticker', 'Cusip / Ticker')

    # TODO: contract x[:2] is NOT robust! CREATE Contract size/Bond Trade methods!
    fut_contract_sizes = ticker_col.map(lambda x: futures_contract_sizes.get(x[:2], '') if isinstance(x, str) else '')

    bond_trades_col = extract_and_stack(as_of_date_data, 'net_bond_trade_usd', None, 'Bond Trades')
    discrete_bond_trades_col = extract_and_stack(as_of_date_data, 'discrete_net_bond_trade_usd', None,
                                                 'Discrete Bond Trades')
    futures_trades_col = extract_and_stack(as_of_date_data, None, 'net_futures_trade_usd', 'Futures Trades')
    discrete_futures_trades_col = extract_and_stack(as_of_date_data, None, 'discrete_net_futures_trade_usd',
                                                    'Discrete Futures Trades')
    '''
    Round futures to contract sizes and bonds to Min Notional Traded
    '''

    futures_trades_col, futures_contracts = discrete_trade_holdings_sizes(futures_trades_col.map(ignore_str),
                                                                          fut_contract_sizes.map(ignore_str),
                                                                          optimizer_parameters)

    futures_contracts = extract_and_stack(as_of_date_data, None, 'discrete_number_futures_contracts_traded',
                                          'Contracts Traded')
    # futures_contracts.name = 'Contracts Traded'
    # futures_trades_col.name = 'Futures Trades (USD)'

    # bond_trades_col, bond_units = discrete_trade_holdings_sizes(bond_trades_col.map(ignore_str),
    #                                                             bond_size,
    #                                                             optimizer_parameters)
    bond_trades_col.name = 'Bonds Traded (USD)'
    # bond_units = (bond_trades_col.map(ignore_str) / bond_size).map(int_ignore_nan)
    # bond_units.name = 'Bond Units Traded'
    # bond_trades_col = (bond_units.map(ignore_str) * bond_size).map(nan_to_str)

    bond_margin_col = extract_and_stack(as_of_date_data,
                                        'discrete_net_bond_margin_usd', None,
                                        'Bond Margin').map(ignore_str).map(int_ignore_str)
    futures_margin_col = extract_and_stack(as_of_date_data, None,
                                           'discrete_net_futures_margin_usd',
                                           'Futures Margin').map(ignore_str).map(int_ignore_str)
    todays_bid = extract_and_stack(as_of_date_data, 'price_bid', 'fut_price_bid', 'Price (Bid)').map(zero_to_str)
    todays_offer = extract_and_stack(as_of_date_data, 'price_ask', 'fut_price_ask', 'Price (Ask)').map(zero_to_str)

    # todays_hedge_ratio = as_of_date_data.loc['frozen_hedge_ratio']

    # unused     ctd_cusip    total_capital_allocation    leveraged_limit    gross_bond_limit

    tear_sheet = pd.concat([ticker_col, yesterday_discrete_notional_col, discrete_bond_trades_col,
                            discrete_futures_trades_col, futures_contracts,
                            todays_bid, todays_offer,
                            todays_discrete_notional_col, bond_margin_col, futures_margin_col], axis=1)
    tear_sheet.index.names = [as_of_date, '']
    return tear_sheet


def report_corr(combo):
    years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016"]
    corr_dict = dict()
    corr_dict['all'] = combo.corr().iloc[0, 1]  # iloc[2, -1]  # 6X6
    for yr in years:
        corr_dict[yr + '-'] = combo.loc[yr:, :].corr().iloc[0, 1]  # iloc[2, -1] # lt st total, lt_f, st_f, total_f

    return pd.Series(corr_dict)
