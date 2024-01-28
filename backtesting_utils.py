# copied from Matteo
# from datetime import time
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.stats import norm

# from typing import Tuple


# reportlab only for fancy PDFs

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    TableStyle, Table
    )

from forecast_utils import *

# Image,
# from smart_open import open


def _convert_to_array(x):
    v = np.asanyarray(x)
    v = v[np.isfinite(v)]
    return v


TRADING_DAY_MSG = 'Trading days needs to be > 0'


def annual_return(returns, price_rets='price', trading_days=252):
    '''
    Computing the average compounded return (yearly)
    '''
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns)
    n_years = v.size / float(trading_days)
    if price_rets == 'strategy_returns':
        return (np.prod((1. + v) ** (1. / n_years)) - 1. if v.size > 0 else np.nan)
    else:
        return (np.sum(v) * (1. / n_years) if v.size > 0 else np.nan)


def annual_volatility(returns, trading_days=252):
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns)
    return (np.std(v) * np.sqrt(trading_days) if v.size > 0 else np.nan)


def value_at_risk(returns, horizon=10, pctile=0.99, mean_adj=False):
    assert horizon > 1, 'horizon>1'
    assert pctile < 1
    assert pctile > 0, 'pctile in [0,1]'
    v = _convert_to_array(returns)
    stdev_mult = norm.ppf(pctile)  # i.e., 1.6449 for 0.95, 2.326 for 0.99
    if mean_adj:
        gains = annual_return(returns, 'price', horizon)
    else:
        gains = 0

    return (np.std(v) * np.sqrt(horizon) * stdev_mult - gains if v.size > 0 else np.nan)


def sharpe_ratio(returns, risk_free=0., trading_days=252):
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns - risk_free)
    return (np.mean(v) / np.std(v) * np.sqrt(trading_days) if v.size > 0 else np.nan)


def max_drawdown(returns, price_rets='price'):
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    if price_rets == 'strategy_returns':
        cumret = np.concatenate(([100.], 100. * np.cumprod(1. + v)))
        maxret = np.fmax.accumulate(cumret)
        return np.nanmin((cumret - maxret) / maxret)
    else:
        cumret = np.concatenate(([1.], np.cumsum(v)))
        maxret = np.fmax.accumulate(cumret)
        return np.nanmin(cumret - maxret)


def max_drawdown_length(returns, price_rets='price'):
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    drawndown = np.zeros(len(v) + 1)
    dd_dict = dict()
    if price_rets == 'strategy_returns':
        cumret = np.concatenate(([100.], 100. * np.cumprod(1. + v)))
        maxret = np.fmax.accumulate(cumret)
        drawndown[(cumret - maxret) / maxret < 0] = 1
    else:
        cumret = np.concatenate(([1.], np.cumsum(v)))  # start at one? no matter
        maxret = np.fmax.accumulate(cumret)
        drawndown[(cumret - maxret) < 0] = 1

    f = np.frompyfunc((lambda x, y: (x + y) * y), 2, 1)
    run_lengths = f.accumulate(drawndown, dtype='object').astype(int)

    trough_position = np.argmin(cumret - maxret)
    peak_to_trough = run_lengths[trough_position]

    next_peak_rel_position = np.argmin(run_lengths[trough_position:])
    next_peak_position = next_peak_rel_position + trough_position

    if run_lengths[next_peak_position] > 0:  # We are probably still in DD
        peak_to_peak = np.nan
    else:
        peak_to_peak = run_lengths[next_peak_position - 1]
        # run_lengths just before it hits 0 (back to peak) is the
        # total run_length of that DD period.

    longest_dd_length = max(run_lengths)  # longest, not nec deepest

    dd_dict['peak_to_trough_maxdd'] = peak_to_trough
    dd_dict['peak_to_peak_maxdd'] = peak_to_peak
    dd_dict['peak_to_peak_longest'] = longest_dd_length
    return dd_dict


def calmar_ratio(returns, price_rets='price', trading_days=252):
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    maxdd = max_drawdown(v, price_rets)
    if np.isnan(maxdd):
        return np.nan
    annret = annual_return(v, price_rets, trading_days=trading_days)
    return annret / np.abs(maxdd)


def stability_of_timeseries(returns, price_rets='price'):
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    if price_rets == 'strategy_returns':
        v = np.cumsum(np.log1p(v))
    else:
        v = np.cumsum(v)
    lin_reg = linregress(np.arange(v.size), v)
    return lin_reg.rvalue ** 2


def omega_ratio(returns, risk_free=0., target_return=0., trading_days=252):
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    return_thresh = (1. + target_return) ** (1. / trading_days) - 1.
    v = v - risk_free - return_thresh
    numer = np.sum(v[v > 0.])
    denom = -np.sum(v[v < 0.])
    return (numer / denom if denom > 0. else np.nan)


def sortino_ratio(returns, target_return=0., trading_days=252):
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    v = v - target_return
    downside_risk = np.sqrt(np.mean(np.square(np.clip(v, np.NINF, 0.))))
    return np.mean(v) * np.sqrt(trading_days) / downside_risk


def tail_ratio(returns):
    v = _convert_to_array(returns)
    if v.size > 0:
        try:
            return np.abs(np.percentile(v, 95.)) / np.abs(np.percentile(v, 5.))
        except FloatingPointError:
            return np.nan
    else:
        return np.nan


def common_sense_ratio(returns):
    # This cannot be compared with pyfolio routines because they implemented a
    # wrong formula CSR = Tail Ratio * Gain-to-Pain Ratio
    # and Gain-to-Pain Raio = Sum(Positive R) / |Sum(Negative R)|
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    return tail_ratio(returns) * np.sum(v[v > 0.]) / np.abs(np.sum(v[v < 0.]))


class PerformanceReport(object):
    def __init__(self, yearly_from_list=None, yearly_list=None, rolling_lookback_list=None, calendar=None):
        if yearly_from_list is None:
            yearly_from_list = ["2010", "2012", "2014", "2016"]
        self.yearly_from_list = yearly_from_list
        if yearly_list is None:
            yearly_list = ["2010", "2011", "2012", "2013", "2014",
                           "2015", "2016", "2017", "2018", "2019", "2020",
                           "2021", "2022", "2023"]
        self.yearly_list = yearly_list
        if rolling_lookback_list is None:
            rolling_lookback_list = [2000, 1500, 1000, 500, 250]
            # approx 1y, 2y, 4y, 6y, 8y lookbacks
        self.rolling_lookback_list = rolling_lookback_list
        self.calendar = calendar
        self.report_dict = dict()

    @staticmethod
    def _convert_to_array(x):
        v = np.asanyarray(x)
        if v.size > 0:
            v = v[np.isfinite(v)]
        return v

    @staticmethod
    def _check_if_inf(x):
        return x if not np.isinf(x) else np.nan

    def _annual_return(self, returns, price_rets='price', trading_days=252):
        '''
        Computing the average compounded return (yearly)
        '''
        assert trading_days > 0, TRADING_DAY_MSG
        v = self._convert_to_array(returns)
        n_years = v.size / float(trading_days)
        if price_rets == 'strategy_returns':
            return (np.prod((1. + v) ** (1. / n_years)) - 1. if v.size > 0 else np.nan)
        else:
            return (np.sum(v) * (1. / n_years) if v.size > 0 else np.nan)

    def _annual_volatility(self, returns, trading_days=252):
        assert trading_days > 0, TRADING_DAY_MSG
        v = self._convert_to_array(returns)
        return (np.std(v) * np.sqrt(trading_days) if v.size > 0 else np.nan)

    def _normal_value_at_risk(self, returns, horizon=10, pctile=0.99, mean_adj=False):
        assert horizon > 1, 'horizon>1'
        assert pctile < 1
        assert pctile > 0, 'pctile in [0,1]'
        v = self._convert_to_array(returns)
        stdev_mult = norm.ppf(pctile)  # i.e., 1.6449 for 0.95, 2.326 for 0.99
        if mean_adj:
            gains = self._annual_return(returns, 'price', horizon)
        else:
            gains = 0

        return (np.std(v) * np.sqrt(horizon) * stdev_mult - gains if v.size > 0 else np.nan)

    def _sharpe_ratio(self, returns, risk_free=0., trading_days=252):
        assert trading_days > 0, TRADING_DAY_MSG
        v = self._convert_to_array(returns - risk_free)
        if v.size == 0 or np.std(v) == 0:
            return np.nan
        return (np.mean(v) / np.std(v) * np.sqrt(trading_days) if v.size > 0 else np.nan)

    def _max_drawdown(self, returns, price_rets='price'):
        v = self._convert_to_array(returns)
        if v.size == 0:
            return np.nan
        if price_rets == 'strategy_returns':
            cumret = np.concatenate(([100.], 100. * np.cumprod(1. + v)))
            maxret = np.fmax.accumulate(cumret)
            return np.nanmin((cumret - maxret) / maxret)
        else:
            cumret = np.concatenate(([1.], np.cumsum(v)))
            maxret = np.fmax.accumulate(cumret)
            return np.nanmin(cumret - maxret)

    def _max_drawdown_length(self, returns, price_rets='price'):
        v = self._convert_to_array(returns)
        if v.size == 0:
            return np.nan
        drawndown = np.zeros(len(v) + 1)
        dd_dict = dict()
        if price_rets == 'strategy_returns':
            cumret = np.concatenate(([100.], 100. * np.cumprod(1. + v)))
            maxret = np.fmax.accumulate(cumret)
            drawndown[(cumret - maxret) / maxret < 0] = 1
        else:
            cumret = np.concatenate(([1.], np.cumsum(v)))  # start at one? no matter
            maxret = np.fmax.accumulate(cumret)
            drawndown[(cumret - maxret) < 0] = 1

        f = np.frompyfunc((lambda x, y: (x + y) * y), 2, 1)
        run_lengths = f.accumulate(drawndown, dtype='object').astype(int)

        trough_position = np.argmin(cumret - maxret)
        peak_to_trough = run_lengths[trough_position]

        next_peak_rel_position = np.argmin(run_lengths[trough_position:])
        next_peak_position = next_peak_rel_position + trough_position

        if run_lengths[next_peak_position] > 0:  # We are probably still in DD
            peak_to_peak = np.nan
        else:
            peak_to_peak = run_lengths[next_peak_position - 1]
            # run_lengths just before it hits 0 (back to peak) is the
            # total run_length of that DD period.

        longest_dd_length = max(run_lengths)  # longest, not nec deepest

        dd_dict['peak_to_trough_maxdd'] = peak_to_trough
        dd_dict['peak_to_peak_maxdd'] = peak_to_peak
        dd_dict['peak_to_peak_longest'] = longest_dd_length
        return dd_dict

    def _common_sense_ratio(self, returns):
        # This cannot be compared with pyfolio routines because they implemented a
        # wrong formula CSR = Tail Ratio * Gain-to-Pain Ratio
        # and Gain-to-Pain Raio = Sum(Positive R) / |Sum(Negative R)|
        v = self._convert_to_array(returns)
        if v.size == 0:
            return np.nan
        return tail_ratio(returns) * np.sum(v[v > 0.]) / np.abs(np.sum(v[v < 0.]))

    def _calmar_ratio(self, returns, price_rets='price', trading_days=252):
        assert trading_days > 0, TRADING_DAY_MSG
        v = self._convert_to_array(returns)
        if v.size == 0:
            return np.nan
        maxdd = self._max_drawdown(v, price_rets)
        if np.isnan(maxdd):
            return np.nan
        annret = self._annual_return(v, price_rets, trading_days=trading_days)
        return annret / np.abs(maxdd)

    def _stability_of_timeseries(self, returns, price_rets='price'):
        v = self._convert_to_array(returns)
        if v.size == 0:
            return np.nan
        if price_rets == 'strategy_returns':
            v = np.cumsum(np.log1p(v))
        else:
            v = np.cumsum(v)
        lin_reg = linregress(np.arange(v.size), v)
        return lin_reg.rvalue ** 2

    def _omega_ratio(self, returns, risk_free=0., target_return=0., trading_days=252):
        assert trading_days > 0, TRADING_DAY_MSG
        v = self._convert_to_array(returns)
        if v.size == 0:
            return np.nan
        return_thresh = (1. + target_return) ** (1. / trading_days) - 1.
        v = v - risk_free - return_thresh
        numer = np.sum(v[v > 0.])
        denom = -np.sum(v[v < 0.])
        return (numer / denom if denom > 0. else np.nan)

    def _sortino_ratio(self, returns, target_return=0., trading_days=252):
        assert trading_days > 0, TRADING_DAY_MSG
        v = self._convert_to_array(returns)
        if v.size == 0:
            return np.nan
        v = v - target_return
        downside_risk = np.sqrt(np.mean(np.square(np.clip(v, np.NINF, 0.))))
        if downside_risk == 0:
            return np.nan
        return np.mean(v) * np.sqrt(trading_days) / downside_risk

    # import warnings

    def _tail_ratio(self, returns):
        v = self._convert_to_array(returns)
        if v.size > 0:
            # TODO: This throws errors SOME of the time, but is never used. catch all errors/warnings
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val = np.abs(np.percentile(v, 95.)) / np.abs(np.percentile(v, 5.))
            except FloatingPointError:
                val = np.nan
            except TypeError:
                val = np.nan
            # except:
            #     val = np.nan
        else:
            val = np.nan
        return val

    '''
    Auction bdays <> Real bdays
    Range	actual count	    auction_bdays	approx real bdays
    2010-17	1022	    1000	        2090
    2012-17	768	        750	            1560
    2014-17	507	        500	            1050
    2016-17	253	        250	            500
    all rolling measures use auction_bdays and may need adjustment if
    using real bdaay cacl
    Note: Will have to change names sr_1000_roll and change in strategy_allocator calculations
    (eg log_max_drawdown_1000_roll). Wen we reindex by bdays, change to 2000, 1500,1000,500
    '''

    def select_year_return(self, yr, yr_from=True):
        if yr in self.return_series.index:
            if yr_from:
                return self.return_series[yr:]
            else:
                return self.return_series[yr]
        else:
            return pd.Series()

    def run_analysis(self, returns_series: pd.Series):
        trading_day_series = returns_series.copy()
        if self.calendar is not None:
            returns_series = returns_series.reindex(self.calendar).fillna(0)
        self.return_series = returns_series
        self.trading_day_returns = trading_day_series
        v = _convert_to_array(returns_series)
        self.sr0 = (np.nan if v.size > 0 or np.std(v) == 0 else
                    np.mean(v) / np.std(v)
                    * np.sqrt(252))

        self.sr0 = self._check_if_inf(self.sr0)
        self.trading_days = (returns_series.map(np.abs) > 0).sum() / (returns_series.count())
        self.sr = self._check_if_inf(self._sharpe_ratio(returns_series))
        self.yearly_from_sr = {yr: self._check_if_inf(self._sharpe_ratio(self.select_year_return(yr, yr_from=True)))
                               for yr in self.yearly_from_list}
        self.yearly_sr = {yr: self._check_if_inf(self._sharpe_ratio(self.select_year_return(yr, yr_from=False)))
                          for yr in self.yearly_list}
        self.trading_sr = self._check_if_inf(self._sharpe_ratio(self.trading_day_returns))
        # TODO: Put in Trading_Day SR calculations approx= SR/sqrt(trading_day_frac)

        # self.yearly_from_trading_sr = {    yr: self._check_if_inf(self._sharpe_ratio(self.select_year_return(yr, yr_from=True)))
        #             for yr in self.yearly_from_list}
        # self.yearly_sr = {yr: self._check_if_inf(self._sharpe_ratio(self.select_year_return(yr, yr_from=False)))
        #                   for yr in self.yearly_list}
        self.trailing_sr = {window: self._check_if_inf(self._sharpe_ratio(returns_series.iloc[-window:]))
                            for window in self.rolling_lookback_list}
        self.annual_return = self._annual_return(returns_series)
        self.yearly_from_annual_ret = {yr: self._annual_return(self.select_year_return(yr, yr_from=True))
                                       for yr in self.yearly_from_list}
        self.yearly_annual_ret = {yr: self._annual_return(self.select_year_return(yr, yr_from=False)) for yr in
                                  self.yearly_list}
        self.trailing_annual_ret = {window: self._annual_return(returns_series.iloc[-window:])
                                    for window in self.rolling_lookback_list}
        self.calmar_ratio = self._check_if_inf(self._calmar_ratio(returns_series))
        self.yearly_from_calmar = {yr: self._check_if_inf(self._calmar_ratio(self.select_year_return(yr, yr_from=True)))
                                   for yr in
                                   self.yearly_from_list}
        self.yearly_calmar = {yr: self._check_if_inf(self._calmar_ratio(self.select_year_return(yr, yr_from=False))) for
                              yr in
                              self.yearly_list}
        self.trailing_calmar = {window: self._check_if_inf(self._calmar_ratio(returns_series.iloc[-window:]))
                                for window in self.rolling_lookback_list}
        self.max_drawdown = self._max_drawdown(returns_series)
        self.yearly_from_max_drawdown = {yr: self._max_drawdown(self.select_year_return(yr, yr_from=True)) for yr in
                                         self.yearly_from_list}
        self.yearly_max_drawdown = {yr: self._max_drawdown(self.select_year_return(yr, yr_from=False)) for yr in
                                    self.yearly_list}
        self.trailing_max_drawdown = {window: self._max_drawdown(returns_series.iloc[-window:])
                                      for window in self.rolling_lookback_list}
        self.dd_analysis = self._max_drawdown_length(returns_series)

        self.normvar_99pct_10day = self._normal_value_at_risk(returns_series, horizon=10,
                                                              pctile=0.99)
        self.normvar_99pct_10day_drift_adj = self._normal_value_at_risk(returns_series, horizon=10,
                                                                        pctile=0.99, mean_adj=True)
        self.stability_of_timeseries = self._stability_of_timeseries(returns_series)
        self.sortino_ratio = self._check_if_inf(self._sortino_ratio(returns_series))
        self.omega_ratio = self._check_if_inf(self._omega_ratio(returns_series))
        self.tail_ratio = self._check_if_inf(self._tail_ratio(returns_series))
        self.volatility = self._annual_volatility(returns_series)
        self.risk = self.volatility ** 2.0

    def run_report(self) -> None:
        # digit_precision = lambda x: '%5.3f' % x
        # dollar_precision = lambda x: '%' \
        # format = {'trading_days': '{:5.3f}',
        # TODO: put in fixed precision for digits vs dollars (note--output is dict)
        ret_dict = dict()
        ret_dict['trading_days'] = self.trading_days
        ret_dict['sr'] = self.sr
        sharpe_from = {'sr_{}+'.format(yr): self.yearly_from_sr[yr] for yr in self.yearly_from_list}
        ret_dict.update(sharpe_from)
        sharpe_yr = {'sr_{}'.format(yr): self.yearly_sr[yr] for yr in self.yearly_list}
        ret_dict.update(sharpe_yr)
        trail_sharpes = {'sr_' + str(wind) + '_roll': self.trailing_sr[wind] for wind in self.rolling_lookback_list}
        ret_dict.update(trail_sharpes)
        ret_dict['calmar_ratio'] = self.calmar_ratio
        calmar_from = {'calmar_{}+'.format(yr): self.yearly_from_calmar[yr] for yr in self.yearly_from_list}
        ret_dict.update(calmar_from)
        calmar_yr = {'calmar_{}'.format(yr): self.yearly_calmar[yr] for yr in self.yearly_list}
        ret_dict.update(calmar_yr)
        trail_calmars = {'calmar_' + str(wind) + '_roll': self.trailing_calmar[wind] for wind in
                         self.rolling_lookback_list}
        ret_dict.update(trail_calmars)
        ret_dict['annual_return'] = self.annual_return
        annual_from = {'annual_ret_{}+'.format(yr): self.yearly_from_annual_ret[yr] for yr in self.yearly_from_list}
        ret_dict.update(annual_from)
        annual_yr = {'annual_ret_{}'.format(yr): self.yearly_annual_ret[yr] for yr in self.yearly_list}
        ret_dict.update(annual_yr)

        trail_annuals = {'annual_ret_' + str(wind) + '_roll': self.trailing_annual_ret[wind] for wind in
                         self.rolling_lookback_list}
        ret_dict.update(trail_annuals)
        ret_dict['NormalVaR_99pct_10day'] = self.normvar_99pct_10day
        ret_dict['NormalVaR_99pct_10day_drift_adj'] = self.normvar_99pct_10day_drift_adj

        ret_dict['stability_of_timeseries'] = self.stability_of_timeseries
        ret_dict['sortino_ratio'] = self.sortino_ratio
        ret_dict['omega_ratio'] = self.omega_ratio
        ret_dict['tail_ratio'] = self.tail_ratio
        ret_dict['max_drawdown'] = self.max_drawdown
        ret_dict['volatility'] = self.volatility
        ret_dict['risk'] = self.risk
        max_dd_from = {'max_drawdown_{}+'.format(yr): self.yearly_from_max_drawdown[yr] for yr in self.yearly_from_list}
        ret_dict.update(max_dd_from)
        max_dd_yr = {'max_drawdown_{}'.format(yr): self.yearly_max_drawdown[yr] for yr in self.yearly_list}
        ret_dict.update(max_dd_yr)
        trail_max_dds = {'max_drawdown_' + str(wind) + '_roll': self.trailing_max_drawdown[wind] for wind in
                         self.rolling_lookback_list}
        ret_dict.update(trail_max_dds)
        ret_dict.update(self.dd_analysis)
        sr = self.sr0
        self.report_dict = ret_dict

    def format_csv(self):
        copy_dict = self.report_dict.copy()
        copy_dict.update({x: np.round(copy_dict[x], 2) for x in copy_dict.keys() if x.startswith('ann')})
        copy_dict.update({x: np.round(copy_dict[x], 2) for x in copy_dict.keys() if x.startswith('max')})
        copy_dict.update({x: np.round(copy_dict[x], 4) for x in copy_dict.keys() if x.startswith('sr')})
        copy_dict.update({x: np.round(copy_dict[x], 4) for x in copy_dict.keys() if x.startswith('calmar')})
        misc_measures = ['omega_ratio', 'sortino_ratio', 'tail_ratio', 'stability_of_timeseries']
        copy_dict.update({x: np.round(copy_dict[x], 4) for x in copy_dict.keys() if x in misc_measures})
        copy_dict.update({'trading_days': np.round(copy_dict['trading_days'], 2)})
        self.formatted_report_dict = copy_dict

    def pdf_report(self, directory_path="./output_dir/Output_554/"):  # aka Brady Report
        # TODO: PDF REport needs updating - how to make it dynamic as we add more fields?
        ret_dict = self.report_dict
        doc = SimpleDocTemplate(directory_path + "returns_report.pdf", pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)

        Story = []

        formatted_time = time.ctime()
        full_name = "Nick Firoozye"
        address_parts = ["Exos Financial", "New York, NY"]

        styles = getSampleStyleSheet()
        orden = ParagraphStyle(name='Justify', alignment=TA_JUSTIFY)
        orden.leading = 14
        orden.borderPadding = 10
        orden.backcolor = colors.grey
        orden.fontsize = 12
        styles.add(orden)
        ptext = '<font size=12>%s</font>' % formatted_time
        Story.append(Paragraph('Performance Report - Auction Strategy', styles['Title']))
        Story.append(Paragraph(ptext, styles["Normal"]))
        Story.append(Spacer(1, 12))
        # directory names as identifier of which run it was
        Story.append(Paragraph(directory_path.replace('output_dir/', ''), styles["Normal"]))
        Story.append(Spacer(1, 12))

        # data = [
        #     ["Activity", "Times/wk", "Time of day", "Description"],
        #     ["B", "01", "ABCD", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"],
        #     ["E", "02", "CDEF", "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"],
        #     ["E", "03", "SDFSDF", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"],
        #     ["e", "04", "SDFSDF", "DDDDDDDDDDDDDDDDDDDDDDDD DDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"],
        #     ["x", "05", "GHJGHJGHJ", "EEEEEEEEEEEEEE EEEEEEEEEEEEEEEEE EEEEEEEEEEEEEEEEEEEE"],
        #     ]

        # files = ['real_vs_gross.jpg', 'pnl_decomp.jpg', 'allocations.jpg']
        # explanations = ['Realized Gains vs Gross (Leveraged) Capital Allocation (initially $250m)',
        #                 'Decomposed PnL from Price Gains (Modeling Edge), Accrued Interest, Funding Costs, '
        #                 'and Transaction Costs',
        #                 'Realized Gains in Red vs Gross allocations in $s to each ONTR']
        #
        # file_dict = dict()
        # for filename, explain in zip(files, explanations):
        #     file_dict[filename] = open(directory_path + filename, "rb")
        #
        #     im = Image(file_dict[filename], 6 * inch, 4 * inch)
        #     # im._restrictSize( 6*inch, 6*inch)
        #     Story.append(im)
        #     # ptext = '''
        #     # <para> .{}  </para><br/>
        #     #     '''.format(explain)
        #
        #     # <para> . </para> Some more test Text
        #     ptext = '''<font size=12> {} </font>'''.format(explain)
        #     Story.append(Paragraph(ptext, styles["Justify"]))
        # for file_link in file_dict.values():
        #     if not file_link.closed:
        #         file_link.close()
        data = [
            ["Metric", "Measure"],
            ['Sharpe Ratio ', "{:10.4f}".format(ret_dict['sr'])],
            # TODO: Automate formatting for each category (in case different yearly, yearly_from and roll params given)
            ['Since 2010 Sharpe Ratio ', "{:10.4f}".format(ret_dict['sr_2010+'])],
            ['Since 2012 Sharpe Ratio ', "{:10.4f}".format(ret_dict['sr_2012+'])],
            ['Since 2014 Sharpe Ratio ', "{:10.4f}".format(ret_dict['sr_2014+'])],
            ['Since 2016 Sharpe Ratio ', "{:10.4f}".format(ret_dict['sr_2016+'])],
            ['2016 Sharpe Ratio ', "{:10.4f}".format(ret_dict['sr_2016'])],
            ['2016 Sharpe Ratio ', "{:10.4f}".format(ret_dict['sr_2017'])],
            ['2016 Sharpe Ratio ', "{:10.4f}".format(ret_dict['sr_2018'])],
            ['2019 Sharpe Ratio ', "{:10.4f}".format(ret_dict['sr_2019'])],
            ['Annual PnL', "${:,.2f}".format(ret_dict['annual_return'])],
            ['Since 2010 Annual PnL ', "${:,.2f}".format(ret_dict['annual_ret_2010+'])],
            ['Since 2012 Annual PnL ', "${:,.2f}".format(ret_dict['annual_ret_2012+'])],
            ['Since 2014 Annual PnL ', "${:,.2f}".format(ret_dict['annual_ret_2014+'])],
            ['Since 2016 Annual PnL ', "${:,.2f}".format(ret_dict['annual_ret_2016+'])],
            ['2016 Annual PnL ', "${:,.2f}".format(ret_dict['annual_ret_2016'])],
            ['2017 Annual PnL ', "${:,.2f}".format(ret_dict['annual_ret_2017'])],
            ['2018 Annual PnL ', "${:,.2f}".format(ret_dict['annual_ret_2018'])],
            ['2019 Annual PnL ', "${:,.2f}".format(ret_dict['annual_ret_2019'])],
            ['Calmar', "{:10.4f}".format(ret_dict['calmar_ratio'])],
            ['Since 2010 Calmar Ratio ', "{:10.4f}".format(ret_dict['calmar_2010+'])],
            ['Since 2012 Calmar Ratio ', "{:10.4f}".format(ret_dict['calmar_2012+'])],
            ['Since 2014 Calmar Ratio ', "{:10.4f}".format(ret_dict['calmar_2014+'])],
            ['Since 2016 Calmar Ratio ', "{:10.4f}".format(ret_dict['calmar_2016+'])],
            ['2016 Calmar Ratio ', "{:10.4f}".format(ret_dict['calmar_2016'])],
            ['2017 Calmar Ratio ', "{:10.4f}".format(ret_dict['calmar_2017'])],
            ['2018 Calmar Ratio ', "{:10.4f}".format(ret_dict['calmar_2018'])],
            ['2019 Calmar Ratio ', "{:10.4f}".format(ret_dict['calmar_2019'])],
            ['Sortino', "{:10.4f}".format(ret_dict['sortino_ratio'])],
            ['Tail Ratio', "{:10.4f}".format(ret_dict['tail_ratio'])],
            ['Omega Ratio ', "{:10.4f}".format(ret_dict['omega_ratio'])],
            ['Stability of Time Series ', "{:10.4f}".format(ret_dict['stability_of_timeseries'])],
            ['Volatility', "{:.2e}".format(ret_dict['volatility'])],
            ['10-day 99th Percentile VaR (normal)', "${:,.2f}".format(ret_dict['NormalVaR_99pct_10day'])],
            ['10-day 99th Percentile VaR (normal-drift adjusted)',
             "${:,.2f}".format(ret_dict['NormalVaR_99pct_10day_drift_adj'])],
            ['Max_DD ', "$({:,.2f})".format(-1 * ret_dict['max_drawdown'])],
            ['Since 2010 Max_DD ', "$({:,.2f})".format(-1 * ret_dict['max_drawdown_2010+'])],
            ['Since 2012 Max_DD ', "$({:,.2f})".format(-1 * ret_dict['max_drawdown_2012+'])],
            ['Since 2014 Max_DD ', "$({:,.2f})".format(-1 * ret_dict['max_drawdown_2014+'])],
            ['Since 2016 Max_DD ', "$({:,.2f})".format(-1 * ret_dict['max_drawdown_2016+'])],
            ['2016 Max_DD ', "$({:,.2f})".format(-1 * ret_dict['max_drawdown_2016'])],
            ['2017 Max_DD ', "$({:,.2f})".format(-1 * ret_dict['max_drawdown_2017'])],
            ['2018 Max_DD ', "(${:,.2f})".format(-1 * ret_dict['max_drawdown_2018'])],
            ['2019 Max_DD ', "(${:,.2f})".format(-1 * ret_dict['max_drawdown_2019'])],
            ['Max_DD_length (peak-to-trough, days)', "{}".format(ret_dict['peak_to_trough_maxdd'])],
            ['Max_DD_length (peak-to-peak, days)', "{}".format(ret_dict['peak_to_peak_maxdd'])],
            ['Longest_DD_length (peak-to-peak, days)', "{}".format(ret_dict['peak_to_peak_longest'])],

        ]

        style = TableStyle(
            [
                ('FONTNAME', (0, 0), (-1, 0), 'Courier-Bold'),
                ('ALIGN', (1, 1), (-2, -2), 'RIGHT'),
                ('TEXTCOLOR', (1, 1), (-2, -2), colors.red),
                ('VALIGN', (0, 0), (0, -1), 'TOP'),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.blue),
                ('ALIGN', (0, -1), (-1, -1), 'CENTER'),
                ('VALIGN', (0, -1), (-1, -1), 'MIDDLE'),
                ('TEXTCOLOR', (0, -1), (-1, -1), colors.green),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
            ])

        s = getSampleStyleSheet()
        s = s["BodyText"]
        s.wordWrap = 'CJK'
        data2 = [[Paragraph(cell, s) for cell in row] for row in data]
        t = Table(data2)
        t.setStyle(style)

        Story.append(t)

        # ptext = '<font size=12>I wanted to code ever since I was young, and
        # have done mods for videogames I liked and played. It’s been my childhood
        # dream to create worlds and stories, and coding is the best way to script events
        # the way you want them to go. While this may not be exactly what I’m looking for,
        # I think it’ll give me an accurate experience of what that life is like.</font>'
        # Story.append(Paragraph(ptext, styles["Justify"]))

        doc.build(Story)

        return


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
        "D": D,
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


