import warnings

import numpy as np
import pandas as pd
# import constants as const
from typing import Any, Callable, Tuple
# from utils.settings import LOGGER




class TradeAllocator(object):
    SMALL_RISK_NUM = 1E-7
    SMALL_POSITION = 100

    # TODO: New TradeAllocator.reset() method to reset history
    def __init__(self, init_allocation=0.0, business_days=None, multiplier_dict=None):
        print('Initialized a TA here')
        SMALL_RISK_NUM = TradeAllocator.SMALL_RISK_NUM

        # multiplier_list = ['risk_aversion', 'funding_multiplier', 'tcost_mult', 'gross_limit']
        self.populate_multipliers(multiplier_dict)

        self.hit_limit = np.nan
        self.signals_hist = pd.DataFrame()
        if isinstance(business_days, pd.DatetimeIndex):
            business_day_list = list(business_days)
        elif isinstance(business_days, pd.Index):
            business_day_list = [pd.to_datetime(x) for x in business_days]
        else:  # if a list, expect already datetimes
            business_day_list = business_days
        self.bus_date_index = business_day_list
        self.init_allocation = init_allocation
        self.target_portfolio_wt = init_allocation  # always start there

        self.allocation_date = None

        self.risk_min = SMALL_RISK_NUM
        self.returns_ser = pd.Series(dtype=float)
        self.returns_frame = pd.DataFrame()

    def populate_multipliers(self, multiplier_dict):
        self.multiplier_dict = multiplier_dict  # save the blob
        self.alpha_multiplier = multiplier_dict['alpha_multiplier']
        self.risk_aversion = multiplier_dict['risk_aversion']
        self.funding_multiplier = multiplier_dict['funding_multiplier']
        self.tcost_multiplier = 1.0
        # tcosts are the only properly scaled element multiplier_dict['tcost_mult']
        # TODO: remove all tcost multiplier dependency
        self.gross_limit = multiplier_dict['gross_limit']
        self.scale_all = multiplier_dict.get('scale_limits', 1)

    def reset_history(self):
        print('We reset history in allocator')
        print('Reset history in allocator')
        self.returns_ser = pd.Series(dtype=float)
        self.returns_frame = pd.DataFrame()
        self.signals_hist = pd.DataFrame()
        self.target_portfolio_wt = self.init_allocation

    def __repr__(self):
        return (f'TA("alpha_mult={self.alpha_mult}", '
                f'"risk_aver={self.risk_aversion}",'
                f'"tcost_mult={self.tcost_mult}",'
                f'"funding_mult={self.funding_multiplier}",'
                f'"scale_all={self.scale_all}"')

    def extend_bus_calendar(self, new_business_date):
        if isinstance(new_business_date, str):
            new_business_date = pd.to_datetime(new_business_date)

        bus_cal = self.bus_date_index
        if not (new_business_date in bus_cal):
            bus_cal.append(new_business_date)
            bus_cal.sort()
            self.bus_date_index = bus_cal
        return

    def update_state(self, signals_ser, allocation_date):
        self.extend_bus_calendar(allocation_date)
        #extend bus cal if earlier set not large enough

        self.signals_hist = pd.concat([self.signals_hist.T, signals_ser], axis=1).T
        SMALL_RISK_NUM = TradeAllocator.SMALL_RISK_NUM
        self.trade_limits = signals_ser['trade_limits']
        # self.swing_limits = signals_ser['swing_trade_stops']
        # floor risk so no np.inf
        self.risk_floor = max(signals_ser['risk'], 0)
        self.risk_min = min(self.risk_floor*(self.risk_floor > 0), self.risk_min, SMALL_RISK_NUM)
        self.risk = (self.risk_aversion / self.scale_all ** 2) * signals_ser['risk']
        self.risk = self.risk if self.risk > 0 else self.risk_min

        self.exp_gain = self.alpha_multiplier * (signals_ser['alpha'] if not np.isnan(signals_ser['alpha']) else 0.0)
        self.expected_gain_net_financing = (self.exp_gain / self.scale_all) - (
                self.funding_multiplier / self.scale_all) * (
                np.heaviside(self.exp_gain, 0) * signals_ser['short_financing_cost'] -
                np.heaviside(-self.exp_gain, 0) * signals_ser['long_financing_cost']
        )

        self.optimal_pfolio_wno_tcosts = self.expected_gain_net_financing * (1 / self.risk)
        self.tcosts = signals_ser['tcosts']
        delta = (self.tcost_multiplier / self.scale_all) * self.tcosts / self.risk
        self.no_trade_zone_ub = self.optimal_pfolio_wno_tcosts + delta
        self.no_trade_zone_lb = self.optimal_pfolio_wno_tcosts - delta
        if allocation_date in self.bus_date_index:
            date_posn = self.bus_date_index.index(allocation_date)
        else:
            print('Allocation Date NOT in index {}'.format(allocation_date))

            # print(self.bus_date_index)
        if self.allocation_date is not None:
            date_before_posn = self.bus_date_index.index(self.allocation_date)
        else:
            date_before_posn = date_posn - 1  # automatically

        self.target_portfolio_wt_tminus1 = 0.0
        if date_before_posn == date_posn - 1:  # 1bd ago
            self.target_portfolio_wt_tminus1 = self.target_portfolio_wt

        target_portfolio_wt = np.clip(self.target_portfolio_wt_tminus1,
                                      self.no_trade_zone_lb,
                                      self.no_trade_zone_ub)
        self.target_portfolio_wt = np.clip(target_portfolio_wt, -1 * self.gross_limit, self.gross_limit)
        self.hit_limit = (np.abs(self.target_portfolio_wt) == self.gross_limit) * 1
        if self.trade_limits == 1:
            target_portfolio_wt = 0
            self.hit_limit = np.nan
        # if self.swing_trade_stops == 1 and np.abs(self.target_portfolio_wt_tminus1) <= TradeAllocator.SMALL_POSITION:
        #     target_portfolio_wt = 0
        #     self.hit_limit = np.nan
        #     # don't bother trading
        self.target_trade = self.target_portfolio_wt - self.target_portfolio_wt_tminus1
        self.allocation_date = allocation_date
        self._create_returns_series(signals_ser=signals_ser)

    def _create_returns_series(self, signals_ser):
        returns_ser = signals_ser.copy()
        returns_ser.name = self.allocation_date
        returns_ser.loc['expected_gain'] = self.exp_gain # overwrite nan
        if 'realized_gain' in returns_ser.index:
            returns_ser['realized_gain'] = np.nan # shouldn't have seen it yet
        # expected gain gross
        returns_ser.loc['expected_gain_net_financing'] = self.expected_gain_net_financing
        # expected net gain (net of tcosts)
        returns_ser.loc['total_risk'] = self.risk
        returns_ser.loc['optimal_pfolio_wno_tcosts'] = self.optimal_pfolio_wno_tcosts
        # opt
        returns_ser.loc['no_trade_zone_lb'] = self.no_trade_zone_lb
        returns_ser.loc['no_trade_zone_ub'] = self.no_trade_zone_ub
        returns_ser.loc['recommended_trade'] = self.target_trade
        returns_ser.loc['target_alloc_usd'] = self.target_portfolio_wt
        missing_fields = ['realized_gain', 'real_short_gain', 'real_long_gain',
                          'realized_usd_gain_gross', 'realized_usd_tcosts',
                          'realized_usd_gain_net', 'realized_usd_no_tcosts',
                          'total_usd_exp_gain', 'total_usd_risk', 'realized_pnl_gain', 'total_pnl']
        all_fields = list(set(returns_ser.index).union(set(missing_fields)))
        returns_ser = returns_ser.reindex(all_fields)
        # placeholders for ex-post info
        self.returns_ser = returns_ser
        # self.returns_frame.loc[self.returns_ser.name, : ] = self.returns_ser
        if self.returns_frame.empty:
            self.returns_frame = pd.DataFrame(self.returns_ser).T
        elif (self.returns_ser.name in self.returns_frame.index):
            self.returns_frame.loc[self.returns_ser.name, : ] = self.returns_ser
        else:
            self.returns_frame = pd.concat([self.returns_frame.T, self.returns_ser], axis=1).T
        self.returns_frame.index.names = ['date']
        # print(self.returns_frame.loc[self.returns_ser.name,:].T)
        # print('136 Type of Returns_frame = {}'.format(type(self.returns_frame)))
        # TODO: Series version not tested after swing_trade_stops used

    def _create_realized_returns_series(self, realized_gain:float):
        self.returns_ser.loc['realized_gain'] = realized_gain
        # signals.loc['realized_price_gain'] = signals['delta_px_price_frac']
        # should be (1/risk_aversion)*(1/var)
        self.returns_ser.loc['real_short_gain'] = self.returns_ser['realized_gain'] + self.returns_ser['short_financing_cost']
        self.returns_ser.loc['real_long_gain'] = self.returns_ser['realized_gain'] - self.returns_ser['long_financing_cost']
        self.returns_ser.loc['realized_usd_gain_gross'] = self.target_portfolio_wt * (
                np.heaviside(self.target_portfolio_wt, 0) * self.returns_ser['real_long_gain'] +
                np.heaviside(-self.target_portfolio_wt, 0) * self.returns_ser['real_short_gain'])

        gross_trade = np.abs(self.target_trade)
        self.returns_ser.loc['realized_usd_tcosts'] = -1 * gross_trade * self.tcosts

        self.returns_ser.loc['realized_usd_gain_net'] = self.returns_ser['realized_usd_gain_gross'] + \
                                                  self.returns_ser['realized_usd_tcosts']
        # with no tcosts, always go to target portfolio
        self.returns_ser.loc['realized_usd_no_tcosts'] = self.optimal_pfolio_wno_tcosts * (
                np.heaviside(self.optimal_pfolio_wno_tcosts, 0) * self.returns_ser['real_long_gain'] +
                np.heaviside(-self.optimal_pfolio_wno_tcosts, 0) * self.returns_ser['real_short_gain'])
        # strategy_returns.loc[:, 'alloc_dv01'] = target_pfolio_wt

        self.returns_ser.loc['total_usd_exp_gain'] = self.expected_gain_net_financing * self.target_portfolio_wt
        self.returns_ser.loc['total_usd_risk'] = (
                                                             1 / 2) * self.risk * self.target_portfolio_wt * self.target_portfolio_wt
        self.returns_ser.loc['hit_limit'] = self.hit_limit
        # self.returns_ser.loc['realized_pnl_gain'] = self.returns_ser['realized_usd_gain_net']  # duplicate
        if 'realized_pnl_gain' in self.returns_ser.index:
            self.returns_ser = self.returns_ser.drop('realized_pnl_gain')
        self.returns_ser = self.returns_ser.rename({'realized_usd_gain_net': 'realized_pnl_gain'})

        # print('163 Type of Returns_frame = {}'.format(type(self.returns_frame)))
        if not self.returns_frame.empty:
            existing_sum = self.returns_frame['realized_pnl_gain'].sum()
            if np.isnan(existing_sum):
                existing_sum = 0.0
            self.returns_ser.loc['total_pnl'] = (existing_sum + self.returns_ser['realized_pnl_gain'])
        else:
            self.returns_ser.loc['total_pnl'] = self.returns_ser['realized_pnl_gain']
        # print(self.returns_frame.loc[self.returns_ser.name, :].T)
        # returns_ser.name = date for time-slice
        self.returns_frame.loc[self.returns_ser.name, :] = self.returns_ser
        # print('173 Type of Returns_frame = {}'.format(type(self.returns_frame)))

    def block_update(self, signals_frame, dataframe_update=False):
        if dataframe_update and isinstance(signals_frame, pd.DataFrame):
            self._update_frame(signals_frame= signals_frame)
            if 'realized_gain' in signals_frame and not signals_frame['realized_gain'].isnull().all():
                self._create_realized_returns_frame(realized_series=signals_frame['realized_gain'])
        elif isinstance(signals_frame, pd.Series):
            self.update_state(signals_ser=signals_frame, allocation_date=signals_frame.name)
        else:
            for idx, signals_row in signals_frame.iterrows():
                self.update_state(signals_ser=signals_row, allocation_date=idx)
                if ('realized_gain' in signals_row) and not np.isnan(signals_row['realized_gain']):
                    self._create_realized_returns_series(signals_row['realized_gain'])

    def set_cut_date(self, start_perf_msmt_date = None ):
        # can only be used after updates started, preferably a block
        self.start_perf_msmt_date = start_perf_msmt_date
        cut_dates = [date for date in list(self.returns_frame.index) if date <= self.start_perf_msmt_date]
        if len(cut_dates) == 0:
            # just fill in if returns_frame also empty!
            self.first_date = (self.returns_frame.index[0] if (self.returns_frame.shape[0] > 0)
                               else const.START_PERFORMANCE_MSMT)
        else:
            self.first_date = max(cut_dates)

    def hit_lim_stats(self, start_perf_msmt_date=None):
        if (start_perf_msmt_date is not None):
            self.set_cut_date(start_perf_msmt_date)
            first_date = self.first_date
        elif hasattr(self, 'first_date'):
            first_date = self.first_date
        else:
            # print('Must use set_cut_date before msm')
            self.set_cut_date()
            first_date = self.first_date
            print('Use Set-Cut_date method to set first date. Default to beginning')
        if 'hit_limit' in self.returns_frame.columns:
            underlimit_pct = (self.returns_frame['hit_limit'].map(lambda x: x == 0).sum() /
                              self.returns_frame['hit_limit'].map(lambda x: x in {0, 1}).sum())
            amount_under = (self.returns_frame['hit_limit'].map(lambda x: x == 0) * (self.gross_limit -
                                                                                     self.returns_frame[
                                                                                         'target_alloc_usd'].abs()))
            amount_under_mean = (amount_under.sum() /
                                 self.returns_frame['hit_limit'].map(lambda x: x in {0, 1}).sum())
            amount_under_null = (amount_under.isnull() | (
                        amount_under == self.gross_limit))  # "* self.returns_frame['hit_limit'].map(lambda x: x == 0)
            non_null_amount_under = amount_under.drop(amount_under.loc[amount_under_null].index)
            amount_under_pctiles = dict()
            for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                amount_under_pctiles[p] = non_null_amount_under.pipe(lambda x: np.percentile(x, p))
                amount_pctiles = pd.Series(amount_under_pctiles)
        else:
            underlimit_pct = np.nan
            amount_under_mean = np.nan
            # amount_under_pctiles = dict()
            amount_pctiles = pd.Series()
        return (underlimit_pct, amount_under_mean, amount_pctiles)

    def sr_cut(self, start_perf_msmt_date=None, LARGE_NUM=np.inf):

        if (start_perf_msmt_date is not None):
            self.set_cut_date(start_perf_msmt_date)
            first_date = self.first_date
        elif hasattr(self, 'first_date'):
            first_date = self.first_date
        else:
            # print('Must use set_cut_date before msm')
            self.set_cut_date()
            first_date=self.first_date
            print('Use Set-Cut_date method to set first date. Default to beginning')
        if 'realized_pnl_gain' in self.returns_frame.columns:
            sr = (self.returns_frame['realized_pnl_gain'].loc[first_date:].pipe(
                lambda x: -1 * LARGE_NUM if (x.std() == 0)
                else x.mean() / x.std() * np.sqrt(252)))
        else:
            sr = -1 * LARGE_NUM# don't mess around
        return sr

    def _update_frame(self, signals_frame: pd.DataFrame):
        '''
        Act on signal as DataFrame/Series. Only map to
        paramaters (current state) at the end of routine.
        @params: signals, optimizer_dict
        signals = dataframe of alpha, risk, trade_limits(stops), tcosts
        This is a full dataframe method for single asset optimal allocation (MVO) with tcosts
        @return strategy_returns dataframe with signals augmented by positions, sr: float sharpe ratio
        '''
        if self.bus_date_index is None:
            self.bus_date_index = list(signals_frame.index)
        else:
            # augment it
            bus_date_index = list(set(self.bus_date_index).union(set(signals_frame.index)))
            bus_date_index.sort()
            self.bus_date_index = bus_date_index

        backup_column = pd.Series(np.zeros(signals_frame.shape[0]),
                                  index=signals_frame.index,
                                  name='placeholder')
        necessary_cols = ['risk', 'alpha', 'tcosts']
        possibly_missing_cols = ['short_financing_cost', 'long_financing_cost', 'trade_limits']
        for col in possibly_missing_cols:
            if col not in signals_frame.columns:
                signals_frame = self.careful_add_col(signals_frame,
                                                     backup_column,
                                                     series_name=col
                                                     )
        self.signals_hist = signals_frame
        # All of these ops are pd.Series / pd.DataFrame ops
        trade_limits = signals_frame['trade_limits']
        # swing_trade_stops = signals_frame['swing_trade_stops']
        # floor risk so no np.inf
        SMALL_RISK_NUM = TradeAllocator.SMALL_RISK_NUM
        risk_floor = signals_frame['risk']
        risk_min = min(risk_floor[risk_floor > 0].min(), SMALL_RISK_NUM)
        risk = (self.risk_aversion / self.scale_all ** 2) * signals_frame['risk']
        risk = risk.map(lambda x: x if x > 0 else self.risk_min)
        exp_gain = signals_frame['alpha'] / self.scale_all
        exp_gain = exp_gain.fillna(0)
        expected_gain_net_financing = exp_gain - (self.funding_multiplier / self.scale_all) * (
                np.heaviside(exp_gain, 0) * signals_frame['short_financing_cost'] -
                np.heaviside(-exp_gain, 0) * signals_frame['long_financing_cost'])

        optimal_pfolio_wno_tcosts = expected_gain_net_financing * (1 / risk)
        tcosts = signals_frame['tcosts']
        delta = (self.tcost_multiplier / self.scale_all) * tcosts / risk
        no_trade_zone_ub = optimal_pfolio_wno_tcosts + delta
        no_trade_zone_lb = optimal_pfolio_wno_tcosts - delta
        # TODO: Allocation date based updating (if not in index, prev day alloc = 0)

        target_portfolio_wt = pd.Series(np.zeros(optimal_pfolio_wno_tcosts.shape),
                                        index=optimal_pfolio_wno_tcosts.index,
                                        name='target_allocation')
        target_trade = pd.Series(np.zeros(optimal_pfolio_wno_tcosts.shape),
                                 index=optimal_pfolio_wno_tcosts.index,
                                 name='trade_recommendation')
        for idx in range(1, len(target_portfolio_wt)):
            date_posn = self.bus_date_index.index(target_portfolio_wt.index[idx])
            date_before_posn = self.bus_date_index.index(target_portfolio_wt.index[idx - 1])
            if date_before_posn == date_posn - 1:
                target_portfolio_wt_tminus1 = target_portfolio_wt.iloc[idx - 1]
                # last posn we record was actually yesterday
            else:
                target_portfolio_wt_tminus1 = 0.0

            target_portfolio_wt.iloc[idx] = np.clip(target_portfolio_wt_tminus1,
                                                    no_trade_zone_lb.iloc[idx],
                                                    no_trade_zone_ub.iloc[idx])

            # target_alloc_usd.iloc[idx] = target_portfolio_wt.iloc[idx]
            # if swing_trade_stops[idx] == 1 and abs(target_portfolio_wt_tminus1) < TradeAllocator.SMALL_POSITION:
            #     target_portfolio_wt[idx] = 0
            if trade_limits[idx] == 1:
                target_portfolio_wt[idx] = 0
                # target_alloc_usd[idx] = 0

            target_portfolio_wt.iloc[idx] = np.clip(target_portfolio_wt.iloc[idx],
                                                    -1 * self.gross_limit,
                                                    self.gross_limit)
            target_trade[idx] = target_portfolio_wt.iloc[idx] - target_portfolio_wt_tminus1
            # apply limit in usd, then go back
            # target_portfolio_wt[idx] = target_alloc_usd.iloc[idx]

        returns_frame = signals_frame.copy()
        returns_frame.loc[:, 'expected_gain'] = exp_gain
        if 'realized_gain' in signals_frame:
            returns_frame.loc[:, 'realized_gain'] = np.nan
        # expected gain gross
        returns_frame.loc[:, 'expected_gain_net_financing'] = expected_gain_net_financing
        # expected net gain (net of tcosts)
        returns_frame.loc[:, 'total_risk'] = risk
        returns_frame.loc[:, 'optimal_pfolio_wno_tcosts'] = optimal_pfolio_wno_tcosts
        # opt
        returns_frame.loc[:, 'no_trade_zone_lb'] = no_trade_zone_lb
        returns_frame.loc[:, 'no_trade_zone_ub'] = no_trade_zone_ub
        returns_frame.loc[:, 'recommended_trade'] = target_trade
        returns_frame.loc[:, 'target_alloc_usd'] = target_portfolio_wt
        returns_frame.loc[:, 'total_usd_exp_gain'] = expected_gain_net_financing * target_portfolio_wt
        returns_frame.loc[:, 'total_usd_risk'] = (1 / 2) * risk * target_portfolio_wt * target_portfolio_wt
        returns_frame.loc[:, 'hit_limit'] = (np.abs(self.target_portfolio_wt) == self.gross_limit) * 1
        must_get_out = returns_frame['trade_limits'].map(lambda x: x == 1)
        returns_frame.loc[must_get_out, 'hit_limit'] = np.nan
        # separate out realized_gain portion
        missing_fields = ['realized_gain', 'real_short_gain', 'real_long_gain',
                          'realized_usd_gain_gross', 'realized_usd_tcosts',
                          'realized_usd_gain_net', 'realized_usd_no_tcosts',
                          'total_usd_exp_gain', 'total_usd_risk', 'realized_pnl_gain', 'total_pnl']
        extra_fields = list(set(missing_fields) - set(returns_frame.columns))
        returns_frame.loc[:, extra_fields] = np.nan
        # placeholders for ex-post info
        if self.returns_frame.empty:
            pass
        else:
            # TODO: allow this method to update existing frame instead - must start differently
            print('overwrote history in TA class instance')
            # print('block_update(dataframe_update=True) misused - overwrote history')
        self.returns_frame = returns_frame
        self.returns_ser = returns_frame.iloc[-1, :]
        self._save_final_row_as_state()
        # print(' 317 Type of Returns_frame = {}'.format(type(self.returns_frame)))

    def _save_final_row_as_state(self):
        self.allocation_date = self.returns_ser.name
        self.exp_gain = self.returns_ser['expected_gain']
        self.expected_gain_net_financing = self.returns_ser['expected_gain_net_financing']
        self.risk = self.returns_ser['total_risk']
        self.optimal_pfolio_wno_tcosts = self.returns_ser['optimal_pfolio_wno_tcosts']
        self.no_trade_zone_lb = self.returns_ser.loc['no_trade_zone_lb']
        self.no_trade_zone_ub =self.returns_ser.loc['no_trade_zone_ub']
        self.target_trade = self.returns_ser.loc['recommended_trade']
        self.target_portfolio_wt = self.returns_ser.loc['target_alloc_usd']

    @staticmethod
    def careful_add_col(frame:pd.DataFrame, series:pd.Series, series_name:str) -> pd.DataFrame:
        # make sure new series is indexed properly,
        series.name = series_name
        if series_name in frame.columns:
            try:
                frame = frame.drop(columns=[series_name])
            except:  # AssertionError or IndexError?
                print(series_name)
                print('columns = {}'.format(frame.columns))

        frame = pd.merge(frame, series, left_index=True, right_index=True, how='left')
        return frame


    def _create_realized_returns_frame(self, realized_series):

        try:
            # print('333 Type of Returns_frame = {}'.format(type(self.returns_frame)))
            # self.returns_frame.loc[:, 'realized_gain'] = realized_series
            if 'realized_gain' in self.returns_frame.columns:
                self.returns_frame = self.returns_frame.drop(columns=['realized_gain'])

            realized_series = realized_series.reindex(self.returns_frame.index)
            self.returns_frame['realized_gain'] = realized_series

            self.returns_frame = self.careful_add_col(self.returns_frame, realized_series, series_name='realized_gain')
            real_short_gain = (self.returns_frame['realized_gain'] +
                               self.returns_frame['short_financing_cost'])
            self.returns_frame = self.careful_add_col(self.returns_frame, real_short_gain, 'real_short_gain')
            real_long_gain = (self.returns_frame['realized_gain'] -
                              self.returns_frame['long_financing_cost'])
            self.returns_frame = self.careful_add_col(self.returns_frame, real_long_gain, 'real_long_gain')
            target_portfolio_wt = self.returns_frame['target_alloc_usd']
            realized_usd_gain_gross = target_portfolio_wt * (
                    np.heaviside(target_portfolio_wt, 0) * self.returns_frame['real_long_gain'] +
                    np.heaviside(-target_portfolio_wt, 0) * self.returns_frame['real_short_gain'])
            self.returns_frame = self.careful_add_col(self.returns_frame, realized_usd_gain_gross, 'realized_usd_gain_gross' )
            target_trade = self.returns_frame['recommended_trade']
            gross_trade = target_trade.fillna(0).map(np.abs)
            tcosts = self.returns_frame['tcosts']
            realized_usd_tcosts = -1 * gross_trade * tcosts
            self.returns_frame = self.careful_add_col(self.returns_frame, realized_usd_tcosts, 'realized_usd_tcosts')

            realized_usd_gain_net = (self.returns_frame['realized_usd_gain_gross']
                                     + self.returns_frame['realized_usd_tcosts'])
            self.returns_frame = self.careful_add_col(self.returns_frame, realized_usd_gain_net, 'realized_usd_gain_net')
            # with no tcosts, always go to target portfolio
            optimal_pfolio_wno_tcosts = self.returns_frame['optimal_pfolio_wno_tcosts'].fillna(0)
            # regular_step = (lambda x: np.heaviside(x, 0))
            realized_usd_no_tcosts = optimal_pfolio_wno_tcosts * (
                    np.heaviside(optimal_pfolio_wno_tcosts, 0) * self.returns_frame['real_long_gain'] +
                    np.heaviside(-1 * optimal_pfolio_wno_tcosts, 0) * self.returns_frame['real_short_gain'])
            self.returns_frame = self.careful_add_col(self.returns_frame,
                                                      realized_usd_no_tcosts,
                                                      'realized_usd_no_tcosts')
            # self.returns_frame.loc[:, 'realized_pnl_gain'] = self.returns_frame['realized_usd_gain_net']  # duplicate
            if 'realized_pnl_gain' in self.returns_frame.columns:
                self.returns_frame = self.returns_frame.drop(columns=['realized_pnl_gain'])
            self.returns_frame = self.returns_frame.rename(columns={'realized_usd_gain_net': 'realized_pnl_gain'})
            self.returns_frame['total_pnl'] = self.returns_frame['realized_pnl_gain'].cumsum()
            # print('360 Type of Returns_frame = {}'.format(type(self.returns_frame)))
        except:  # ValueError broadcast (6,5) as (6,)
            print('MASSIVE ERROR IN Trade Allocator')
            # print('363 Type of Returns_frame = {}'.format(type(self.returns_frame)))
            four_digit_random_number = int(np.random.uniform(1000,9999))
            print('Random  id = {}'.format(four_digit_random_number))
            if not self.returns_frame.empty:
                print('attempting to write returns_frame')

                try:
                    self.returns_frame.to_csv(const.ERROR_DIR + 'error_in_ta_returns_{}.csv'.format(four_digit_random_number))
                except:
                    print('Cant Print returns_frame')
            else:
                pd.DataFrame().to_csv(const.ERROR_DIR + 'error_in_ta_returns_{}.csv'.format(four_digit_random_number))
                print('No returns_frame to save')
            realized_frame = pd.DataFrame(realized_series)
            if not realized_frame.empty:
                print('output returns frame and series')
                try:
                    realized_frame.to_csv(const.ERROR_DIR + 'error_in_ta_realized_series_{}.csv'.format(four_digit_random_number))
                except:
                    print('cant print realized-series')
            else:
                print('Relurns Series null')
                pd.DataFrame().to_csv(
                    const.ERROR_DIR + 'error_in_ta_realized_series_{}.csv'.format(four_digit_random_number))

            # raise FloatingPointError
        return 1


# from pandas.testing import assert_frame_equal, assert_series_equal
import time

def main():
    optimizer_dict = {
        'alpha_multiplier': 1.00,
        'scale_limits': 1.5,
        'funding_multiplier': 0.22,
        'tcost_multiplier': 1.00,  # unused
        'risk_aversion': 0.00046794,
        'gross_limit': 50000000,
    }

    test_data = pd.read_csv('~/Dropbox/FX/trade_allocation/strategy_test_df.csv', index_col = [0])
    test_data.index = pd.to_datetime(test_data.index)
    otr_date_index = test_data.index
    ta = TradeAllocator(business_days=otr_date_index, multiplier_dict=optimizer_dict)
    signals = test_data

    signals['risk'] = signals['volatility']**2
    '''
    Can be in returns or in price changes (only matters how you sum the final strategy returns)
    alpha = E[returns(t+1)|t] = E[p(t+1)|t] - p(t)
    realized_gain = returns(t+1) = p(t+1) - p(t)
    volatility = returns.rolling(x).std() or ewma(span=x).std() etc
    Preferable to just input risk = vol**2
    tcosts = (ask(t)-bid(t))/2
    tcost_multiplier should usually be 1 but the tcosts are so small w/respect to the alphas
    '''
    signals = signals.drop(columns=['volatility'])
    signals = signals.rename(columns={'target':'realized_gain','forecast':'alpha', 'spread':'tcosts'})
    ta.block_update(signals,dataframe_update=True)
    returns = ta.returns_frame
    returns.to_csv('test_returns.csv')
    #TODO: Test incremental update gives same results as block update


    print('finished our little test')


if __name__ == '__main__':
    main()


