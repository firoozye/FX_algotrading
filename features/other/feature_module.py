from features.other.calendar_info import *
from features.other.past_returns import *
from features.other.sample_stats import *
from features.other.quantile import *
from features.technicals.higher_order_moments import *
from features.technicals.trend import *
# from features.DFA import *
# from features.lyapunov_exponent import *
# from features.entropy import *
# from features.lppls import *
from features.technicals.spectrum import *


class Features:
    def __init__(self, df, list_of_windows, momentum_periods, quantiles, predict_periods,
                 offsets):
        ticker_ohlcv_df = df
        self.index = ticker_ohlcv_df.index
        self.moving_windows = list_of_windows
        self.predict_periods = predict_periods
        self.offsets = offsets
        self.quantiles = quantiles
        self.momentum_periods = momentum_periods
        self.open = ticker_ohlcv_df['Open']
        self.high = ticker_ohlcv_df['High']
        self.low = ticker_ohlcv_df['Low']
        self.close = ticker_ohlcv_df['Close']
        self.spread_close = ticker_ohlcv_df['spread_close']
        self.spread_open = ticker_ohlcv_df['spread_open']
        self.spread_high = ticker_ohlcv_df['spread_high']
        self.spread_low = ticker_ohlcv_df['spread_low']
        
        self.stock_features = pd.DataFrame()
        self.stock_features['open'] = self.open
        self.stock_features['high'] = self.high
        self.stock_features['low'] = self.low
        self.stock_features['close'] = self.close
        self.stock_features['spread_close'] = self.spread_close
        self.stock_features['spread_open'] = self.spread_open
        self.stock_features['spread_high'] = self.spread_high
        self.stock_features['spread_low'] = self.spread_low
        self.stock_features.index = pd.to_datetime(self.index)
      

    def get_features(self):
        # technical features
        # print('technical features')
        for window_len in self.moving_windows:
            try:
                self.stock_features['sma_w_{}'.format(window_len)] = get_sma(series=self.stock_features['close'],
                                                                             use_custom_window=True,
                                                                             window=window_len)
            except:
                self.stock_features['sma_w_{}'.format(window_len)] = np.nan

            try:
                self.stock_features['diff_close_sma_w_{}'.format(window_len)] = \
                    (self.close - self.stock_features['sma_w_{}'.format(window_len)]) / \
                    self.stock_features['sma_w_{}'.format(window_len)]
            except:
                self.stock_features['diff_close_sma_w_{}'.format(window_len)] = np.nan

            try:
                self.stock_features['ema_w_{}'.format(window_len)] = get_ema(series=self.stock_features['close'],
                                                                             use_custom_window=True,
                                                                             window=window_len)
            except:
                self.stock_features['ema_w_{}'.format(window_len)] = np.nan

            try:
                self.stock_features['diff_close_ema_w_{}'.format(window_len)] = \
                    (self.close - self.stock_features['ema_w_{}'.format(window_len)]) / \
                    self.stock_features['ema_w_{}'.format(window_len)]
            except:
                self.stock_features['diff_close_ema_w_{}'.format(window_len)] = np.nan

            try:
                self.stock_features['osc_w_{}'.format(window_len)] = get_oscillator(close=self.stock_features['close'],
                                                                                    low=self.stock_features['low'],
                                                                                    high=self.stock_features['high'],
                                                                                    use_custom_window=True,
                                                                                    window=window_len)
            except:
                self.stock_features['osc_w_{}'.format(window_len)] = np.nan

            try:
                self.stock_features['boll_high_w_{}'.format(window_len)], \
                    self.stock_features['boll_low_w_{}'.format(window_len)] = \
                    get_bollinger_bands(series_close=self.stock_features['close'],
                                        use_custom_window=True,
                                        window=window_len)
            except:
                self.stock_features['boll_high_w_{}'.format(window_len)] = np.nan
                self.stock_features['boll_low_w_{}'.format(window_len)] = np.nan

            try:
                self.stock_features['diff_close_boll_high_w_{}'.format(window_len)] = \
                    (self.close - self.stock_features['boll_high_w_{}'.format(window_len)]) / \
                    self.stock_features['boll_high_w_{}'.format(window_len)]
            except:
                self.stock_features['diff_close_boll_high_w_{}'.format(window_len)] = np.nan

            try:
                self.stock_features['diff_close_boll_low_w_{}'.format(window_len)] = \
                    (self.close - self.stock_features['boll_low_w_{}'.format(window_len)]) / \
                    self.stock_features['boll_low_w_{}'.format(window_len)]
            except:
                self.stock_features['diff_close_boll_low_w_{}'.format(window_len)] = np.nan

            try:
                self.stock_features['rsi_w_{}'.format(window_len)] = get_rsi(series_close=self.stock_features['close'],
                                                                             use_custom_window=True,
                                                                             window=window_len)
            except:
                self.stock_features['rsi_w_{}'.format(window_len)] = np.nan

            try:
                self.stock_features['adx_w_{}'.format(window_len)] = get_adx(close=self.stock_features['close'],
                                                                             low=self.stock_features['low'],
                                                                             high=self.stock_features['high'],
                                                                             use_custom_window=True,
                                                                             window=window_len)
            except:
                self.stock_features['adx_w_{}'.format(window_len)] = np.nan

            # pct change of the technicals
            for mom_period in self.momentum_periods:
                try:
                    self.stock_features['sma_w_{}_change_{}'.format(window_len, mom_period)] = \
                        self.stock_features['sma_w_{}'.format(window_len)].pct_change(mom_period)
                except:
                    self.stock_features['sma_w_{}_change_{}'.format(window_len, mom_period)] = np.nan

                try:
                    self.stock_features['ema_w_{}_change_{}'.format(window_len, mom_period)] = \
                        self.stock_features['ema_w_{}'.format(window_len)].pct_change(mom_period)
                except:
                    self.stock_features['ema_w_{}_change_{}'.format(window_len, mom_period)] = np.nan

                try:
                    self.stock_features['osc_w_{}_change_{}'.format(window_len, mom_period)] = \
                        self.stock_features['osc_w_{}'.format(window_len)].pct_change(mom_period)
                except:
                    self.stock_features['osc_w_{}_change_{}'.format(window_len, mom_period)] = np.nan

                try:
                    self.stock_features['boll_w_{}_change_{}'.format(window_len, mom_period)] = \
                        self.stock_features['boll_high_w_{}'.format(window_len)].pct_change(mom_period)
                except:
                    self.stock_features['boll_w_{}_change_{}'.format(window_len, mom_period)] = np.nan

                try:
                    self.stock_features['rsi_w_{}_change_{}'.format(window_len, mom_period)] = \
                        self.stock_features['rsi_w_{}'.format(window_len)].pct_change(mom_period)
                except:
                    self.stock_features['rsi_w_{}_change_{}'.format(window_len, mom_period)] = np.nan

                try:
                    self.stock_features['adx_w_{}_change_{}'.format(window_len, mom_period)] = \
                        self.stock_features['adx_w_{}'.format(window_len)].pct_change(mom_period)
                except:
                    self.stock_features['adx_w_{}_change_{}'.format(window_len, mom_period)] = np.nan

        try:
            self.stock_features['macd'] = get_macd(series=self.stock_features['close'],
                                                   period1=12,
                                                   period2=26)
        except:
            self.stock_features['macd'] = np.nan

        for mom_period in self.momentum_periods:
            try:
                self.stock_features['macd_change_{}'.format(mom_period)] = \
                    self.stock_features['macd'].pct_change(mom_period)
            except:
                self.stock_features['macd_change_{}'.format(mom_period)] = np.nan

        # datetime artefacts
        # print('datetime artefacts')
        try:
            artefacts_dict = get_calendar_info(datetime_series=pd.Series(self.stock_features.index))
            self.stock_features['month'] = np.array(artefacts_dict['month'])
            self.stock_features['day'] = np.array(artefacts_dict['day'])
            self.stock_features['day_of_year'] = np.array(artefacts_dict['day_of_year'])
            self.stock_features['week_of_year'] = np.array(artefacts_dict['week_of_year'])
            self.stock_features['day_of_week'] = np.array(artefacts_dict['day_of_week'])
            self.stock_features['quarter'] = np.array(artefacts_dict['quarter'])
            self.stock_features['is_month_start'] = np.array(artefacts_dict['is_month_start'])
            self.stock_features['is_month_end'] = np.array(artefacts_dict['is_month_end'])
            self.stock_features['is_quarter_start'] = np.array(artefacts_dict['is_quarter_start'])
            self.stock_features['is_quarter_end'] = np.array(artefacts_dict['is_quarter_end'])
        except:
            self.stock_features['month'] = np.nan
            self.stock_features['day'] = np.nan
            self.stock_features['day_of_year'] = np.nan
            self.stock_features['week_of_year'] = np.nan
            self.stock_features['day_of_week'] = np.nan
            self.stock_features['quarter'] = np.nan
            self.stock_features['is_month_start'] = np.nan
            self.stock_features['is_month_end'] = np.nan
            self.stock_features['is_quarter_start'] = np.nan
            self.stock_features['is_quarter_end'] = np.nan

        # momentum features - simple returns, cumulative returns with offset
        # print('momentum features')
        for period in self.momentum_periods:
            try:
                self.stock_features['return_{}'.format(period)] = \
                    np.array(get_return(prices=pd.Series(self.close), period=period))
            except:
                self.stock_features['return_{}'.format(period)] = np.nan

        for window_len in self.moving_windows:
            for offset in self.offsets:
                try:
                    self.stock_features['cum_return_{}_offset_{}'.format(window_len, offset)] = \
                        np.array(get_cum_return(returns=pd.Series(self.close).pct_change(),
                                                window=window_len, offset=offset))
                except:
                    self.stock_features['cum_return_{}_offset_{}'.format(window_len, offset)] = np.nan

        for period_1 in self.momentum_periods:
            for period_2 in self.momentum_periods:
                if period_1 != period_2:
                    try:
                        self.stock_features['diff_returns_{}_{}'.format(period_1, period_2)] = \
                            self.stock_features['return_{}'.format(period_1)] - \
                            self.stock_features['return_{}'.format(period_2)]
                    except:
                        self.stock_features['diff_returns_{}_{}'.format(period_1, period_2)] = np.nan

                    try:
                        self.stock_features['pct_returns_{}_{}'.format(period_1, period_2)] = \
                            self.stock_features['return_{}'.format(period_1)] / \
                            self.stock_features['return_{}'.format(period_2)]
                    except:
                        self.stock_features['pct_returns_{}_{}'.format(period_1, period_2)] = np.nan

        # sample statistics
        # print('sample statistics')
        for window_len in self.moving_windows:
            for mom_period in self.momentum_periods:
                try:
                    stats_list = get_sample_statistics(data=pd.Series(self.close).pct_change(mom_period),
                                                       window_length=window_len)
                except:
                    stats_list = [np.nan * 20]

                self.stock_features['ret_{}_w_{}_mean'.format(mom_period, window_len)] = np.array(stats_list[0])
                self.stock_features['ret_{}_w_{}_median'.format(mom_period, window_len)] = np.array(stats_list[1])
                self.stock_features['ret_{}_w_{}_var'.format(mom_period, window_len)] = np.array(stats_list[2])
                self.stock_features['ret_{}_w_{}_std'.format(mom_period, window_len)] = np.array(stats_list[3])
                self.stock_features['ret_{}_w_{}_min'.format(mom_period, window_len)] = np.array(stats_list[4])
                self.stock_features['ret_{}_w_{}_max'.format(mom_period, window_len)] = np.array(stats_list[5])
                self.stock_features['ret_{}_w_{}_dev_mean_max'.format(mom_period, window_len)] = \
                    np.array(stats_list[6])
                self.stock_features['ret_{}_w_{}_dev_mean_min'.format(mom_period, window_len)] = \
                    np.array(stats_list[7])
                self.stock_features['ret_{}_w_{}_dev_min_max'.format(mom_period, window_len)] = \
                    np.array(stats_list[8])
                self.stock_features['ret_{}_w_{}_dev_mean_std'.format(mom_period, window_len)] = \
                    np.array(stats_list[9])
                self.stock_features['ret_{}_w_{}_diff_mean_median'.format(mom_period, window_len)] = \
                    np.array(stats_list[10])
                self.stock_features['ret_{}_w_{}_diff_max_min'.format(mom_period, window_len)] = \
                    np.array(stats_list[11])
                self.stock_features['ret_{}_w_{}_diff_max_mean'.format(mom_period, window_len)] = \
                    np.array(stats_list[12])
                self.stock_features['ret_{}_w_{}_diff_min_mean'.format(mom_period, window_len)] = \
                    np.array(stats_list[13])
                self.stock_features['ret_{}_w_{}_dev_max_std'.format(mom_period, window_len)] = \
                    np.array(stats_list[14])
                self.stock_features['ret_{}_w_{}_dev_min_std'.format(mom_period, window_len)] = \
                    np.array(stats_list[15])

        # risk
        # print('risk measures')
        for window_len in self.moving_windows:
            for q in self.quantiles:
                try:
                    self.stock_features['q_{}_w_{}'.format(q, window_len)] = \
                        np.array(get_rolling_quantile(series=pd.Series(self.close).pct_change(),
                                                      window=window_len, q=q))
                except:
                    self.stock_features['q_{}_w_{}'.format(q, window_len)] = np.nan

                try:
                    self.stock_features['diff_close_q_{}_w_{}'.format(q, window_len)] = \
                        (self.close - self.stock_features['q_{}_w_{}'.format(q, window_len)]) / \
                        self.stock_features['q_{}_w_{}'.format(q, window_len)]
                except:
                    self.stock_features['diff_close_q_{}_w_{}'.format(q, window_len)] = np.nan

        # higher order moments
        for period in self.momentum_periods:
            returns = self.close.pct_change(period)
            for window in self.moving_windows:
                try:
                    self.stock_features['ret_{}_skew'.format(period)] = get_rolling_skew(returns=returns, win=window)
                except:
                    self.stock_features['ret_{}_skew'.format(period)] = np.nan

                try:
                    self.stock_features['ret_{}_kurtosis'.format(period)] = get_rolling_kurtosis(returns=returns,
                                                                                                 win=window)
                except:
                    self.stock_features['ret_{}_kurtosis'.format(period)] = np.nan

                for n in [5, 6, 7, 8, 9, 10]:
                    try:
                        self.stock_features['ret_{}_mom_{}'.format(period, n)] = get_rolling_moment(returns=returns,
                                                                                                    win=window, n=n)
                    except:
                        self.stock_features['ret_{}_mom_{}'.format(period, n)] = np.nan

        # linear and polynomial trend predictions
        # print('trends')
        for window in self.moving_windows:
            for pred_period in self.predict_periods:
                for poly_degree in [1, 2, 3, 4, 5, 6]:
                    try:
                        self.stock_features['trend_ret_{}_pred_deg_{}_win_{}'.format(pred_period, poly_degree, window)] = \
                            get_rolling_trend(price_array=self.close, win=window, deg=poly_degree, step=pred_period,
                                              is_return=True)
                    except:
                        self.stock_features[
                            'trend_ret_{}_pred_deg_{}_win_{}'.format(pred_period, poly_degree, window)] = np.nan

        # Hurst parameter
#         for window in self.moving_windows:
#             for mom_period in self.momentum_periods:
#                 try:
#                     self.stock_features['ret_{}_hurst_w_{}'.format(mom_period, window)] = \
#                         get_rolling_dfa(time_series=self.close.pct_change(mom_period), win=window)
#                 except:
#                     self.stock_features['ret_{}_hurst_w_{}'.format(mom_period, window)] = np.nan

#                 try:
#                     self.stock_features['ret_{}_lyap_exp_w_{}'.format(mom_period, window)] = \
#                          get_rolling_lyapunov_exponent(returns=self.close.pct_change(mom_period), win=window)
#                 except:
#                     self.stock_features['ret_{}_lyap_exp_w_{}'.format(mom_period, window)] = np.nan

        # Entropy
#         for window in self.moving_windows:
#             for mom_period in self.momentum_periods:
#                 try:
#                     self.stock_features['ret_{}_pe_w_{}'.format(mom_period, window)] = \
#                         rolling_pe(time_series=self.close.pct_change(mom_period),
#                                    win=window,
#                                    order=1,
#                                    delay=0,
#                                    normalize=False)
#                 except:
#                     self.stock_features['ret_{}_pe_w_{}'.format(mom_period, window)] = np.nan

                # try:
                #   self.stock_features['ret_{}_wpe_w_{}'.format(mom_period, window)] = \
                #        rolling_wpe(time_series=self.close.pct_change(mom_period),
                #                    win=window,
                #                    order=1,
                #                    delay=0,
                #                    normalize=False)
                # except:
                #    self.stock_features['ret_{}_wpe_w_{}'.format(mom_period, window)] = np.nan

        # LPPLS
        #try:
        #    lppls_pos, lppls_neg = get_lppls_confidence_indicators(price_ts=self.close)
        #    self.stock_features['lppls_pos'] = lppls_pos
        #    self.stock_features['lppls_neg'] = lppls_neg
        #except:
        #    self.stock_features['lppls_pos'] = np.nan
        #    self.stock_features['lppls_neg'] = np.nan

        # SSA
#         components = [3, 5, 10]
#         for window in self.moving_windows:
#             for l in components:
#                 for i in [0, -1]:
#                     try:
#                         ssa_mean = rolling_ssa_decomposition_stat(timeseries=self.close,
#                                                                   win=window,
#                                                                   l=l,
#                                                                   component_num=i)

#                         self.stock_features['ssa_w_{}_l_{}_i_{}_mean'.format(window, l, i)] = ssa_mean

#                         if i <= 1:
#                             self.stock_features['diff_close_ssa_w_{}_l_{}_i_{}_mean'.format(window, l, i)] = \
#                                 (self.close - ssa_mean) / ssa_mean

#                             self.stock_features['diff_sma_ssa_w_{}_l_{}_i_{}'.format(window, l, i)] = \
#                                 (self.stock_features['sma_w_{}'.format(window)] - ssa_mean) / ssa_mean

#                             self.stock_features['diff_ema_ssa_w_{}_l_{}_i_{}'.format(window, l, i)] = \
#                                 (self.stock_features['ema_w_{}'.format(window)] - ssa_mean) / ssa_mean

#                             self.stock_features['diff_close_ssa_w_{}_l_{}_i_{}_mean'.format(window, l, i)] = \
#                                 (self.close - ssa_mean) / ssa_mean
#                     except:
#                         self.stock_features['ssa_w_{}_l_{}_i_{}_mean'.format(window, l, i)] = np.nan

#                         if i <= 1:
#                             self.stock_features['diff_close_ssa_w_{}_l_{}_i_{}_mean'.format(window, l, i)] = np.nan
#                             self.stock_features['diff_sma_ssa_w_{}_l_{}_i_{}'.format(window, l, i)] = np.nan
#                             self.stock_features['diff_ema_ssa_w_{}_l_{}_i_{}'.format(window, l, i)] = np.nan
#                             self.stock_features['diff_close_ssa_w_{}_l_{}_i_{}_mean'.format(window, l, i)] = np.nan

        return self.stock_features
