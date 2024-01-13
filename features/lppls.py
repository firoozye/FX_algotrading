import multiprocessing
from numba import njit
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize


class LPPLS(object):
    def __init__(self, observations):
        """
        Parameters
        ----------
            observations : np.array/pd.DataFrame of shape (2, M)
                2xM matrix with timestamp and observed value.
        """
        assert isinstance(observations, (np.ndarray, pd.DataFrame)), \
            f'Expected observations to be <pd.DataFrame> or <np.ndarray>, got :{type(observations)}'

        self.observations = observations
        self.coef_ = {}
        self.indicator_result = []

    @staticmethod
    @njit
    def lppls(t, tc, m, w, a, b, c1, c2):
        return a + np.power(tc - t, m) * (b + ((c1 * np.cos(w * np.log(tc - t))) + (c2 * np.sin(w * np.log(tc - t)))))

    def func_restricted(self, x, *args):
        """
        Finds the least square difference.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        Parameters
        ----------
            x : np.ndarray (n,)
                1-D array with shape (n,).
            args : tuple
                Tuple of the fixed parameters needed to completely specify the function.
        Returns
        -------
            ls_difference : float
                least square difference
        """

        tc = x[0]
        m = x[1]
        w = x[2]
        obs = args[0]

        a, b, c1, c2 = self.matrix_equation(obs, tc, m, w)

        delta = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in obs[0, :]]
        delta = np.subtract(delta, obs[1, :])
        delta = np.power(delta, 2)
        ls_difference = np.sum(delta)

        return ls_difference

    @staticmethod
    @njit
    def matrix_equation(observations, tc, m, w):
        """
        Derive linear parameters in LPPLs from nonlinear ones.
        """
        T = observations[0]
        P = observations[1]
        deltaT = tc - T
        phase = np.log(deltaT)
        fi = np.power(deltaT, m)
        gi = fi * np.cos(w * phase)
        hi = fi * np.sin(w * phase)
        A = np.stack((np.ones_like(deltaT), fi, gi, hi))

        return np.linalg.lstsq(A.T, P, rcond=-1.0)[0]

    def fit(self, observations, max_searches, minimizer='Nelder-Mead'):
        """
        Parameters
        ----------
            observations : np.ndarray (M,2)
                the observed time-series data
            max_searches : int
                The maxi amount of searches to perform before giving up. The literature suggests 25
            minimizer : str
                See list of valid methods to pass to scipy.optimize.minimize:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        Returns
        -------
            tc, m, w, a, b, c, c1, c2
                LPPLS model parameters' estimation
        """
        search_count = 0
        # find bubble
        while search_count < max_searches:
            tc_init_min, tc_init_max = self._get_tc_bounds(observations, 0.20, 0.20)

            # set random initialization limits for non-linear params
            init_limits = [
                (tc_init_min, tc_init_max),  # tc : Critical Time
                (0, 2),  # m : 0.1 ≤ m ≤ 0.9
                (1, 50),  # ω : 6 ≤ ω ≤ 13
            ]

            # randomly choose vals within bounds for non-linear params
            non_lin_vals = [random.uniform(a[0], a[1]) for a in init_limits]

            tc = non_lin_vals[0]
            m = non_lin_vals[1]
            w = non_lin_vals[2]
            seed = np.array([tc, m, w])

            # Increment search count on SVD convergence error, but raise all other exceptions.
            try:
                tc, m, w, a, b, c, c1, c2 = self.minimize(observations, seed, minimizer)
                return tc, m, w, a, b, c, c1, c2
            except (np.linalg.LinAlgError, UnboundLocalError, ValueError):
                search_count += 1

        return 0, 0, 0, 0, 0, 0, 0, 0

    def minimize(self, observations, seed, minimizer):
        """
        Parameters
        ----------
            observations : np.ndarray
                the observed time-series data
            seed : list
                time-critical, omega, and m.
            minimizer : str
                See list of valid methods to pass to scipy.optimize.minimize:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        Returns
        -------
            tc, m, w, a, b, c, c1, c2
                LPPLS model parameters' estimation
        """

        cofs = minimize(
            args=observations,
            fun=self.func_restricted,
            x0=seed,
            method=minimizer
        )

        if cofs.success:

            tc = cofs.x[0]
            m = cofs.x[1]
            w = cofs.x[2]

            a, b, c1, c2 = self.matrix_equation(observations, tc, m, w)
            c = (c1 ** 2 + c2 ** 2) ** 0.5

            # Use sklearn format for storing fit params
            for coef in ['tc', 'm', 'w', 'a', 'b', 'c', 'c1', 'c2']:
                self.coef_[coef] = eval(coef)
            return tc, m, w, a, b, c, c1, c2
        else:
            raise UnboundLocalError

    def mp_compute_indicator(self, workers, window_size=80, smallest_window_size=20, increment=5, max_searches=25,
                             filter_conditions_config=[]):
        obs_copy = self.observations
        obs_copy_len = len(obs_copy[0, :]) - window_size

        func = self._func_compute_indicator

        func_arg_map = [(
            obs_copy[:, i:window_size + i],  # obs
            i,  # n_iter
            window_size,  # window_size
            smallest_window_size,  # smallest_window_size
            increment,  # increment
            max_searches,  # max_searches
            filter_conditions_config,
        ) for i in range(obs_copy_len)]

        pool = multiprocessing.Pool(processes=workers)

        result = pool.map(func, func_arg_map)
        pool.close()

        self.indicator_result = result
        return result

    def _func_compute_indicator(self, args):

        obs, n_iter, window_size, smallest_window_size, increment, max_searches, filter_conditions_config = args

        n_fits = (window_size - smallest_window_size) // increment

        res = []

        # run n fits on the observation slice.
        for j in range(n_fits):
            obs_shrinking_slice = obs[:, j * increment:window_size + n_iter]

            # fit the model to the data and get back the params
            tc, m, w, a, b, c, c1, c2 = self.fit(obs_shrinking_slice, max_searches, minimizer='SLSQP')

            first = obs_shrinking_slice[0][0]
            last = obs_shrinking_slice[0][-1]

            qualified = {}
            # filter_conditions_config = [
            #   {'condition_1':[tc_range, m_range, w_range, O_min, D_min]},
            #   {'condition_2':[tc_range, m_range, w_range, O_min, O_min]}
            # ]
            for condition in filter_conditions_config:
                for value in condition:
                    tc_min, tc_max = condition[value][0]
                    m_min, m_max = condition[value][1]
                    w_min, w_max = condition[value][2]
                    O_min = condition[value][3]
                    D_min = condition[value][4]

                    tc_init_min, tc_init_max = self._get_tc_bounds(obs_shrinking_slice, tc_min, tc_max)

                    tc_in_range = last - tc_init_min < tc < last + tc_init_max
                    m_in_range = m_min < m < m_max
                    w_in_range = w_min < w < w_max

                    O_in_range = ((w / (2 * np.pi)) * np.log(abs(tc / (tc - last)))) > O_min

                    D_in_range = (m * abs(b)) / (w * abs(c)) > D_min if m > 0 and w > 0 else False

                    if tc_in_range and m_in_range and w_in_range and O_in_range and D_in_range:
                        is_qualified = True
                    else:
                        is_qualified = False

                    qualified[value] = is_qualified

            sign = 1 if b < 0 else -1

            res.append({
                'tc': tc,
                'm': m,
                'w': w,
                'a': a,
                'b': b,
                'c': c,
                'c1': c1,
                'c2': c2,
                'qualified': qualified,
                'sign': sign,
                't1': first,
                't2': last,
            })

        return res

    def _get_tc_bounds(self, obs, lower_bound_pct, upper_bound_pct):
        """
        Parameters
        ----------
            obs : np.ndarray (M,2)
                the observed data
            lower_bound_pct : float
                percent of (t_2 - t_1) to use as the LOWER bound initial value for the optimization
            upper_bound_pct : float
                percent of (t_2 - t_1) to use as the UPPER bound initial value for the optimization
        Returns
        -------
            tc_init_min, tc_init_max
        """
        t_first = obs[0, 0]
        t_last = obs[0, -1]
        t_delta = t_last - t_first
        pct_delta_min = t_delta * lower_bound_pct
        pct_delta_max = t_delta * upper_bound_pct
        tc_init_min = t_last - pct_delta_min
        tc_init_max = t_last + pct_delta_max
        return tc_init_min, tc_init_max

    def res_to_df(self, res, condition_name):
        """
        Parameters
        ----------
            res : list
                result from mp_compute_indicator
            condition_name : str
                the name you assigned to the filter condition in your config
        Returns
        -------
            pd.DataFrame()
        """
        idx = self.observations[0, :]
        price = self.observations[1, :]
        n = len(price) - len(res)
        # pos_fits = [0] * n
        # neg_fits = [0] * n
        pos_conf_lst = [0] * n
        neg_conf_lst = [0] * n
        fits_ = [0] * n

        for r in res:
            pos_count = 0
            neg_count = 0
            pos_true_count = 0
            neg_true_count = 0
            for fits in r:
                if fits['sign'] > 0:
                    pos_count += 1
                    if fits['qualified'][condition_name]:
                        pos_true_count += 1
                if fits['sign'] < 0:
                    neg_count += 1
                    if fits['qualified'][condition_name]:
                        neg_true_count += 1
            # pos_conf_lst.append(pos_true_count / len(r))
            # neg_conf_lst.append(neg_true_count / len(r))
            fits_.append(fits)
            pos_conf_lst.append(pos_true_count / pos_count if pos_count > 0 else 0)
            neg_conf_lst.append(neg_true_count / neg_count if neg_count > 0 else 0)

        return pd.DataFrame({
            'idx': idx,
            'price': price,
            'pos_conf': pos_conf_lst,
            'neg_conf': neg_conf_lst,
            'fit_params': fits_,
        }).set_index('idx')


def get_lppls_confidence_indicators(price_ts):
    """
    Parameters
    ----------
        price_ts : list/pd.Series/np.ndarray
            historical price data (not in logs!)
    Returns
    -------
        lppls_ci : list
            list with 2 np.ndarray's (positive bubble confidence indicators, negative bubble confidence indicators)
    """
    price = np.log(price_ts)
    time = np.arange(price.shape[0])
    observations = np.array([time, price])
    lppls_model = LPPLS(observations=observations)

    # define custom filter condition
    filter_conditions_config = [
        {'condition_1': [
            (0.0, 0.1),  # tc_range
            (0, 1),  # m_range
            (4, 25),  # w_range
            2.5,  # O_min
            0.5,  # D_min
        ]},
    ]

    # compute the confidence indicator
    res = lppls_model.mp_compute_indicator(
        workers=4,
        window_size=120,
        smallest_window_size=30,
        increment=5,
        max_searches=25,
        filter_conditions_config=filter_conditions_config
    )

    res_df = lppls_model.res_to_df(res, condition_name='condition_1')
    lppls_ci = [res_df.pos_conf.values, res_df.neg_conf.values]

    return lppls_ci
