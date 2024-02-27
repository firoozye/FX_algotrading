import pandas as pd


class TradingStrategy(object):
    '''
    TradingStrategy is merely to hold all the elements
    prices,
    positions
    costs
    PnL(T) = Sum_1^T \theta_t \cdot dP_t - \sum |\Delta \theta_t |\cdot costs(t)
    where \theta_t = weights conditioned on F_t and
    dP_t = P(t+1) - P(t)  = p.diff().shift(-1)

    There is a separate horizon, in case we want to consider interacting with
    dP_{t+horizon} = P(t+horizon-1) - P(t)
    '''
    def __init__(self, prices: pd.DataFrame | None = None,
                 costs: pd.DataFrame | None = None,
                 position: pd.DataFrame | None = None,
                 horizon: int = 1):
        assert prices.index.equals(position.index)
        assert set(prices.keys()) == set(position.keys())
        # assert horizon >= 1
        # avoid duplicates
        assert not prices.index.has_duplicates, "Price Index has duplicates"
        assert not position.index.has_duplicates, "Position Index has duplicates"

        assert prices.index.is_monotonic_increasing, "Price Index is not increasing"
        assert position.index.is_monotonic_increasing, "Position Index is not increasing"

        if costs is None:
            costs = prices.copy() * 0.0

        self.__prices = prices
        self.__price_costs = costs
        self.__costs = costs / prices
        self.__position = position
        self.__horizon = horizon

    @property
    def prices(self):
        return self.__prices

    @property
    def position(self):
        return self.__position

    @property
    def horizon(self):
        return self.__horizon

    @property
    def costs(self):
        return self.__costs

    @property
    def profit(self)->pd.Series:
        steps_forward = -1 * self.horizon
        costs_steps_forward = -1 * (self.horizon - 1)
        return (
            (
                    (self.prices.pct_change().shift(periods= steps_forward) * self.position) -
                    (self.costs.shift(periods=costs_steps_forward) * self.position.diff().abs())
            ).sum(axis=1)
        )


    @property
    def gross_profit(self)->pd.Series:
        steps_forward = -1 * self.horizon
        return (
            (
                (self.prices.pct_change().shift(periods=steps_forward) * self.position)
            ).sum(axis=1)
        )

    def cumulative_returns(self, init_capital=None, gross=False):

        if gross:
            init_capital = init_capital or 100 * self.gross_profit.std()
            # We assume we start every day with the same initial capital!
            r = self.gross_profit / init_capital
        else:
            # common problem for most CTAs.
            init_capital = init_capital or 100 * self.profit.std()
            # We assume we start every day with the same initial capital!
            r = self.profit / init_capital
        # We then simply compound the nav!
        # We could also achieve the same by scaling the positions with increasing fundsize...
        return (1 + r).cumprod()

    def individual_asset_pnl(self)->pd.DataFrame:
        steps_forward = -1 * self.horizon
        costs_steps_forward = -1 * (self.horizon - 1)
        return (
            (
                    (self.prices.pct_change().shift(periods= steps_forward) * self.position) -
                    (self.costs.shift(periods=costs_steps_forward) * self.position.diff().abs())
            )
        )

    def turnover(self) -> pd.Series:
        pass