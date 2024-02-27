import pandas as pd


def forecast_performance(combo_fcast):
    combo_fcast['year'] = combo_fcast.index.map(lambda x: x.year)

    yearly_performance = combo_fcast.groupby('year')
    corr_row = (lambda x: pd.DataFrame(x.corr().loc[
                                           ['auction_avg', 'diff', 'response'],
                                           ['long_term', 'short_term', 'forecast']].values.diagonal(),
                                       index=['lt_corr', 'st_corr', 'total_corr']
                                       ))

    yrly_corr = yearly_performance.apply(corr_row).unstack()
    yrly_corr.columns = yrly_corr.columns.droplevel(0)
    total_corr = combo_fcast.pipe(corr_row).T
    total_corr.index = ['total']
    yrly_corr = pd.concat([yrly_corr, total_corr], axis=0)

    # for yearly_slice  in yearly_performance:

    return yrly_corr
