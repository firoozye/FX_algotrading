from features.feature_module import *
import warnings
warnings.filterwarnings('ignore')

ticker = 'AAPL'
startdate='2020-01-01'
df = Features(ticker=ticker,
         startdate=startdate,
         interval='1d',
         list_of_windows=[120],
         momentum_periods=[21],
         quantiles=[0.01],
         predict_periods=[21],
         offsets=[21]).get_features()
print(df)
