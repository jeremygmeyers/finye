#source: http://pandas.pydata.org/pandas-docs/stable/remote_data.html
# git  : https://github.com/pydata/pandas-datareader

import pandas
import pandas.io.data as web
import datetime
import matplotlib.pyplot as plt
import numpy

# TASK LIST
# 1. CONTEMPLATE SWITCHING TO QUANDL
# 2. ADD MORE FUNCTIONALITY FROM GOOGLE
# 2A. STDEV                                DONE
# 2B. PEARSON CORREL & BETA                DONE
# 2C. SPEARMAN CORREL & BETA

# Note: Historical TLT data pulled disagrees with Google & Yahoo Finance for much of 2014-15
    # Perhaps this is due to frequent dividend adjustments. Has 12 dividends per year.

def import_data(symbols, end_date): # interval 365 days hardcoded
    start = end_date.replace(year=end_date.year-1)
    data = pandas.DataFrame()
    for sym in symbols:
        data[sym] = web.DataReader(sym, 'yahoo', start, end_date)['Adj Close']
    return data
    #Adj Close adjusts for dividends in the close price

def daily_returns(data):
    new_data = (-1) * numpy.log((data).shift(1) / data ) # differences for returns
    new_data = new_data[1:] # removes top row of NaNs
    return new_data

# main method

end_date = datetime.date.today()
symbols = ['SPY','GOOG','IBM','TLT','GLD','^VIX','VXX','UVXY','IWM','RWM','SH']
data = import_data(symbols, datetime.date.today())
returns = daily_returns(data)

returns_std = numpy.std(returns) # stdevs
returns_corr = returns.corr() # Pearson correlations
returns_beta = returns_std # easily creates new series of same shape
market_std = returns_std[0]

for i in xrange(0,len(returns_std)):
    returns_beta[i] = returns_std[i] * returns_corr.iloc[i,0] / market_std

print returns_beta
