#source: http://pandas.pydata.org/pandas-docs/stable/remote_data.html
# git  : https://github.com/pydata/pandas-datareader

import pandas
import pandas.io.data as web
import datetime
import matplotlib.pyplot as plt
import numpy

# TASK LIST
# 1. CONTEMPLATE SWITCHING TO QUANDL
    # READ https://blog.thinknewfound.com/2014/06/tracking-down-data/1:
# 2. ADD MORE FUNCTIONALITY FROM GOOGLE
# 2A. STDEV
# 2B. PEARSON CORREL & BETA
# 2C. SPEARMAN CORREL & BETA

# Note: Historical TLT data pulled disagrees with Google & Yahoo Finance for much of 2014-15
    # Perhaps this is due to frequent dividend adjustments. Has 12 dividends per year.

# data import, with hardcoded:
    # end = today
    # symbols

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
print data
print "RETURNS"
print daily_returns(data)

