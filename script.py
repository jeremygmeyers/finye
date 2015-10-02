#source: http://pandas.pydata.org/pandas-docs/stable/remote_data.html
# git  : https://github.com/pydata/pandas-datareader

import pandas
import pandas.io.data as web
import datetime
import matplotlib.pyplot as plt
import numpy

# data import, with hardcoded:
    # interval = 365
    # end = today
    # symbols

def import_data(interval, end_date):


interval = 365
end = datetime.date.today()
start = end.replace(year=end.year-1)

symbols = ['SPY','GOOG','IBM','TLT','GLD','^VIX','VXX','UVXY','IWM','RWM','SH']

data = pandas.DataFrame()
for sym in symbols:
    data[sym] = web.DataReader(sym, 'yahoo', start, end)['Adj Close']
    #Adj Close adjusts for dividends in the close price

# end of data import

def daily_returns(data):
    new_data = (-1) * numpy.log(data.shift(1) / data ) # differences for returns
    new_data = new_data[1:] # removes top row of NaNs
    return new_data

# main method

data = import_data(365, datetime.date.today())
print daily_returns(data)
