import pandas
import pandas.io.data as web
import datetime
import matplotlib.pyplot as plt
import numpy

# TASK LIST
# 1. CONTEMPLATE SWITCH TO QUANDL
#       does it adjust for stock splits?
#       does it adjust for dividends (see how yahoo does this) ?
#       is the data accurate? compare with what i'm currently getting from yahoo?
# 2. ADD MORE FUNCTIONALITY FROM GOOGLE
# 3. HISTORICALS TAB                        DONE
#       STDEV
#       PEARSON CORREL & BETA
#       SPEARMAN CORREL & BETA
# 4. CORRELATION & BETA ANALYSIS TAB
#       notes, graphs, benchmark notes
# 5. HISTORICAL PORTFOLIO ANALYSIS

# perhaps switch to quandl (though I'd want to create adjusted close myself then)
# would be good experience working with api

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

def collinearity_pearson(returns):
    returns_std = numpy.std(returns) # stdevs
    returns_corr = returns.corr() # Pearson correlations
    returns_beta = returns_std.copy() # easily creates new series of same shape
    market_std = returns_std[0]
    for i in xrange(0,len(returns_std)):
        returns_beta[i] = returns_std[i] * returns_corr.iloc[i,0] / market_std
    return returns_std, returns_corr, returns_beta

def collinearity_spearman(returns):
    returns_std = numpy.std(returns) # stdevs
    spearman_corr = returns_std.copy() # easily creates new series of same shape
    spearman_beta = returns_std.copy() # see above
    for i in xrange(0,len(returns_std)):
        temp = returns.iloc[:,i]
        temp_sort = temp.argsort()
        ranks = numpy.empty(len(temp), int)
        ranks[temp_sort] = numpy.arange(len(temp))
        if i == 0: market_ranks = ranks
        diff = (ranks - market_ranks)**2
        spearman_corr[i] = 1- 6*diff.sum() / float((len(diff)) * (len(diff)**2-1))
        market_std = returns_std[0]
        spearman_beta[i] = spearman_corr[i] * returns_std[i] / market_std
    return returns_std, spearman_corr, spearman_beta

# main method

end_date = datetime.date.today()
symbols = ['SPY','GOOG','IBM','TLT','GLD','^VIX','VXX','UVXY','IWM','RWM','SH']

data = import_data(symbols, datetime.date.today())
returns = daily_returns(data)
returns_std, returns_corr, returns_beta = collinearity_spearman(returns)

print returns_std, returns_corr, returns_beta



