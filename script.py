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
# 4. CORRELATION & BETA ANALYSIS TAB        DONE
#       notes, graphs, benchmark notes
# 5. HISTORICAL PORTFOLIO ANALYSIS

# programing technique - create portfolio object

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

def correlation_analysis(stockA, stockB, setPrint):
    # if you want stock beta against index, set stockA=index
    # assumes 1 year interval since import_data assumes that
    loc_data = import_data([stockA,stockB],datetime.date.today())
    loc_returns = daily_returns(loc_data)
    loc_stds, loc_pearson_corrs, loc_pearson_betas = collinearity_pearson(loc_returns)
    loc_stds, loc_spearman_corrs, loc_spearman_betas = collinearity_spearman(loc_returns)
    if setPrint == 1:
        print '\nCorrelation Analysis of ', stockA, 'and ', stockB, '(1 year of data) \n'
        print 'Standard Deviations: \n', loc_stds, '\n'

        result = pandas.concat([loc_spearman_betas, loc_spearman_corrs, loc_pearson_betas, loc_pearson_corrs.iloc[0]], axis=1)
        result.columns = ['Spearman Beta','Spearman Corr','Pearson Beta','Pearson Corr']
        print 'Spearman Beta & Correlation: \n', result

        graph = plt.figure()
        plt.grid(True)
        plt.title('Correlation measures spread around beta line',fontsize=20,fontweight='bold')
        plt.scatter(loc_returns.iloc[:,0],loc_returns.iloc[:,1])
        plt.xlabel(stockA, fontsize=20)
        plt.ylabel(stockB, fontsize=20)

        # add regression line (uses pearson data, fits data better)
        x_min = min(loc_returns.iloc[:,0])
        x_max = max(loc_returns.iloc[:,0])
        y_line_max = x_max * loc_pearson_corrs.iloc[1]
        y_line_min = x_min * loc_pearson_corrs.iloc[1]

        plt.plot( (x_min, x_max), (y_line_min, y_line_max), '-', color='red')

        plt.show()
        graph.savefig('graph.jpg')
        graph.clear()
        graph.close()
        graph.clf()

class Portfolio(object):
    def __init__(self, symbols, amounts):
        self.symbols = symbols # stock symbols
        self.amounts = amounts # amount of stock held in each

def stockAmount(symbol,value):
    loc_data = import_data(symbol, datetime.date.today())
    loc_price = loc_data.iloc[len(loc_data)-1]
    loc_price = loc_price.iloc[0]
    amount = value / loc_price
    return amount

def stockValue(symbol,amount):
    loc_data = import_data(symbol, datetime.date.today())
    loc_price = loc_data.iloc[len(loc_data)-1]
    loc_price = loc_price.iloc[0]
    value = amount * loc_price
    return value

def zeroBetaPortfolio(symbols,value,spearman):
    # this will go long all positions
    # spearman = 0 uses pearson, spearman = 1 uses spearman
    loc_data = import_data(symbols, datetime.date.today())
    loc_returns = daily_returns(loc_data)
    if spearman == 0:
        loc_stds, loc_corrs, loc_betas = collinearity_pearson(loc_returns)
    if spearman == 1:
        loc_stds, loc_corrs, loc_betas = collinearity_spearman(loc_returns)
    abs_sum_beta = numpy.sum(numpy.absolute(loc_betas))
    loc_values = loc_stds.copy() # easily creates new array with same shape
    loc_amounts = loc_stds.copy() # "
    for i in xrange(0,len(symbols)):
        loc_values[i] = ( abs_sum_beta - numpy.absolute(loc_betas[i]) ) / abs_sum_beta * value
        loc_amounts[i] = stockAmount(symbols[i],loc_values[i])
    return Portfolio(symbols,loc_amounts)

def analyzePortfolio(port):
    totalValue = 0
    print '\nthis portfolio has '
    for i in xrange(0,len(port.symbols)):
        shareValue = stockValue(port.symbols[i],port.amounts[i])
        totalValue = totalValue + shareValue
        print port.amounts[i], ' shares of ', port.symbols[i], ' which is $', shareValue
    print '\n total value is $', totalValue

# main method
'''
end_date = datetime.date.today()
symbols = ['SPY','GOOG','IBM','TLT','GLD','^VIX','VXX','UVXY','IWM','RWM','SH']

data = import_data(symbols, datetime.date.today())
returns = daily_returns(data)
returns_std, returns_corr, returns_beta = collinearity_spearman(returns)
print 'hereC'
#correlation_analysis('SPY','GOOG',1)
#print returns_std, returns_corr, returns_beta
'''

x = zeroBetaPortfolio(['SPY','TLT'],100000,1)
analyzePortfolio(x)
