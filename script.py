import pandas
import pandas.io.data as web
import datetime
import matplotlib.pyplot as plt
import numpy

# ADD FUNCTIONALITY FROM GOOGLE SPREADSHEET
# 1. HISTORICALS TAB                        DONE
#       STDEV
#       PEARSON CORREL & BETA
#       SPEARMAN CORREL & BETA
# 2. CORRELATION & BETA ANALYSIS TAB        DONE
#       notes, graphs, benchmark notes
# 3. HISTORICAL PORTFOLIO ANALYSIS          in progress
#       current port analysis
#       historical analysis
#       port stats
#       compare multiple portfolios simultaneously!!!!!
# 4. DOWNLOAD DATA TO SAVE FOR LATER USE W/OUT INTERNET

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
    for i in range(0,len(returns_std)):
        returns_beta[i] = returns_std[i] * returns_corr.iloc[i,0] / market_std
    return returns_std, returns_corr, returns_beta

def collinearity_spearman(returns):
    returns_std = numpy.std(returns) # stdevs
    spearman_corr = returns_std.copy() # easily creates new series of same shape
    spearman_beta = returns_std.copy() # see above
    for i in range(0,len(returns_std)):
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

def historicalsPort(port):
    data = import_data(port.symbols, datetime.date.today())
    data.loc[:,'port'] = pandas.Series(0, index=data.index)
    for i in range(0,len(data.iloc[:,0])):
        for z in range(0,len(data.iloc[0,:])-1):
            data.iloc[i,len(port.symbols)] = port.amounts[z] * data.iloc[i,z] + data.iloc[i,len(port.symbols)]
    return data

def stockAmount(symbol,value):
    loc_data = import_data([symbol], datetime.date.today())
    loc_price = loc_data.iloc[len(loc_data)-1]
    loc_price = loc_price.iloc[0]
    amount = value / loc_price
    return amount

def stockValue(symbol,amount,daysAgo):
    loc_data = import_data([symbol], datetime.date.today())
    loc_price = loc_data.iloc[len(loc_data)-1-daysAgo]
    loc_price = loc_price.iloc[0]
    value = amount * loc_price
    return value

def printPort(port):
    rows = port.symbols + ['total']
    df = pandas.DataFrame(0, index = rows, columns = ['shares','price','curVal','cur%'])
    df.shares = port.amounts
    for i in range(0,len(port.symbols)) :
        df.iloc[i,0] = round (df.iloc[i,0], 2)
        df.iloc[i,1] = round ( stockValue(port.symbols[i],1,0) , 2 )
        df.iloc[i,2] = round ( stockValue(port.symbols[i],port.amounts[i],0) , 2 )
    df.iloc[len(rows)-1,0] = float('NaN')
    df.iloc[len(rows)-1,1] = float('NaN')
    df.iloc[len(rows)-1,2] = df.sum(axis=0)['curVal']
    for i in range(0,len(port.symbols)) :
        df.iloc[i,3] = round ( df.iloc[i,2] / df.iloc[len(rows)-1,2] , 2 )
    df.iloc[len(rows)-1,3] = df.sum(axis=0)['cur%']
    print '\nPortfolio components:', '\n', df, '\n'

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
    new = pandas.DataFrame(0,index=symbols+['port'], columns=['beta','absBeta','sumABOthers','priorNorm','weightedBeta','amount'])
    new['beta'] = loc_betas
    new['absBeta'] = numpy.absolute(loc_betas)
    for i in range(0,len(symbols)):
        for z in range(0,len(symbols)):
            if i != z:
                new.iloc[i,2] += new.iloc[z,1]

    new.iloc[len(symbols),1] = numpy.sum(numpy.absolute(new.absBeta))
    new.iloc[len(symbols),2] = numpy.sum(numpy.absolute(new.sumABOthers))

    for i in range(0,len(symbols)):
        new.iloc[i,3] = new.iloc[i,2] / new.iloc[len(symbols),2]
        new.iloc[i,4] = new.iloc[i,0] * new.iloc[i,3]
        new.iloc[i,5] = value * new.iloc[i,3] / loc_data.iloc[0,i]

    new.iloc[len(symbols),3] = numpy.sum(numpy.absolute(new.priorNorm))
    new.iloc[len(symbols),4] = numpy.sum(new.weightedBeta)

    return Portfolio(symbols,new.iloc[:len(new)-1,5])

def analyzePortfolio(port):
    printPort(port)

    data = pandas.DataFrame(historicalsPort(port), columns=port.symbols+['port','% change to today']) # adds cols to DF
    values = data.iloc[::-1]
    val_index = [0,1,2,3,4,5,21,63,125,len(data)-1]
    values = values.iloc[val_index,:]
    values.index = val_index

    for z in values.index:
        values['% change to today'][z] = round ( (values['port'][0] - values['port'][z]) / values['port'][z] * 100 , 1 )

    values = numpy.round(values, 2)

    print 'Historical values:\n', values, '\n'

    returns = daily_returns(data)
    evalDF = pandas.DataFrame(0, index= port.symbols+['port'], columns = ['std%','range%','1yGain%','beta','corr'])
    evalDF['std%'], evalDF['corr'], evalDF['beta'] = collinearity_spearman(returns)


    evalDF['range%'] = (data.max(axis=0) - data.min(axis=0)) / data.iloc[len(data)-1]
    evalDF['1yGain%'] = (data.iloc[len(data)-1] - data.iloc[0] ) / data.iloc[0]

    evalDF.iloc[:,0:3] = numpy.round(100*evalDF.iloc[:,0:3],2)
    evalDF.iloc[:,3:5] = numpy.round(evalDF.iloc[:,3:5], 2)
    # note: corr & beta are calculated here against stock 1
    print 'Historical statistics: \n', evalDF, '\n'

def comparePorts(arrayOfPorts):
    stocklist = []
    cols = []
    for i in range(0,len(arrayOfPorts)):
        stocklist += arrayOfPorts[i].symbols
        newCol = ['Port '+str(i)]
        cols += newCol
        printPort(arrayOfPorts[i])
    stocklist = [x.upper() for x in stocklist]
    stocks = set(stocklist)
    stocks = sorted(list(stocks))
    compare = pandas.DataFrame(0, index=stocks+['beta'], columns=cols)
    print compare
# main method
'''
end_date = datetime.date.today()
symbols = ['SPY','GOOG','IBM','TLT','GLD','^VIX','VXX','UVXY','IWM','RWM','SH']

data = import_data(symbols, datetime.date.today())
returns = daily_returns(data)
returns_std, returns_corr, returns_beta = collinearity_spearman(returns)
correlation_analysis('SPY','GOOG',1)
print returns_std, returns_corr, returns_beta
'''

x = zeroBetaPortfolio(['IWM','eem','GLD','TLT'],100000,1)
y = zeroBetaPortfolio(['SPY','TLT'],100000,1)
comparePorts([x,y])
#analyzePortfolio(x)
