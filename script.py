import pandas
import pandas.io.data
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
# 3. HISTORICAL PORTFOLIO ANALYSIS          DONE
#       current port analysis
#       historical analysis
#       port stats
#       compare multiple portfolios
# 4. DOWNLOAD DATA & REUSE (for sans wifi)  next
#       stock data -- done
#       options data -- wait
# 5. MARKET AWARENESS (see google/port)     in progress
#       need to pull options data/IV/IVR
#       will need to further rework import_data functionality
# 6. DEVELOP INTERFACE
#       input should be happening interactively, perhaps with web app
# 7. FURTHER UNDERSTAND THE DATA
#       how does div adj happen? does it explain the difference for TLT?
#       Historical TLT data pulled disagrees with Google & Yahoo Finance for much of 2014-15. Perhaps this is due to frequent dividend adjustments. Has 12 dividends per year.


# NEXT
# how to calculate delta?

def option_data(symbol):
    data = pandas.io.data.Options(symbol,'yahoo')
    data = data.get_call_data(expiry='2015-11-20')
    # EXPIRY SHOULD UPDATE FLEXIBLY!!!!!!!!
    data['IV'] = data['IV'].replace('%','',regex=True).astype('float')/100
    dataIV = pandas.Series(data['IV']).reset_index(drop=True)
    # should prob keep the strikes as the index!!!, so i can use later
    print symbol, ' ', numpy.median(dataIV)
    # median is not really what i want, it's IV with delta = 30

def options_data(symbols):
    for x in range(0,len(symbols)):
        option_data(symbols[x])


options_data(['aapl','goog'])
# i also want price of the option, that div by stock price

def import_data(symbols, end_date): # interval 365 days hardcoded
    name = ''.join(symbols)+str(end_date)+'.csv'
    try:
        data = pandas.read_csv(name,index_col=0)
    except IOError:
        start = end_date.replace(year=end_date.year-1)
        data = pandas.DataFrame()
        for sym in symbols:
            data[sym] = pandas.io.data.DataReader(sym, 'yahoo', start, end_date)['Adj Close']
    data.to_csv(name)
    return data
    #Adj Close adjusts for dividends in the close price

def daily_returns(data):
    new_data = (-1) * numpy.log((data).shift(1) / data ) # differences for returns
    new_data = new_data[1:] # removes top row of NaNs
    return new_data

def collinearity(returns,spearman): #will add a symbol for beta/corr calc.
    #assumes returns input are up to date, since compares with uptodate SPY
    numsym = len(returns.columns)
    spy = import_data(['SPY'], datetime.date.today() )
    returnsSPY = daily_returns(spy)
    returns = pandas.DataFrame(returns,index=returns.index)
    returns = pandas.concat([returns,returnsSPY],axis=1)
    returns_std = numpy.std(returns) # stdevs
    returns_corr = returns.corr() # Pearson corrs
    returns_corr = returns.corr().iloc[:,numsym] # Pearson corrs series
    returns_beta = returns_std.copy() # easily creates new series of same shape
    market_std = returns_std[numsym]
    if spearman == 0:
        for i in range(0, numsym):
            returns_beta[i] = returns_std[i] * returns_corr.iloc[i] / market_std
    if spearman == 1:
        for i in range(numsym,-1,-1):
            temp = returns.iloc[:,i]
            temp_sort = temp.argsort()
            ranks = numpy.empty(len(temp), int)
            ranks[temp_sort] = numpy.arange(len(temp))
            if i == numsym: market_ranks = ranks
            diff = (ranks - market_ranks)**2
            returns_corr[i] = 1- 6*diff.sum() / float((len(diff)) * (len(diff)**2-1))
            market_std = returns_std[0]
            returns_beta[i] = returns_corr[i] * returns_std[i] / market_std
    return returns_std.iloc[:-1], returns_corr.iloc[:-1], returns_beta.iloc[:-1]

def correlation_analysis(stockA, stockB, setPrint):
    # if you want stock beta against index, set stockA=index
    # assumes 1 year interval since import_data assumes that
    loc_data = import_data([stockA,stockB], datetime.date.today())
    loc_returns = daily_returns(loc_data)

    loc_stds, loc_pearson_corrs, loc_pearson_betas = collinearity(loc_returns,0)
    loc_stds, loc_spearman_corrs, loc_spearman_betas = collinearity(loc_returns,1)
    if setPrint == 1:
        print '\nCorrelation Analysis of ', stockA, 'and ', stockB, '(1 year of data) \n'
        print 'Standard Deviations: \n', loc_stds, '\n'

        result = pandas.concat([loc_spearman_betas, loc_spearman_corrs, loc_pearson_betas, loc_pearson_corrs], axis=1)
        result.columns = ['Spearman Beta','Spearman Corr','Pearson Beta','Pearson Corr']
        print 'Spearman Beta & Correlation: \n', result

        loc_data = import_data([stockA,stockB],datetime.date.today())
        loc_returns = daily_returns(loc_data)
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
        graph.clf()

class Portfolio(object):
    def __init__(self, symbols, amounts):
        self.symbols = symbols # stock symbols
        self.amounts = amounts # amount of stock held in each

def historicalsPort(port):
    data = import_data(port.symbols, datetime.date.today())
    data.loc[:,'port'] = pandas.Series(0, index=data.index)
    for i in range(0,len(data.iloc[:,0])): # across rows
        for z in range(0,len(data.iloc[0,:])-1): # across cols
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

def portSum(port,toPrint):
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
    if toPrint==1 :
        print '\nPortfolio components:', '\n', df, '\n'
    return df

def zeroBetaPortfolio(symbols,value,spearman):
    # this will go long all positions
    # spearman = 0 uses pearson, spearman = 1 uses spearman
    loc_data = import_data(symbols, datetime.date.today())
    loc_returns = daily_returns(loc_data)
    loc_stds, loc_corrs, loc_betas = collinearity(loc_returns,spearman)
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

def analyzePortfolio(port,printToScreen):
    data = pandas.DataFrame(historicalsPort(port), columns=port.symbols+['port','% change to today']) # adds cols to DF
    values = data.iloc[::-1]
    val_index = [0,1,2,3,4,5,21,63,125,len(data)-1]
    values = values.iloc[val_index,:]
    values.index = val_index

    for z in values.index:
        values['% change to today'][z] = round ( (values['port'][0] - values['port'][z]) / values['port'][z] * 100 , 1 )

    values = numpy.round(values, 2)

    returns = daily_returns(data)
    evalDF = pandas.DataFrame(0, index= port.symbols+['port'], columns = ['std%','range%','1yGain%','beta','corr'])
    evalDF['std%'], evalDF['corr'], evalDF['beta'] = collinearity(returns,1)

    evalDF['range%'] = (data.max(axis=0) - data.min(axis=0)) / data.iloc[len(data)-1]
    evalDF['1yGain%'] = (data.iloc[len(data)-1] - data.iloc[0] ) / data.iloc[0]

    evalDF.iloc[:,0:3] = numpy.round(100*evalDF.iloc[:,0:3],2)
    evalDF.iloc[:,3:5] = numpy.round(evalDF.iloc[:,3:5], 2)
    # note: corr & beta are calculated here against stock 1
    #NEED TO FIX THAT!!!!!!!!!
    overview = portSum(port,printToScreen)
    if printToScreen == 1:
        print 'Historical values:\n', values, '\n'
        print 'Historical statistics: \n', evalDF, '\n'
    return overview, values, evalDF

def comparePorts(arrayOfPorts):
    stocklist = []
    cols = []
    output = []
    for i in range(0,len(arrayOfPorts)):
        stocklist += arrayOfPorts[i].symbols
        newCol = ['Port '+str(i)]
        cols += newCol
        output.append( analyzePortfolio(arrayOfPorts[i],0) )
    stocklist = [x.upper() for x in stocklist]
    stocks = set(stocklist)
    stocks = sorted(list(stocks))
    compare = pandas.DataFrame(0, index=stocks+['std%','range%','1yGain%','beta','corr'], columns=cols)
    for c in range(0,len(compare.columns)):      # across cols
        for r in range(0, len(compare.index)):   # across rows

            try:
                if output[c][0]['cur%'].loc[compare.iloc[r].name]:
                    compare.iloc[r,c] = output[c][0]['cur%'].loc[compare.iloc[r].name]        # requires all caps
                #print 'yay'
            except KeyError:
                keyError = 1
                #print 'bo!'
            try:
                if output[c][2].iloc[len(arrayOfPorts[c].symbols),:].loc[compare.iloc[r].name] :
                    compare.iloc[r,c] = output[c][2].iloc[len(arrayOfPorts[c].symbols),:].loc[compare.iloc[r].name]
            except KeyError:
                keyError = 1
    print compare


# main method

#end_date = datetime.date.today()
#symbols = ['GOOG','SPY']#,'IBM','TLT','SPY','GLD','^VIX','VXX','UVXY','IWM','RWM','SH']
#data = import_data(symbols,end_date)
#returns = daily_returns(data)
#returns_std, returns_corr, returns_beta = collinearity(returns,0)
#correlation_analysis('SPY','GOOG',1)
#print returns_std, '\n',returns_corr,'\n', returns_beta

#aapl = pandas.io.data.Options('AAPL')
#puts,calls = aapl.get_options_data()
#print puts,calls

'''
x = zeroBetaPortfolio(['IWM','EEM','GLD','TLT'],100000,1)
y = zeroBetaPortfolio(['IWM','SH','X'],100000,1)
#print collinearity(daily_returns(historicalsPort(x)),1)
comparePorts([x,y])
#analyzePortfolio(x,1)
#correlation_analysis('IWM','TLT',1)
'''
