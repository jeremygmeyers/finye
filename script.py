import pandas
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates
import numpy

# problem - latest 1dmove is incorrect, prob b/c i'm losing last day price
# somewhere in datafeed from stocks - historicalPort / analyzePort

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
# 4. DOWNLOAD DATA & REUSE (for sans wifi)  in progress
#       stock data -- done
#       options data -- wait                in progress!!!
# 5. MARKET AWARENESS + OPTIONS             in progress
#       pull options data
#
# 6. DEVELOP INTERFACE
#       input should be happening interactively, perhaps with web app
# 7. FURTHER UNDERSTAND THE DATA

# LONG TERM
# how to calculate delta?

# convert csv from TOS into a different format I can analyze

#       how does div adj happen? does it explain the difference for TLT?
#       Historical TLT data pulled disagrees with Google & Yahoo Finance for much of 2014-15. Perhaps this is due to frequent dividend adjustments. Has 12 dividends per year.

def import_data(symbols, start_date, end_date): # vary close type!
    name = ''.join(symbols)+str(start_date)+str(end_date)+'.csv'
    try:
        data = pandas.read_csv(name,index_col=0)
    except IOError:
        data = pandas.DataFrame()
        for sym in symbols:
            data[sym] = web.DataReader(sym, 'yahoo', start_date, end_date)['Adj Close']
            last_price = round ( option_data(sym).iloc[0,11] , 2)
            # on occasion the latest day close will not appear
            if round ( float ( data[sym].iloc[len(data)-1] ) , 2) != last_price :
                data = data.append(pandas.Series(last_price,[end_date]).to_frame(sym))
        data.to_csv(name)
    return data
    # Adj Close adjusts for dividends in the close price

def daily_returns(data):
    new_data = (-1) * numpy.log((data).shift(1) / data ) # differences for returns
    new_data = new_data[1:] # removes top row of NaNs
    return new_data

def collinearity(returns,spearman): #will add a symbol for beta/corr calc.
    #assumes returns input are up to date, since compares with uptodate SPY
    numsym = len(returns.columns)-1
    spy = import_data(['SPY'], date_num_years_ago(datetime.date.today(),1), datetime.date.today() )
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
            returns_beta[i] = returns_corr[i] * returns_std[i] / market_std
    return returns_std.iloc[:-1], returns_corr.iloc[:-1], returns_beta.iloc[:-1]

def correlation_analysis(stockA, stockB, setPrint):
    # if you want stock beta against index, set stockA=index
    # assumes 1 year interval since import_data assumes that
    loc_data = import_data([stockA,stockB], date_num_years_ago(datetime.date.today(),1), datetime.date.today())
    loc_returns = daily_returns(loc_data)

    loc_stds, loc_pearson_corrs, loc_pearson_betas = collinearity(loc_returns,0)
    loc_stds, loc_spearman_corrs, loc_spearman_betas = collinearity(loc_returns,1)
    if setPrint == 1:
        print '\nCorrelation Analysis of ', stockA, 'and ', stockB, '(1 year of data) \n'
        print 'Standard Deviations: \n', loc_stds, '\n'

        result = pandas.concat([loc_spearman_betas, loc_spearman_corrs, loc_pearson_betas, loc_pearson_corrs], axis=1)
        result.columns = ['Spearman Beta','Spearman Corr','Pearson Beta','Pearson Corr']
        print 'Spearman Beta & Correlation: \n', result

        loc_data = import_data([stockA,stockB],date_num_years_ago(datetime.date.today(),1), datetime.date.today())
        loc_returns = daily_returns(loc_data)
        graph = plt.figure()
        plt.grid(True)
        plt.title('Correlation measures spread around beta line',fontsize=20,fontweight='bold')
        plt.scatter(loc_returns.iloc[:,0],loc_returns.iloc[:,1])
        plt.xlabel(stockA, fontsize=20)
        plt.ylabel(stockB, fontsize=20)

        # adds regression line (uses pearson data, fits data better)
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
        self.symbols = []
        self.amounts = []
        for x in range(0,len(amounts)):
            self.symbols.append( symbols[x] ) # stock symbols
            self.amounts.append( amounts[x] ) # amount of stock held in each

def historicalsPort(port):
    data = import_data(port.symbols, date_num_years_ago(datetime.date.today(),1), datetime.date.today())
    data.loc[:,'port'] = pandas.Series(0, index=data.index)
    for i in range(0,len(data.iloc[:,0])): # across rows
        for z in range(0,len(data.iloc[0,:])-1): # across cols
            data.loc[data.index[i],'port'] = port.amounts[z] * data.iloc[i,z] + data.iloc[i,len(port.symbols)]
    return data

def stockAmount(symbol,value):
    loc_data = import_data([symbol], date_num_years_ago(datetime.date.today(),1), datetime.date.today())
    loc_price = loc_data.iloc[len(loc_data)-1]
    loc_price = loc_price.iloc[0]
    amount = value / loc_price
    return amount

def stockValue(symbol,amount,daysAgo):
    loc_data = import_data([symbol], date_num_years_ago(datetime.date.today(),1), datetime.date.today())
    loc_price = loc_data.iloc[len(loc_data)-1-daysAgo]
    loc_price = loc_price.iloc[0]
    value = amount * loc_price
    return value

def portSum(port,toPrint):
    rows = port.symbols + ['total']
    df = pandas.DataFrame(0, index = rows, columns = ['shares','price','curVal','cur%'])
    for i in range(0,len(port.symbols)) :
        df.iloc[i,0] = round (port.amounts[i], 2)
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
    loc_data = import_data(symbols, date_num_years_ago(datetime.date.today(),1), datetime.date.today())
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

    min_cur_max = [ values.min()['port'] , values.iloc[0,len(port.symbols)], values.max()['port'] ]

    val_index = [0,1,2,3,4,5,21,63,125,len(data)-1]
    values = values.iloc[val_index,:]
    values.index = val_index

    for z in values.index:
        #if z == 0:
            #print values['port'][0], values['port'][1]
        #    values.loc[z,'% change to today'] = round ( (values['port'][0] - values['port'][1]) / values['port'][1] * 100 , 1 )
        #else:
        values.loc[z,'% change to today'] = round ( (values['port'][0] - values['port'][z]) / values['port'][z] * 100 , 1 )

    values = numpy.round(values, 2)

    returns = daily_returns(data)
    evalDF = pandas.DataFrame(0, index= port.symbols+['port'], columns = ['std%','range%','1yGain%','beta','corr'])
    evalDF['std%'], evalDF['corr'], evalDF['beta'] = collinearity(returns,1)
    evalDF['range%'] = (data.max(axis=0) - data.min(axis=0)) / data.iloc[len(data)-1]
    evalDF['1yGain%'] = (data.iloc[len(data)-1] - data.iloc[0] ) / data.iloc[0]

    evalDF.iloc[:,0:3] = numpy.round(100*evalDF.iloc[:,0:3],2)
    evalDF.iloc[:,3:5] = numpy.round(evalDF.iloc[:,3:5], 2)

    overview = portSum(port,printToScreen)
    if printToScreen == 1:
        print 'Historical values:\n', values, '\n'
        print 'Historical statistics: \n', evalDF, '\n'
    return overview, values, evalDF, min_cur_max

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
                    compare.iloc[r,c] = output[c][0]['cur%'].loc[compare.iloc[r].name]
            except KeyError:
                keyError = 1
            try:
                if output[c][2].iloc[len(arrayOfPorts[c].symbols),:].loc[compare.iloc[r].name] :
                    compare.iloc[r,c] = output[c][2].iloc[len(arrayOfPorts[c].symbols),:].loc[compare.iloc[r].name]
            except KeyError:
                keyError = 1
    return compare

def option_data(symbol):
    data = web.Options(symbol,'yahoo')
    data = data.get_call_data(expiry='2015-12-18')
    # EXPIRY SHOULD UPDATE FLEXIBLY!!!!!!!!
    return data

def options_data(symbols):          # NOT BEING USED ANYMORE
    dataArray = []
    for x in range(0,len(symbols)):
        dataArray.append( option_data(symbols[x]) )
    return dataArray

def options_analysis(symbols):
    new = pandas.DataFrame(0, index=symbols, columns=['IV','Bid','Ask','Strike','Price','Ratio','Beta','Corr','DayMove','MonthMove','YearMove','pRank52'])
    for x in range(0,len(symbols)):
        data = option_data(symbols[x])
        data['IV'] = data['IV'].replace('%','',regex=True)
        data['IV'] = data['IV'].replace(',','',regex=True).astype('float')/100
        strikes = data.index.get_level_values('Strike')

        # get nearest OTM call and output IV, I'd rather us option w/ delta=30, but this is a good proxy for now
        y = 0
        while float(strikes[y]) < data['Underlying_Price'][0]:
            y = y+1
        nearestOTMstrike = strikes[y]

        new.loc[symbols[x],'IV'] = round (100*data.iloc[y,7], 2)
        new.loc[symbols[x],'Bid'] = data.iloc[y,1]
        new.loc[symbols[x],'Ask'] = data.iloc[y,2]
        new.loc[symbols[x],'Strike'] = strikes[y]
        new.loc[symbols[x],'Price'] = data.iloc[y,11]
        #price differs from close after hours, this is from real time options data
        new.loc[symbols[x],'Ratio'] = round (100*new.loc[symbols[x],'Bid'] / new.loc[symbols[x],'Price'], 2)
        # create temp portfolio to analyze
        port = Portfolio([symbols[x]], [100])
        analyzed = analyzePortfolio(port,0)
        new.loc[symbols[x],'Beta'] = analyzed[2].loc[symbols[x],'beta']
        new.loc[symbols[x],'Corr'] = analyzed[2].loc[symbols[x],'corr']
        new.loc[symbols[x],'DayMove'] = analyzed[1].loc[1,"% change to today"]
        new.loc[symbols[x],'MonthMove'] = analyzed[1].loc[21,"% change to today"]
        values_flipped = analyzed[1].iloc[::-1]
        new.loc[symbols[x],'YearMove'] = values_flipped.iloc[0,2]

        # calculate price as % within range of 52 week movement
        port_max = analyzed[3][2]
        port_min = analyzed[3][0]
        port_cur = analyzed[3][1]
        p_in_52 = (port_cur - port_min) / ( port_max - port_min )
        new.loc[symbols[x],'pRank52'] = round (100*p_in_52, 0)

    return new


# NEEDS WORK
def calculate_iv_on_exp(symbol):
    # attempts to calculate the composite IV for a given expiration
    data = pandas.Series(option_data(symbol)['IV'])
    data = data.reset_index(drop=True)
    sum_IV = 0
    for x in range(0,len(data)):
        if x <= len(data)/2:
            weight = float(x)**0.5 / (len(data)/2)**0.5
        else:
            weight = float(len(data) - x)**0.5 / (len(data)/2)**0.5
        print weight
        sum_IV += float( str(data[x])[:-1] ) * weight / 100

    avg = sum_IV / len(data)
    return avg
    #compIV = data.mean(axis=1)
    #return compIV

def uppercase(symbols):
    for x in range(0,len(symbols)):
        symbols[x] = symbols[x].upper()
    return symbols

#print calculate_iv_on_exp('TLT')

#print options_analysis(uppercase(['spy','tgt','gld','tlt','eem','iwm','goog','ibm','yhoo','x','twtr','ung','gm','qqq']))

def current_price(symbol):
    return round ( option_data(symbol).iloc[0,11] , 2)

def correlation_portfolio(symbols):
    # get returns data for symbols, ignore amounts of each in portfolio
    loc_data = import_data(symbols, date_num_years_ago(datetime.date.today(),1), datetime.date.today())
    loc_returns = daily_returns(loc_data)

    # analyze the correlations against each other
    corr_table = pandas.DataFrame(index=symbols,columns=symbols)

    for x in range(0,len(symbols)):
        for y in range(0,len(symbols)):
            # returns pearson correlation
            corr_table.iloc[x,y] =  numpy.corrcoef(loc_returns.iloc[:,x],
                    loc_returns.iloc[:,y])[0,1]

    # flips table for easy viewing
    corr_table = corr_table[::-1]
    corr_table = corr_table.transpose()

    corr_table.to_csv('corr_table.csv')

    print corr_table


def get_correlation(returns,spearman):
    numsym = len(returns.columns)-1
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
            returns_beta[i] = returns_corr[i] * returns_std[i] / market_std
    return returns_std.iloc[:-1], returns_corr.iloc[:-1], returns_beta.iloc[:-1]

# Code 30d, 60d, 90d, 180d, 1y correlations between different underlyings and SPY // chart of 1m and 3m  moving correlation
def corr_plot(symbols,moving_avg_days, start_date, end_date):
    loc_data = import_data(symbols, start_date, end_date)
    loc_returns = daily_returns(loc_data)
    array_of_correlations = []
    for x in range(0,len(loc_returns)-moving_avg_days):
        try:
            array_of_correlations.append(get_correlation(loc_returns[x:x+moving_avg_days],0)[1])
        except ValueError:
            donothing = 0

    graph = plt.figure()
    plt.grid(True)
    plt.title(str(moving_avg_days) +' day moving Correlation of ' + symbols[0] + ' and ' + symbols[1],fontsize=20,fontweight='bold')
    '''
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d/%Y'))
    dates = []
    for x in range(moving_avg_days,len(loc_returns)):
        dates.append(matplotlib.dates.DateFormatter(loc_returns.index[x]))
    print type(dates)
    '''
    plt.scatter(xrange(0,len(array_of_correlations)),array_of_correlations)
    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Correlation", fontsize=20)

    plt.show()
    graph.savefig('graph.jpg')
    graph.clear()
    graph.clf()
    return array_of_correlations

def date_num_years_ago(end_date,num_years_ago):
    return datetime.date.today().replace(year=datetime.date.today().year-num_years_ago)

corr_plot(['USO','SPY'],40, date_num_years_ago(datetime.date.today(),2), datetime.date.today())
#corr_plot(['GLD','IBM','IWM','TWTR','USO','X'])

#'BA','BP','BAC','BK','CL','DAL','HON','HOT','NOC','SBUX','HPQ','T','VZ','COST','C','GE','UAL'])

#correlation_portfolio(['GLD','IBM','IWM','TWTR','USO','X'])

# main method

#end_date = datetime.date.today()
#symbols = ['GOOG','SPY']#,'IBM','TLT','SPY','GLD','^VIX','VXX','UVXY','IWM','RWM','SH']
#data = import_data(symbols,end_date)
#returns = daily_returns(data)
#returns_std, returns_corr, returns_beta = collinearity(returns,0)
#correlation_analysis('UAL','USO',1)

#x = zeroBetaPortfolio(['IWM','EEM','GLD','TLT'],100000,1)
#portSum(x,1)
#y = zeroBetaPortfolio(['IWM','SH','X'],100000,1)
#print collinearity(daily_returns(historicalsPort(x)),1)
#print comparePorts([x,y])
#analyzePortfolio(x,1)
#correlation_analysis('IWM','TLT',1)
