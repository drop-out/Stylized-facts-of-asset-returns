
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# In[2]:

def autocorrelation(dataframe,column,date_begin,date_end,n_lags,title):
    data=dataframe.loc[(dataframe['date']>=date_begin)&(dataframe['date']<=date_end),:].copy()
    a=[]
    for i in range(1,n_lags+1):
        data['diff_lag']=data[column].shift(i)
        temp=data.dropna()
        corr,p=scipy.stats.pearsonr(temp[column],temp.diff_lag)
        a.append([i,corr])
    corr=pd.DataFrame(a)
    corr.columns=['auto_correlation','lag_order']
    plt.figure(figsize=(12,6))
    plt.plot(corr.auto_correlation,corr.lag_order)
    plt.title('''%s 
    %s to %s'''%(title,date_begin,date_end))
    plt.savefig("output/%s %s.jpg"%(title,date_begin))
    
def histogram(dataframe,column,date_begin,date_end,freq):
    if freq=='Daily':
        data=dataframe.loc[(dataframe['date']>=date_begin)&(dataframe['date']<=date_end),:].copy()
    else:
        data=dataframe.loc[(dataframe['year']>=date_begin)&(dataframe['year']<=date_end),:].copy()
    sigma=np.std(data[column])
    length=len(data[column])
    hist=pd.DataFrame()
    hist['Return']=np.linspace(-0.07,0.07,29)
    hist['normal']=scipy.stats.norm.pdf(hist.Return,scale=sigma)
    hist['COUNT']=0
    for i in data[column]:
        position=int(i//0.005+14)
        if (position>=0) and (position<=28):
            hist.iloc[position,-1]+=1
    hist['COUNT']=hist['COUNT']/length*200
    plt.figure(figsize=(12,6))
    plt.plot(hist.Return,hist.normal)
    plt.plot(hist.Return,hist.COUNT,'.')
    plt.title('''Comparison of %s Return to Normal Distribution
    %s to %s'''%(freq,date_begin,date_end))
    plt.savefig("output/%s %s %s.jpg"%(freq,date_begin,date_end))


# In[3]:

#Stylized fact 1: Abcense of significant autocorrelations on daily return
data=pd.read_csv('sp500.csv')
data['difference']=np.log(data.value)-np.log(data.value.shift(1))

autocorrelation(data,'difference','1990-01-01','1999-12-31',100,'Autocorrelation of Daily S&P 500 Returns')
autocorrelation(data,'difference','2000-01-01','2009-12-31',100,'Autocorrelation of Daily S&P 500 Returns')
autocorrelation(data,'difference','2010-01-01','2017-11-30',100,'Autocorrelation of Daily S&P 500 Returns')

#Stylized fact 2: Volatility culstering
data=pd.read_csv('sp500.csv')
data['difference']=np.square(np.log(data.value)-np.log(data.value.shift(1)))
autocorrelation(data,'difference','1990-01-01','1999-12-31',100,'Autocorrelation of Squared Daily S&P 500 Returns')
autocorrelation(data,'difference','2000-01-01','2009-12-31',100,'Autocorrelation of Squared Daily S&P 500 Returns')
autocorrelation(data,'difference','2010-01-01','2017-11-30',100,'Autocorrelation of Squared Daily S&P 500 Returns')


# In[4]:

#Stylized fact 3: Sharp peak and heavy tails
data=pd.read_csv('sp500.csv')
data['difference']=np.log(data.value)-np.log(data.value.shift(1))
histogram(data,'difference','1990-01-01','1999-12-31','Daily')
histogram(data,'difference','2000-01-01','2009-12-31','Daily')
histogram(data,'difference','2010-01-01','2017-11-30','Daily')


# In[5]:

#Stylized fact 4: Aggregational Gaussian
data['year']=pd.to_datetime(data['date']).dt.year
data['week']=pd.to_datetime(data['date']).dt.week

data_week=data.groupby(['year','week'],as_index=False)['difference'].sum()
histogram(data_week,'difference',1990,2017,'Weekly')

histogram(data,'difference','1990-01-01','2017-11-30','Daily')



