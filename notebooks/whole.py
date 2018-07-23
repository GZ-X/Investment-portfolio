import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *
from scipy import stats
#%matplotlib inline

#DATA_DIR = '../../Data_for_invest/'

#Read new data
print('start..')
newcrsp = pd.read_csv('NewData.csv',low_memory =False)

print('import done')
crsp1=newcrsp[newcrsp['SHRCD']!=10.0]
crsp1=newcrsp[newcrsp['SHRCD']!=11.0]

cleanedcrsp = newcrsp.drop(crsp1.index)

print('clean done')
print(len(newcrsp),len(cleanedcrsp))

#1
#Start point of processing of mergeddata
cleanedcrsp[['PERMCO','PERMNO','SHRCD','EXCHCD']]=\
    cleanedcrsp[['PERMCO','PERMNO','SHRCD','EXCHCD']].astype(int)

# Line up date to be end of month
cleanedcrsp['date']=pd.to_datetime(cleanedcrsp['date'])
crsp_m = cleanedcrsp

#3
J = 6 # Formation Period Length: J can be between 3 to 12 months
K = 6 # Holding Period Length: K can be between 3 to 12 months

_tmp_crsp = crsp_m[['PERMNO','date','RET']].sort_values(['PERMNO','date'])\
    .set_index('date')

# Replace missing return with 0
_tmp_crsp['RET']=_tmp_crsp['RET'].fillna(0)

_tmp_crsp['RET'] = pd.to_numeric(_tmp_crsp['RET'], errors='coerce')

#4
# Calculate rolling cumulative return
# by summing log(1+ret) over the formation period

#pd.to_numeric(s, errors='coerce')
_tmp_crsp['logret']=np.log(1+_tmp_crsp['RET'])
umd = _tmp_crsp.groupby(['PERMNO'])['logret'].rolling(J, min_periods=J).sum()
umd = umd.reset_index()
umd['cumret']=np.exp(umd['logret'])-1



########################################
# Formation of 10 Momentum Portfolios  #
########################################

# For each date: assign ranking 1-10 based on cumret
# 1=lowest 10=highest cumret
umd=umd.dropna(axis=0, subset=['cumret'])
umd['momr']=umd.groupby('date')['cumret'].transform(lambda x: pd.qcut(x, 10, labels=False))

umd.momr=umd.momr.astype(int)
umd['momr'] = umd['momr']+1
umd['hdate1']=umd['date']+MonthBegin(1)
umd['hdate2']=umd['date']+MonthEnd(K)
umd=umd.rename(columns={'date':'form_date'})
umd = umd[['PERMNO','form_date','momr','hdate1','hdate2']]

# join rank and return data together
# note: this step consumes a lot of memory so takes a while
_tmp_ret = crsp_m[['PERMNO','date','RET']]
print('start merge')

port = pd.merge(_tmp_ret, umd, on=['PERMNO'], how='inner')
print('merge done')
port = port[(port['hdate1']<=port['date']) & (port['date']<=port['hdate2'])]

umd2 = port.sort_values(by=['date','momr','form_date','PERMNO']).drop_duplicates()
print('start group')
umd2['momr'] = umd2['momr'].astype(float)
umd2['form_date'] = pd.to_datetime(umd2['form_date'])
umd2['PERMNO'] = umd2['PERMNO'].astype(float)
#umd2['RET'] = umd2['RET'].astype(float)
umd2['RET'] = pd.to_numeric(umd2['RET'], errors='coerce')
umd3 = umd2.groupby(['date','momr','form_date'])['RET'].mean().reset_index()
print('group done')

# Skip first two years of the sample 
start_yr = umd3['date'].dt.year.min()+2
umd3 = umd3[umd3['date'].dt.year>=start_yr]
umd3 = umd3.sort_values(by=['date','momr'])

print('start create')
# Create one return series per MOM group every month
ewret = umd3.groupby(['date','momr'])['RET'].mean().reset_index()
ewstd = umd3.groupby(['date','momr'])['RET'].std().reset_index()
ewret = ewret.rename(columns={'RET':'ewret'})
ewstd = ewstd.rename(columns={'RET':'ewretstd'})
ewretdat = pd.merge(ewret, ewstd, on=['date','momr'], how='inner')
ewretdat = ewretdat.sort_values(by=['momr'])
print('create done')

# portfolio summary
ewretdat.groupby(['momr'])['ewret'].describe()[['count','mean', 'std']]
print(len(ewretdat),ewretdat.columns)
#################################
# Long-Short Portfolio Returns  #
#################################

print('Long-Short Portfolio Returns start..')
# Transpose portfolio layout to have columns as portfolio returns
ewretdat2 = ewretdat.pivot(index='date', columns='momr', values='ewret')

# Add prefix port in front of each column
ewretdat2 = ewretdat2.add_prefix('port')
print('add prefix')
print(ewretdat2.columns)
ewretdat2 = ewretdat2.rename(columns={'port1.0':'losers', 'port10.0':'winners'})
print('after rename')
print(ewretdat2.columns)
ewretdat2['long_short'] = ewretdat2['winners'] - ewretdat2['losers']

# Compute Long-Short Portfolio Cumulative Returns
ewretdat3 = ewretdat2
ewretdat3['1+losers']=1+ewretdat3['losers']
ewretdat3['1+winners']=1+ewretdat3['winners']
ewretdat3['1+ls'] = 1+ewretdat3['long_short']

ewretdat3['cumret_winners']=ewretdat3['1+winners'].cumprod()-1
ewretdat3['cumret_losers']=ewretdat3['1+losers'].cumprod()-1
ewretdat3['cumret_long_short']=ewretdat3['1+ls'].cumprod()-1
print('done')

print('Portfolio Summary Statistics start..')
#################################
# Portfolio Summary Statistics  #
################################# 

# Mean 
mom_mean = ewretdat3[['winners', 'losers', 'long_short']].mean().to_frame()
mom_mean = mom_mean.rename(columns={0:'mean'}).reset_index()

# T-Value and P-Value
t_losers = pd.Series(stats.ttest_1samp(ewretdat3['losers'],0.0)).to_frame().T
t_winners = pd.Series(stats.ttest_1samp(ewretdat3['winners'],0.0)).to_frame().T
t_long_short = pd.Series(stats.ttest_1samp(ewretdat3['long_short'],0.0)).to_frame().T

t_losers['momr']='losers'
t_winners['momr']='winners'
t_long_short['momr']='long_short'

t_output =pd.concat([t_winners, t_losers, t_long_short])\
    .rename(columns={0:'t-stat', 1:'p-value'})
print('done')
    
# Combine mean, t and p
print('start merge result')
mom_output = pd.merge(mom_mean, t_output, on=['momr'], how='inner')
print('merge done')

print('start write csv')
mom_output.to_csv('output.csv',index=False)
print('all done')
