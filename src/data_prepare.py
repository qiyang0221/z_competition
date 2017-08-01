# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 10:07:59 2017

@author: leland
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
from sklearn.cross_validation import train_test_split

df1 = pd.read_csv('properties_2016.csv')
#df1.head()
#df1.describe().T

def CountNa(x):
    n = 0
    for i in x:
        if i != i:
            n += 1
    return n

df1['CountNa'] = df1.apply(lambda x:CountNa(x),axis=1)
#df1[df1['CountNa'] < 20].head(3).T
#set(df1['CountNa'])

#处理非数值型变量
#总共有以下列：hashottuborspa,propertyzoningdesc,taxdelinquencyflag,fireplaceflag,taxdelinquencyflag
#,propertycountylandusecode
df1['hashottuborspa'] = df1.hashottuborspa.replace(True,1).replace(False, 0)

#propertyzoningdesc = pd.DataFrame(df1.propertyzoningdesc.drop_duplicates().reset_index(drop=True)).reset_index()
#propertyzoningdesc.to_csv('propertyzoningdesc.csv',index=0)
propertyzoningdesc = pd.read_csv('propertyzoningdesc.csv')
df1 = pd.merge(df1,propertyzoningdesc,how='left',on='propertyzoningdesc')
del df1['propertyzoningdesc']
df1.rename(columns={'index':'propertyzoningdesc'},inplace=True)

df1['fireplaceflag'] = df1.fireplaceflag.replace(True,1).replace(False, 0)

df1['taxdelinquencyflag'] = df1.taxdelinquencyflag.replace('Y', 1).replace('N', 0)

#propertycountylandusecode = pd.DataFrame(df1.propertycountylandusecode.drop_duplicates().reset_index(drop=True)).reset_index()
#propertycountylandusecode.to_csv('propertycountylandusecode.csv',index=0)
propertycountylandusecode = pd.read_csv('propertycountylandusecode.csv')
df1 = pd.merge(df1,propertycountylandusecode,how='left',on='propertycountylandusecode')
del df1['propertycountylandusecode']
df1.rename(columns={'index':'propertycountylandusecode'},inplace=True)

#set(df1.propertycountylandusecode)
df1['rawcensustractandblock'] = df1['rawcensustractandblock']/10000000000
df1.to_csv('data_prepare_wait_20170730.csv',index=0)


df2 = pd.read_csv('train_2016_v2.csv')
#a = df1.taxdelinquencyyear[df1.taxdelinquencyyear.notnull()]
#set(a)

sold_home = pd.merge(df1,df2,how='inner',on='parcelid')
sold_home['transactiondate'] = pd.to_datetime(sold_home['transactiondate'])
sold_home['txn_dateadd'] = sold_home['transactiondate'] - pd.to_datetime('2016-01-01')
sold_home['txn_dateadd'] = sold_home['txn_dateadd'].astype('timedelta64[D]').astype(int)

sold_home.to_csv('data_prepare_train_20170730.csv',index=0)











