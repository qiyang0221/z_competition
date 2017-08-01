# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 16:12:25 2017

@author: leland
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

sold_home = pd.read_csv('data_prepare_train_20170730.csv')
df1 = pd.read_csv('data_prepare_wait_20170730.csv')

cols = sold_home.columns.tolist()
cols.remove('parcelid')
cols.remove('transactiondate')
cols.remove('logerror')

'''
for i in cols:
    print('Columns name is: ',i)
    print(df1[i][df1[i].notnull()].head())
'''

#pd.to_datetime('2016-10-15') - pd.to_datetime('2016-01-01')
#pd.to_datetime('2016-11-15') - pd.to_datetime('2016-01-01')
#pd.to_datetime('2016-12-15') - pd.to_datetime('2016-01-01')
#pd.to_datetime('2017-10-15') - pd.to_datetime('2016-01-01')
#pd.to_datetime('2017-11-15') - pd.to_datetime('2016-01-01')
#pd.to_datetime('2017-12-15') - pd.to_datetime('2016-01-01')

#sold_home = sold_home[cols].astype(float)
#df201610 = df201610[cols].astype(float)

X_train,X_test,Y_train,Y_test = train_test_split(sold_home[cols],sold_home['logerror'],random_state=1)


dtrain = xgb.DMatrix(X_train,label=Y_train)
dtest = xgb.DMatrix(X_test,label=Y_test)

def Custom_val(preds,dtrain):
    labels = dtrain.get_label()
    val = np.array(map(lambda x:abs(x[0] - x[1]),zip(labels,preds)))
    total = val.sum()
    return 'Custom val',total


#用xgb.train做
params = {
        'booster':'gblinear'
        ,'objective':'reg:linear'
        ,'eval_mertic' : 'rmse'
        ,'max_depth' : 8
        ,'alpha' : 3
        ,'lambda' : 5   #L2正则
        ,'subsample' : 0.8
        ,'colsample_bytree' : 0.9
        ,'eta' : 0.002
        ,'seed' : 1
        ,'nthread' : -1
        ,'min_child_weight':5
        }

plst = list(params.items())
watchlist = [(dtrain,'train'),(dtest,'val')]
model = xgb.train(plst,dtrain,100000,watchlist,early_stopping_rounds=100)  #feval=Custom_val,

model.dump_model('model/xgb_20170801.raw.txt')

final = pd.DataFrame(df1['parcelid'])
for i in [288,319,349,653,684,714]:
    df1['txn_dateadd'] = i
    preds = pd.DataFrame(model.predict(xgb.DMatrix(df1[cols])))
    preds.columns = [i]
    preds = preds[i].apply(lambda x:round(x,4))
    final = final.join(preds)

final.head()
final.rename(columns={288:'201610',319:'201611',349:'201612',653:'201710',684:'201711',714:'201712'},inplace=True)
final.to_csv('predict_data_20170801_1.csv',index=0)




#用xgb.XGBRegressor做，利用sklearn接口调参
from sklearn.model_selection import RandomizedSearchCV as RCV
import pickle


params_rcv = {
        'max_depth' : [6,7,8,9,10]
        ,'learning_rate' : [0.001,0.002,0.003,0.004,0.005]
        ,'n_estimators' :[500,600,800,1000]
        ,'gamma' : np.linspace(0,0.01,10)
        ,'min_child_weight':[2,3,4,5,6,7]
        ,'subsample' : [0.7,0.8,0.9,1.0]
        ,'colsample_bytree' : [0.7,0.8,0.9,1.0]
        ,'reg_alpha' : np.linspace(2,12,num=10)
        ,'reg_lambda' : np.linspace(3,13,num=10)
        ,'base_score' : np.linspace(0,2,10)
        }

xgbreg = xgb.XGBRegressor(max_depth=5,learning_rate=0.001,n_estimators=1000,gamma=0.001
                          ,min_child_weight=2,subsample=0.7,colsample_bytree=0.7
                          ,reg_alpha=0.5,reg_lambda=2,base_score=0,seed=6)

rcv = RCV(xgbreg,params_rcv,n_jobs=-1,scoring='neg_mean_absolute_error',cv=3,random_state=6)
rcv.fit(X_train,Y_train)
print(rcv.best_estimator_)
print(rcv.best_score_)
print(rcv.best_params_)
'''
XGBRegressor(base_score=0.0, colsample_bylevel=1, colsample_bytree=0.9,
       gamma=0.0033333333333333331, learning_rate=0.002, max_delta_step=0,
       max_depth=8, min_child_weight=5, missing=None, n_estimators=500,
       nthread=-1, objective='reg:linear', reg_alpha=10.888888888888889,
       reg_lambda=9.6666666666666679, scale_pos_weight=1, seed=6,
       silent=True, subsample=0.8)
'''



xgbreg = rcv.best_estimator_
xgbreg.fit(X_train,Y_train)

from sklearn.metrics import mean_absolute_error as MSE
print(MSE(Y_test,xgbreg.predict(X_test)))

pickle.dump(xgbreg,open('model/xgbreg_20170730.pkl','wb'))

final = pd.DataFrame(df1['parcelid'])
for i in [288,319,349,653,684,714]:
    df1['txn_dateadd'] = i
    preds = pd.DataFrame(xgbreg.predict(df1[cols]))
    preds.columns = [i]
    preds = preds[i].apply(lambda x:round(x,4))
    final = final.join(preds)

final.head()
final.rename(columns={288:'201610',319:'201611',349:'201612',653:'201710',684:'201711',714:'201712'},inplace=True)
final.to_csv('predict_data_20170801_2.csv',index=0)


















