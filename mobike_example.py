# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:47:44 2017

@author: ZHILANGTAOSHA
"""

import pandas as pd
import numpy as np
import geohash
from sklearn.metrics import average_precision_score


def cal_mht_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = Lx + Ly
    return L

def binary_error(preds, train_data):
    labels = train_data.get_label()
    return 'map', average_precision_score(labels,preds,average='weighted'), True
def fhjg(pred):
    pred = pred.sort_values('pred',ascending=False)
    del pred['pred']
    d1 = pred.groupby('orderid').head(1)
    pred = pred[~pred.index.isin(d1.index)]
    d2 = pred.groupby('orderid').head(1)
    pred = pred[~pred.index.isin(d2.index)]
    d3 = pred.groupby('orderid').head(1)
    dd = pd.merge(d1,d2,on = 'orderid',how = 'outer')
    dd = pd.merge(dd,d3,on = 'orderid',how = 'outer')
    return(dd)

def zdy_pf(pred,ture):
    pred = pred.sort_values('pred',ascending=False)
    d1 = pred.groupby('orderid').head(1)
    pred = pred[~pred.index.isin(d1.index)]
    d2 = pred.groupby('orderid').head(1)
    pred = pred[~pred.index.isin(d2.index)]
    d3 = pred.groupby('orderid').head(1)
    dd = pd.merge(d1,d2,on = 'orderid',how = 'outer')
    dd = pd.merge(dd,d3,on = 'orderid',how = 'outer')
    pf = pd.merge(ture[['orderid','geohashed_end_loc']],dd,how = 'left')
    s1 = sum(pf.geohashed_end_loc == pf.tj_dd_x)
    p1 = pf[~(pf.geohashed_end_loc == pf.tj_dd_x)]
    s2 = sum(p1.geohashed_end_loc == p1.tj_dd_y)
    p1 = p1[~(p1.geohashed_end_loc == p1.tj_dd_y)]
    s3 = sum(p1.geohashed_end_loc == p1.tj_dd_y)
    jg = s1 + s2/2 + s3/3
    jg = jg / pf.shape[0]
    return(jg)
def train_tztq(train):
    zh = train[['geohashed_start_loc','geohashed_end_loc']]
    zh.loc[:,'pl'] = 1
    zh = zh.groupby(['geohashed_start_loc','geohashed_end_loc'])['pl'].count().reset_index()
    zh['max_dd'] = zh[['geohashed_start_loc','geohashed_end_loc']].max(axis=1)
    zh['min_dd'] = zh[['geohashed_start_loc','geohashed_end_loc']].min(axis=1)
    #统计相同两个点的关联个数
    zh = zh.groupby(['max_dd','min_dd'])['pl'].agg({'zz':np.sum,'gs':np.size}).reset_index()
    print(zh.shape)
    #换算为经纬度
    zh.loc[:,'start_jw'] = zh.max_dd.map(geohash.decode)
    zh.loc[:,'start_j'] = zh.start_jw.map(lambda x:x[0])
    zh.loc[:,'start_w'] = zh.start_jw.map(lambda x:x[1])
    zh.loc[:,'end_jw'] = zh.min_dd.map(geohash.decode)
    zh.loc[:,'end_j'] = zh.end_jw.map(lambda x:x[0])
    zh.loc[:,'end_w'] = zh.end_jw.map(lambda x:x[1])
    del zh['start_jw'],zh['end_jw']
    #计算经纬度的具体距离
#    zh.loc[:,'juli'] = zh.apply(lambda x:cal_mht_distance(x['start_j'],x['start_w'],x['end_j'],x['end_w']),axis=1)
    #建立最后的地点画像
    #计算大的维度的地点对应个数
    max_dd_tz = zh.groupby('max_dd').agg({'gs':np.size})
    #计算训练集中地点出现的总次数
    max_dd_tz['zz_gs'] = zh.groupby('max_dd')['zz'].agg({'zz_gs':np.sum})
    #计算
    #地点所至最大距离
#    max_dd_tz['juli_max'] = zh.groupby('max_dd')['juli'].max()
#    #地点所至最小距离
#    max_dd_tz['juli_min'] = zh.groupby('max_dd')['juli'].min()
#    #地点所至中值
#    max_dd_tz['juli_median'] = zh.groupby('max_dd')['juli'].median()
    #计算大的维度的地点对应个数
    min_dd_tz = zh.groupby('min_dd').agg({'gs':np.size})
    #计算训练集中地点出现的总次数
    min_dd_tz['zz_gs'] = zh.groupby('min_dd')['zz'].agg({'zz_gs':np.sum})
    #计算
    #地点所至最大距离
#    min_dd_tz['juli_max'] = zh.groupby('min_dd')['juli'].max()
#    #地点所至最小距离
#    min_dd_tz['juli_min'] = zh.groupby('min_dd')['juli'].min()
#    #地点所至中值
#    min_dd_tz['juli_median'] = zh.groupby('min_dd')['juli'].median()
    #拼接所有地点
    dd_tz = pd.concat([max_dd_tz,min_dd_tz])
    return dd_tz
def shijian_chuli(train):
    hour_sx = {0:1,1:1,2:1,3:1,4:1,5:1,6:2,7:4,8:4,9:4,10:2,11:2,12:5,13:5,14:2,15:3,16:3,17:6,18:6,19:6,20:3,21:3,22:1,23:1}
    week_sx = {0:1,1:1,2:1,3:1,4:1,5:2,6:2}
    train.loc[:,'starttime'] = pd.to_datetime(train.starttime)
    train['weekday_time'] = train.starttime.dt.weekday
    train['hour_time'] = train.starttime.dt.hour
    train['hour_sx'] = train.hour_time.map(hour_sx)
    train['week_sx'] = train.weekday_time.map(week_sx)
    return(train)
    
def jsjl(zh):
    js = zh[~zh.tj_dd.isnull()].drop_duplicates()
    #换算为经纬度
    js.loc[:,'start_jw'] = js.iloc[:,0].map(geohash.decode)
    js.loc[:,'start_j'] = js.start_jw.map(lambda x:x[0])
    js.loc[:,'start_w'] = js.start_jw.map(lambda x:x[1])
    js.loc[:,'end_jw'] = js.iloc[:,1].map(geohash.decode)
    js.loc[:,'end_j'] = js.end_jw.map(lambda x:x[0])
    js.loc[:,'end_w'] = js.end_jw.map(lambda x:x[1])
    del js['start_jw'],js['end_jw']    
    #计算经纬度的具体距离
    js.loc[:,'juli'] = js.apply(lambda x:cal_mht_distance(x['start_j'],x['start_w'],x['end_j'],x['end_w']),axis=1)
    return js  

def pj_sj(df_true,df_tj_liebiao,df_his,yucj = 0):
    #==============================================================================
    # 拼接起始位置
    #==============================================================================    
    #df_tj = pd.merge(df_tj_liebiao,df_true[['orderid','geohashed_start_loc']],on = 'orderid',how = 'left')    
    #==============================================================================
    # 添加特征
    #==============================================================================    
    #添加时间    
    df = shijian_chuli(df_true)
    df_tj = pd.merge(df_tj_liebiao,df[['orderid','hour_sx','week_sx','userid','bikeid']],on = 'orderid',how = 'left')    
    #添加两个点的距离    
    zh = df_tj.loc[:,('geohashed_start_loc','tj_dd')]
    js = jsjl(zh)   
    df_tj = pd.merge(df_tj,js[['geohashed_start_loc','tj_dd','juli']],on = ['geohashed_start_loc','tj_dd'],how = 'left')
    #添加地点特征    
    train_dd_tz = train_tztq(df_his)
    train_dd_tz = train_dd_tz.reset_index()
    train_dd_tz = train_dd_tz.rename(columns={'index':'geohashed_start_loc'})
#    df_tj = pd.merge(df_tj,train_dd_tz,on = 'geohashed_start_loc')
    train_dd_tz = train_dd_tz.rename(columns={'geohashed_start_loc':'tj_dd'})
    df_tj = pd.merge(df_tj,train_dd_tz,on = 'tj_dd')    
    if yucj == 1:return(df_tj)        
    #==============================================================================
    # 给数据标记label
    #==============================================================================
    df['label'] = 1
    df_tj = pd.merge(df_tj,df[['orderid','geohashed_end_loc','label']].rename(columns={'geohashed_end_loc':'tj_dd'}),on = ['orderid','tj_dd'],how = 'left')    
    df_tj.label = df_tj.label.fillna(0)
    return(df_tj)
def chuli_ls_gs(eval_train_his,dds = 3,mm = 3):
    list_dy = eval_train_his[['geohashed_start_loc','geohashed_end_loc']]
    list_dy['max_dd'] = list_dy[['geohashed_start_loc','geohashed_end_loc']].max(axis = 1)
    list_dy['min_dd'] = list_dy[['geohashed_start_loc','geohashed_end_loc']].min(axis = 1)
    del list_dy['geohashed_start_loc']
    del list_dy['geohashed_end_loc']
    list_dy['gs'] = 1
    l1 = list_dy.groupby(['max_dd','min_dd'])['gs'].count().reset_index()
    l2 = list_dy.groupby(['max_dd','min_dd'])['gs'].count().reset_index()
    l2.columns = ['min_dd','max_dd','gs']
    l2 = pd.concat([l1,l2])
    l2 = l2.groupby(['max_dd','min_dd'])['gs'].sum().reset_index()
    l2 = l2.sort_values(['max_dd','gs','min_dd'],ascending=False)
    l2.columns = ['geohashed_start_loc', 'tj_dd%s'%str(mm), 'gs']
    return l2.groupby('geohashed_start_loc').head(dds),l2

def tj_chuli(df_true,df_his,n = 10):
    #==============================================================================
    #推荐规则函数,用于从历史数据推荐地点
    #==============================================================================
    #==============================================================================
    # 推荐userid的地点（出发，和结束）
    #==============================================================================
    eval_df_his1 = df_his.rename(columns={'geohashed_start_loc':'tj_dd1', 'geohashed_end_loc':'tj_dd2'})
    print(eval_df_his1.shape)
    eval_df = pd.merge(df_true,eval_df_his1,on='userid',how='outer')
    print(eval_df.shape)
    #==============================================================================
    # 按出发地点推荐
    #地点最近的3个热门地点
    #==============================================================================
    tj_zj,l2 = chuli_ls_gs(df_his,dds = 10,mm=3)
    eval_df2 = pd.merge(df_true,tj_zj,on='geohashed_start_loc',how='outer')
    eval_df2['zj_dd'] = eval_df2.geohashed_start_loc.map(lambda x: x[:-1])
    rmgs = l2.groupby('geohashed_start_loc')['gs'].sum().reset_index()
    rmgs['zj_dd'] = rmgs.geohashed_start_loc.map(lambda x: x[:-1])
    rmgs.columns = ['tj_dd4', 'gs', 'zj_dd']
    rmgs = rmgs.groupby('zj_dd').head(n)
    del rmgs['gs']
    eval_df2 = pd.merge(eval_df2,rmgs,on='zj_dd',how='outer')    
    #eval_df_his = eval_df_his.rename(columns={'bikeid_x':'bikeid_y'})
    #eval_df = pd.merge(eval_df,eval_df_his,on='bikeid_y')
    eval_df_tj_1 = eval_df[['orderid','tj_dd1']].drop_duplicates().rename(columns={'tj_dd1':'tj_dd'})
    eval_df_tj_2 = eval_df[['orderid','tj_dd2']].drop_duplicates().rename(columns={'tj_dd2':'tj_dd'})
    eval_df_tj_3 = eval_df2[['orderid','tj_dd3']].drop_duplicates().rename(columns={'tj_dd3':'tj_dd'})
    eval_df_tj_4 = eval_df2[['orderid','tj_dd4']].drop_duplicates().rename(columns={'tj_dd4':'tj_dd'})
    del l2['tj_dd3']
    l2.columns = ['tj_dd','gs']
    l2 = l2.groupby('tj_dd')['gs'].sum().reset_index()
    eval_df_tj_1 = pd.merge(eval_df_tj_1,l2,on = 'tj_dd',how = 'left')
    eval_df_tj_1 = eval_df_tj_1.sort_values(['orderid', 'gs', 'tj_dd'],ascending=False)
    eval_df_tj_1 = eval_df_tj_1.groupby('orderid').head(n)
    eval_df_tj_2 = pd.merge(eval_df_tj_2,l2,on = 'tj_dd',how = 'left')
    eval_df_tj_2 = eval_df_tj_2.sort_values(['orderid', 'gs', 'tj_dd'],ascending=False)
    eval_df_tj_2 = eval_df_tj_2.groupby('orderid').head(n)
    eval_df_tj_1['tj_fs1'] = 1
    eval_df_tj_2['tj_fs2'] = 1
    eval_df_tj_3['tj_fs3'] = 1
    eval_df_tj_4['tj_fs4'] = 1
    eval_df_tj = pd.concat([eval_df_tj_1,eval_df_tj_2,eval_df_tj_3,eval_df_tj_4]) #
    eval_df_tj = eval_df_tj.groupby(['orderid','tj_dd']).sum().reset_index().fillna(0)
    eval_df_tj = pd.merge(eval_df_tj,df_true[['orderid','geohashed_start_loc']],on = 'orderid',how='left')
    eval_df_tj = eval_df_tj[~(eval_df_tj.tj_dd == eval_df_tj.geohashed_start_loc)]
    #eval_df_tj = eval_df_tj[(eval_df_tj.tj_fs3 > 0) | (eval_df_tj.tj_fs4 > 0)]
    return (eval_df_tj)

def lssjsc(train,list_his,list_label):
    eval_train = train[train.starttime.map(lambda x:x[:10]).isin(list_label)]
    eval_train_his = train[train.starttime.map(lambda x:x[:10]).isin(list_his)]
    del eval_train_his['orderid']
    return eval_train,eval_train_his

def lssjsc_test(train,list_his):
    eval_train_his = train[train.starttime.map(lambda x:x[:10]).isin(list_his)]
    del eval_train_his['orderid']
    return eval_train_his



def train_fit(eval_train_tj,eval_test_tj,featurelist):
    import lightgbm as lgb
    lgb_train = lgb.Dataset(eval_train_tj[featurelist],eval_train_tj.label)
    lgb_eval = lgb.Dataset(eval_test_tj[featurelist],eval_test_tj.label, reference=lgb_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth':10,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print('Start training...')
    # train
    evals_result = {}
    gbm = lgb.train(params,lgb_train,num_boost_round=1000,
                    valid_sets=lgb_eval,feval=binary_error,
                    early_stopping_rounds=20,
                    evals_result = evals_result)
    return gbm,evals_result


def train_fit_xgb(eval_train_tj,eval_test_tj,featurelist):
    import xgboost as xgb
    params = {
        'objective': 'binary:logistic',
        'eta': 0.1,
        'colsample_bytree': 0.886,
        'min_child_weight': 2,
        'max_depth': 10,
        'subsample': 0.886,
        'alpha': 10,
        'gamma': 30,
        'lambda':50,
        'verbose_eval': True,
        'nthread': 8,
        'eval_metric': 'auc',
        'scale_pos_weight': 10,
        'seed': 201703,
        'missing':-1
    }
    xgtrain = xgb.DMatrix(eval_train_tj[featurelist],eval_train_tj.label)
    xgtest = xgb.DMatrix(eval_test_tj[featurelist],eval_test_tj.label)
    watchlist = [(xgtrain,'train'), (xgtest, 'val')]
    df_list = {}
    gbdt = xgb.train(params, xgtrain, 10000, evals = watchlist, verbose_eval = 10, early_stopping_rounds = 100,evals_result=df_list)
    return gbdt,df_list


print('.....开始.....')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('.....开始.....')


#==============================================================================
# 
#==============================================================================
'''
训练集 特征区间 ：'2017-05-10 00:00:00'~'2017-05-21 00:00:00'
训练集 标签区间 ：'2017-05-21 00:00:00'~'2017-05-23 00:00:00'
测试集 特征区间 ：'2017-05-10 00:00:00'~'2017-05-23 00:00:00'
测试集 标签区间 ：'2017-05-23 00:00:00'~'2017-05-25 00:00:00'
'''

#eval_list1  = ['2017-05-10', '2017-05-11','2017-05-12', '2017-05-13','2017-05-14']
#eval_list2 = ['2017-05-15', '2017-05-16','2017-05-17', '2017-05-18','2017-05-19']
#eval_list3  = ['2017-05-20', '2017-05-21','2017-05-22', '2017-05-23','2017-05-24']


#eval_list1  = ['2017-05-10', '2017-05-11','2017-05-12', '2017-05-13','2017-05-14']
#eval_list2 = ['2017-05-15', '2017-05-16','2017-05-17', '2017-05-18','2017-05-19']
#eval_list3  = ['2017-05-15', '2017-05-16','2017-05-17', '2017-05-18','2017-05-19']
#eval_list4 = ['2017-05-20', '2017-05-21','2017-05-22', '2017-05-23','2017-05-24']
#eval_list5  = ['2017-05-20', '2017-05-21','2017-05-22', '2017-05-23','2017-05-24']

eval_list1  = ['2017-05-10', '2017-05-11','2017-05-12', '2017-05-13','2017-05-14',
               '2017-05-15', '2017-05-16','2017-05-17', '2017-05-18','2017-05-19',
               '2017-05-20']
eval_list2 = ['2017-05-21','2017-05-22']
eval_list3  = ['2017-05-10', '2017-05-11','2017-05-12', '2017-05-13','2017-05-14',
               '2017-05-15', '2017-05-16','2017-05-17', '2017-05-18','2017-05-19',
               '2017-05-20','2017-05-21','2017-05-22']
eval_list4 = ['2017-05-23','2017-05-24']

eval_list5  = ['2017-05-10', '2017-05-11','2017-05-12', '2017-05-13','2017-05-14',
               '2017-05-15', '2017-05-16','2017-05-17', '2017-05-18','2017-05-19',
               '2017-05-20','2017-05-21','2017-05-22','2017-05-23','2017-05-24']

eval_train,eval_train_his = lssjsc(train,eval_list1,eval_list2)
for i in range(10)[1:]:
    print(i)
    eval_train_tj = tj_chuli(eval_train,eval_train_his,n = i)
    jg_jc = eval_train_tj[['orderid','tj_dd']]
    jg_jc2 = eval_train[['orderid','geohashed_end_loc']]
    jg_jc = jg_jc.drop_duplicates()
    jg_jc2.columns = jg_jc.columns
    jg_jc2['zq']  = 1
    jg_jc['zq']  = 1
    qs = pd.merge(jg_jc2,jg_jc,on=['orderid', 'tj_dd'],how = 'left').isnull().sum()['zq_y']
    print (qs / jg_jc2.shape[0])


eval_list7  = ['2017-05-10', '2017-05-11','2017-05-12', '2017-05-13','2017-05-14','2017-05-15', '2017-05-16','2017-05-17']

eval_list8  = ['2017-05-18','2017-05-19','2017-05-20','2017-05-21','2017-05-22','2017-05-23','2017-05-24']

eval_train,eval_train_his = lssjsc(train,eval_list7,eval_list8)


'''
eval_train_tj = pj_sj(eval_train,eval_train_tj,eval_train_his,yucj = 0)

eval_eval,eval_eval_his = lssjsc(train,eval_list3,eval_list4)
eval_eval_tj = tj_chuli(eval_eval,eval_eval_his)
eval_eval_tj = pj_sj(eval_eval,eval_eval_tj,eval_eval_his,yucj = 0)

jg_jc = eval_eval_tj[['orderid','tj_dd']]
jg_jc2 = eval_eval[['orderid','geohashed_end_loc']]
jg_jc = jg_jc.drop_duplicates()
jg_jc2.columns = jg_jc.columns
jg_jc2['zq']  = 1
jg_jc['zq']  = 1
qs = pd.merge(jg_jc2,jg_jc,on=['orderid', 'tj_dd'],how = 'left').isnull().sum()['zq_y']
print (qs / jg_jc2.shape[0])



featurelist = [i for i in eval_train_tj.columns if i not in ['orderid', 'geohashed_start_loc', 'geohashed_end_loc','bikeid','userid','tj_dd','label','week_sx']]

gbm,eval_list = train_fit(eval_train_tj,eval_eval_tj,featurelist)
eval_eval_tj['pred'] = gbm.predict(eval_eval_tj[featurelist])
pred = eval_eval_tj[['orderid','tj_dd','pred']]
print(zdy_pf(pred,eval_eval))
...
#gbm1,eval_list = train_fit(eval_eval_tj,eval_train_tj,featurelist)
#eval_eval_tj['pred'] = gbm1.predict(eval_eval_tj[featurelist])
#pred = eval_eval_tj[['orderid','tj_dd','pred']]
#print(zdy_pf(pred,eval_eval))
#
#
eval_train_tj['pred'] = gbm.predict(eval_train_tj[featurelist])
pred = eval_train_tj[['orderid','tj_dd','pred']]
print(zdy_pf(pred,eval_train))
#
#
###
del eval_train,eval_train_tj,eval_train_his
del eval_eval,eval_eval_tj,eval_eval_his
test_his = lssjsc_test(train,eval_list5)
test_tj = tj_chuli(test,test_his)
test_tj = pj_sj(test,test_tj,test_his,yucj = 1)

test_tj['pred'] = gbm.predict(test_tj[featurelist])
pred = test_tj[['orderid','tj_dd','pred']]
jg = fhjg(pred)
print(jg.isnull().sum())
#
#j = 3815984
#for i in range(10):
#    z = j + 3815984
#    if i == 0 :
#        test_tj.loc[:j,'pred'] =  gbm.predict(test_tj.loc[:j,:][featurelist])
#        j = 0
#    elif i == 19:
#        test_tj.loc[j:,'pred'] = gbm.predict(test_tj.loc[j:,:][featurelist])
#    else:
#        test_tj.loc[j:z,'pred'] = gbm.predict(test_tj.loc[j:z,:][featurelist])
#    j = j + 3815984
zh = pd.concat([jg,test[~test.orderid.isin(jg.orderid)][['orderid']]])
zh = pd.merge(zh,test[['orderid','geohashed_start_loc']],on = 'orderid')
    
    
zh = zh.sort_values('geohashed_start_loc')
zh = zh.fillna(method='pad')
zh = zh.sort_values('geohashed_start_loc',ascending=False)
zh = zh.fillna(method='pad')

del zh['geohashed_start_loc']
zh.orderid = zh.orderid.astype(int)
zh.to_csv('jg5.csv',index=None,header=None)
    
    
## 相差的分钟数
#def diff_of_minutes(time1, time2):
#    d = {'5': 0, '6': 31, }
#    try:
#        days = (d[time1[6]] + int(time1[8:10])) - (d[time2[6]] + int(time2[8:10]))
#        try:
#            minutes1 = int(time1[11:13]) * 60 + int(time1[14:16])
#        except:
#            minutes1 = 0
#        try:
#            minutes2 = int(time2[11:13]) * 60 + int(time2[14:16])
#        except:
#            minutes2 = 0
#        return (days * 1440 - minutes2 + minutes1)
#    except:
#        return np.nan
#
## 分组排序
#def rank(data, feat1, feat2, ascending):
#    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
#    data['rank'] = range(data.shape[0])
#    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
#    data = pd.merge(data,min_rank,on=feat1,how='left')
#    data['rank'] = data['rank'] - data['min_rank']
#    del data['min_rank']
#    return data
#    
#def get_user_next_loc(train,test):
#    train_temp = train.copy()
#    test_temp = test.copy()
#    all_data = pd.concat([train_temp,test_temp]).drop_duplicates()
#    all_data = rank(all_data,'userid','starttime',ascending=True)
#    all_data['rank'] = range(all_data.shape[0])
#    min_rank = all_data.groupby('userid',as_index=False)['rank'].agg({'min_rank':'min'})
#    all_data = pd.merge(all_data,min_rank,on='userid',how='left')
#    all_data['rank'] = all_data['rank']-all_data['min_rank']
#    all_data_temp = all_data.copy()
#    all_data_temp['rank'] = all_data_temp['rank']-1
#    result = pd.merge(all_data[['orderid','userid','rank','starttime']],
#                      all_data_temp[['userid','rank','geohashed_start_loc','starttime']],on=['userid','rank'],how='inner')
#    result = result[result['orderid'].isin(test_temp['orderid'].values)]
#    result['user_sep_time'] = result.apply(lambda x: diff_of_minutes(x['starttime_y'],x['starttime_x']),axis=1)
#    result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
#    result = result[['orderid','geohashed_end_loc','user_sep_time']]
#    return result

'''
'''

jg_jc = eval_eval_tj[['orderid','tj_dd']]
jg_jc2 = eval_eval[['orderid','geohashed_end_loc']]
jg_jc = jg_jc.drop_duplicates()
jg_jc2.columns = jg_jc.columns
jg_jc2['zq']  = 1
jg_jc['zq']  = 1
qs = pd.merge(jg_jc2,jg_jc,on=['orderid', 'tj_dd'],how = 'left').isnull().sum()['zq_y']
print (qs / jg_jc2.shape[0])
'''
