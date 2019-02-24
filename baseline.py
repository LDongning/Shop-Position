# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import datetime 
import gc
from joblib import Parallel, delayed
import lightgbm as lgb
import xgboost as xgb
import math
import matplotlib.pylab as plt
import os
from sklearn.model_selection import train_test_split
import seaborn as sns

cache = './data/'
rec_shop_id_train_path = './data/rec_shop_id_801_831.csv'
rec_shop_id_test_path = './data/rec_shop_id_829_831.csv'
rec_shop_id_pre_path = './data/rec_shop_id_901_915.csv'
shop_info_path = './data/train_ccf_first_round_shop_info.csv'
ave_shop_wifi_infos_path = './data/ave_shop_wifi_infos.csv'

ave_shop_wifi_infos = pd.read_csv(ave_shop_wifi_infos_path)
shop_info = pd.read_csv(shop_info_path).set_index('shop_id')
ave_shop_wifi_infos['ave_value_wifi'] = ave_shop_wifi_infos['ave_value_wifi'].map(lambda x:dict(sorted(eval(x).items(),key=lambda ele:ele[1],reverse=True)))
ave_shop_wifi_infos.rename(columns={'ave_value_wifi':'ave_bssid_infos','shop_id':'tj_shop_id'},inplace=True)

def applyParallel(dfGrouped, func):
     with Parallel(n_jobs=3) as parallel:
        retLst = parallel( delayed(func)(pd.Series(value)) for key, value in dfGrouped )
        return pd.concat(retLst, axis=0)

# 分组排序
def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    
    return data

# 对结果进行整理
def reshape(pred):
    result = pred.copy()
    result = rank(result,'row_id','pred',ascending=False)
    result = result[result['rank']==0][['row_id','tj_shop_id','rank']]
    result = result.set_index(['row_id','rank']).unstack()
    result.reset_index(inplace=True)
    result['row_id'] = result['row_id'].astype('int')
    result.columns = ['row_id', 'pre_shop_id']
    
    return result

def feature_map(features):
    # Compute the correlation matrix
    corr = features.corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 20))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
def dealtime_hour(ele):
        morning = [7,8,9,10]
        moon = [11,12,13,14]
        afternoon = [15,16,17,18]
        night = [19,20,21,22,23,0]
        if ele.hour in morning:
            return 0
        elif ele.hour in moon:
            return 1
        elif ele.hour in afternoon:
            return 2 
        elif ele.hour in night:
            return 3
        else:
            return 4
#==============================================================================
#         if datetime.datetime.strptime(ele,'%Y-%m-%d %H:%M').hour in morning:
#             return 0
#         elif datetime.datetime.strptime(ele,'%Y-%m-%d %H:%M').hour in moon:
#             return 1
#         elif datetime.datetime.strptime(ele,'%Y-%m-%d %H:%M').hour in afternoon:
#             return 2
#         elif datetime.datetime.strptime(ele,'%Y-%m-%d %H:%M').hour in night:
#             return 3
#         else:
#             return 4
#==============================================================================

def cal_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = Lx + Ly
    return L

def calon_degree(lat1,lon1,lat2,lon2):
    dx = (lon2-lon1)*math.cos(lat1/57.2958)
    dy = lat2 - lat1
    if((dx==0) and (dy==0)):
        degree = -1
    else:
        if (dy==0.0):
            dy=1.0e-24
        degree = 180 -90*(math.copysign(1,dy)) - math.atan(dx/dy)*57.2958
        degree = round(degree/45)*45
        if degree>315:
            degree = 0
    return degree


    
def get_feature(row):
    wifi_infos = row.wifi_infos
    ave_bssid_infos = row.ave_bssid_infos
    df = pd.Series([])
    df['index'] = row.index
    
    df_rss,ave_rss = [],[]
    for bssid in wifi_infos.keys():
        df_rss.append(wifi_infos[bssid][0])
        if bssid in ave_bssid_infos.keys():
            ave_rss.append(ave_bssid_infos[bssid])
        else:
            ave_rss.append(-999)
    
    df_rss = pd.Series(df_rss)
    ave_rss = pd.Series(ave_rss)
    
    subtract_rss = df_rss - ave_rss
    subtract_square_rss = np.square(subtract_rss)
        
    #添加wifi_infos数据特征
    #df['subtract_rss_mean'] = subtract_rss.mean()
    df['subtract_rss_var'] = subtract_rss.var()
    df['subtract_rss_min'] = subtract_rss.min()
    df['subtract_rss_max'] = subtract_rss.max()
    
    df['std_subtract_square_rss'] = np.sqrt(sum(subtract_square_rss))
    df['subtract_square_rss_mean'] = subtract_square_rss.mean()
    #df['subtract_square_rss_var'] = subtract_square_rss.min()
    df['subtract_square_rss_min'] = subtract_square_rss.max()
    df['subtract_square_rss_max'] = subtract_square_rss.var()
    
    #df['diff_max_bssid'] = int(list(wifi_infos.items())[0][0] == list(ave_bssid_infos.items())[0][0])
    #df['diff_max_bssid_value'] = list(wifi_infos.items())[0][1][0] - list(ave_bssid_infos.items())[0][1]
    
    #计算两个序列的相关性
    df['pearson_ave_rss_df_rss'] = ave_rss.corr(df_rss,method='pearson')
    df['kendall_ave_rss_df_rss'] = ave_rss.corr(df_rss,method='kendall')
    df['spearman_ave_rss_df_rss'] = ave_rss.corr(df_rss,method='spearman')
    
    #添加flag连接特征
    connect = [wifi_infos[bssid][1] for bssid in wifi_infos.keys()]
    df['flag_true'] =  connect.count(1)
    
    df['wifi_counts'] = len(wifi_infos)
    
#==============================================================================
#     #添加时间特征
#     df['hour'] = dealtime_hour(df.time_stamp)
#     df['minute'] = df.time_stamp.minute/60
#     df['weekday'] = df.time_stamp.weekday()
#==============================================================================
    
#==============================================================================
#     tj_shop_info = shop_info.loc[df.tj_shop_id]
#     #添加 tj_shop_id与 真实每条消费记录数据之间的距离特征
#     df['distance'] = np.square(cal_distance(df.latitude,df.longitude,tj_shop_info.latitude,tj_shop_info.longitude))
#     
#     #计算角度特征
#     df['angle'] = calon_degree(df.latitude,df.longitude,tj_shop_info.latitude,tj_shop_info.longitude)
#==============================================================================
    
#==============================================================================
#     #添加 tj_shop_id与 真实shop_id之间的价格差异特征
#     df['diff_price'] = np.square(real_shop_info.price - tj_shop_info.price)
#       
#     #添加 tj_shop_id与 真实shop_id之间的商店类型差异特征
#     df['diff_category'] = np.square(int(real_shop_info.category_id[2:]) - int(tj_shop_info.category_id[2:]))
#==============================================================================
    
    return df

def split_wifi_infos(ele):
    dict_wifi_infos = {}
    for ele in ele.split(';'):
        temp = ele.split('|')
        if 'true' in temp[2]:
            dict_wifi_infos[temp[0]] = (int(temp[1]),1)
        else:
            dict_wifi_infos[temp[0]] = (int(temp[1]),0)
    #返回由rss值由大到小排序后的wifi序列    
    dict_wifi_infos = sorted(dict_wifi_infos.items(),key=lambda x:x[1][0],reverse=True)    
    return dict(dict_wifi_infos)


        
def make_train_set():
    dump_path = os.path.join(cache, 'train2.hdf')
    if os.path.exists(dump_path):
        result = pd.read_hdf(dump_path, 'all')
    else:
        print('构造训练数据特征集....')
        train = pd.read_csv(rec_shop_id_train_path)
        train.reset_index(inplace=True)
        train['time_stamp']=pd.to_datetime(train['time_stamp'])
        train['wifi_infos'] = train.wifi_infos.map(lambda x: split_wifi_infos(x))
        train = pd.merge(train,ave_shop_wifi_infos,on='tj_shop_id',how='left')
        #result = applyParallel(train.iterrows(), get_feature).sort_values(by='index')
        print('multiprocessing...')
        result = Parallel(n_jobs=3)( delayed(get_feature)(pd.Series(value)) for key, value in train.iterrows() )
        #print(result)
        result = pd.DataFrame(result)
        result = pd.merge(train[['index','label']],result,on='index',how='left')
        result.to_hdf(dump_path, 'all')
        
    return result

def add_feature(row):
    wifi_infos = row.wifi_infos
    ave_bssid_infos = row.ave_bssid_infos
    
    df_rss,ave_rss = [],[]
    for bssid in wifi_infos.keys():
        df_rss.append(wifi_infos[bssid][0])
        if bssid in ave_bssid_infos.keys():
            ave_rss.append(ave_bssid_infos[bssid])
        else:
            ave_rss.append(-999)
    
    df_rss = pd.Series(df_rss)
    ave_rss = pd.Series(ave_rss)
    
    pearson_ave_rss_df_rss = ave_rss.corr(df_rss,method='pearson')
    kendall_ave_rss_df_rss = ave_rss.corr(df_rss,method='kendall')
    spearman_ave_rss_df_rss = ave_rss.corr(df_rss,method='spearman')
    
    return [pearson_ave_rss_df_rss,kendall_ave_rss_df_rss,spearman_ave_rss_df_rss]


def change_test_feature():
    
    test = pd.read_csv(rec_shop_id_pre_path)
    test.reset_index(inplace=True)
    test['wifi_infos'] = test.wifi_infos.map(lambda x: split_wifi_infos(x))
    test = pd.merge(test,ave_shop_wifi_infos,on='tj_shop_id',how='left')
    
    print('开始添加特征....')
    t0 = time.time()
    pearson_ave_rss_df_rss,kendall_ave_rss_df_rss,spearman_ave_rss_df_rss = [],[],[]
    for key,row in test.iterrows():
        feature_list = add_feature(row)
        pearson_ave_rss_df_rss.append(feature_list[0])
        kendall_ave_rss_df_rss.append(feature_list[1])
        spearman_ave_rss_df_rss.append(feature_list[2])
        if key % 10000 ==0 :
            print(key,time.time()-t0)
            t0 = time.time()
            
    feature = pd.DataFrame({'index':test.index,'pearson_ave_rss_df_rss':pearson_ave_rss_df_rss,
                            'kendall_ave_rss_df_rss':kendall_ave_rss_df_rss,
                            'spearman_ave_rss_df_rss':spearman_ave_rss_df_rss})
            
    #feature = applyParallel(test.iterrows(), add_feature).sort_values(by='index')
    
    print('Union and Out....')
    dump_path = os.path.join(cache, 'test_offline.csv')
    result = pd.read_csv(dump_path)
    delate_features = ['angle','distance','hour','minute','weekday','diff_max_bssid',
     'diff_max_bssid_value','subtract_rss_mean','subtract_square_rss_var','corr_ave_rss_df_rss']
    result = result.drop(delate_features, axis=1)
    result = pd.merge(result,feature,on = 'index',how = 'left')
    result.to_csv('./data/pre_.csv',index = False)
    
def make_test_set():
    #dump_path = os.path.join(cache, 'test_offline.csv')
    dump_path = os.path.join(cache, 'test.hdf')
    if os.path.exists(dump_path):
        result = pd.read_hdf(dump_path, 'all')
        #result = pd.read_csv(dump_path)
    else:
        print('构造测试数据特征集....')
        #test = pd.read_csv(rec_shop_id_test_path)
        test = pd.read_csv(rec_shop_id_pre_path)
        test.reset_index(inplace=True) 
        test['time_stamp']=pd.to_datetime(test['time_stamp'])
        test['wifi_infos'] = test.wifi_infos.map(lambda x: split_wifi_infos(x))
        test = pd.merge(test,ave_shop_wifi_infos,on='tj_shop_id',how='left')
        result = applyParallel(test.iterrows(), get_feature).sort_values(by='index')
        #result = pd.merge(test[['index','row_id','tj_shop_id','label']],result,on='index',how='left')   #用于线下测试
        result = pd.merge(test[['index','row_id','tj_shop_id']],result,on='index',how='left')
        result.to_hdf(dump_path, 'all')
        gc.collect()
        
    return result  
  
def train_fit(train,label):
    
    train_x,train_y,test_x,test_y = train_test_split(train, label, test_size=0.2, random_state=0)
    #print(train_x.columns())
    
    lgb_train = lgb.Dataset(train_x,test_x)
    lgb_eval = lgb.Dataset(train_y,test_y, reference=lgb_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth':7,
        'num_leaves': 6,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print('Start training...')
    
    # train
    gbm = lgb.train(params,lgb_train,num_boost_round=1000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=20)
    
    lgb.plot_importance(gbm,figsize =(15,10))
    plt.show()
    
    return gbm

if __name__ =="__main__":
    print('读取数据......')
#==============================================================================
#     train_columns = ['row_id', 'tj_shop_id', 'user_id', 'shop_id', 'time_stamp', 'longitude',
#        'latitude', 'wifi_infos','label']
#     shop_info_columns = ['shop_id', 'category_id', 'longitude', 'latitude', 'price', 'mall_id']
#     
#==============================================================================
    #构造训练数据和测试数据
    print('构造训练数据和测试数据的特征集....')
    #train,test = make_train_set(),make_test_set()
    data = pd.read_csv('./data/train_813_826.csv')
    train = data.copy()
    #train = data[data['time_stamp']<='2017-08-19 23:50']
#==============================================================================
#     #线下验证数据区间
#     test = data[(data['time_stamp']>='2017-08-20 00:00') & (data['time_stamp']<='2017-08-26 23:50')]
#     shop_id_df = test.shop_id.drop_duplicates().to_frame().reset_index(drop=True)
#     shop_id_df['row_id'] = shop_id_df.index
#     test = pd.merge(test,shop_id_df,on = 'shop_id',how = 'left')
#==============================================================================
    
    feature_map(train.drop(['index','tj_shop_id','time_stamp','shop_id'], axis=1).astype(float))
    #Training
    train_data,label = train.drop(['index','label','tj_shop_id','time_stamp','shop_id'], axis=1).astype(float),train['label']
    
    gbm = train_fit(train_data,label)
#==============================================================================
#     #Predicting
#     print('线下验证...')
#     test.loc[:,'pred'] = gbm.predict(test.drop(['index','row_id','label','time_stamp','tj_shop_id','shop_id'],axis=1).astype(float), num_iteration=gbm.best_iteration)
#     result_test = reshape(test[['row_id','tj_shop_id','pred']])
#     result_test = pd.merge(test[test['label'] == 1],result_test,on='row_id',how='left')
#     result_test['right_flag'] = (result_test['tj_shop_id']==result_test['pre_shop_id']).astype(int)
#     print('Acc: ',result_test.right_flag.values.tolist().count(1)/result_test.shape[0])
#==============================================================================
    print('线下预测...')
    test = pd.read_csv('./data/test_offline.csv')
    test.loc[:,'pred'] = gbm.predict(test.drop(['index','row_id','tj_shop_id'], axis=1).astype(float), num_iteration=gbm.best_iteration)
    result = reshape(test[['row_id','tj_shop_id','pred']])
    result_9072 = pd.read_csv('./result/result_90.72.csv')
    result_9072 = pd.merge(result_9072,result,on = 'row_id',how = 'left')
    result_9072['val'] = result_9072['shop_id'] == result_9072['pre_shop_id']
    print(result_9072.val.value_counts())
    result = result.rename(columns = {'pre_shop_id':'shop_id'})
    result.to_csv('./result/result1.csv',index=False)
    
    
    