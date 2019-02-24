# -*- coding: utf-8 -*-

import pandas as pd
import time
from joblib import Parallel, delayed

def applyParallel(dfGrouped, func):
    with Parallel(n_jobs=7) as parallel:
        retLst = parallel( delayed(func)(value) for value in dfGrouped.values )
        return pd.concat(retLst, axis=0)
    
#==============================================================================
# 构造规则
#==============================================================================

def train_caculate(df):  
    mall_id = df[6]
    evaluate_wifi_infos = df[5]
                                     
    def get_shop_weight_dict(shop,evaluate_wifi_infos):
        match_wifi_info = ave_shop_wifi_infos.loc[shop]['ave_value_wifi']
        weight_count = calculate_weight(match_wifi_info,evaluate_wifi_infos)
        shop_weight1[shop] = weight_count                  
                                                  
    match_shops = pd.Series(shop_id_mall.loc[mall_id].shop_ids)
    #策略1
    shop_weight1 = {}
    match_shops.map(lambda shop : get_shop_weight_dict(shop,evaluate_wifi_infos))
    #字典sorted排序后返回的是一个list数组
    shop_weight1 = sorted(shop_weight1.items(),key=lambda item:item[1],reverse=True)
    
    #取权重前三的所有bssid
    ####去重,并按照原来顺序排序
    temp_weight = [ele[1] for ele in shop_weight1]
    weight = list(set(temp_weight))
    weight.sort(key=temp_weight.index)
    
    tj_shop_ids = []
    if len(weight)>=3:
        for ele in shop_weight1:
            if ele[1] == weight[0]:
                tj_shop_ids.append(ele[0])
            if ele[1] == weight[1]:
                tj_shop_ids.append(ele[0])
            if ele[1] == weight[2]:
                tj_shop_ids.append(ele[0])
            
    if len(weight)==2:
        for ele in shop_weight1:
            if ele[1] == weight[0]:
                tj_shop_ids.append(ele[0])
            if ele[1] == weight[1]:
                tj_shop_ids.append(ele[0])
                if len(tj_shop_ids)>3:
                    break
                
    if len(weight)==1:
        for ele in shop_weight1:
            if ele[1] == weight[0]:
                tj_shop_ids.append(ele[0])
                if len(tj_shop_ids)>3:
                    break
    
    #若推荐的shop_id与真实的不同，修改其中一个shop_id
    shop_id = df[1]
    if shop_id not in set(tj_shop_ids):
        del tj_shop_ids[len(tj_shop_ids)-1]
        tj_shop_ids.append(shop_id)
    
    temp_df = []  
    for index,ele in enumerate(set(tj_shop_ids)):
        if index == 0:
            temp_df = pd.DataFrame({'row_id':[df[7]],'tj_shop_id':[ele]})
        else:
            temp_df = pd.concat([temp_df,pd.DataFrame({'row_id':[df[7]],'tj_shop_id':[ele]})])
    
    return temp_df

def test_caculate(df):  
    mall_id = df[2]
    evaluate_wifi_infos = df[6]
                                     
    def get_shop_weight_dict(shop,evaluate_wifi_infos):
        match_wifi_info = ave_shop_wifi_infos.loc[shop]['ave_value_wifi']
        weight_count = calculate_weight(match_wifi_info,evaluate_wifi_infos)
        shop_weight1[shop] = weight_count                  
                                                  
    match_shops = pd.Series(shop_id_mall.loc[mall_id].shop_ids)
    #策略1
    shop_weight1 = {}
    match_shops.map(lambda shop : get_shop_weight_dict(shop,evaluate_wifi_infos))
    #字典sorted排序后返回的是一个list数组
    shop_weight1 = sorted(shop_weight1.items(),key=lambda item:item[1],reverse=True)
    
    #取权重前三的所有bssid
    ####去重,并按照原来顺序排序
    temp_weight = [ele[1] for ele in shop_weight1]
    weight = list(set(temp_weight))
    weight.sort(key=temp_weight.index)
    
    tj_shop_ids = []
    if len(weight)>=3:
        for ele in shop_weight1:
            if ele[1] == weight[0]:
                tj_shop_ids.append(ele[0])
            if ele[1] == weight[1]:
                tj_shop_ids.append(ele[0])
            if ele[1] == weight[2]:
                tj_shop_ids.append(ele[0])
            
    if len(weight)==2:
        for ele in shop_weight1:
            if ele[1] == weight[0]:
                tj_shop_ids.append(ele[0])
            if ele[1] == weight[1]:
                tj_shop_ids.append(ele[0])
                if len(tj_shop_ids)>3:
                    break
                
    if len(weight)==1:
        for ele in shop_weight1:
            if ele[1] == weight[0]:
                tj_shop_ids.append(ele[0])
                if len(tj_shop_ids)>3:
                    break
                
    temp_df = []  
    for index,ele in enumerate(set(tj_shop_ids)):
        if index == 0:
            temp_df = pd.DataFrame({'row_id':[df[0]],'tj_shop_id':[ele]})
        else:
            temp_df = pd.concat([temp_df,pd.DataFrame({'row_id':[df[0]],'tj_shop_id':[ele]})])
    
    return temp_df

#==============================================================================
# 根据match_wifi_info和evaluate_wifi_infos进行计算一个店铺的加权数
#==============================================================================

def calculate_weight(match_wifi_info,evaluate_wifi_infos):
    weight_count = 0       
    wifi = sorted([wifi.split('|') for wifi in evaluate_wifi_infos.split(';')],key=lambda x:int(x[1]),reverse=True)
    for wifi_info in wifi:
        bssid,rss = wifi_info[0],wifi_info[1]  
        if bssid in match_wifi_info:           
            #策略1
            if abs(int(rss)-int(match_wifi_info.get(bssid)))<Value:
                weight_count += 1

    return weight_count


    
evaluation_path = './data/test_evaluation_public.csv'
ave_shop_wifi_infos_path = './data/ave_shop_wifi_infos.csv'
shop_id_mall_path = './data/shop_id_mall.csv'
shop_info_path = './data/train_ccf_first_round_shop_info.csv'

shop_info = pd.read_csv(shop_info_path)
user_shop_hehavior = pd.read_csv('./data/train_ccf_first_round_user_shop_behavior.csv')
evaluation_data = pd.read_csv(evaluation_path)
ave_shop_wifi_infos = pd.read_csv(ave_shop_wifi_infos_path)
shop_id_mall = pd.read_csv(shop_id_mall_path)

#字符串转换
ave_shop_wifi_infos['ave_value_wifi'] = ave_shop_wifi_infos['ave_value_wifi'].map(lambda x:eval(x))
shop_id_mall['shop_ids'] = shop_id_mall['shop_ids'].map(lambda x :eval(x))
shop_id_mall.set_index('mall_id',inplace=True)
ave_shop_wifi_infos.set_index('shop_id',inplace=True)

                 
#设置阈值并且计算
Value = 14

if __name__=='__main__':
    #划分训练数据区间,区分周末和工作日，8-28情人节，剔除
    #训练集 特征区间：2017-08-01 00:00 ~ 2017-08-31 23:50
    #训练集 标签区间：2017-08-22 00:00 ~ 2017-08-24 23:50
    #测试集 标签区间：2017-08-29 00:00 ~ 2017-08-31 23:50
    print('********开始计算********')
    t0 = time.time()
    print('取训练数据3天进行样本推荐构造.......')
    #test = user_shop_hehavior[user_shop_hehavior['time_stamp']<='2017-08-24 23:50']
    #test = user_shop_hehavior[(user_shop_hehavior['time_stamp']>'2017-08-28 23:50')]
    test = pd.merge(user_shop_hehavior,shop_info[['shop_id','mall_id']],on='shop_id',how='left')
    test['row_id'] = range(len(test))
    pre_test = applyParallel(test,train_caculate)
    
#==============================================================================
#     pre_test = pd.DataFrame({'row_id':[0],'shop_id':[0],'weight':[0],'shop_ids':[0]})
#     for line in test.values:
#         pre_test = pd.concat([pre_test,train_caculate(line)])
#         print('********')
#==============================================================================
    
    pre_test = pd.merge(pre_test,test,on='row_id',how='left')
    pre_test['label'] = (pre_test['shop_id']==pre_test['tj_shop_id']).astype(int)
    print('正负样本比例 : ',pre_test['label'].value_counts(),'耗时： ',time.time()-t0)
    pre_test.to_csv('./data/rec_shop_id_801_831.csv',index=False)
    
#==============================================================================
#     pre_test.to_csv('./data/rec_shop_id_822_824.csv',index=False)
#     pre_test = applyParallel(evaluation_data,test_caculate)
#     pre_test = pd.merge(pre_test,evaluation_data,on = 'row_id',how = 'left')
#     pre_test.to_csv('./data/rec_shop_id_901_915.csv',index=False)
#==============================================================================
    print('耗时： ',time.time()-t0)
    