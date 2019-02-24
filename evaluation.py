# -*- coding: utf-8 -*-

import pandas as pd
import time
from joblib import Parallel, delayed
from collections import defaultdict
import numpy as np

def applyParallel(dfGrouped, func):
    with Parallel(n_jobs=7) as parallel:
        retLst = parallel( delayed(func)(value) for value in dfGrouped.values )
        return pd.concat(retLst, axis=0)
    
#==============================================================================
# 构造规则
#==============================================================================

def caculate(df):
    #t1 = time.time()
    #mall_id = df.mall_id
    #evaluate_wifi_infos = df.wifi_infos
    
    mall_id = df[3]
    evaluate_wifi_infos = df[2]
                                     
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
    #print(shop_weight1)
#==============================================================================
#     #策略2
#     index = 0
#     shop_weight2 = np.nan
#     shop_weight = []
#     while True:
#         try:
#             if index==5:
#                 #print('...')
#                 break
#             wifi = sorted([wifi.split('|') for wifi in evaluate_wifi_infos.split(';')],key=lambda x:int(x[1]),reverse=True)[index]
#             counter = wifi_to_shops[wifi[0]]
#             shop_weight2 = sorted(counter.items(),key=lambda x:x[1],reverse=True)
#             break
#         except:
#             index += 1
#     
#     #策略1 和 策略2 加权融合  
#     if shop_weight2 !=np.nan:
#         weight_dict = defaultdict(lambda :[])
#         for index,ele in enumerate(shop_weight1):
#             shop_id = ele[0]
#             weight_dict[shop_id].append(1.0-0.05*index)
#         for index,ele in enumerate(shop_weight2):
#             shop_id = ele[0]
#             weight_dict[shop_id].append(0.9-0.05*index)
#             
#         #权重相乘    
#         shop_weight = defaultdict(lambda :0)
#         for item in weight_dict.items():
#             if len(item[1])>1:
#                 shop_weight[item[0]] = item[1][0]*item[1][1]
#             else:
#                 shop_weight[item[0]] = item[1][0]
#                 
#         #按照权重大小排序
#         shop_weight = sorted(shop_weight.items(),key=lambda x:x[1],reverse=True)
#     else:
#         shop_weight = shop_weight1
#==============================================================================
    
    
    first_weight = shop_weight1[0][1]
    temp_shop_id = []
    for ele in shop_weight1:
        if first_weight == ele[1]:
            temp_shop_id.append(ele[0])
        else:   
            break
    
    temp_df = pd.DataFrame({'row_id':[df[4]],'shop_id':[shop_weight1[0][0]],'weight':[shop_weight1[0][1]],'shop_ids':[temp_shop_id]})
#==============================================================================
#     df['row_id'] = df.row_id
#     df['shop_id'] = shop_weight1[0][0]
#     df['weight'] = shop_weight1[0][1]
#     df['shop_ids'] = temp_shop_id
#==============================================================================
    
    #print('计算一条记录用时{}s'.format(time.time()-t1))
    
    #print("index:",i,"shop_id:",shop_weight_dict[0][0],"appearance_times:",shop_weight_dict[0][1],"row_id:",row_id[len(row_id)-1])    
    
    #return df[['row_id','shop_id','weight','shop_ids']].to_frame().T
    
    return temp_df


#==============================================================================
# 根据match_wifi_info和evaluate_wifi_infos进行计算一个店铺的加权数
#==============================================================================

def calculate_weight(match_wifi_info,evaluate_wifi_infos):
    weight_count = 0
    
    
    wifi = sorted([wifi.split('|') for wifi in evaluate_wifi_infos.split(';')],key=lambda x:int(x[1]),reverse=True)
    #temp_count = 0
    for wifi_info in wifi:
        bssid,rss = wifi_info[0],wifi_info[1]  
        if bssid in match_wifi_info:           
            #策略1
            if abs(int(rss)-int(match_wifi_info.get(bssid)))<Value:
                weight_count += 1
                #temp_count = np.log1p(1+abs(int(rss)-int(match_wifi_info.get(bssid)))) + temp_count
#==============================================================================
#             #策略2
#             try:
#                 if bssid == wifi[0][0]:
#                     weight_count += 1
#                 if bssid == wifi[0][1]:
#                     weight_count += 0.8
#                 if bssid == wifi[0][2]:
#                     weight_count += 0.5
#                 #print('加权')
#             except:
#                 continue
#==============================================================================
#    if (weight_count == 0) | (temp_count == 0):
#       return 100000.0
#    else:
#        return temp_count/weight_count

    return weight_count

def get_category(shop_info,select):
    def find(x):
        cate = []
        for ele in x:
            cate.append(shop_info.loc[ele]['category_id']) 
        return cate

    shop_info.set_index('shop_id',inplace=True)
    select['category'] = select.shop_ids.map(lambda x :find(x))
    
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

wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))
for line in user_shop_hehavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1
                 
#设置阈值并且计算
Value = 14

if __name__=='__main__':
    print('********开始计算********')
    t0 = time.time()
    print('取 2017-08-25 00:00 ~ 2017-08-31 11:50 训练数据进行线下验证.......')
    test = user_shop_hehavior[(user_shop_hehavior['time_stamp']>'2017-08-25 00:00')][['shop_id','wifi_infos']].reset_index()
    test = pd.merge(test,shop_info[['shop_id','mall_id']],on='shop_id',how='left')
    test['row_id'] = range(len(test))
    test = test.rename(columns={'shop_id':'orignal_shop_id'})
    pre_test = applyParallel(test,caculate)
#==============================================================================
#     pre_test = pd.DataFrame({'row_id':[0],'shop_id':[0],'weight':[0],'shop_ids':[0]})
#     for line in test.values:
#         pre_test = pd.concat([pre_test,caculate(line)])
#==============================================================================
    
    pre_test = pd.merge(pre_test,test[['row_id','orignal_shop_id']],on='row_id',how='left')
    pre_test['value'] = pre_test['shop_id']==pre_test['orignal_shop_id']
    print('Acc : ',len(pre_test[pre_test['value']==True])/len(pre_test),'耗时： ',time.time()-t0)
    
#==============================================================================
#     print('线上预测.......')
#     
#     evaluate_result = applyParallel(evaluation_data.iterrows(), caculate).sort_values(by='row_id')
#     print('******总计算耗时{}h'.format((time.time()-t0)/3600)) 
#     evaluate_result.to_csv('./result/c1_c2_result_less{}.csv'.format('_'+str(Value)+'_'+time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))),index=False)
#==============================================================================
    #multi_evaluate_result.to_csv('./result/multi_evaluate_result_less{}.csv'.format('_'+str(Value)+'_'+time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))),index=False)
