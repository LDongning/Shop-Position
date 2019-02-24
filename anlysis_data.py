# -*- coding: utf-8 -*-
'''
Anlysis datas 
'''

import pandas as pd
import seaborn as sn
import numpy as np
import time
shop_info_path = './data/train_ccf_first_round_shop_info.csv'
shop_behavior_path = './data/train_ccf_first_round_user_shop_behavior.csv'
test_path = './data/test_evaluation_public.csv'

#处理wifi_info
def deal_wifi_info(df):
    
    def split_str(ele):
        temp_list = ele.split(';')
        wifi_infos_df = pd.DataFrame(index=list(range(len(temp_list))),columns=['bssid','RSS','flag'])
        for index,ele in enumerate(temp_list):
            temp = ele.split('|')
            wifi_infos_df.loc[index]=[temp[0],int(temp[1]),temp[2]]
        #按照RSS强度排序，选出前三个强度的bssid
        #print('排序前',wifi_infos_df['RSS'][:3])
        wifi_infos_df.sort_values(by='RSS',ascending = False,inplace=True)
        #print('排序后',wifi_infos_df['RSS'][:3])
        if len(temp_list)>=3:
            #return (tuple(wifi_infos_df.iloc[0]),tuple(wifi_infos_df.iloc[1]),tuple(wifi_infos_df.iloc[2])),wifi_infos_df['RSS'][:3].mean()
            return wifi_infos_df['RSS'][:3].mean()
        elif len(temp_list)==2:
            #return (tuple(wifi_infos_df.iloc[0]),tuple(wifi_infos_df.iloc[1])),wifi_infos_df['RSS'][:2].mean()
            return wifi_infos_df['RSS'][:2].mean()
        else:
            #return (tuple(wifi_infos_df.iloc[0])),wifi_infos_df['RSS'][:1].mean()
            return wifi_infos_df['RSS'][:1].mean()
    
    #df['wifi_infos_sel'],df['rss_sel_aver'] = df.wifi_infos.map(lambda ele:split_str(ele))
    df['rss_sel_aver'] = df.wifi_infos.map(lambda ele:split_str(ele))
    
    return df

def split_str(wifi_info):
    set_bssid = set({})
    temp_list = wifi_info.split(';')
    for ele in temp_list:
        set_bssid.add(int(ele.split('|')[0][2:]))
    return set_bssid
    
def count_bssid(df,key):
    
    def get_id_bssid(key,value):
        temp_set = set({})
        for ele in value.values:
            for el in ele:
                temp_set.add(el)
        
        df_id_bssid.loc[key] = [temp_set]
                          
    df.set_index(key,inplace=True)
    df_id_bssid = pd.DataFrame(index=df.index.drop_duplicates(),columns=['bssid'])
                      
    df.index.drop_duplicates().map(lambda x:
        get_id_bssid(x,pd.Series(df.loc[x]['wifi_infos']).map(lambda wifi_info:split_str(wifi_info))))
           
    return df_id_bssid

def compare_mall(test,df,key):
    def compare(row,df):
        #print(row['bssid_test'],df.loc[row['mall_id']])
        same_bssid = set(df.loc[row[key]]['bssid']).intersection(set(row['bssid_test']))
        return same_bssid
    
    test['bssid_test'] = test.wifi_infos.map(lambda x :split_str(x))
    test['same_bssid'] = test.index.map(lambda index : compare(test.loc[index],df))
    test['same_count'] = test.same_bssid.map(lambda x : len(x))
    
    return test[['shop_id','bssid_test','same_bssid','same_count']]

def compare_shop(test,df,key):
    def compare(row,df):
        temp_df = df[(df['mall_id']==row['mall_id'])]
        same_bssid = temp_df.bssid.map(lambda x :set(x).intersection(set(row['bssid_test'])))
        return same_bssid
    
    test['bssid_test'] = test.wifi_infos.map(lambda x :split_str(x))
    test['same_bssid'] = test.index.map(lambda index : compare(test.loc[index],df))
    test['same_count'] = test.same_bssid.map(lambda x : x.map(lambda ele:len(ele)))
    
    return test[['shop_id','bssid_test','same_bssid','same_count']]

def plot(compare_test_train):
    sn.set_style("white")
    sn.set_context({"figure.figsize": (24, 10)})
    sn.barplot(x=compare_test_train.mall_id,y=compare_test_train.train_bssid_counts,color='red')
    sn.barplot(x=compare_test_train.mall_id,y=compare_test_train.test_bssid_counts,color='blue')
    
if __name__ =="__main__":
    #读取数据
    shop_info = pd.read_csv(shop_info_path)
    shop_behavior = pd.read_csv(shop_behavior_path)
    test_data = pd.read_csv(test_path)
    print('....连接shop_info和shop_behavior两个表')
    shop_behavior = pd.merge(shop_behavior,shop_info,on=['shop_id'],how='left')
    t0 = time.time()
    
#==============================================================================
#     print('**************统计训练数据中每个商场所有bssid************')
#     train_mall_id_bssid = count_bssid(shop_behavior,key='mall_id')
#     t1 = time.time()
#     print('********用时{} s'.format(time.time()-t0))
#==============================================================================

#==============================================================================
#     print('**************统计测试数据中每个商场所有bssid************')
#     
#     test_mall_id_bssid = count_bssid(test_data,key='mall_id')
#     print('********用时{} s'.format(time.time()-t1))
#==============================================================================

#==============================================================================
#     print('**************统计测试数据中每条记录数据的bssid与对应商场的匹配程度********')
#     test_compare_train = compare_mall(test_data,train_mall_id_bssid)
#     print('********用时{} s'.format(time.time()-t1))
#==============================================================================
    
#==============================================================================
#     print('**************统计训练数据中每个店铺的所有bssid************')
#     train_shop_id_bssid = count_bssid(shop_behavior,key='shop_id')
#     t1 = time.time()
#     print('********用时{} s'.format(time.time()-t0))
#     train_shop_id_bssid = pd.merge(train_shop_id_bssid.reset_index(),shop_info[['shop_id','mall_id']],on = 'shop_id',how='left')
#     
#     print('**************统计测试数据中每条记录数据的bssid与对应商店的匹配程度********')
#     test_compare_train = compare_shop(test_data,train_shop_id_bssid,'mall_id')
#     print('********用时{} s'.format(time.time()-t1))
#==============================================================================
    
    #*******************将一个月的数据分为以下三个时段******************
    #level1:2017-08-01 00:00 ~ 2017-08-10 23:50
    #level2:2017-08-11 00:00 ~ 2017-08-21 23:50
    #level2:2017-08-21 00:00 ~ 2017-08-31 23:50 
    


#    print('...处理wifi_info,选出前三个强度的bssid,添加为新的一列')
#    t0 = time.time()
#    shop_behavior = deal_wifi_info(shop_behavior)
#    test_data = deal_wifi_info(test_data)
#    print('用时{}s'.format(time.time()-t0))
