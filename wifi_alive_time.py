# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
import time
user_shop_behavior_path = './data/train_ccf_first_round_user_shop_behavior.csv'
shop_wifi_infos_path = './data/shop_wifi_infos.csv'
#==============================================================================
# 得到大于等于指定天数以上稳定的wifi数据
#==============================================================================
def get_data_largger_15_bssid(day_value,wifi_data_counts):
    bssid_list = []
    for item in wifi_data_counts.items():
        if len(item[1]) >= day_value:
            bssid_list.append(item[0])
    return bssid_list
#==============================================================================
# 方式一：dataFrame 处理掉非稳定wifi数据
#==============================================================================
def get_stable_shop_wifi_infos(bssid_list,shop_wifi_infos):
    stable_shop_wifi_infos = pd.DataFrame(columns=['shop_id','stable_bssid_wifi_info'])
    counts = 0
    for wifi_line in shop_wifi_infos.values:
        counts += 1
        print("处理掉%d间商铺信息。。。"%counts)
        shop_id = wifi_line[0]
        one_shop_wifi_dict = {}
        for wifi in wifi_line[1].split(';')[:-1]:
            bssid = wifi.split('|')[0]
            rss = wifi.split('|')[1]
            if bssid in bssid_list:
                if bssid in one_shop_wifi_dict:
                    one_shop_wifi_dict[bssid].append(rss)
                else:
                    one_shop_wifi_dict.setdefault(bssid,[]).append(rss)
        one_shop_wifi = pd.DataFrame({'shop_id':shop_id,'stable_bssid_wifi_info':[one_shop_wifi_dict]})
        stable_shop_wifi_infos = stable_shop_wifi_infos.append(one_shop_wifi,ignore_index=True)
    return stable_shop_wifi_infos
#==============================================================================
# 方式二：利用deafultdict数据形式处理
#==============================================================================

def get_stable_shop_wifi_infos_defaultdict(wifi_line,bssid_list):
    dict_temp = defaultdict(lambda :[])
    for wifi in wifi_line['wifi_infos_add']:
        if wifi[0] in bssid_list:
            dict_temp[wifi[0]].append(wifi[1])
            
    wifi_line['stable_bssid_infos'] = dict_temp      
             
    return wifi_line.to_frame().T

#def get_stable_shop_wifi_infos_defaultdict(index,bssid_list,shop_wifi_infos):
#    def get_table(bssid_list,wifi):
#        if wifi[0] in bssid_list: 
#            stable_shop_wifi_infos_defaultdict[shop_id][wifi[0]].append(wifi[1])
#            
#    shop_id = shop_wifi_infos.loc[index]['shop_id']
#    print("处理掉%d间商铺信息。。。"%(index+1))
#    t0 = time.time()
#    pd.Series(shop_wifi_infos.loc[index]['wifi_infos_add']).map(lambda wifi : get_table(bssid_list,wifi))

def applyParallel(dfGrouped, bssid_list):
    with Parallel(n_jobs=6) as parallel:
        retLst = parallel( delayed(get_stable_shop_wifi_infos_defaultdict)(pd.Series(value),bssid_list) for key,value in dfGrouped)   
        return pd.concat(retLst, axis=0)
    
if __name__=='__main__':
    shop_wifi_infos = pd.read_csv(shop_wifi_infos_path)
    wifi_data_counts = defaultdict(lambda : defaultdict(lambda : 0))
    user_shop_behavior = pd.read_csv(user_shop_behavior_path)
    for line in user_shop_behavior.values:
        data = line[2].split(' ')[0]
        wifi_list = [wifi.split('|') for wifi in line[5].split(';')]
        for wifi in wifi_list:
            wifi_data_counts[wifi[0]][data] += 1
    print("统计bssid在8月份出现日期次数完毕")
#   day_value指定天数
    day_value = 15
    print("获得稳定bssid值......")
    bssid_list = get_data_largger_15_bssid(day_value,wifi_data_counts)
    print('获得每个店铺稳定wifi统计信息......')
    shop_wifi_infos['wifi_infos_add'] = shop_wifi_infos.wifi_infos_add.map(lambda line: [wifi.split('|') for wifi in line.split(';')[:-1]])
#    stable_shop_wifi_infos = get_stable_shop_wifi_infos(bssid_list,shop_wifi_infos)
    t0 = time.time()
    stable_shop_wifi_infos_defaultdict = applyParallel(shop_wifi_infos.iterrows(),bssid_list)
    print('花费时间：',time.time()-t0)    
#    stable_shop_wifi_infos_defaultdict = defaultdict(lambda:defaultdict(lambda : []))
#    shop_wifi_infos.index.map(lambda index: get_stable_shop_wifi_infos_defaultdict(index,bssid_list,shop_wifi_infos))