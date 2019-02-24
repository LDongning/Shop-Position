# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict
path='./data/'
user_shop_hehavior=pd.read_csv(path+'train_ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv(path+'train_ccf_first_round_shop_info.csv')

#test=pd.read_csv(path+'test_evaluation_public.csv')

mall_list=list(set(list(shop.mall_id)))


# {max_rss_bssid:{shop_id:count}}
bssid_counts = defaultdict(lambda : 0)
bssid_valid = set([])
for line in user_shop_hehavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    bssid_counts[wifi[0]]= bssid_counts[wifi[0]] + 1
    if bssid_counts[wifi[0]] > 20:
       bssid_valid.add(wifi[0])
              
#生成index = shop_id,columns = bssid1,bssid2....,values = rss的表
#其中bssid是历史出现次数超过20次的

    
