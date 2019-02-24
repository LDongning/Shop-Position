# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
user_shop_hehavior = pd.read_csv('./data/train_ccf_first_round_user_shop_behavior.csv')
evalution = pd.read_csv('./data/test_evaluation_public.csv')

#让WIFI关联商铺

#构造规则
print('规则构造....')
#选择每条历史数据中的最强的 bssid所对应的shop_id ，每出现一次，就加一
wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))
for line in user_shop_hehavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1

#线下验证
print('线下验证....')
right_count = 0
for line in user_shop_hehavior[(user_shop_hehavior['time_stamp']>'2017-08-25 00:00')].values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0] 
    counter = wifi_to_shops[wifi[0]]
    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count/len(user_shop_hehavior[(user_shop_hehavior['time_stamp']>'2017-08-25 00:00')])) 

#预测
print('预测....')
preds = []
for line in evalution.values:
    index = 0
    while True:
        try:
            if index==5:
                #print('...')
                pred_one = np.nan
                break
            wifi = sorted([wifi.split('|') for wifi in line[6].split(';')],key=lambda x:int(x[1]),reverse=True)[index]
            counter = wifi_to_shops[wifi[0]]
            pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
            break
        except:
            index+=1
    preds.append(pred_one)

result = pd.DataFrame({'row_id':evalution.row_id,'shop_id':preds})
result.fillna('s_666').to_csv('wifi_baseline.csv',index=None) #随便填的 这里还能提高不少