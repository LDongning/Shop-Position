# -*- coding: utf-8 -*-
import pandas as pd
import time
import datetime
shop_info_path = './data/train_ccf_first_round_shop_info.csv'
shop_behavior_path = './data/train_ccf_first_round_user_shop_behavior.csv'
evaluation_path = './data/test_evaluation_public.csv'
multi_result_path = './result/multi_result.csv'
result_path = './result/evaluate_result_less_14_2017-10-24_14_46_14[shop_ids].csv'

def get_Time_Feature(data):
    
    def func(ele):
        morning = [7,8,9,10]
        moon = [11,12,13,14]
        afternoon = [15,16,17,18]
        night = [19,20,21,22,23,0]
        
        if datetime.datetime.strptime(ele,'%Y-%m-%d %H:%M').hour in morning:
            return 0
        elif datetime.datetime.strptime(ele,'%Y-%m-%d %H:%M').hour in moon:
            return 1
        elif datetime.datetime.strptime(ele,'%Y-%m-%d %H:%M').hour in afternoon:
            return 2
        elif datetime.datetime.strptime(ele,'%Y-%m-%d %H:%M').hour in night:
            return 3
        else:
            return 4
        
    data['time_class'] = data['time_stamp'].map(lambda ele: func(ele))
    del data['time_stamp']
    return data
 
def judge(i,select_multi_result,table_count):
    row = select_multi_result.loc[i]
    cate_count_dict,cate_shop_id_dict = {},{}
    for index,ele in enumerate(row['category']):
        #print((ele,row['time_class']))
        if (ele,row['time_class']) in list(table_count.index):
            cate_count_dict[ele] = table_count.loc[(ele,row['time_class'])]['counts']
        else:
            cate_count_dict[ele] = 0
        
        cate_shop_id_dict[ele] = row['shop_ids'][index]
        
    #找出最大的那个类
    #print(cate_count_dict)
    cate_count_dict = sorted(cate_count_dict.items(),key=lambda item:item[1],reverse=True)
    
    #返回最大的类对应的shop_id
    return cate_shop_id_dict[cate_count_dict[0][0]] 
        
if __name__=='__main__':
    shop_info = pd.read_csv(shop_info_path)
    train_data = pd.read_csv(shop_behavior_path)
    print('连接shop_info和shop_behavior两个表...')
    train_data = pd.merge(train_data,shop_info[['shop_id','category_id','mall_id']],on=['shop_id'],how='left')

    multi_result = pd.read_csv(multi_result_path) 
    result = pd.read_csv(result_path)
    #字符串转换
    result['shop_ids'] = result['shop_ids'].map(lambda x:eval(x))
    multi_result['shop_ids'] = multi_result['shop_ids'].map(lambda x:eval(x))
    multi_result['category'] = multi_result['category'].map(lambda x:eval(x))
    print('开始计算...')
    t0 = time.time()
    
    #将训练数据时间离散化，分为5个类别，
    print('训练数据时间离散化...')
    train_data = get_Time_Feature(train_data)
    train_data['gs'] = 1 
    print('按时间类别和商店类型分类统计...')
    table_count = train_data.groupby(by = ['category_id','time_class'])['gs'].count().reset_index()
    table_count.columns = ['category_id','time_class','counts']
    
    #将测试数据时间离散化
    print('测试数据时间离散化...')
    evaluation_data = pd.read_csv(evaluation_path)
    evaluation_data = get_Time_Feature(evaluation_data)
    print('连接multi_result和evaluation_data两个表....')
    multi_result = pd.merge(multi_result,evaluation_data[['row_id','time_class']],on='row_id',how='inner')
    
    #选择multi_result中没有重复类别商店的数据
    print('选择multi_result中没有重复类别商店的数据...')
    multi_result['value'] = multi_result.index.map(lambda i : len(multi_result.loc[i]['shop_ids'])-len(set(multi_result.loc[i]['category'])))
    select_multi_result = multi_result[(multi_result['value']==0)]
    print('multi_result中,非重复商店类别的个数',len(select_multi_result),'占总multi_result比例：{}%'.format(len(select_multi_result)*100/len(multi_result)))
    
    #将选取的数据进行按照 "时间类别和商店类型的统计频次" 进行判断，挑出出现频次最高的商店
    print('按条件进行筛选判断...')
    table_count.set_index(['category_id','time_class'],inplace =True)
    select_multi_result['shop_id'] = select_multi_result.index.map(lambda i : judge(i,select_multi_result,table_count))
    
    #将挑选出来的数据，与result联合
    rest_df = result[~(result['row_id'].isin(select_multi_result['row_id']))][['row_id','shop_id']]
    new_result = rest_df.append(select_multi_result[['row_id','shop_id']])
    new_result.to_csv('./result/result_change.csv',index=False)
    
    
    