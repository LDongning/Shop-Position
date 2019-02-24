import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import keras
from keras.models import Model,Sequential,load_model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Flatten,Dense,concatenate
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import metrics
from sklearn.preprocessing import scale
import os
os.chdir("user/tianchi")
def haversine(lon1, lat1, lon2, lat2):
    from math import radians, cos, sin, asin, sqrt
    lon1= map(radians, np.array(lon1))  
    lat1= map(radians, np.array(lat1))
    lon2= map(radians, np.array(lon2))
    lat2= map(radians, np.array(lat2))
    lon1 = np.array(list(lon1)).reshape(-1,1)
    lon2 = np.array(list(lon2)).reshape(-1,1)
    lat1 = np.array(list(lat1)).reshape(-1,1)
    lat2 = np.array(list(lat2)).reshape(-1,1)
    dlon = lon2 - lon1
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2  
    c = 2 * np.arcsin(np.sqrt(a))   
    r = 6371
    return c * r * 1000  
df=pd.read_csv("data10-14.csv",encoding="gbk")
df1=pd.read_csv("shop10-14.csv",encoding="gbk").set_index("店铺ID")
df1["ran"]=range(len(df1))
df=pd.merge(df,df1,left_on="shop_id",right_index=True,how="left")
dfx2=pd.pivot_table(df1,"人均消费指数",index="店铺所在商场ID",columns="按商场排名").fillna(0).astype(int)
inde=np.random.permutation(np.arange(len(df)))
trainindex=inde[:900000]
testindex=inde[900000:]
trainindex1=trainindex[:450000]
trainindex2=trainindex[450000:]
dfs=df.iloc[trainindex1].wifi_infos.str.split(";",expand=True)
dfs1=dfs[0]
for i in dfs.columns[1:]:
    dfs1=pd.concat([dfs1,dfs[i]])
dfs1=dfs1.dropna()
dfs1=dfs1.str.split("|",expand=True)
dfs1[2]=dfs1[2].map(lambda x:50 if x=="true" else 0)
dfs1[1]=dfs1[1].astype(int)
dfs1[1]=dfs1[1]+dfs1[2]+100
dfs1=dfs1.drop(2,axis=1)
dfs1[0]=dfs1[0].map(lambda x:x+"133" if len(x)<6 else x)
dfs1["index1"]=dfs1[0].map(lambda x:x[:4])
dfs1["column1"]=dfs1[0].map(lambda x:x[4:6])
dfs1=dfs1.drop(0,axis=1)
dfs1=dfs1.rename(columns={1:"va"})
dfs1=dfs1.reset_index()
dfs2=dfs1.groupby(["index","index1","column1"]).va.max().unstack().unstack().fillna(0).astype(int)

dfs=df.iloc[trainindex2].wifi_infos.str.split(";",expand=True)
dfs1=dfs[0]
for i in dfs.columns[1:]:
    dfs1=pd.concat([dfs1,dfs[i]])
dfs1=dfs1.dropna()
dfs1=dfs1.str.split("|",expand=True)
dfs1[2]=dfs1[2].map(lambda x:50 if x=="true" else 0)
dfs1[1]=dfs1[1].astype(int)
dfs1[1]=dfs1[1]+dfs1[2]+100
dfs1=dfs1.drop(2,axis=1)
dfs1[0]=dfs1[0].map(lambda x:x+"133" if len(x)<6 else x)
dfs1["index1"]=dfs1[0].map(lambda x:x[:4])
dfs1["column1"]=dfs1[0].map(lambda x:x[4:6])
dfs1=dfs1.drop(0,axis=1)
dfs1=dfs1.rename(columns={1:"va"})
dfs1=dfs1.reset_index()
dfs3=dfs1.groupby(["index","index1","column1"]).va.max().unstack().unstack().fillna(0).astype(int)

dfs2=pd.concat([dfs2,dfs3])
del dfs3
dfs2=dfs2.loc[trainindex]

dfs=df.iloc[testindex].wifi_infos.str.split(";",expand=True)
dfs1=dfs[0]
for i in dfs.columns[1:]:
    dfs1=pd.concat([dfs1,dfs[i]])
dfs1=dfs1.dropna()
dfs1=dfs1.str.split("|",expand=True)
dfs1[2]=dfs1[2].map(lambda x:50 if x=="true" else 0)
dfs1[1]=dfs1[1].astype(int)
dfs1[1]=dfs1[1]+dfs1[2]+100
dfs1=dfs1.drop(2,axis=1)
dfs1[0]=dfs1[0].map(lambda x:x+"133" if len(x)<6 else x)
dfs1["index1"]=dfs1[0].map(lambda x:x[:4])
dfs1["column1"]=dfs1[0].map(lambda x:x[4:6])
dfs1=dfs1.drop(0,axis=1)
dfs1=dfs1.rename(columns={1:"va"})
dfs1=dfs1.reset_index()
dfs4=dfs1.groupby(["index","index1","column1"]).va.max().unstack().unstack().fillna(0).astype(int)
dfs4=dfs4.loc[testindex]
dict1=df1.ran.to_dict()
df["ra"]=df.shop_id.map(dict1)
dfs=pd.read_csv("trainaway.csv").drop("Unnamed: 0",axis=1)
target=pd.get_dummies(df.按商场排名)
df["time_stamp"]=pd.to_datetime(df.time_stamp,format="%Y-%m-%d %H:%M")
df["rank1"]=df.groupby("user_id").time_stamp.rank(ascending=False,method="first").astype(int)
for i in range(1,df.rank1.max()):
    print(i)
    dfu=df[df.rank1==i]
    dfu1=df[df.rank1==i+1].rename(columns={"店铺所在商场ID":"oldid","按商场排名":"oldrank"})
    dfu2=pd.merge(pd.DataFrame(dfu["user_id"]),dfu1[["user_id","oldid","oldrank"]].set_index("user_id"),left_on="user_id",right_index=True,how="left")
    if i==1:
        dfu3=dfu2.copy()
    else:
        dfu3=pd.concat([dfu3,dfu2])
df=pd.merge(df,dfu3.drop("user_id",axis=1),left_index=True,right_index=True,how="left")
dfo1=pd.get_dummies(df.oldid).fillna(0).astype(int)
dfo2=pd.get_dummies(df.oldrank).loc[:,range(220)].fillna(0).astype(int)
dfd=pd.read_csv("trainwifi.csv").drop("Unnamed: 0",axis=1)
dff=df.groupby("user_id").按商场排名.value_counts().unstack().fillna(0).astype(int)
dff=pd.merge(pd.DataFrame(df.user_id),dff,left_on="user_id",right_index=True,how="left").drop("user_id",axis=1)
dff=dff-pd.get_dummies(df.按商场排名)
dfx4=pd.read_csv("shop10-26.csv").drop("Unnamed: 0",axis=1)
dfx5=pd.read_csv("trainwifi2.csv").drop("Unnamed: 0",axis=1)

input1=Input((220,))
dense1=Dense(512,activation="tanh")(input1)
input2=Input((220,))
dense2=Dense(512,activation="tanh")(input2)
input3=Input((97,))
dense3=Dense(512,activation="tanh")(input3)
input31=Input((97,))
dense31=Dense(512,activation="tanh")(input31)
input32=Input((220,))
dense32=Dense(512,activation="tanh")(input32)
input32=Input((220,))
dense32=Dense(512,activation="tanh")(input32)
input34=Input((220,))
dense34=Dense(512,activation="tanh")(input34)
input35=Input((220,))
dense35=Dense(512,activation="tanh")(input35)
input36=Input((220,))
dense36=Dense(512,activation="tanh")(input36)
merge1=concatenate([dense1,dense2,dense3,dense31,dense32,dense34,dense35,dense36],axis=-1)
dense4=Dense(1024,activation="tanh")(merge1)
drop1=Dropout(0.5)(dense4)
input33=Input((220,))
dense33=Dense(512,activation="tanh")(input33)
input4=Input((90,100,1))
conv1 = Conv2D(16, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(input4)
print ("conv1 shape:",conv1.shape)
conv1 = Conv2D(16, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(conv1)
print ("conv1 shape:",conv1.shape)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
fla=Flatten()(pool1)
dense5=Dense(512,activation="tanh")(fla)
drop2=Dropout(0.5)(dense5)
merge2=concatenate([drop1,drop2,dense33],axis=-1)
dense7=Dense(1024,activation="tanh")(merge2)
drop7=Dropout(0.5)(dense7)
out=Dense(220,activation="softmax")(drop7)
model=Model([input1,input2,input3,input4,input31,input32,input33,input34,input35,input36],out)
model.compile(optimizer = Adam(1e-3), loss = 'categorical_crossentropy', metrics = [metrics.categorical_accuracy])
tensorboard = TensorBoard(log_dir='log2', histogram_freq=0)
checkpoint = ModelCheckpoint("dengqi2.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
rate=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0000001, cooldown=0, min_lr=0)
callbacks_list = [checkpoint, tensorboard,rate]
model.fit([dfs.iloc[trainindex].values,dfx2.loc[df.店铺所在商场ID].iloc[trainindex].values,pd.get_dummies(df.店铺所在商场ID).iloc[trainindex].values,dfs2.values.reshape(900000,90,100,1),dfo1.iloc[trainindex].values,dfo2.iloc[trainindex].values,dfd.iloc[trainindex].values,dff.iloc[trainindex].values,dfx4.iloc[trainindex].values,dfx5.iloc[trainindex].values],target.iloc[trainindex].values,batch_size=512,epochs=100,verbose=1,validation_data=([dfs.iloc[testindex].values,dfx2.loc[df.店铺所在商场ID].iloc[testindex].values,pd.get_dummies(df.店铺所在商场ID).iloc[testindex].values,dfs4.values.reshape(238015,90,100,1),dfo1.iloc[testindex].values,dfo2.iloc[testindex].values,dfd.iloc[testindex].values,dff.iloc[testindex].values,dfx4.iloc[testindex].values,dfx5.iloc[testindex].values],target.iloc[testindex].values),callbacks=callbacks_list)
