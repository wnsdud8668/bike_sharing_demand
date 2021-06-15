#%%

from datetime import date
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

import missingno as msno
plt.style.use('seaborn')
import warnings
warnings.filterwarnings("ignore")
mpl.rcParams['axes.unicode_minus'] = False

#%%
# %matplotlib inline




# +
#데이터 가져오기
import pandas as pd
import os 

#현재 경로 얻어오기
path = os.getcwd()

#df_train = pd.read_csv("train.csv")
train_df = pd.read_csv("train.csv", parse_dates = ["datetime"])
#df_test = pd.read_csv("test.csv")
test_df = pd.read_csv("test.csv", parse_dates = ["datetime"])

#%%

#sns.heatmap(df_train[['season', 'holiday', 'workingday', 'weather', 'temp',
#       'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']])
#plt.show()

msno.matrix(train_df, figsize=(12,5))


#%%

# month, hour 데이터 생성

train_df['month']=[ i.month for i in train_df['datetime']]
train_df['hour']=[ i.hour for i in train_df['datetime']]
test_df['month']=[ i.month for i in test_df['datetime']]
test_df['hour']=[ i.hour for i in test_df['datetime']]


#%%
# 데이터 분할
X_train_df=np.asarray(train_df[['season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed', 
       'month', 'hour']])

Y_train_df=np.asarray(train_df[['count']])

X_test_df=np.asarray(test_df[['season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed', 
       'month', 'hour']])

#%%

from sklearn.model_selection import train_test_split


# shuffle = False

X_train, X_test, y_train, y_test = train_test_split(X_train_df, Y_train_df, test_size=0.4, shuffle=False, random_state=1004)

#%%

from sklearn.ensemble import GradientBoostingRegressor 
regressor = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05, 
       max_depth=4, min_samples_leaf=15, min_samples_split=10, random_state =42) 
       
regressor.fit(X_train,y_train)

#%%
y_hat = regressor.predict(X_train) 

plt.scatter(y_train, y_hat, alpha = 0.2) 

plt.xlabel('Targets (y_tr)',size=18) 

plt.ylabel('Predictions (y_hat)',size=18) 

plt.show()

#%%

y_predict = regressor.predict(X_test) 

#%%