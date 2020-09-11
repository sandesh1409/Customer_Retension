import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from time import strftime
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv('raw.csv')

group = df.groupby('username')

count = 0
df1 = pd.DataFrame()
T = dt.datetime(2050, 1, 1, 0, 0)
df[['signup_date','ref_date']] = df[['signup_date','ref_date']].astype('datetime64[ns]')

for i,j in group:
    count += 1
    if count % 7 == 0:
        j['churn_date'] = j['signup_date'] + dt.timedelta(days=332)
        
    elif count % 9 == 0:
        j['churn_date'] = j['signup_date'] + dt.timedelta(days=557)
        
    elif count % 11 == 0:
        j['churn_date'] = j['signup_date'] + dt.timedelta(days=786)
        
    elif count % 13 == 0:
        j['churn_date'] = j['signup_date'] + dt.timedelta(days=871)
        
    elif count % 17 == 0 or count % 19 == 0:
        j['churn_date'] = j['signup_date'] + dt.timedelta(days=987)
        
    elif count % 23 == 0 or count % 27 == 0 or count % 29 == 0:
        j['churn_date'] = j['signup_date'] + dt.timedelta(days=1001)
        
    else:
        j['churn_date'] = T 
        
    
    df1 = pd.concat([df1, j], axis=0) 


df1['churn'] = 0
for i in range(len(df1)):
    if df1['churn_date'].iloc[i] == T:
        df1['churn'][i] = 0
    else:
        df1['churn'][i] = 1

    

def convert_date_to_ordinal(date):
    return date.toordinal()

for i in range(len(df1)):
    df1['signup_date'].iloc[i] = convert_date_to_ordinal(df1['signup_date'].iloc[i])
    df1['ref_date'].iloc[i] = convert_date_to_ordinal(df1['ref_date'].iloc[i])
    df1['churn_date'].iloc[i] = convert_date_to_ordinal(df1['churn_date'].iloc[i])
    
df1 = df1.astype({'signup_date':'int64', 'ref_date':'int64', 'churn_date':'int64'})


X = df1.drop(['username','churn_date', 'churn'], 1)
y = df1['churn_date']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from catboost import CatBoostRegressor

model=CatBoostRegressor(iterations=500, depth=3, learning_rate=0.1, loss_function='RMSE')
cat_features = np.where(X.dtypes != np.int64)[0]
# Fit model
model.fit(X_train, y_train, cat_features,eval_set=(X_test, y_test), plot=True, verbose=False)


fet_input = np.array([77820,89452,12,45,'Canada'])
pred = model.predict(fet_input)

import pickle
mod = '/home/sandynote/Desktop/Intern_task/Nakunj_Inc/modelfile1.pkl'
with open(mod, 'wb') as file:
    pickle.dump(model,file)

print("model created")