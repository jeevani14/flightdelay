#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Reading allstation data
station = pd.read_csv("AllStationsData_PHD.txt",sep='|')


# In[3]:


station


# In[4]:


#Reading train data
train = pd.read_csv("Train (1).csv")


# In[5]:


train


# In[6]:


#Visualizaion using scatterplot to get inference
train.plot.scatter(x='ScheduledTravelTime', y='Distance', alpha=0.5)
plt.show


# In[7]:


#Read test data
test = pd.read_csv("Test.csv")


# In[8]:


test


# In[9]:


#Merging train and allstation data 
df_1 = pd.merge(train, station, left_on='Destination', right_on='AirportID', how='left')


# In[10]:


df_1


# In[11]:


#Merging test and allstation data
df_2 = pd.merge(test, station, left_on='Destination', right_on='AirportID', how='left')


# In[12]:


df_2


# In[13]:


#Reading 2004 hourly data
h1 = pd.read_csv("200401hourly.txt")
h2 = pd.read_csv("200403hourly.txt")
h3 = pd.read_csv("200405hourly.txt")
h4 = pd.read_csv("200407hourly.txt")
h5 = pd.read_csv("200409hourly.txt")
h6 = pd.read_csv("200411hourly.txt")


# In[14]:


#Concatenating all the 2004 hourly data into one dataframe
hourly = pd.concat([h1,h2,h3,h4,h5,h6], axis=0, ignore_index = True, sort = False)


# In[15]:


#Reading 2005 hourly data
i1 = pd.read_csv("200503hourly.txt")
i2 = pd.read_csv("200507hourly.txt")
i3 = pd.read_csv("200507hourly.txt")
i4 = pd.read_csv("200511hourly.txt")


# In[16]:


#Concatinationg all the 2005 hourly data into one dataframe
hourly_1 = pd.concat([i1,i2,i3,i4], axis=0, ignore_index = True, sort = False)


# In[17]:


hourly.head(1)


# In[18]:


df_1.head(1)


# In[19]:


hourly = hourly.sort_values(by=['Time'])
df_1 = df_1.sort_values(by=['ScheduledArrTime'])


# In[20]:


hourly_1 = hourly_1.sort_values(by=['Time'])
df_2 = df_2.sort_values(by=['ScheduledArrTime'])


# In[21]:


#Merging dataframe 1 with concatinated hourly dataframe
df = pd.merge_asof(df_1, hourly, left_on='ScheduledArrTime', right_on='Time', by=['WeatherStationID'])


# In[22]:


df


# In[23]:


df.dtypes


# In[24]:


df.drop(['AirportID','YearMonthDay','Time','FlightNumber','Origin','SkyConditions'], axis=1, inplace=True)


# In[25]:


#Merging dataframe 1 with concatinated hourly dataframe
td = pd.merge_asof(df_2, hourly_1, left_on='ScheduledArrTime', right_on='Time', by=['WeatherStationID'])


# In[26]:


td


# In[27]:


td.drop(['AirportID','YearMonthDay','Time','FlightNumber','Origin','SkyConditions'], axis=1, inplace=True)


# In[28]:


df.isna().sum()


# In[29]:


df.dtypes


# In[30]:


#Creating target column  
df[['Date','time']] = df['ActualArrivalTimeStamp'].str.split(' ', expand= True)


# In[31]:


df[['A','B']] = df['time'].str.split(':', expand= True)


# In[32]:


df['Actual'] = df['A']+df['B']


# In[33]:


df.drop(['A','B'],axis=1,inplace=True)


# In[34]:


df['Actual'].astype(int)


# In[35]:


df['Actual'] = df['Actual'].astype(int)


# In[36]:


df['ScheduledArrTime'] = df['ScheduledArrTime'].astype(int)


# In[37]:


df['target'] = df['Actual'] - df['ScheduledArrTime']


# In[38]:


df['target'].nunique()


# In[39]:


def Target(x):
    if x>=15:
        return 1
    else:
        return 2
df['Target'] = df['target'].apply(Target)


# In[40]:


#visualization using scatterplot
df.plot.scatter(x='Target', y='ScheduledTravelTime', alpha=0.5)
plt.show


# In[41]:


df.drop(['Destination'],axis=1, inplace=True)
td.drop(['Destination'],axis=1, inplace=True)


# In[42]:


df.drop(['ActualArrivalTimeStamp','Year','Month','DayofMonth'], axis=1, inplace=True)


# In[43]:


#Heatmap to find correlation between attributes
fig, axs = plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(), cmap ='RdYlGn', annot = True)


# In[44]:


#Preprocessing the data
df['Visibility'] = df['Visibility'].str.replace('SM','')


# In[45]:


df['Visibility'].fillna(df['Visibility'].median(), inplace=True)


# In[46]:


df['Visibility'] = pd.to_numeric(df['Visibility'])


# In[47]:


df['TimeZone'].value_counts()


# In[48]:


timezone = {'+5':5,'+6':6,'+8':8,'+7':7,'+4':4,'+9':9,'+10':10}
df['timezone'] = df.TimeZone.replace(timezone)


# In[49]:


df.drop(['TimeZone'],axis=1,inplace=True)


# In[50]:


df['DBT'].fillna(df['DBT'].mean(),inplace=True)


# In[51]:


df['DewPointTemp'].fillna(df['DewPointTemp'].mean(),inplace=True)


# In[52]:


df['RelativeHumidityPercent'].fillna(df['RelativeHumidityPercent'].mean(), inplace=True)


# In[53]:


df['WindSpeed'].fillna(df['WindSpeed'].median(),inplace=True)


# In[54]:


df['WindSpeed'] = df['WindSpeed'].astype(object).astype(int)


# In[55]:


df['WindDirection'] = df['WindDirection'].apply(lambda x: 0 if x =='VRB' else x)


# In[56]:


df['WindDirection'] = pd.to_numeric(df['WindDirection'])


# In[57]:


df['WindDirection'].fillna(df['WindDirection'].mean(), inplace=True)


# In[58]:


df['WindGustValue'].fillna(df['WindGustValue'].mean(),inplace=True)


# In[59]:


df['StationPressure'].fillna(df['StationPressure'].mean(), inplace=True)


# In[60]:


td.isna().sum()


# In[61]:


td.dtypes


# In[62]:


td.drop(['Year','Month','DayofMonth'], axis=1, inplace=True)


# In[63]:


td['Visibility'] = td['Visibility'].str.replace('SM','')


# In[64]:


td['Visibility'].fillna(td['Visibility'].median(), inplace=True)


# In[65]:


td['Visibility'] = pd.to_numeric(td['Visibility'])


# In[66]:


td['TimeZone'].value_counts()


# In[67]:


timezone = {'+5':5,'+6':6,'+8':8,'+7':7,'+4':4,'+9':9,'+10':10}
td['timezone'] = td.TimeZone.replace(timezone)


# In[68]:


td.drop(['TimeZone'],axis=1,inplace=True)


# In[69]:


td['DBT'].fillna(td['DBT'].mean(),inplace=True)


# In[70]:


td['DewPointTemp'].fillna(td['DewPointTemp'].mean(),inplace=True)


# In[71]:


td['RelativeHumidityPercent'].fillna(td['RelativeHumidityPercent'].mean(), inplace=True)


# In[72]:


td['WindSpeed'].fillna(td['WindSpeed'].median(),inplace=True)


# In[73]:


td['WindSpeed'] = td['WindSpeed'].astype(object).astype(int)


# In[74]:


td['WindDirection'] = td['WindDirection'].apply(lambda x: 0 if x =='VRB' else x)


# In[75]:


td['WindDirection'] = pd.to_numeric(td['WindDirection'])


# In[76]:


td['WindDirection'].fillna(td['WindDirection'].mean(), inplace=True)


# In[77]:


td['WindGustValue'].fillna(td['WindGustValue'].mean(),inplace=True)


# In[78]:


td['StationPressure'].fillna(td['StationPressure'].mean(), inplace=True)


# In[79]:


df.drop(['Date','time','Actual','target'],axis=1,inplace=True)


# In[80]:


#Standardizing the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#model = scaler.fit(df)
#scaled_data = model.transform(df)


# In[81]:


df.drop(['DayOfWeek', 'WeatherStationID'], axis=1, inplace=True)
td.drop(['DayOfWeek', 'WeatherStationID'], axis=1, inplace=True)


# In[82]:


df


# In[83]:


td.head(1)


# In[84]:


df_x = df.drop(columns = ['Target'])


# In[85]:


df_y = df.iloc[:,17:18]


# In[86]:


df_x.head()


# In[87]:


df_y.head()


# In[88]:


#ML Modelling

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[89]:


X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 42)


# In[90]:


X_train = scaler.fit_transform(X_train)


# In[91]:


rf = RandomForestClassifier(max_depth=8, n_estimators = 500, criterion='gini',max_features= 'sqrt',class_weight='balanced')
rf.fit(X_train, y_train)


# In[92]:


y_train_prediction = rf.predict(X_train)
y_test_prediction = rf.predict(X_test)


# In[93]:


#Predicting the accuracy
print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))


# In[94]:


print(classification_report(y_test, y_test_prediction))


# In[95]:


#https://www.kaggle.com/code/sociopath00/random-forest-using-gridsearchcv
#param_grid = { 
   # 'n_estimators': [100, 500],
   # 'max_features': ['auto', 'sqrt', 'log2'],
    #'max_depth' : [4,5,6,7,8],
    #'criterion' :['gini', 'entropy']
#}


# In[96]:


#from sklearn.model_selection import GridSearchCV
#CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
#CV_rf.fit(X_train, y_train)


# In[97]:


#CV_rf.best_params_


# In[98]:


from sklearn.metrics import f1_score 
print(f1_score(y_test, y_test_prediction))


# In[99]:


rf_prediction = rf.predict(td)
print(rf_prediction)


# In[100]:


result1 = test.iloc[:,0:1]


# In[101]:


result1


# In[102]:


result2 = pd.DataFrame(rf_prediction,columns = ['FlightDelayStatus'])


# In[103]:


result2


# In[104]:


result2.value_counts()


# In[105]:


output = pd.concat([result1, result2], axis=1)


# In[106]:


#output.to_csv("output4", index=False)


# In[107]:


#output.to_csv("output5", index=False) accuracy increasing


# In[108]:


#output.to_csv("output6", index=False)


# In[109]:


#output.to_csv("output7", index=False) less score


# In[110]:


#output.to_csv("output8", index=False)


# In[111]:


#output.to_csv("output9", index=False)


# In[112]:


#output.to_csv("output10", index=False) improved aacuracy


# In[113]:


#output.to_csv("output11", index=False)


# In[114]:


#output.to_csv("output12", index=False)


# In[115]:


output.to_csv("output13", index=False)


# In[116]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()


# In[117]:


log.fit(X_train,y_train)


# In[118]:


prediction = log.predict(X_test)


# In[126]:


classification_report(y_test, prediction)


# In[127]:


from sklearn.metrics import f1_score 
print(f1_score(y_test, prediction))


# In[120]:


logpredict = log.predict(td)
print(logpredict)


# In[121]:


result11 = test.iloc[:,0:1]


# In[122]:


result22 = pd.DataFrame(logpredict,columns = ['FlightDelayStatus'])


# In[123]:


result22.value_counts()


# In[124]:


output = pd.concat([result11, result22], axis=1)


# In[125]:


output.to_csv("output14", index=False)


# In[ ]:





# In[ ]:




