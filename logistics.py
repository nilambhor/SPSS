#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[3]:


wea=pd.read_csv("C:\\Users\\Administrator\\Downloads\\weather.csv")


# In[4]:


wea.head()


# In[5]:


def score_to_numeric(x):
    if x=='Yes':
        return 1
    if x=='No':
        return 0
   
    
wea['RainTomorrow'] = wea['RainTomorrow'].apply(score_to_numeric)


# In[6]:


wea.head()


# In[7]:


wea.info()


# In[9]:


X=wea[['Pressure3pm', 'Cloud3pm', 'Temp3pm', 'RISK_MM']]


# In[10]:


y=wea[['RainTomorrow']]


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=0)


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


model=LogisticRegression()


# In[15]:


model.fit(X_train, y_train)


# In[16]:


prediction=model.predict(X_test)


# In[18]:


pred=pd.DataFrame(prediction)


# In[19]:


pred


# In[21]:


from sklearn import metrics


# In[23]:


cm = metrics.confusion_matrix(y_test, prediction)
print(cm)


# In[24]:


print(metrics.accuracy_score(y_test, prediction))


# In[ ]:




