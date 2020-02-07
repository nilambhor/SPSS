#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score


# In[2]:


from imblearn.over_sampling import SMOTE


# In[3]:


cr = pd.read_csv('E:\\data science project\\UCI_Credit_Card.csv')


# In[4]:


cr.head()


# In[5]:


cr['default.payment.next.month'].value_counts()


# In[6]:


cr.info()


# In[7]:


X=cr[['PAY_0','PAY_2','PAY_3','BILL_AMT1','BILL_AMT2','BILL_AMT3']]


# In[8]:


y=cr[['default.payment.next.month']]


# In[9]:


x_train, x_val, y_train, y_val = train_test_split(X,y,
                                                  test_size = 0.3,
                                                  random_state=0)


# In[10]:


sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)


# In[11]:


from sklearn.tree import DecisionTreeClassifier


# In[12]:


from sklearn import metrics


# In[13]:


lm=DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=6)


# In[14]:


lm.fit(x_train_res, y_train_res)


# In[15]:


print ('Validation Results')
print (lm.score(x_val, y_val))
print (recall_score(y_val, lm.predict(x_val)))


# In[16]:


print ('\nTest Results')
print (lm.score(x_val, y_val))
print (recall_score(y_val, lm.predict(x_val)))


# In[17]:


from sklearn import metrics


# In[18]:


prediction=lm.predict(x_val)


# In[19]:


print(metrics.accuracy_score(prediction, y_val))


# In[ ]:





