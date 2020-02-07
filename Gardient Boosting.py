#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score


# In[46]:


from imblearn.over_sampling import SMOTE


# In[47]:


cr = pd.read_csv('E:\\data science project\\UCI_Credit_Card.csv')


# In[48]:


cr.head()


# In[49]:


cr['default.payment.next.month'].value_counts()


# In[50]:


cr.info()


# In[51]:


X=cr[['PAY_0','PAY_2','PAY_3','BILL_AMT1','BILL_AMT2','BILL_AMT3']]


# In[52]:


y=cr[['default.payment.next.month']]


# In[53]:


x_train, x_val, y_train, y_val = train_test_split(X,y,
                                                  test_size = 0.3,
                                                  random_state=0)


# In[54]:


sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)


# In[55]:


from sklearn import metrics


# In[56]:


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}


# In[57]:


from sklearn import ensemble 


# In[58]:


model=ensemble.GradientBoostingRegressor(**params)


# In[59]:


model.fit(x_train_res, y_train_res)


# In[60]:


prediction=model.predict(x_val)


# In[61]:


pred=pd.DataFrame(prediction)


# In[62]:


pred


# In[63]:


print ('Validation Results')
print (model.score(x_val, y_val))


# In[64]:


print ('\nTest Results')
print (model.score(x_val, y_val))
#print (recall_score(y_val, model.predict(x_val)))


# In[65]:


model_score=model.score(x_train_res, y_train_res)


# In[66]:


print(model_score)


# In[67]:


import matplotlib.pyplot as plt


# In[68]:


from sklearn.model_selection import cross_val_predict

fig, ax = plt.subplots()
ax.scatter(y_val, prediction, edgecolors=(0, 0, 0))
ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()


# In[ ]:




