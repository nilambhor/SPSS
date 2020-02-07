#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[31]:


train_data = pd.read_csv(r'Downloads\train_big.csv')
test_data = pd.read_csv(r'Downloads\test_big.csv')


# In[32]:


train_data.head(5)


# In[33]:



train_data['Outlet_Size'].value_counts()


# In[34]:


def score_to_numeric(x):
    if x=='High':
        return 2
    if x=='Medium':
        return 1
    if x=='Small':
        return 0
    
train_data['Outlet_Size'] = train_data['Outlet_Size'].apply(score_to_numeric)


# In[35]:


train_data.describe()


# In[36]:


train_data.head()


# In[37]:


train_data.dropna(inplace=True)


# In[38]:


train_data.head()


# In[39]:


X=train_data[['Item_MRP', 'Outlet_Size']]


# In[40]:


y=train_data[['Item_Outlet_Sales']]


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=0)


# In[45]:


model=LinearRegression()


# In[47]:


model.fit(X_train, y_train)


# In[49]:


prediction=model.predict(X_test)


# In[50]:


pd.DataFrame(prediction)


# In[51]:


plt.scatter(y_test, prediction)


# In[53]:


model.score(X_train, y_train)


# In[ ]:




