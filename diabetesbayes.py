#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np


# In[37]:


df = pd.read_csv("C:\\Users\\Admin\\Desktop\\NEHACDACAssigment\\ML\\diabetes.csv")
df.head()


# In[38]:


df.info()


# In[39]:


df.isnull().sum()


# In[40]:


y=df['Outcome']
x=df.drop('Outcome',axis='columns')


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


# In[42]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
print(X_train,y_train)


# In[43]:


model.fit(X_train,y_train)


# In[44]:


model.score(X_test,y_test)


# In[45]:


X_test[0:10]


# In[46]:


y_test[0:10]


# In[47]:


model.predict(X_test[0:10])


# In[48]:


model.predict_proba(X_test[:10])


# In[49]:


from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(),X_train, y_train, cv=5)


# In[50]:


a=cross_val_score(GaussianNB(),X_train, y_train, cv=5)

print(np.mean(a))


# In[ ]:




