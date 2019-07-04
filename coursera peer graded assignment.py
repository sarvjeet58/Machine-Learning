#!/usr/bin/env python
# coding: utf-8

# In[100]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[101]:


df = pd.read_csv("I:\sarv\kc_house_data.csv")
df.head()


# In[102]:


df.dtypes


# In[103]:


df.drop(["id"],axis=1,inplace=True)


# In[104]:


df.describe()


# In[158]:


df["floors"].value_counts()


# In[155]:


df.to_frame()


# In[106]:


sns.boxplot(df['waterfront'],df['price'])
plt.show()


# In[107]:


sns.regplot(df['sqft_above'],df['price'])
plt.show()


# In[108]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
X=df[['sqft_living']]
Y=df[['price']]
LR.fit(X,Y)


# In[109]:


Yhat=LR.predict(Y)


# In[110]:


LR.score(X,Y)


# In[111]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
X=df[["floors","waterfront","lat","bedrooms","sqft_basement","view","bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
Y=df[['price']]
LR.fit(X,Y)


# In[112]:


LR.score(X,Y)


# In[117]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),('mode',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(X,Y)


# In[119]:


pipe.score(X,Y)


# In[121]:


from sklearn.linear_model import Ridge
RigeModel = Ridge(alpha=0.1)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

RigeModel.fit(X_train,Y_train)


# In[132]:


RigeModel.score(X_test,Y_test)


# In[153]:


from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
X_train_poly = pr.fit_transform(X_train)
X_test_poly = pr.fit_transform(X_test)
from sklearn.linear_model import Ridge
poly_model = Ridge(alpha=0.1)
poly_model.fit(X_train_poly, Y_train)


# In[154]:



poly_model.score(X_test_poly,Y_test)


# In[ ]:




