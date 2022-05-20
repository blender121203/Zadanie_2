#!/usr/bin/env python
# coding: utf-8

# # Статистика курса валют

# ### Парсинг данных

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings ('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import max_error


# In[13]:


data=pd.read_excel("Valute.xlsx")


# In[14]:


data


# In[15]:


data.shape


# ### Предобработка данных

# In[17]:


data['data']=pd.to_datetime(data['data'], errors='ignore')


# In[22]:


data['year'] = data['data'].dt.year
data['month'] = data['data'].dt.month
data['day'] = data['data'].dt.day


# In[23]:


data


# ### Аналитика

# In[29]:


plt.scatter(data.month, data.curs , color ='wheat')
plt.xlabel("month")
plt.ylabel("curs")
plt.grid()
plt.show()


# In[31]:


plt.scatter(data.year, data.curs , color ='wheat')
plt.xlabel("year")
plt.ylabel("curs")
plt.grid()
plt.show()


# In[33]:


plt.scatter(data.day, data.curs , color ='wheat')
plt.xlabel("day")
plt.ylabel("curs")
plt.grid()
plt.show()


# ### Линейная регрессия

# *Регрессионный анализ:*

# In[35]:


lbl=LabelEncoder()
nomeric=data.select_dtypes(exclude=[np.number])
cols=nomeric.columns.values
for col in cols:
    data[col]=lbl.fit_transform(data[col].astype(str))


# In[36]:


scal=StandardScaler()
df_scaler=scal.fit_transform(data.drop("curs",axis=1))
pd.DataFrame(df_scaler, columns=data.drop("curs",axis=1).columns)


# In[37]:


msk=np.random.rand(len(data)) <0.8
train=data[msk]
test=data[~msk]


# In[39]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x=np.asanyarray(train[['month']])
train_y=np.asanyarray(train[['curs']])
regr.fit(train_x, train_y)
print('Coefficients:',regr.coef_)
print('Intercept:',regr.intercept_)


# In[40]:


plt.scatter(train.month, train.curs , color = 'blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("month")
plt.ylabel("curs")
plt.grid()


# *Зависимость курса от года*

# In[41]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x=np.asanyarray(train[['year']])
train_y=np.asanyarray(train[['curs']])
regr.fit(train_x, train_y)
print('Coefficients:',regr.coef_)
print('Intercept:',regr.intercept_)


# In[43]:


plt.scatter(train.year, train.curs , color = 'blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("year")
plt.ylabel("curs")
plt.grid()


# *Зависимость курса валют от дня*

# In[44]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x=np.asanyarray(train[['day']])
train_y=np.asanyarray(train[['curs']])
regr.fit(train_x, train_y)
print('Coefficients:',regr.coef_)
print('Intercept:',regr.intercept_)


# In[45]:


plt.scatter(train.day, train.curs , color = 'blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("day")
plt.ylabel("curs")
plt.grid()


# *Зависимость курса валют от общей даты*

# In[46]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x=np.asanyarray(train[['data']])
train_y=np.asanyarray(train[['curs']])
regr.fit(train_x, train_y)
print('Coefficients:',regr.coef_)
print('Intercept:',regr.intercept_)


# In[47]:


plt.scatter(train.data, train.curs , color = 'blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("data")
plt.ylabel("curs")
plt.grid()


# *Лучше всего от курса валют зависит день, так как там больше всего точек пересеклись с линейной регрессией.*

# ### Статистика курса валют

# In[50]:


data.describe()


# *По статистике можно сказать, что выбросов в данных нету, так как медианное и среднее значение значительно не отличаются, дисперсия в данных плохая только в признаке data, но это нормально, так как остальные атрибуты имеют хорошее стандартное отклонение.*
