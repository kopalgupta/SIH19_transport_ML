#!/usr/bin/env python
# coding: utf-8
# from jupyter notebook

# applying multiple linear regression, weighted least squares (weighted linear regression) and lambda architecture
# In[73]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.linear_model import Ridge
# from yellowbrick.regressor import ResidualsPlot


# In[74]:


from mpl_toolkits.mplot3d import Axes3D


# In[75]:


from statsmodels.formula.api import ols

from statsmodels.stats.anova import anova_lm


# In[76]:


sample = pd.read_csv('sample.csv')


# In[77]:


sample.head()
sample.info()
sample.describe()
sample.columns


# In[78]:


import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[37]:


sns.pairplot(sample)


# In[79]:

from sklearn import linear_model
clf = linear_model.LinearRegression()


# In[94]:


X = sample[['stop', 'month', 'week',
               'day', 'time']]
y = sample['occupancy']


# In[81]:


clf.fit(X,y)


# In[82]:


import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


# In[83]:


clf.coef_


# In[84]:


test = pd.read_csv('test.csv')


# In[85]:


test.head()
test.info()
test.describe()
test.columns


# In[86]:


test


# In[87]:


X_test = test[['stop', 'month', 'week',
               'day', 'time']]


# In[88]:


predictions = clf.predict(X_test)


# In[89]:


predictions


# In[90]:


import statsmodels.api as sm


# In[91]:

# For outliers on the lower side (not much useful in this case)
outliers = pd.read_csv('sample_outliers.csv')


# In[111]:


x_low = outliers[['stop', 'month', 'week',
               'day', 'time']]
ymod = outliers[['occupancy']]


# In[112]:


X = sm.add_constant(x_low)


# In[113]:


X


# In[127]:


weights = np.random.uniform(35, 55, size=(10,))
# setting low, high and size of weights 


# In[128]:


weights = weights/weights.sum()


# In[133]:


from sklearn.linear_model import LinearRegression
WLS = LinearRegression()
x_time = outliers['time']
WLS.fit(X_low, ymod, sample_weight=x_time)


# In[178]:


print weights


# In[135]:


print(WLS.intercept_)


# In[136]:


print(WLS.coef_)


# In[137]:


predictions_new = WLS.predict(X_test)


# In[138]:


predictions_new


# In[139]:


print X_test


# In[140]:

# For outliers on the higher side (Better results as compared to previous approaches for this case)
outliers_high = pd.read_csv('sample_outliers_high.csv')


# In[149]:


x_high = outliers_high[['stop', 'month', 'week',
               'day', 'time']]
ymod2 = outliers_high[['occupancy']]


# In[150]:


X2 = sm.add_constant(x_high)


# In[151]:


X2


# In[152]:


weights2 = np.random.uniform(88, 120, size=(10,))
# setting low, high and size of weights2 


# In[153]:


weights2 = weights2/weights2.sum()


# In[154]:


from sklearn.linear_model import LinearRegression
WLS = LinearRegression()
x_time = outliers_high['time']
WLS.fit(x_high, ymod2, sample_weight=x_time)


# In[157]:


print X_test


# In[155]:


predictions_high = WLS.predict(X_test)


# In[177]:


print predictions_high


# In[175]:





# In[176]:


# Lambda Arch

# Batch -> w = 0.7 (obtained above)

# Realtime -> w = 0.3
total_tickets = 125 # w = 0.1
get_down_my_stop_or_before = 42 # w = 0.1
crowd_at_my_stop = 8 # w = 0.1
views = 120
net_people = total_tickets - get_down_my_stop_or_before
score = 0.1*net_people + 0.1*crowd_at_my_stop + 0.1*views + 0.7*predictions_high[1][0]
print score


# In[ ]:





# In[ ]:




