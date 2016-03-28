
# coding: utf-8

# # Initialization of libraries

# In[1]:

# %matplotlib inline
import os
import sys
# import matplotlib

import numpy as np
import pandas as pd
import time
# import matplotlib.pyplot as plt
# import networkx as nx

from datetime import datetime
from collections import defaultdict
from sktensor import dtensor, cp_als

# matplotlib.style.use('ggplot')


# # Manufacturing company emails

# ### Importing the dataset

# In[2]:

# Looking for the dataset
if not os.path.isdir('radoslaw_email'):
    print 'The dataset has not be found'
else:
    # Loading the dataset
    tab = pd.read_csv('radoslaw_email/out.radoslaw_email_email', sep=' ', skiprows=2, header=None, index_col=False)


# In[3]:

# Display
tab.head()


# In[4]:

# Keeping desired columns and naming them
tab.drop(2, axis=1, inplace=True)
tab.columns = ['sender', 'receiver', 'weight', 'datetime']

# Date Time format
tab.datetime = pd.to_datetime(tab.datetime,unit='s')


# In[5]:

# Display
tab.drop('weight', axis=1, inplace=True)
tab.head()


# In[8]:

mail_list = pd.unique(tab.iloc[:,2])

# Get the indexes of each author
mail_indexes = [(element, index) for index, element in enumerate(tab.datetime)]
mail_index = defaultdict(list)

for k, v in mail_indexes:
    mail_index[k].append(v)
    
mail_index_ML = []

for date in mail_index.keys() :
    if len(mail_index[date]) >= 10 :
        mail_index_ML += mail_index[date]

for i in range(len(mail_index_ML)) :
    tab = tab[tab.index != mail_index_ML[i]]
    
tab.reset_index(inplace=True)

tab.drop(['index'], axis=1, inplace=True)


# In[10]:

tab['month'] = tab.datetime.dt.month


# In[11]:

tab.head()


# ### Quick Data Analysis

# In[10]:

print '%d mails exchanged between %d employees from %s to %s' %(tab.shape[0],
                                                                len(pd.unique(pd.concat([tab.sender,tab.receiver]))),
                                                                str(min(tab.datetime)),
                                                                str(max(tab.datetime))) 


# In[11]:

# tab.datetime.groupby(tab.datetime.dt.week).count().plot(
#     title='Temporal Distribution per week',
#     kind='bar',
#     figsize=(16, 10),
#     color='#348ABD',
#     alpha=0.6,
#     lw='1',
#     edgecolor='#348ABD',
#     grid=True)


# ## Creating a tensor

# ### Listing all the unique values of the dataset

# In[12]:

X = tab.iloc[:, [0, 1, 3]]

i_list = pd.unique(X.iloc[:, 0])
j_list = pd.unique(X.iloc[:, 1])
k_list = pd.unique(X.iloc[:, 2])

i_list.sort()
j_list.sort()
k_list.sort()


# In[13]:

i_ = list(tab.sender)
j_ = list(tab.receiver)
i_.sort()
j_.sort()


# In[14]:

users = pd.unique(i_ + j_)
users.sort()
users


# In[15]:

len(users)


# ### Creating and filling the tensor

# In[16]:

T = np.zeros((len(users), len(users), len(k_list)))
T = dtensor(T)


# In[17]:

# Going through the dataframe
t_temp = time.time()

for i, j, k in zip(X.sender, X.receiver, X.month):
    # Finding the corresponding index in the tensor
    i_ind = [i_ind for i_ind, x in enumerate(i_list) if x == i][0]
    j_ind = [j_ind for j_ind, x in enumerate(j_list) if x == j][0]
    k_ind = [k_ind for k_ind, x in enumerate(k_list) if x == k][0]
    
    # Modifying the tensor value for the tuple (i_ind, j_ind, k_ind)
    T[i_ind, j_ind, k_ind] += 1

print time.time()-t_temp    


# In[18]:

# Logarithmic Transformation
nonz = T.nonzero()
for ind in range(len(nonz[0])):
    i_ind = nonz[0][ind]
    j_ind = nonz[1][ind] 
    k_ind = nonz[2][ind]
    
    T[i_ind, j_ind, k_ind] = 1 + np.log(T[i_ind, j_ind, k_ind]) 


# In[19]:

T.shape


# In[ ]:



