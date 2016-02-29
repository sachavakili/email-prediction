
# coding: utf-8

# # Initialization of libraries

# In[2]:

# %matplotlib inline
import os
import sys
# import matplotlib

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import networkx as nx

from datetime import datetime

from sktensor import dtensor, cp_als

# matplotlib.style.use('ggplot')


# # Manufacturing company emails

# ### Importing the dataset

# In[3]:

# Looking for the dataset
if not os.path.isdir('radoslaw_email'):
    print 'The dataset has not be found'
else:
    # Loading the dataset
    tab = pd.read_csv('radoslaw_email/out.radoslaw_email_email', sep=' ', skiprows=2, header=None, index_col=False)


# In[4]:

# Display
# tab.head()


# In[5]:

# Keeping desired columns and naming them
tab.drop(2, axis=1, inplace=True)
tab.columns = ['sender', 'receiver', 'weight', 'datetime']

# Date Time format
tab.datetime = pd.to_datetime(tab.datetime,unit='s')


# In[6]:

# Display
# tab.head()


# ### Quick Data Analysis

# In[7]:

# print '%d mails exchanged between %d employees from %s to %s' %(tab.shape[0],
#                                                                 len(pd.unique(pd.concat([tab.sender,tab.receiver]))),
#                                                                 str(min(tab.datetime.dt.date)),
#                                                                 str(max(tab.datetime.dt.date))) 


# In[8]:

# tab.datetime.groupby(tab.datetime.dt.week).count().plot(
#     title='Temporal Distribution per week',
#     kind='bar',
#     figsize=(16, 10),
#     color='#348ABD',
#     alpha=0.6,
#     lw='1',
#     edgecolor='#348ABD',
#     grid=True)


# ### Number of active days / sender

# In[9]:

# sent_mails = tab.datetime.dt.date.groupby(tab.sender).nunique()

# sent_mails.describe()


# ### Number of receivers / sender

# In[9]:

# sent_mails = tab.receiver.groupby(tab.sender).nunique()

# sent_mails.describe()


# In[10]:

# sent_mails.argmax()
# max(sent_mails) == len(pd.unique(tab.ix[tab.sender == 38, 1]))


# ### Top senders

# In[27]:

# top_senders = tab.sender.groupby(tab.sender).count()
# top_senders.sort(ascending=False)
# top_senders


# ### Top receivers

# In[28]:

# top_receivers = tab.receiver.groupby(tab.receiver).count()
# top_receivers.sort(ascending=False)
# top_receivers


# ### Creating a graph (Dipou's Birthday example)

# In[11]:

# dipou_brthd = datetime.strptime('2010-01-19', '%Y-%m-%d')


# In[12]:

# row_list = tab.datetime.dt.date == dipou_brthd.date()
# edge_list = zip(tab.sender[row_list], tab.receiver[row_list])

# G=nx.Graph()
# G.add_edges_from(edge_list)


# In[13]:

# print '%d mails exchanged between %d employees the %s' %(nx.number_of_edges(G), 
#                                                          nx.number_of_nodes(G), 
#                                                          str(dipou_brthd.date()))


# In[14]:

# nx.draw(G)


# ### Converting datetimes into integers

# In[15]:

dates = tab.datetime.dt.date[:]
tab.datetime = dates.apply(lambda dates: dates.toordinal())


# ## Creating a tensor

# ### Listing all the unique values of the dataset

# In[ ]:

X = tab.iloc[:, [0, 1, 3]]

i_list = pd.unique(X.iloc[:, 0])
j_list = pd.unique(X.iloc[:, 1])
k_list = pd.unique(X.iloc[:, 2])

i_list.sort()
j_list.sort()
k_list.sort()


# ### Creating and filling the tensor

# In[ ]:

T = np.zeros((len(i_list), len(j_list), len(k_list)))
T = dtensor(T)


# In[ ]:

# Going through the dataframe
for i, j, k in zip(X.sender, X.receiver, X.datetime):
    # Finding the corresponding index in the tensor
    i_ind = int(np.where(i_list == i)[0])
    j_ind = int(np.where(j_list == j)[0])
    k_ind = int(np.where(k_list == k)[0])
    
    # Modifying the tensor value for the tuple (i_ind, j_ind, k_ind)
    T[i_ind, j_ind, k_ind] += 1

# Logarithmic Transformation
nonz = T.nonzero()
for ind in range(len(nonz[0])):
    i_ind = nonz[0][ind]
    j_ind = nonz[1][ind] 
    k_ind = nonz[2][ind]
    
    T[i_ind, j_ind, k_ind] = 1 + np.log(T[i_ind, j_ind, k_ind]) 

# ### Useful Code for finding dataframe lines corresponding to given values

# In[ ]:

i = 1  # sender "1"
j = 10  # receiver "10"
# k = 

X.ix[(X.sender == i)&(X.receiver == j)].head()  # using "head()" for only showing the 5 first results
# X.ix[(X.sender == i)&(X.receiver == j)&(X.datetime == k)]

