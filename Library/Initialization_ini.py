
# coding: utf-8

# # Initialization of libraries

# In[3]:

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

# In[8]:

# Looking for the dataset
if not os.path.isdir('radoslaw_email'):
    print 'The dataset has not be found'
else:
    # Loading the dataset
    tab = pd.read_csv('radoslaw_email/out.radoslaw_email_email', sep=' ', skiprows=2, header=None, index_col=False)


# In[9]:

# Display
tab.head()


# In[10]:

# Keeping desired columns and naming them
tab.drop(2, axis=1, inplace=True)
tab.columns = ['sender', 'receiver', 'weight', 'datetime']

# Date Time format
tab.datetime = pd.to_datetime(tab.datetime,unit='s')


# In[11]:

# Display
tab[0:30]


# ### Quick Data Analysis

# In[12]:

print '%d mails exchanged between %d employees from %s to %s' %(tab.shape[0],
                                                                len(pd.unique(pd.concat([tab.sender,tab.receiver]))),
                                                                str(min(tab.datetime)),
                                                                str(max(tab.datetime))) 


# In[13]:

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

# In[14]:

# sent_mails = tab.datetime.dt.date.groupby(tab.sender).nunique()

# sent_mails.describe()


# ### Number of receivers / sender

# In[15]:

# sent_mails = tab.receiver.groupby(tab.sender).nunique()

# sent_mails.describe()


# In[16]:

# sent_mails.argmax()
# max(sent_mails) == len(pd.unique(tab.ix[tab.sender == 9, 1]))


# ### Top senders

# In[36]:

# top_senders = tab.sender.groupby(tab.sender).count()
# top_senders.sort(ascending=False)
# top_senders


# ### Top receivers

# In[37]:

# top_receivers = tab.receiver.groupby(tab.receiver).count()
# top_receivers.sort(ascending=False)
# top_receivers


# ### Creating a graph (Dipou's Birthday example)

# In[16]:

# dipou_brthd = datetime.strptime('2010-01-19', '%Y-%m-%d')


# In[17]:

# row_list = tab.datetime.dt.date == dipou_brthd.date()
# edge_list = zip(tab.sender[row_list], tab.receiver[row_list])

# G=nx.Graph()
# G.add_edges_from(edge_list)


# In[18]:

# print '%d mails exchanged between %d employees the %s' %(nx.number_of_edges(G), 
#                                                          nx.number_of_nodes(G), 
#                                                          str(dipou_brthd.date()))


# In[19]:

# nx.draw(G)


# ### Converting datetimes into integers

# In[17]:

dates = tab.datetime.dt.date[:]
tab.datetime = dates.apply(lambda dates: dates.toordinal())


# ## Creating a tensor

# ### Listing all the unique values of the dataset

# In[18]:

X = tab.iloc[:, [0, 1, 3]]

i_list = pd.unique(X.iloc[:, 0])
j_list = pd.unique(X.iloc[:, 1])
k_list = pd.unique(X.iloc[:, 2])

i_list.sort()
j_list.sort()
k_list.sort()


# ### Creating and filling the tensor

# In[19]:

T = np.zeros((len(i_list), len(j_list), len(k_list)))
T = dtensor(T)


# In[20]:

# Going through the dataframe
for i, j, k in zip(X.sender, X.receiver, X.datetime):
    # Finding the corresponding index in the tensor
    i_ind = [i_ind for i_ind, x in enumerate(i_list) if x == i][0]
    j_ind = [j_ind for j_ind, x in enumerate(j_list) if x == j][0]
    k_ind = [k_ind for k_ind, x in enumerate(k_list) if x == k][0]
    
    # Modifying the tensor value for the tuple (i_ind, j_ind, k_ind)
    T[i_ind, j_ind, k_ind] += 1


# ### Useful Code for finding dataframe lines corresponding to given values

# In[26]:

i = 1  # sender "1"
j = 10  # receiver "10"
# k = 

X.ix[(X.sender == i)&(X.receiver == j)].head()  # using "head()" for only showing the 5 first results
# X.ix[(X.sender == i)&(X.receiver == j)&(X.datetime == k)]


# In[ ]:



