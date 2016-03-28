
# coding: utf-8

# # Initialization of libraries

# In[163]:

# %matplotlib inline
import os
import sys
# import matplotlib

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import networkx as nx
from collections import defaultdict, Counter
import time 
from datetime import datetime

from sktensor import dtensor, cp_als

# matplotlib.style.use('ggplot')


# # Manufacturing company emails

# ### Importing the dataset

# In[164]:

# Looking for the dataset
if not os.path.isdir('radoslaw_email'):
    print 'The dataset has not be found'
else:
    # Loading the dataset
    tab = pd.read_csv('radoslaw_email/out.radoslaw_email_email', sep=' ', skiprows=2, header=None, index_col=False)


# In[165]:

# Display
# tab.head()


# In[166]:

# Keeping desired columns and naming them
tab.drop(2, axis=1, inplace=True)
tab.columns = ['sender', 'receiver', 'weight', 'datetime']

# Date Time format
tab.datetime = pd.to_datetime(tab.datetime,unit='s')


# In[167]:

# Display
tab.head()


# In[168]:

len(tab)


# In[169]:

mail_list = pd.unique(tab.iloc[:,3])

# Get the indexes of each author
mail_indexes = [(element, index) for index, element in enumerate(tab.datetime)]
mail_index = defaultdict(list)

for k, v in mail_indexes:
    mail_index[k].append(v)


# In[170]:

mail_index_ML = []

for date in mail_index.keys() :
    if len(mail_index[date]) >= 10 :
        mail_index_ML += mail_index[date]


# In[171]:

len(mail_index_ML)


# In[172]:

for i in range(len(mail_index_ML)) :
    tab = tab[tab.index != mail_index_ML[i]]


# In[173]:

len(tab)


# In[174]:

tab.reset_index(inplace=True)


# In[175]:

tab.head()


# In[176]:

tab.drop(['index','weight'], axis=1, inplace=True)


# In[177]:

tab.head()


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

# In[7]:

# sent_mails = tab.receiver.groupby(tab.sender).nunique()

# sent_mails.describe()


# In[8]:

# sent_mails.argmax()
# max(sent_mails) == len(pd.unique(tab.ix[tab.sender == 38, 1]))


# ### Top senders

# In[9]:

# top_senders = tab.sender.groupby(tab.sender).count()
# top_senders.sort(ascending=False)
# top_senders


# ### Top receivers

# In[10]:

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

# In[178]:

dates = tab.datetime.dt.date[:]
tab.datetime = dates.apply(lambda dates: dates.toordinal())


# ## Creating a tensor

# ### Listing all the unique values of the dataset

# In[180]:

X = tab.iloc[:, [0, 1, 2]]

i_list = pd.unique(X.iloc[:, 0])
j_list = pd.unique(X.iloc[:, 1])
k_list = pd.unique(X.iloc[:, 2])

i_list.sort()
j_list.sort()
k_list.sort()


# In[181]:

i_ = list(tab.sender)
j_ = list(tab.receiver)
i_.sort()
j_.sort()


# In[182]:

users = pd.unique(i_ + j_)
users.sort()
users


# In[183]:

X.head()


# ### Creating and filling the tensor

# In[184]:

T = np.zeros((len(users), len(users), len(k_list)))
T = dtensor(T)


# In[185]:

# Going through the dataframe
t_temp = time.time()

for i, j, k in zip(X.sender, X.receiver, X.datetime):
    # Finding the corresponding index in the tensor
    i_ind = [i_ind for i_ind, x in enumerate(i_list) if x == i][0]
    j_ind = [j_ind for j_ind, x in enumerate(j_list) if x == j][0]
    k_ind = [k_ind for k_ind, x in enumerate(k_list) if x == k][0]
    
    # Modifying the tensor value for the tuple (i_ind, j_ind, k_ind)
    T[i_ind, j_ind, k_ind] += 1

print time.time()-t_temp    


# In[186]:

# Logarithmic Transformation
nonz = T.nonzero()
for ind in range(len(nonz[0])):
    i_ind = nonz[0][ind]
    j_ind = nonz[1][ind] 
    k_ind = nonz[2][ind]
    
    T[i_ind, j_ind, k_ind] = 1 + np.log(T[i_ind, j_ind, k_ind]) 


# In[187]:

T.shape


# In[ ]:



