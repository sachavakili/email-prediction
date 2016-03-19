
# coding: utf-8

# ### Imports

# In[1]:

get_ipython().magic(u'matplotlib inline')

import os
import sys
import time
import pickle
import matplotlib

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from collections import Counter

matplotlib.style.use('ggplot')


# ### Creating a list of all edges

# In[2]:

with open('wikipedia_admin/rfa_all.NL-SEPARATED.txt', 'rb') as rf:
    readfile = rf.read().split('\n\n')


# In[3]:

nb_edges = len(readfile)
print('%d edges in the file') %(nb_edges)

nb_edges = int(1.0*nb_edges)


# ### Creating an empty dataframe to host the edges attributes

# In[4]:

columns = ['SRC', 'TGT', 'VOT', 'RES', 'YEA', 'DAT', 'TXT']
df = pd.DataFrame(index   = range(nb_edges),
                  columns = columns)


# In[5]:

df.head()


# ### Filling the dataframe with the actual edges attributes

# In[6]:

temp = time.time()

for i in range(nb_edges):
    if i % 10000 == 1:
        print('%d/%d edges treated - %.0f seconds elapsed') %(i, nb_edges, time.time() - temp)
   
    line = readfile[i]
    contents = line.split('\n')

    if len(contents) == len(columns):
        for k in range(len(columns)):
            df.iat[i, k] = contents[k][4:]
    else:
        df = df.ix[df.index != i]        


# In[7]:

df.head()


# ### Dumping / Loading the result in a file

# In[8]:

pickle.dump(df, open("wikipedia_admin/edges_df.p", "wb"))
# df = pickle.load(open("wikipedia_admin/edges_df.p", "rb"))


# ### Creating the graph

# In[9]:

G = nx.from_pandas_dataframe(df, 'SRC', 'TGT', ['VOT', 'RES', 'YEA', 'DAT', 'TXT'])


# ### Degree Analysis

# In[10]:

degree_sequence = Counter(nx.degree(G).values()) # degree sequence
degree_df = pd.DataFrame.from_dict(degree_sequence, orient='index')


# In[11]:

plt.figure(figsize=(14, 10))
plt.title('Log-Log Degree Distribution in the Wikipedia Graph' )
plt.ylabel('Frequency')
plt.xlabel('degree')
# plt.xlim([k_list[t_0], k_list[t_1]+10])
plt.grid('on')
plt.loglog(degree_df, 'bo')
# plt.xticks(np.arange(k_list[t_0], k_list[t_1]+10, 5), rotation=90)
# plt.ticklabel_format(useOffset=False)

# plt.axvline(k_list[t_1-tau])
# plt.axvline(k_list[t_1-1])


# In[ ]:



        