
# coding: utf-8

# In[1]:

# %matplotlib inline
import os
import sys
# import matplotlib
import time

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
#import networkx as nx
import pickle


#from datetime import datetime

from sktensor import dtensor, cp_als
# matplotlib.style.use('ggplot')


# ### Creating a list of all edges

# In[2]:

#with open('dblp_inpr.xml', 'rb') as rf:
with open('dblp.xml', 'rb') as rf:
    readfile = rf.read().split('<inproceedings')


# In[3]:

nb_edges = len(readfile)
print('%d edges in the file') %(nb_edges)

# Only take 1000 edges (publications)
# nb_edges = 10001


# ### Creating an empty dataframe to host the edges attributes

# In[4]:

#columns = ['author', 'title', 'year', 'crossref']
#df = pd.DataFrame(index   = 4*range(nb_edges),
#                  columns = columns)

#df2 = pd.DataFrame(index   = 5*range(nb_edges),
#                   columns = columns)


# In[74]:

#df2.head()


# ### Filling the dataframe with the actual edges attributes

# In[84]:

#dftest.append(dftest.iloc[0, :], ignore_index=True)


# In[83]:

temp = time.time()
counter = 0
nb_authors = 0

for i in range(1,nb_edges):
    if i % 1000 == 1:
        print('%d/%d edges treated - %.0f seconds elapsed') %(i, nb_edges, time.time() - temp)
   
    contents = readfile[i].splitlines()
    author_list = []
    title = ''
    year = ''
    cross = ''
    
    if contents != '':
        for content in contents:
            if content[:8] == '<author>':
                author_list += [content[8:-9]]
            elif content[:7] == '<title>':
                title = content[7:-8]
            elif content[:6] == '<year>':
                year = content[6:-7]
            elif content[:10] == '<crossref>':
                cross = content[10:-11]
            if (i == nb_edges -1)&(content[:16] == '</inproceedings>') :
                break
        
        # Consider only the publications of type inproceedings between 1991 and 2007 :
        if ((int(year[:4]) >= 1991)and(int(year[:4]) <= 2007)) : 
            for k in range(len(author_list)):
                df.iat[counter, 0] = author_list[k]
                df.iat[counter, 1] = title
                df.iat[counter, 2] = year
                df.iat[counter, 3] = cross
                counter += 1
            nb_authors += len(author_list)
            
    else:
        df = df.ix[df.index != i]        


# In[85]:

nb_authors


# In[8]:

#df2.iloc[nb_authors-10: nb_authors +5,:]


# In[165]:

#df = df[df.index != len(df) - 1]


# In[89]:

df.reset_index(inplace=True)


# In[91]:

df = df[df.index < nb_authors]


# In[94]:

df.iloc[-100:]


# In[93]:

df.drop(['index'], axis=1, inplace=True)


# ### Dumping the result in a file

# In[95]:

pickle.dump(df, open("dblp_inproceeding.p", "wb"))


# ### Open inproceeding file

# In[2]:

df = pickle.load( open( "dblp_inproceeding.p", "rb" ) )


# In[14]:

# Keeping desired columns and naming them
#df.columns = ['nan','index_', 'author', 'title', 'year', 'crossref']
#df.drop(['index'], axis=1, inplace=True)

# Date Time format
#tab.datetime = pd.to_datetime(tab.datetime,unit='s')


# ### Loading the dataset

# In[2]:

df = pickle.load( open( "dblp_inproceeding.p", "rb" ) )


# ### Creating a tensor

# In[3]:

author_list = pd.unique(df.iloc[:, 0])
conf_list = pd.unique(df.iloc[:, 3])
year_list = pd.unique(df.iloc[:, 2])

year_list.sort()

author_dic = {element : index for index, element in enumerate(author_list)}
conf_dic =   {element : index for index, element in enumerate(conf_list)}
year_dic =   {element : index for index, element in enumerate(year_list)}

T = np.zeros((len(author_list), len(conf_list), len(year_list)))
T = dtensor(T)


# In[4]:

# Going through the dataframe
t_temp = time.time()
i = 0
limit = 1000000
L = len(zip(df.author, df.crossref, df.year))

for a, c, y in zip(df.author, df.crossref, df.year)[:limit]:
    if i % 100000 == 1:
        print('%d/%d (df.author, df.crossref, df.year) treated - %.0f seconds elapsed') %(i, L, time.time() - t_temp)
        
    # Finding the corresponding index in the tensor
    a_ind = author_dic[a]
    c_ind = conf_dic[c]
    y_ind = year_dic[y]
    
    # Modifying the tensor value for the tuple (i_ind, j_ind, k_ind)
    T[a_ind, c_ind, y_ind] += 1
    
    i +=1

print time.time()-t_temp    


# In[ ]:

# Logarithmic Transformation
nonz = T.nonzero()
for ind in range(len(nonz[0])):
    i_ind = nonz[0][ind]
    j_ind = nonz[1][ind] 
    k_ind = nonz[2][ind]
    
    T[i_ind, j_ind, k_ind] = 1 + np.log(T[i_ind, j_ind, k_ind]) 


# In[ ]:

#pickle.dump(T, open("tensor_inproceeding.p", "wb"))


# In[ ]:



