"""
@author: svakili

Purpose : Research Seminar - Link prediction
Reading function

Notes
------
colnames of edges_df.p
# (1) SRC source name (string)
# (2) TGT target name (string)
# (3) VOT vote of the SRC for the TGT (integer)
# (4) RES vote of the majority for the TGT (integer)
# (5) YEA year of the vote (integer)
# (6) DAT date of the vote (string)
# (7) TXT text explaining the vote (string)
"""

import pickle
import random
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

def load_data(max_data_size=None, balance=True):
    """ Loads the data and splits into train and test set """
    reader = pickle.load(open("../wikipedia_admin/edges_df.p", "rb"))
    
    if balance:
        reader = balance_data(reader)

    all_data_set = np.asarray(reader)
    
    if max_data_size:
        all_data_set = all_data_set[:min(len(all_data_set), max_data_size)]

    itrain, itest, ytrain, ytest = train_test_split(
        all_data_set[:, [0, 1, 3, 4, 5, 6]], all_data_set[:, 2])

    ytrain = np.asarray(ytrain, dtype=int)
    ytest = np.asarray(ytest, dtype=int)
    edges_info = list(reader.itertuples())

    return itrain, itest, ytrain, ytest, edges_info

def balance_data(data):
    """ Balance the data with half +1 and half -1 
        by selecting all -1 from the dataset) 
    """
    if not isinstance(data, pd.core.frame.DataFrame):
        print 'Error: Not a DataFrame'
        return None
    else:
        data_pos = data.ix[data.VOT == 1, :].reset_index(drop=True)
        data_neg = data.ix[data.VOT == -1, :].reset_index(drop=True)

        data_pos.reindex(range(len(data_pos)))
        data_neg.reindex(range(len(data_neg)))

        selected_index = random.sample(data_pos.index, len(data_neg))
        data_balanced = pd.concat([data_neg, 
                                   data_pos.iloc[selected_index, :]], 
                                  ignore_index=True)

        data_balanced = data_balanced.iloc\
        [np.random.permutation(len(data_balanced))].reset_index(drop=True)

        return data_balanced


