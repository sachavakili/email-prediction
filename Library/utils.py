"""
@author: svakili

Purpose : Research Seminar - Link prediction
Reading function

Notes
------
colnames of node_information.csv
# (1) SRC source name (string)
# (2) TGT target name (string)
# (3) VOT vote of the SRC for the TGT (integer)
# (4) RES vote of the majority for the TGT (integer)
# (5) YEA year of the vote (integer)
# (6) DAT date of the vote (string)
# (7) TXT text explaining the vote (string)
"""

import pickle
import numpy as np
from sklearn.cross_validation import train_test_split


def load_data(max_data_size=None):
    """ Loads the data and splits into train and test set """
    reader = pickle.load(open("../wikipedia_admin/edges_df.p", "rb"))
 
    all_data_set = np.asarray(reader)
    if max_data_size:
        all_data_set = all_data_set[:min(len(all_data_set), max_data_size)]

    itrain, itest, ytrain, ytest = train_test_split(
        all_data_set[:, [0, 1, 3, 4, 5, 6]], all_data_set[:, 2])

    node_info = list(reader.itertuples())

    return itrain, itest, ytrain, ytest, node_info

