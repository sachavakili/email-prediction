"""
@author: svakili

Purpose : Research Seminar - Link prediction
Reading function

Notes
------
colnames of edges_df.p
# (1) SRC source id (integer)
# (2) TGT target id (integer)
"""
import csv
import pickle
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

def load_data(max_data_size=None, balance=True):
    """ Loads the data and splits into train and test set """
    if balance:
        with open('../enron_emails/Email-EuAll.txt', 'r') as filename:
            reader = csv.reader(filename)
            # Skip the header
            for _ in range(4):
                next(reader, None)  
            # whole Enron datasett
            all_data_set = np.asarray([element[0].split('\t') \
                                           for element in reader], dtype=int)
            
            if max_data_size:
                all_data_set = all_data_set[:min(len(all_data_set), \
                                                         max_data_size)]

            all_data_set = balance_data(all_data_set)
            edges_info = list(pd.DataFrame(all_data_set).itertuples())
    else:
        all_data_set = pickle.load(open("../enron_emails/enron_np.p", "rb"))
        edges_info = list(pd.DataFrame(all_data_set).itertuples())


    itrain, itest, ytrain, ytest = train_test_split(
        all_data_set[:, [0, 1]], all_data_set[:, 2])

    ytrain = np.asarray(ytrain, dtype=int)
    ytest = np.asarray(ytest, dtype=int)

    return itrain, itest, ytrain, ytest, edges_info

def balance_data(data):
    """ Balance the data with half 1 and half 0 
        by selecting all 1 from the dataset) 
    """
    if not isinstance(data, np.ndarray):
        print 'Error: Not a NdArray'
        return None
    else:
        data = pd.DataFrame(data, columns = ['SRC', 'TGT'])
        nrow = len(data)
        ids = np.unique(data)
        # ratio = 0.9

        data_pos = pd.concat([data, pd.DataFrame(1, index = range(nrow), columns = ['VOT'])], 1)
        data_neg = pd.DataFrame(0, index = range(int(ratio*nrow)), columns = ['SRC', 'TGT', 'VOT'])

        selected_sender = np.random.choice(ids, int(ratio*nrow))
        selected_receiver = np.random.choice(ids, int(ratio*nrow))
        data_neg.iloc[:, 0] = selected_sender
        data_neg.iloc[:, 1] = selected_receiver

        data_balanced = pd.concat([data_pos, data_neg], 0)
        data_balanced = data_balanced.drop_duplicates(['SRC', 'TGT'], take_last=False)

        data_balanced = np.asarray(data_balanced, dtype=int)
        np.random.shuffle(data_balanced)

        return data_balanced

