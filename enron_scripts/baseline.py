"""
@author: svakili

Purpose : Research Seminar - Link prediction
Degree Features

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
# Python standard
import csv
import numpy as np
# Machine learning libs
from sklearn import linear_model
from sklearn import metrics

# User scripts
from load import load_data
from features import GraphDegreeFeatures


# Limit size of training set (None for no limit)
MAX_DATA_SIZE = 10000
# Output the prediction (False for no printing)
PRINT_OUTPUT = True

#%% Load data

# paper id to index in EDGE_INFO
ITRAIN, ITEST, YTRAIN, YTEST, EDGES_INFO = load_data(max_data_size=MAX_DATA_SIZE)

# Precict on test set
PTEST = np.zeros(len(YTEST))

# Evaluation metrics
ACCURACY = metrics.accuracy_score(YTEST, PTEST)
F1 = metrics.f1_score(YTEST, PTEST)
print 'Accuracy', ACCURACY
print 'F1', F1

# Output
if PRINT_OUTPUT:
    with open('../enron_emails/degree_predictions.csv', 
              'wb') as pred1:
        CSV_OUT = csv.writer(pred1)
        CSV_OUT.writerow(['id', 'category'])  # write header
        for i, row in enumerate(PTEST):
            CSV_OUT.writerow([i, row])
    