"""
@author: svakili

Purpose : Research Seminar - Link prediction
Baseline Test

Notes
------
colnames of NODE_INFOrmation.csv
# (1) SRC source name (string)
# (2) TGT target name (string)
# (3) VOT vote of the SRC for the TGT (integer)
# (4) RES vote of the majority for the TGT (integer)
# (5) YEA year of the vote (integer)
# (6) DAT date of the vote (string)
# (7) TXT text explaining the vote (string)
"""
# Python standard
# import random
# import numpy as np
import csv
# Machine learning libs
from sklearn import svm
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics.pairwise import linear_kernel
# from sklearn import preprocessing
from sklearn import metrics

# User scripts
from utils import load_data

from features import Year


# Limit size of training set (None for no limit)
MAX_DATA_SIZE = None


#%% Load data

# the columns of the data frame below are:
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

# paper id to index in NODE_INFO
ITRAIN, ITEST, YTRAIN, YTEST, NODE_INFO = load_data(max_data_size=MAX_DATA_SIZE)

# #%% Compute features using abstracts 
# ft = TextOnly(NODE_INFO)
# tmp_xtrain = ft.get_features(ITRAIN, 'train')
# tmp_xvalid = ft.get_features(ivalid, 'valid')
# tmp_xtest = ft.get_features(ITEST, 'test')
# print 'Fitting Linear SVM for ABSTRACTS'
# classifier = svm.LinearSVC(C=1)
# classifier.fit(tmp_xtrain, YTRAIN)
# pvalid = classifier.predict(tmp_xvalid)
# accuracy = metrics.accuracy_score(yvalid, pvalid)
# f1 = metrics.f1_score(yvalid, pvalid)
# print 'Accuracy', accuracy
# print 'F1', f1

# #%%
# ftrain = classifier.decision_function(tmp_xtrain)[:, np.newaxis]
# fvalid = classifier.decision_function(tmp_xvalid)[:, np.newaxis]
# ftest = classifier.decision_function(tmp_xtest)[:, np.newaxis]

# Concatenate previous prediction scores to Title/Author/Year features
FT = Year(NODE_INFO)
XTRAIN = FT.get_features(ITRAIN, 'train')
XTEST = FT.get_features(ITEST, 'test')

# Reshaping because 1D
XTRAIN = XTRAIN.reshape(-1, 1)
XTEST = XTEST.reshape(-1, 1)

#%%
# Train linear SVM
#print 'Fitting Linear SVM'
#classifier = svm.LinearSVC(C=1)

print 'Fitting Gaussian SVM'
CLASSIFIER = svm.SVC(kernel='rbf', C=100, gamma=10, cache_size=4000, verbose=True)

#print 'Fitting SGD Linear SVM'
#classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-5, n_iter=25, random_state=42)

CLASSIFIER.fit(XTRAIN, YTRAIN)
# Precict on test set
PTEST = CLASSIFIER.predict(XTEST)
# Evaluation metrics
ACCURACY = metrics.accuracy_score(YTEST, PTEST)
F1 = metrics.f1_score(YTEST, PTEST)
print 'Accuracy', ACCURACY
print 'F1', F1

# #%%
# # Predict on test set
# ptest = classifier.predict(xtest)

with open('../wikipedia_admin/results/baseline_predictions.csv', 'wb') as pred1:
    CSV_OUT = csv.writer(pred1)
    CSV_OUT.writerow(['id', 'category'])  # write header
    for i, row in enumerate(PTEST):
        CSV_OUT.writerow([i, row])
