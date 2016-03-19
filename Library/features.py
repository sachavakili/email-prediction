"""
@author: svakili

Purpose : Research Seminar - Link prediction
Model definition

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
from abc import abstractmethod
import numpy as np
import nltk
import scipy.sparse
# from process_authors import process_authors, split_authors
# from sklearn.feature_extraction.text import TfidfVectorizer


# Load STEMMER
STEMMER = nltk.PorterStemmer()

# Load STOPWORDS
# nltk.download('stopwords')  # uncomment if running for first time
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

class BaseFeatureExtractor(object):
    """ Elementary class that requires 
        defining get_feature"""
    def __init__(self, node_info):
        self.ids = {'//'.join(element[1:3]): element[0] for element in node_info}
        self.node_info = node_info
    
    def get_features(self, data, msg='dataset?'):
        """ Aggregation of the features"""
        features = []
        for i, row in enumerate(data):
            edge_id = '//'.join(row[0:2])
            edge_idx = self.ids[edge_id]
            edge_info = self.node_info[edge_idx]
            feature = self.get_feature(edge_idx, edge_info)        
            features.append(feature)
            if i % 1000:
                print '[{}] ({:.0f}%) processed (#{})'.format(msg, 100.*i/len(data), i)
        if isinstance(features[0], scipy.sparse.coo.coo_matrix):
            return scipy.sparse.vstack(features)
        else:
            return np.asarray(features)
        
    @abstractmethod
    def get_feature(self, edge_idx, edge_info):
        """ To be defined for every model"""
    # def get_feature(source_idx, source_info, target_idx, target_info):
        pass


class Year(BaseFeatureExtractor):
    """ Class using only 3 features"""
    def clean_info(self, info):
        """Treat year"""
        yea = info[5]
        # return year
        year_ = yea
        return year_
        
    def get_feature(self, edge_idx, edge_info):
        """I don't know"""
        edge_year = self.clean_info(edge_info)
        n_years = edge_year
        return np.asarray((n_years))
    

# class TextOnly(BaseFeatureExtractor):
#     def __init__(self, node_info):
#         BaseFeatureExtractor.__init__(self, node_info)
#         # initialize TFIDF vectorizer
#         self.abstracts = [node[5] for node in node_info]        
#         self.tv = TfidfVectorizer(stop_words='english')
#         print 'Fitting tf.idf vectorizer'
#         self.transformed_abstracts = self.tv.fit_transform(self.abstracts)
    
#     def get_feature(self, source_idx, source_info, target_idx, target_info):
#         sft = self.transformed_abstracts[source_idx]
#         tft = self.transformed_abstracts[target_idx]
#         feature = scipy.sparse.hstack([sft, tft])
#         return feature


# from utils import load_data

# ITRAIN, ITEST, YTRAIN, YTEST, NODE_INFO = load_data(max_data_size=100)
# TRYME = BaseFeatureExtractor(NODE_INFO)
# print NODE_INFO[2]
# print TRYME.ids['INeverCry/BDD']
# # for k in range(3):
# #     idx, element = enumerate(NODE_INFO)
# #     print element[k]
# # print TRYME.ids[]
