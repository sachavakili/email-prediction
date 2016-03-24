"""
@author: svakili

Purpose : Research Seminar - Link prediction
Model definition

Notes
------
colnames of edges_edge_df.p
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
import networkx as nx
import pandas as pd
# from process_authors import process_authors, split_authors
# from sklearn.feature_extraction.text import Tfiedge_dfVectorizer

# Load STEMMER
STEMMER = nltk.PorterStemmer()

# Load STOPWORDS
# nltk.download('stopwords')  # uncomment if running for first time
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

class BaseFeatureExtractor(object):
    """ Elementary class that requires 
        defining get_feature """
    def __init__(self, edges_info):
        self.ids = {'//'.join(element[1:3]): element[0] for element in edges_info}
        self.edges_info = edges_info
    
    def get_features(self, data, msg='dataset?'):
        """ Aggregation of the features"""
        features = []
        for i, row in enumerate(data):
            edge_id = '//'.join(row[0:2])
            edge_idx = self.ids[edge_id]
            edge_info = self.edges_info[edge_idx]
            feature = self.get_feature(edge_idx, edge_info)        
            features.append(feature)
            if i % 1000 is True:
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
    """ Class using only the year """
    def clean_info(self, info):
        """Treat year"""
        yea = info[5]
        # return year
        year_ = (float(yea) - 2003.) / (2013. - 2003.)  # mapping to [0;1]
        return year_
        
    def get_feature(self, edge_idx, edge_info):
        """I don't know"""
        edge_year = self.clean_info(edge_info)
        n_years = edge_year
        return np.asarray((n_years))
    

class GraphDegreeFeatures(BaseFeatureExtractor):
    """ Class using Graph Features
        # (1) d_in_p(SRC)     number of + incoming edges to SRC
        # (2) d_in_n(TGT)     number of - incoming edges to TGT
        # (3) d_out_p(SRC)    number of + outgoing edges from SRC
        # (4) d_out_n(TGT)    number of - outgoing edges from TGT
        # (5) common(SRC/TGT) number of common neighbors (undirected)
        # (6) d_out(SRC)      total number of outgoing edges from SRC
        # (7) d_in(TGT)       total number of incoming edges to TGT
    """    
    def __init__(self, edges_info):
        BaseFeatureExtractor.__init__(self, edges_info)
        # initialize the Graph
        print 'Fitting Degree Feature Graph'
        columns = ['ind', 'SRC', 'TGT', 'VOT',
                   'RES', 'YEA', 'DAT', 'TXT']
        edges_df = pd.DataFrame(edges_info, columns=columns)
        self.graph = nx.from_pandas_dataframe(edges_df,
                                              'SRC', 'TGT', 
                                              'VOT',
                                              nx.DiGraph())      
        self.ugraph = self.graph.to_undirected()
    
    def get_feature(self, edge_idx, edge_info):
        """Graph Features"""
        source = edge_info[1]
        target = edge_info[2]
        year   = (float(edge_info[5]) - 2003.) / (2013. - 2003.)  # mapping to [0;1]

        in_source = self.graph.in_edges(source, data=True)
        in_target = self.graph.in_edges(target, data=True)
        out_source = self.graph.out_edges(source, data=True) 
        out_target = self.graph.out_edges(target, data=True)

        d_in_p = len([1 for _, _, w in in_source if w['VOT'] == 1])
        d_in_n = len([1 for _, _, w in in_target if w['VOT'] == -1])
        d_out_p = len([1 for _, _, w in out_source if w['VOT'] == 1])
        d_out_n = len([1 for _, _, w in out_target if w['VOT'] == -1])
        common = len(list(nx.common_neighbors(self.ugraph, source, target)))
        d_out = self.graph.out_degree(source) 
        d_in = self.graph.in_degree(target) 
        return np.asarray((d_in_p, d_in_n, d_out_p, 
                           d_out_n, common, d_out, d_in))
        # return np.asarray((year, d_in_p, d_in_n, d_out_p, 
        #                    d_out_n, common, d_out, d_in))


class GraphTriadFeatures(BaseFeatureExtractor):
    """ Class using Graph Features
        For every edge (u,v), 16 features counting the number
        of triads (u,v,w) s.t.:
            (u,w) is an edge (2 directions, 2 signs)
                AND
            (v,w) is an edge (2 directions, 2 signs)
    """    
    def __init__(self, edges_info):
        BaseFeatureExtractor.__init__(self, edges_info)
        # initialize the Graph
        print 'Fitting Triad Feature Graph'
        columns = ['ind', 'SRC', 'TGT', 'VOT',
                   'RES', 'YEA', 'DAT', 'TXT']
        edges_df = pd.DataFrame(edges_info, columns=columns)
        self.graph = nx.from_pandas_dataframe(edges_df,
                                              'SRC', 'TGT', 
                                              'VOT',
                                              nx.DiGraph())      
        self.ugraph = self.graph.to_undirected()
    
    def get_feature(self, edge_idx, edge_info):
        """Graph Features"""
        source = edge_info[1]
        target = edge_info[2]

        common = len(list(nx.common_neighbors(self.ugraph, source, target)))

        in_source = self.graph.in_edges(source, data=True)
        in_target = self.graph.in_edges(target, data=True)
        out_source = self.graph.out_edges(source, data=True) 
        out_target = self.graph.out_edges(target, data=True)

        n_in_p_in_p = len(set([u for u, _, w in in_source if w['VOT'] == 1])\
                .intersection([u for u, _, w in in_target if w['VOT'] == 1]))
        n_in_p_in_n = len(set([u for u, _, w in in_source if w['VOT'] == 1])\
                .intersection([u for u, _, w in in_target if w['VOT'] == -1]))
        n_in_n_in_p = len(set([u for u, _, w in in_source if w['VOT'] == -1])\
                .intersection([u for u, _, w in in_target if w['VOT'] == 1]))
        n_in_n_in_n = len(set([u for u, _, w in in_source if w['VOT'] == -1])\
                .intersection([u for u, _, w in in_target if w['VOT'] == -1]))
        n_in_p_out_p = len(set([u for u, _, w in in_source if w['VOT'] == 1])\
                .intersection([v for _, v, w in out_target if w['VOT'] == 1]))
        n_in_p_out_n = len(set([u for u, _, w in in_source if w['VOT'] == 1])\
                .intersection([v for _, v, w in out_target if w['VOT'] == -1]))
        n_in_n_out_p = len(set([u for u, _, w in in_source if w['VOT'] == -1])\
                .intersection([v for _, v, w in out_target if w['VOT'] == 1]))
        n_in_n_out_n = len(set([u for u, _, w in in_source if w['VOT'] == -1])\
                .intersection([v for _, v, w in out_target if w['VOT'] == -1]))
        n_out_p_in_p = len(set([v for _, v, w in out_source if w['VOT'] == 1])\
                .intersection([u for u, _, w in in_target if w['VOT'] == 1]))
        n_out_p_in_n = len(set([v for _, v, w in out_source if w['VOT'] == 1])\
                .intersection([u for u, _, w in in_target if w['VOT'] == -1]))
        n_out_n_in_p = len(set([v for _, v, w in out_source if w['VOT'] == -1])\
                .intersection([u for u, _, w in in_target if w['VOT'] == 1]))
        n_out_n_in_n = len(set([v for _, v, w in out_source if w['VOT'] == -1])\
                .intersection([u for u, _, w in in_target if w['VOT'] == -1]))
        n_out_p_out_p = len(set([v for _, v, w in out_source if w['VOT'] == 1])\
                .intersection([v for _, v, w in out_target if w['VOT'] == 1]))
        n_out_p_out_n = len(set([v for _, v, w in out_source if w['VOT'] == 1])\
                .intersection([v for _, v, w in out_target if w['VOT'] == -1]))
        n_out_n_out_p = len(set([v for _, v, w in out_source if w['VOT'] == -1])\
                .intersection([v for _, v, w in out_target if w['VOT'] == 1]))
        n_out_n_out_n = len(set([v for _, v, w in out_source if w['VOT'] == -1])\
                .intersection([v for _, v, w in out_target if w['VOT'] == -1]))
        return np.asarray((n_in_p_in_p, n_in_p_in_n, n_in_n_in_p, n_in_n_in_n, 
                           n_in_p_out_p, n_in_p_out_n, n_in_n_out_p, n_in_n_out_n, 
                           n_out_p_in_p, n_out_p_in_n, n_out_n_in_p, n_out_n_in_n, 
                           n_out_p_out_p, n_out_p_out_n, n_out_n_out_p, n_out_n_out_n))
