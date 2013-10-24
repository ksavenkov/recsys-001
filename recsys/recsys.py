from scipy import sparse
import numpy as np

import dataset as ds
import score as sc
import suggest as sg

class Recommender:
    '''Recommender that compute and store models and provide recommendations.'''

    def __init__(self, dataset, model, score_algo, suggest_algo, verbose = True):
        '''Initialize the recommender, but don't do any data import or processing.
            - dataset provides ratings (see dataset.py)
            - recommender models (see model.py)
            - score_algo is one of the scoring algorithms (see score.py)
            - suggest_algo is one of the sort-n-cut algorithms (see suggest.py)
            NB: item-to-item, user-to-item, user-to-user etc should be implemented as distinct score functions
            input data conversion is performed by DataIO object,
        '''
        self.__built = False
        self.__dataset = dataset
        self.__model = model
        self.__score = score_algo
        self.__suggest = suggest_algo
        self.__verbose = verbose
        pass
    
    def build(self):
        '''Convert the dataset to the efficient internal representation and build all the models.'''
        self.__log('Building the recommender model')
        self.__model.build(self.__dataset) 
        self.__log('\t...done.')
        self.__built = True
        pass

    def recommend(self, ids, n):
        '''Return top n recommendations for a given list of (normalized) ids.'''
        if not self.__built:
            self.build()
            self.__built == True
        # score all possible recommendations
        P = self.__score(self.__model, ids)
        # select n from them
        return self.__suggest(P, n)

    def recommend_from(self, ids, choose_from_ids):
        '''For a given list of (normalized) ids, return scores of ids from choose_from_ids set'''
        if not self.__built:
            self.build()
            self.__built == True
        # score all possible recommendations
        P = self.__score(self.__model, ids, choose_from_ids)
        # select n from them
        return P
 
        
    def __log(self, msg):
        if self.__verbose:
            print msg
