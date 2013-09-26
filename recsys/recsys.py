from scipy import sparse
import numpy as np

import dataset as ds
import score as sc
import suggest as sg
import printer as pr

class Recommender:
    '''Recommender that compute and store models and provide recommendations.'''

    def __init__(self, dataset, score_algo, suggest_algo, printer, verbose = True):
        '''Initialize the recommender, but don't do any data import or processing.
            - dataset provides ratings (see dataset.py)
            - score_algo is one of the scoring algorithms (see score.py)
            - suggest_algo is one of the sort-n-cut algorithms (see suggest.py)
            - printer is a pretty-printer for recommender output (see printer.py)
        '''
        self.__built = False
        self.__dataset = dataset
        self.__score = score_algo
        self.__suggest = suggest_algo
        self.__printer = printer
        self.__verbose = verbose
        pass
    
    def build(self):
        '''Convert the dataset to the efficient internal representation and build all the models.'''
        self.__log('Building the recommender')
        self.__extract_facts() 
        self.__log('\t...done.')
        self.__built = True
        pass

    def recommend(self, ids, n):
        '''Return top n recommendations for a given list of ids.'''
        if not self.__built:
            self.build()
            self.__built == True
        # don't forget to convert given item idx to a normalized form
        ids_norm = [self.__dataset.new_item_idx(i) for i in ids]
        # score all possible recommendations
        P = self.__score(self.__R, ids_norm)
        # select n from them
        recommended = self.__suggest(P, n)
        #return them in desired format
        return self.__printer(self.__dataset, ids_norm, P, recommended)
        

    def __extract_facts(self):
        '''Converts dataset ratings to a sparse.csr_matrix with '1' for each fact when a user rated an item''' 
        # construct coo_matrix((V,(I,J))) and convert it to CSR for better performance
        self.__R = sparse.coo_matrix(([1,]*len(self.__dataset.ratings), zip(*self.__dataset.ratings)[:2] )).tocsr()
        self.__log('\tratings loaded to %dx%d matrix' % self.__R.shape)
    
    def __log(self, msg):
        if self.__verbose:
            print msg
