from scipy import sparse
import numpy as np

import dataset as ds
import score as sc
import suggest as sg
import printer as pr
import unittest

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

## functors

class RecommenderTest(unittest.TestCase):

    def setUp(self):
        self.data = ds.Dataset(verbose = False)
        self.data.load('../data/ratings.csv')

    def test_extract(self):
        self.rs = Recommender(self.data, sc.cooc_simple, sg.top_ns, pr.coursera_pa1_printer, verbose = False)
        self.rs.build()
        # test __extract_facts:
        # overall number of ratings
        self.assertTrue(self.rs._Recommender__R.sum() == len(self.data.ratings))
        # item and user checksums
        user_sums = list([0,]*self.data.num_users())
        item_sums = list([0,]*self.data.num_items())
        for (u,i,r) in self.data.ratings:
            user_sums[u] += 1
            item_sums[i] += 1
        self.assertTrue(np.array_equal(self.rs._Recommender__R.sum(1), np.matrix(user_sums).T))
        self.assertTrue(np.array_equal(self.rs._Recommender__R.sum(0), np.matrix(item_sums)))

    def test_simple(self):
        (n, given_items, expected) = [5, [11,121,8587], '11,603,0.96,1892,0.94,1891,0.94,120,0.93,1894,0.93\n121,120,0.95,122,0.95,603,0.94,597,0.89,604,0.88\n8587,603,0.92,597,0.90,607,0.87,120,0.86,13,0.86']
        self.rs = Recommender(self.data, sc.cooc_simple, sg.top_ns, pr.coursera_pa1_printer, verbose = False)
        self.rs.build()
        self.assertTrue(self.rs.recommend(given_items,n) == expected)
        
    def test_advanced(self):
        (n, given_items, expected) = [5, [11,121,8587], '11,1891,5.69,1892,5.65,243,5.00,1894,4.72,2164,4.11\n121,122,4.74,120,3.82,2164,3.40,243,3.26,1894,3.22\n8587,10020,4.18,812,4.03,7443,2.63,9331,2.46,786,2.39']
        self.rs = Recommender(self.data, sc.cooc_advanced, sg.top_ns, pr.coursera_pa1_printer, verbose = False)
        self.rs.build()
        self.assertTrue(self.rs.recommend(given_items,n) == expected)
        
if __name__ == '__main__':
    unittest.main()
