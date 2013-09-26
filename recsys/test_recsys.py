import unittest
import numpy as np

from recsys import Recommender
import dataset as ds
import score as sc
import suggest as sg
import printer as pr

class RecommenderTest(unittest.TestCase):

    def setUp(self):
        self.data = ds.Dataset(verbose = False)
        self.data.load('testdata/ratings.csv')

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
