import unittest
import numpy as np

from recsys import Recommender
import dataset as ds
import model
import score as sc
import suggest as sg
import printer as pr

class RecommenderTest(unittest.TestCase):

    def setUp(self):
        self.data = ds.DataIO(verbose = False)

    def test_simple(self):
        (n, given_items, expected) = [5, [11,121,8587], '11,603,0.96,1892,0.94,1891,0.94,120,0.93,1894,0.93\n121,120,0.95,122,0.95,603,0.94,597,0.89,604,0.88\n8587,603,0.92,597,0.90,607,0.87,120,0.86,13,0.86']
        self.data.load('testdata/ratings.csv')
        self.rs = Recommender(self.data, model.PreferenceModel(verbose = False), sc.cooc_simple, 
                              sg.top_ns, verbose = False)
        self.rs.build()
        given_items = self.data.translate_items(given_items)
        recs = self.rs.recommend(given_items,n)
        self.assertTrue(self.data.print_recs(recs, 
                            given_items = given_items, 
                            printer = pr.coursera_pa1_printer)  == expected)
        
    def test_advanced(self):
        (n, given_items, expected) = [5, [11,121,8587], '11,1891,5.69,1892,5.65,243,5.00,1894,4.72,2164,4.11\n121,122,4.74,120,3.82,2164,3.40,243,3.26,1894,3.22\n8587,10020,4.18,812,4.03,7443,2.63,9331,2.46,786,2.39']
        self.data.load('testdata/ratings.csv')
        self.rs = Recommender(self.data, model.PreferenceModel(verbose = False), sc.cooc_advanced, 
                              sg.top_ns, verbose = False)
        self.rs.build()
        given_items = self.data.translate_items(given_items)
        recs = self.rs.recommend(given_items,n)
        self.assertTrue(self.data.print_recs(recs, 
                            given_items = given_items, 
                            printer = pr.coursera_pa1_printer)  == expected)

    def test_tfidf_unweighted(self):
        (n, given_users, expected) = [5, [4045,144,3855,1637,2919], 'recommendations for user 4045:\n  11: 0.3596\n  63: 0.2612\n  807: 0.2363\n  187: 0.2059\n  2164: 0.1899\nrecommendations for user 144:\n  11: 0.3715\n  585: 0.2512\n  38: 0.1908\n  141: 0.1861\n  807: 0.1748\nrecommendations for user 3855:\n  1892: 0.4303\n  1894: 0.2958\n  63: 0.2226\n  2164: 0.2119\n  604: 0.1941\nrecommendations for user 1637:\n  2164: 0.2272\n  141: 0.2225\n  745: 0.2067\n  601: 0.1995\n  807: 0.1846\nrecommendations for user 2919:\n  11: 0.3659\n  1891: 0.3278\n  640: 0.1958\n  424: 0.1840\n  180: 0.1527']
        self.data.load('testdata/ratings.csv', 'testdata/movie-tags.csv')
        self.rs = Recommender(self.data, model.TFIDFModel(verbose = False), sc.tfidf_unweighted, 
                              sg.top_ns, verbose = False)
        self.rs.build()
        given_users = self.data.translate_users(given_users)
        recs = self.rs.recommend(given_users,n)
        self.assertTrue(self.data.print_recs(recs, 
                            given_users = given_users, 
                            printer = pr.coursera_pa2_printer)  == expected)
        
if __name__ == '__main__':
    unittest.main()
