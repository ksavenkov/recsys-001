import unittest
import numpy as np
from scipy import sparse
from score import cooc_simple, cooc_advanced, tfidf_unweighted

class DummyPreferenceModel:
    def __init__(self, R):
        self.R = R
    
    def P(self):
        return self.R

class DummyTFIDFModel:
    def U(self):
        return sparse.csr_matrix(
                     [[ 2., 5., 5./3                  , 20./3, 3.75, 5./3 ],
                      [ 5., 5., (5./3 + 20./3 + 10./3), 40./3, 7.5 , 5.   ],
                      [ 2., 5., 5./3                  , 25./3, 1.25, 10./3]])
    def I(self):
        return sparse.csr_matrix(
                     [[ 1., 5., 5./3 ,  5./3 ,  1.25, 5./3],
                      [ 1., 0., 20./3,  0.   ,  1.25, 0.  ],
                      [ 1., 0., 10./3,  0.   ,  2.5 , 5./3],
                      [ 1., 0., 0.   ,  5.   ,  2.5 , 0.  ],
                      [ 1., 0., 0.   ,  20./3,  0.  , 5./3]])

class CoocTest(unittest.TestCase):

    def setUp(self):
        self.M1 = DummyPreferenceModel(
                   sparse.csr_matrix([[1,1,1,1,1],
                                     [1,1,1,1,0],
                                     [1,1,1,0,0],
                                     [1,1,0,0,0],
                                     [1,0,0,0,0]])
                  )
        self.M2 = DummyPreferenceModel(
                  sparse.csr_matrix([[1,1,0,1,0,1],
                                     [1,0,1,0,1,1]])
                  )
        self.M3 = DummyPreferenceModel(
                  sparse.csr_matrix([[1,1],
                                     [0,1],
                                     [1,0],
                                     [0,1],
                                     [1,0],
                                     [1,1]])
                  )

    def test_simple(self):
        R1_simple = np.matrix([[ 0., 0.8, 0.6 ,  0.4 ,  0.2 ],
                               [ 1., 0. , 0.75,  0.5 ,  0.25],
                               [ 1., 1. , 0.  ,  2./3,  1./3],
                               [ 1., 1. , 1.  ,  0.  ,  0.5 ],
                               [ 1., 1. , 1.  ,  1.  ,  0.  ]])
        R2_simple = np.matrix([[ 0. ,  0.5,  0.5,  0.5,  0.5,  1. ],
                               [ 1. ,  0. ,  0. ,  1. ,  0. ,  1. ],
                               [ 1. ,  0. ,  0. ,  0. ,  1. ,  1. ],
                               [ 1. ,  1. ,  0. ,  0. ,  0. ,  1. ],
                               [ 1. ,  0. ,  1. ,  0. ,  0. ,  1. ],
                               [ 1. ,  0.5,  0.5,  0.5,  0.5,  0. ]])
        R3_simple = np.matrix([[ 0. ,  0.5],
                               [ 0.5,  0. ]])

        self.assertTrue(np.array_equal(cooc_simple(self.M1,range(self.M1.P().shape[1])), R1_simple))
        self.assertTrue(np.array_equal(cooc_simple(self.M2,range(self.M2.P().shape[1])), R2_simple))
        self.assertTrue(np.array_equal(cooc_simple(self.M3,range(self.M3.P().shape[1])), R3_simple))
        self.assertTrue(np.array_equal(cooc_simple(self.M1,[0,4]), R1_simple[[0,4],:]))
        self.assertTrue(np.array_equal(cooc_simple(self.M1,[1,2,3]), R1_simple[[1,2,3],:]))
        self.assertTrue(np.array_equal(cooc_simple(self.M1,[0,2,4]), R1_simple[[0,2,4],:]))
        self.assertTrue(np.array_equal(cooc_simple(self.M1,[2,3]), R1_simple[[2,3],:]))
        self.assertTrue(np.array_equal(cooc_simple(self.M3,[0]), R3_simple[0,:]))
        self.assertTrue(np.array_equal(cooc_simple(self.M3,[1]), R3_simple[1,:]))

    def test_advanced(self):
        R1_advanced = np.matrix([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                                 [ 1.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                                 [ 1.        ,  2.        ,  0.        ,  0.        ,  0.        ],
                                 [ 1.        ,  1.5       ,  3.        ,  0.        ,  0.        ],
                                 [ 1.        ,  4./3      ,  2.        ,  4.        ,  0.        ]])
        R2_advanced = np.matrix([[ 0.,  0.,  0.,  0.,  0.,  0.],
                                 [ 1.,  0.,  0.,  0.,  0.,  1.],
                                 [ 1.,  0.,  0.,  0.,  0.,  1.],
                                 [ 1.,  0.,  0.,  0.,  0.,  1.],
                                 [ 1.,  0.,  0.,  0.,  0.,  1.],
                                [  0.,  0.,  0.,  0.,  0.,  0.]])
        R3_advanced = np.matrix([[ 0. ,  0.5],
                                 [ 0.5,  0. ]])

        # full matrix computation
        self.assertTrue(np.array_equal(cooc_advanced(self.M1,range(self.M1.P().shape[1])), R1_advanced))
        self.assertTrue(np.array_equal(cooc_advanced(self.M2,range(self.M2.P().shape[1])), R2_advanced))
        self.assertTrue(np.array_equal(cooc_advanced(self.M3,range(self.M3.P().shape[1])), R3_advanced))
        self.assertTrue(np.array_equal(cooc_advanced(self.M1,[0,4]), R1_advanced[[0,4],:]))
        # different sets of items
        self.assertTrue(np.array_equal(cooc_advanced(self.M1,[1,2,3]), R1_advanced[[1,2,3],:]))
        self.assertTrue(np.array_equal(cooc_advanced(self.M1,[0,2,4]), R1_advanced[[0,2,4],:]))
        self.assertTrue(np.array_equal(cooc_advanced(self.M1,[2,3]), R1_advanced[[2,3],:]))
        # single items
        self.assertTrue(np.array_equal(cooc_advanced(self.M3,[0]), R3_advanced[0,:]))
        self.assertTrue(np.array_equal(cooc_advanced(self.M3,[1]), R3_advanced[1,:]))


        self.assertTrue(True)

class TFIDFTest(unittest.TestCase):
    def setUp(self):
        self.model = DummyTFIDFModel()
    
    def test_unweighted(self):
        expected = np.matrix([[ 0.83661284,  0.26910251,  0.44434619,  0.81608663,  0.73476803],
                              [ 0.70704371,  0.63705899,  0.73164356,  0.75463312,  0.69770838],
                              [ 0.79424985,  0.20054049,  0.33089257,  0.77205766,  0.85158558]]) 
        # convert to string form to avoid problems with float precision
        self.assertTrue(
            ' '.join(['%.8f' % f for f in list(np.asarray(tfidf_unweighted(self.model, [0,1,2])).reshape(-1,))])
            ==
            ' '.join(['%.8f' % f for f in list(np.asarray(expected).reshape(-1,))]))
        self.assertTrue(
            ' '.join(['%.8f' % f for f in list(np.asarray(tfidf_unweighted(self.model, [0,2])).reshape(-1,))])
            ==
            ' '.join(['%.8f' % f for f in list(np.asarray(expected[[0,2],:]).reshape(-1,))]))
        self.assertTrue(
            ' '.join(['%.8f' % f for f in list(np.asarray(tfidf_unweighted(self.model, [1])).reshape(-1,))])
            ==
            ' '.join(['%.8f' % f for f in list(np.asarray(expected[[1],:]).reshape(-1,))]))

if __name__ == '__main__':
    unittest.main()
