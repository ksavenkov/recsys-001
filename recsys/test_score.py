import unittest
import numpy as np
from scipy import sparse
from score import cooc_simple, cooc_advanced

class CoocTest(unittest.TestCase):

    def setUp(self):
        self.R1 = sparse.csr_matrix([[1,1,1,1,1],
                                     [1,1,1,1,0],
                                     [1,1,1,0,0],
                                     [1,1,0,0,0],
                                     [1,0,0,0,0]])
        self.R2 = sparse.csr_matrix([[1,1,0,1,0,1],
                                     [1,0,1,0,1,1]])
        self.R3 = sparse.csr_matrix([[1,1],
                                     [0,1],
                                     [1,0],
                                     [0,1],
                                     [1,0],
                                     [1,1]])

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

        self.assertTrue(np.array_equal(cooc_simple(self.R1,range(self.R1.shape[1])), R1_simple))
        self.assertTrue(np.array_equal(cooc_simple(self.R2,range(self.R2.shape[1])), R2_simple))
        self.assertTrue(np.array_equal(cooc_simple(self.R3,range(self.R3.shape[1])), R3_simple))
        self.assertTrue(np.array_equal(cooc_simple(self.R1,[0,4]), R1_simple[[0,4],:]))
        self.assertTrue(np.array_equal(cooc_simple(self.R1,[1,2,3]), R1_simple[[1,2,3],:]))
        self.assertTrue(np.array_equal(cooc_simple(self.R1,[0,2,4]), R1_simple[[0,2,4],:]))
        self.assertTrue(np.array_equal(cooc_simple(self.R1,[2,3]), R1_simple[[2,3],:]))
        self.assertTrue(np.array_equal(cooc_simple(self.R3,[0]), R3_simple[0,:]))
        self.assertTrue(np.array_equal(cooc_simple(self.R3,[1]), R3_simple[1,:]))

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
        self.assertTrue(np.array_equal(cooc_advanced(self.R1,range(self.R1.shape[1])), R1_advanced))
        self.assertTrue(np.array_equal(cooc_advanced(self.R2,range(self.R2.shape[1])), R2_advanced))
        self.assertTrue(np.array_equal(cooc_advanced(self.R3,range(self.R3.shape[1])), R3_advanced))
        self.assertTrue(np.array_equal(cooc_advanced(self.R1,[0,4]), R1_advanced[[0,4],:]))
        # different sets of items
        self.assertTrue(np.array_equal(cooc_advanced(self.R1,[1,2,3]), R1_advanced[[1,2,3],:]))
        self.assertTrue(np.array_equal(cooc_advanced(self.R1,[0,2,4]), R1_advanced[[0,2,4],:]))
        self.assertTrue(np.array_equal(cooc_advanced(self.R1,[2,3]), R1_advanced[[2,3],:]))
        # single items
        self.assertTrue(np.array_equal(cooc_advanced(self.R3,[0]), R3_advanced[0,:]))
        self.assertTrue(np.array_equal(cooc_advanced(self.R3,[1]), R3_advanced[1,:]))


        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
