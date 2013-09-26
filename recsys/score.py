__all__ = ['cooc_simple', 'cooc_advanced']

import numpy as np
from scipy import sparse
import unittest

np.seterr(all='ignore',invalid='ignore')

#TODO: should be turned into classes to precompute and reuse some information (i.e. build the model)

#returns dense matrix: [x,y] = number of users rated both item x and item y
def rated_X_and_Y(R, given_items):
    #highlight the fact the matrix is dense
    return ( R[:,given_items].T * R ).todense()

#returns dense matrix: [x,:] = number of users rated item x
def rated_X(R, given_items):
    return R[:,given_items].sum(0).T * np.ones((1,R.shape[1]))

def cooc_simple(R,given_items):
    '''Builds matrix M1 given_items x num_items, where M1[X,Y] = (rated both X and Y) / rated X:
       - rated both X and Y: computed by dot product of binary matrix
       - rated X: computed by column-wise sum, then duplicating that as a column
       - note that the matrix are int, so we use true_divide to get float results
    '''

    P = np.true_divide ( rated_X_and_Y(R,given_items) , 
                           rated_X(R,given_items) )

    # cooccurence algorithms assume the given items aren't scored
    P[range(len(given_items)),given_items] = 0 

    return P

def cooc_advanced(R,given_items):
    '''Builds matrix M2 given_items x num_items, where 
       M2[X,Y] = ( (rated both X and Y) / 
                        rated X) / 
                  ( (rated A not X) / 
                       not rated X )
     Let's avoid some divisions:
               = ((rated both X and Y) * (not rated X)) / 
                   ( (rated X) * (rated Y not X) )
     Theoretically, both numerator and denominator can be computed using the same function cooc_simple
        and swapping 0s and 1s in X. However, it is not a good idea to do the swap in a sparse matrix ;-) 
        Instead, let's notice that 'not rated X = total users - rated X'
        In a similar fashion, 'rated Y but not X = rated Y - rated Y and X'
               = ((rated both X and Y) * (total users - rated X)) / 
                   ( (rated X) * (rated Y - rated X and Y) )
    '''

    rated_x = rated_X(R, given_items)
    rated_x_and_y = rated_X_and_Y(R, given_items)
    rated_y = np.ones((len(given_items),1)) * R.sum(0)
    total_users = R.shape[0]

    # extract here to handle division by zero
    cooc_x = np.multiply( rated_x_and_y , total_users - rated_x )
    cooc_not_x = np.multiply( rated_x , rated_y - rated_x_and_y )

    # For some y, there are no users rated y but not x (at least, for x = y). 
    # mask zero values in the denominator
    zero_mask = cooc_not_x == 0

    P = np.true_divide ( cooc_x,
               np.ma.masked_array(cooc_not_x, zero_mask) )
    
    # cooccurence algorithms assume the given items aren't scored
    P[range(len(given_items)),given_items] = 0 

    # fill missing vlaues (x/0 and 0/0) with 0 
    return P.filled(0)


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
