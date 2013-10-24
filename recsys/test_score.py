import unittest
import numpy as np
from scipy import sparse
from scipy import stats
from score import cooc_simple, cooc_advanced, tfidf_unweighted, tfidf_weighted, cosine, pearson, user_based_knn, leave_top_n

# PA3 doesn't actually invoke the recommender, just rating predictior, so put the test here
from dataset import DataIO
from model import UserModel
from suggest import top_ns

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
    def R(self):
        return sparse.csr_matrix(
                     [[4.0, 1.0, 0.0, 4.6, 0.0],
                      [4.5, 4.0, 3.5, 3.7, 4.0],
                      [5.0, 0.0, 3.4, 0.0, 4.0]])

    def P(self):
        return sparse.csr_matrix(
                     [[1, 0, 0, 1, 0],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1]])

class DummyUserModel:
    def R(self):
        return sparse.csr_matrix(
                      [[ 0.8       , -2.2       , 0.0       ,  1.4       ,  0.0       ],
                       [ 0.56      ,  0.06      , -0.44     , -0.24      ,  0.06      ],
                       [ 0.86666667,  0.0       , -0.73333333, 0.0       , -0.13333333]])
    def mean(self):
        return np.matrix([[ 3.2       ],
                       [ 3.94      ],
                       [ 4.13333333]])

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
        R1_simple = np.matrix([[-1., 0.8, 0.6 ,  0.4 ,  0.2 ],
                               [ 1.,-1. , 0.75,  0.5 ,  0.25],
                               [ 1., 1. ,-1.  ,  2./3,  1./3],
                               [ 1., 1. , 1.  , -1.  ,  0.5 ],
                               [ 1., 1. , 1.  ,  1.  , -1.  ]])
        R2_simple = np.matrix([[-1. ,  0.5,  0.5,  0.5,  0.5,  1. ],
                               [ 1. , -1. ,  0. ,  1. ,  0. ,  1. ],
                               [ 1. ,  0. , -1. ,  0. ,  1. ,  1. ],
                               [ 1. ,  1. ,  0. , -1. ,  0. ,  1. ],
                               [ 1. ,  0. ,  1. ,  0. , -1. ,  1. ],
                               [ 1. ,  0.5,  0.5,  0.5,  0.5, -1. ]])
        R3_simple = np.matrix([[-1. ,  0.5],
                               [ 0.5, -1. ]])
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
        R1_advanced = np.matrix([[-1.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                                 [ 1.        , -1.        ,  0.        ,  0.        ,  0.        ],
                                 [ 1.        ,  2.        , -1.        ,  0.        ,  0.        ],
                                 [ 1.        ,  1.5       ,  3.        , -1.        ,  0.        ],
                                 [ 1.        ,  4./3      ,  2.        ,  4.        , -1.        ]])
        R2_advanced = np.matrix([[-1.,  0.,  0.,  0.,  0.,  0.],
                                 [ 1., -1.,  0.,  0.,  0.,  1.],
                                 [ 1.,  0., -1.,  0.,  0.,  1.],
                                 [ 1.,  0.,  0., -1.,  0.,  1.],
                                 [ 1.,  0.,  0.,  0., -1.,  1.],
                                [  0.,  0.,  0.,  0.,  0., -1.]])
        R3_advanced = np.matrix([[-1. ,  0.5],
                                 [ 0.5, -1. ]])

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
        expected = np.matrix([[ -999.         , -999.         , 0.44434619  , -999.         , 0.73476803],
                              [ -999.         , -999.         , -999.         , -999.         , -999.       ],
                              [ -999.         , 0.20054049  , -999.         , 0.77205766  , -999.       ]])
        
        # convert to string form to avoid problems with float precision
        self.assertTrue(
            stringify_matrix(tfidf_unweighted(self.model, [0,1,2]))
            ==
            stringify_matrix(expected))
        self.assertTrue(
            stringify_matrix(tfidf_unweighted(self.model, [0,2]))
            ==
            stringify_matrix(expected[[0,2],:]))
        self.assertTrue(
            stringify_matrix(tfidf_unweighted(self.model, [1]))
            ==
            stringify_matrix(expected[[1],:]))

    def test_weighted(self):
        expected = np.matrix([[ -999.         , -999.         , -0.50277642  , -999.         , 0.50818190],
                              [ -999.         , -999.         , -999.         , -999.         , -999.       ],
                              [ -999.         , -0.24407424  , -999.         , 0.03498383  , -999.       ]])

        # convert to string form to avoid problems with float precision
        self.assertTrue(
            stringify_matrix(tfidf_weighted(self.model, [0,1,2]))
            ==
            stringify_matrix(expected))
        self.assertTrue(
            stringify_matrix(tfidf_weighted(self.model, [0,2]))
            ==
            stringify_matrix(expected[[0,2],:]))
        self.assertTrue(
            stringify_matrix(tfidf_weighted(self.model, [1]))
            ==
            stringify_matrix(expected[[1],:]))

    def test_cosine(self):
        U = sparse.csr_matrix([[1,1,1,1],
                               [0,1,1,1],
                               [0,0,0,1]])
        I = sparse.csr_matrix([[1,1,1,1],
                               [2,2,2,2],
                               [4,4,4,4]])
        expected = np.matrix([[ 1.        , 1.        , 1.       ],
                              [ 0.8660254 , 0.8660254 , 0.8660254],
                              [ 0.5       , 0.5       , 0.5      ]])
        self.assertTrue(stringify_matrix(cosine(U,I).todense()) 
                        ==
                        stringify_matrix(expected))

class UserTest (unittest.TestCase):

    def setUp(self):
        self.model = DummyUserModel()

    def test_user_knn(self):
        expected = np.matrix([[ 2.47335263, 2.72],
                              [ 3.20666667, 5.34]])
        self.assertTrue(stringify_matrix(user_based_knn(self.model, 30, [0,2], [2,3], cosine))
                        ==
                        stringify_matrix(expected))

    def test_leave_top_n(self):
        a = sparse.csr_matrix([[1,6],[2,5],[3,4]])
        n = 2
        expected = sparse.csr_matrix([[0,6],[2,5],[3,0]])
        leave_top_n(a,n)
        self.assertTrue(np.array_equal(a.todense(), expected.todense()))

class UserKNNTest(unittest.TestCase):
    def test_pa3(self):
        testdata = zip([(1024,77),(1024,268),(1024,462),(1024,393),(1024,36955),(2048,77),(2048,36955),(2048,788)],
                       [
                        "1024,77,4.3848,Memento (2000)",
                        "1024,268,2.8646,Batman (1989)",
                        "1024,462,3.1082,Erin Brockovich (2000)",
                        "1024,393,3.8722,Kill Bill: Vol. 2 (2004)",
                        "1024,36955,2.3524,True Lies (1994)",
                        "2048,77,4.8493,Memento (2000)",
                        "2048,36955,3.9698,True Lies (1994)",
                        "2048,788,3.8509,Mrs. Doubtfire (1993)",
                        ])

        data = DataIO(verbose = False)
        data.load('testdata/ratings.csv', items_file = 'testdata/movie-titles.csv')
        model = UserModel(verbose = False, normalize = True)
        model.build(data)
        
        for ((u,i),s) in testdata:
            self.assertTrue('%s' % s ==
                            '%d,%d,%.4f,%s' % (u,i,user_based_knn(model, 30, [data.new_user_idx(u)],[data.new_item_idx(i)], 
                                                cosine, promote_users = True, normalize = 'centered'), data.title(i)))


class WA4Test(unittest.TestCase):

    def setUp(self):
        self.data = DataIO(verbose = False)
        self.data.load('testdata/ratings-ma4.csv')
        self.model = UserModel(normalize = False, verbose = False)
        self.model.build(self.data)

    def test_pearson(self):
        # test correlation
        S = pearson(self.model.R(), self.model.R()).todense()
        # 1. check we don't have numbers more than 1
        # user string comparison to avoid float nuances
        self.assertTrue('%.2f' % S.max() == '1.00');

        # 2. check there are only '1' on the diagonal
        self.assertTrue(sum([S[i,i] for i in range(S.shape[0])]) == S.shape[0])
        
        # 3. check a couple of correlation coefficients
        corr_test = [(1648, 5136, 0.40298),
                     (918, 2824, -0.31706)]
        for (u1,u2,c) in corr_test:
            # check what's in the full matrix 
            u1 = self.data.new_user_idx(u1)
            u2 = self.data.new_user_idx(u2)
            # check precomputed
            self.assertTrue('%.5f' % S[u1,u2] == '%.5f' % c)
            # compute here
            self.assertTrue('%.5f' % pearson(self.model.R()[u1,:], self.model.R()[u2,:]).todense() == '%.5f' % c)

    def test_5nn(self):
        u = 3712
        nns = [(2824,0.46291), (3867,0.400275), (5062,0.247693), (442,0.22713), (3853,0.19366)]
        S = pearson(self.model.R(), self.model.R())
        leave_top_n(S,6)
        top_neighbours = [(self.data.old_user_idx(i),S[i,self.data.new_user_idx(u)]) 
                                    for i in S[:,self.data.new_user_idx(u)].nonzero()[0]]
        top_neighbours.sort(key = lambda a: a[1], reverse = True)
        # skip the first element (corr = 1)
        self.assertTrue(','.join(['%d,%.6f' % a for a in top_neighbours[1:]]) == 
                        ','.join(['%d,%.6f' % a for a in nns]))
    
    # consider moving this test to test_recsys.py
    def test_unnormalized(self):
       u = 3712
       expected = [(641,5.000), (603,4.856), (105,4.739)]
       R = user_based_knn(self.model, 5, [self.data.new_user_idx(u)], range(self.data.num_items()), 
                pearson, promote_users = False)
       recs = top_ns([R],3, keep_order = True)
       self.assertTrue(','.join(['%d,%.3f' % (self.data.old_item_idx(a),b) for (a,b) in recs[0]]) == 
                       ','.join(['%d,%.3f' % a for a in expected]))

    # consider moving this test to test_recsys.py
    def test_normalized(self):

       u = 3712
       expected = [(641,5.900), (603,5.546), (105,5.501)]
       R = user_based_knn(self.model, 5, [self.data.new_user_idx(u)], range(self.data.num_items()), 
                pearson, promote_users = False, normalize = 'normalize')
       recs = top_ns([R],3, keep_order = True)
       self.assertTrue(','.join(['%d,%.3f' % (self.data.old_item_idx(a),b) for (a,b) in recs[0]]) == 
                       ','.join(['%d,%.3f' % a for a in expected]))


def stringify_matrix(m):
    return ' '.join(['%.8f' % f for f in list(np.asarray(m).reshape(-1,))])

if __name__ == '__main__':
    unittest.main()
