import unittest
import numpy as np
from scipy import sparse

import dataset as ds
from model import PreferenceModel, TFIDFModel, UserModel

class PreferenceModelTest(unittest.TestCase):

    def setUp(self):
        self.data = ds.DataIO(verbose = False)
        self.data.load('testdata/ratings.csv')
        self.model = PreferenceModel(verbose = False)
        self.model.build(self.data)

    def test(self):
        # test __extract_facts:
        # overall number of ratings
        self.assertTrue(self.model.P().sum() == len(self.data.ratings))
        # item and user checksums
        user_sums = list([0,]*self.data.num_users())
        item_sums = list([0,]*self.data.num_items())
        for (u,i,r) in self.data.ratings:
            user_sums[u] += 1
            item_sums[i] += 1
        self.assertTrue(np.array_equal(self.model.P().sum(1), np.matrix(user_sums).T))
        self.assertTrue(np.array_equal(self.model.P().sum(0), np.matrix(item_sums)))

class DummyDataset:
    def __init__(self):
        self.ratings = [(0,0,4.0), (0,1,1),            (0,3,4.6),
                        (1,0,4.5), (1,1,4), (1,2,3.5), (1,3,3.7), (1,4,4.0),
                        (2,0,5.0),          (2,2,3.4),            (2,4,4.0)
                       ]
        self.item_tags = [(0,0,1), (0,1,1), (0,2,1), (0,3,1), (0,4,1), (0,5,1),
                          (1,0,1),          (1,2,4),          (1,4,1), 
                          (2,0,1),          (2,2,2),          (2,4,2), (2,5,1),
                          (3,0,1),                   (3,3,3), (3,4,2), 
                          (4,0,1),                   (4,3,4),          (4,5,1)
                         ]

class TFIDFModelTest(unittest.TestCase):

    def setUp(self):
        self.data = DummyDataset()
        self.model = TFIDFModel(verbose = False)

    def test_tocsr(self):
        # test item/tag matrix conversion
        TF_expected = [[1, 1, 1, 1, 1, 1],
                       [1, 0, 4, 0, 1, 0],
                       [1, 0, 2, 0, 2, 1],
                       [1, 0, 0, 3, 2, 0],
                       [1, 0, 0, 4, 0, 1]]
        TF = self.model._TFIDFModel__convert_tocsr(self.data.item_tags)
        self.assertTrue(np.array_equal(TF.todense(), TF_expected))

    def test_extract(self):
        # test fact extraction without a threshold
        DF_expected = [[1, 1, 1, 1, 1, 1],
                       [1, 0, 1, 0, 1, 0],
                       [1, 0, 1, 0, 1, 1],
                       [1, 0, 0, 1, 1, 0],
                       [1, 0, 0, 1, 0, 1]]
        DF = self.model._TFIDFModel__extract_facts(self.data.item_tags)
        self.assertTrue(np.array_equal(DF.todense(), DF_expected))

    def test_extract_threshold(self):
        # test preference extraction with a threshold
        P_expected = [[1, 0, 0, 1, 0],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1]]
        P = self.model._TFIDFModel__extract_facts(self.data.ratings, 3.5)
        self.assertTrue(np.array_equal(P.todense(), P_expected))

    def test_tfidf_profiles(self):
        I_expected = np.matrix([[ 0.,          0.86991409,  0.27610534,  0.27610534,  0.12061088,  0.27610534],
                                [ 0.,          0.        ,  0.9940897 ,  0.        ,  0.10856185,  0.        ],
                                [ 0.,          0.        ,  0.83309624,  0.        ,  0.36392077,  0.41654812],
                                [ 0.,          0.        ,  0.        ,  0.96011533,  0.27960428,  0.        ],
                                [ 0.,          0.        ,  0.        ,  0.9701425 ,  0.        ,  0.24253563]])
        self.model.build(self.data)
        # test TFIDF item profile extraction
        self.assertTrue(stringify_matrix(self.model.I().todense()) == 
                        stringify_matrix(I_expected))

class TestUserModel(unittest.TestCase):
    def setUp(self):
        self.data = DummyDataset()
        self.model = UserModel(verbose = False)
        self.model.build(self.data)

    def test_mean(self):
        expected = np.matrix([[ 3.2       ],
                              [ 3.94      ],
                              [ 4.13333333]])
        self.assertTrue(stringify_matrix(self.model.mean()) == stringify_matrix(expected))

    def test_users(self):
        expected = sparse.csr_matrix(
                      [[ 0.8       , -2.2       , 0.0       ,  1.4       ,  0.0       ],
                       [ 0.56      ,  0.06      , -0.44     , -0.24      ,  0.06      ],
                       [ 0.86666667,  0.0       , -0.73333333, 0.0       , -0.13333333]])
        self.assertTrue(stringify_matrix(self.model.R().todense()) == stringify_matrix(expected.todense()))

def stringify_matrix(m):
    return ' '.join(['%.8f' % f for f in list(np.asarray(m).reshape(-1,))])

if __name__ == '__main__':
    unittest.main()
