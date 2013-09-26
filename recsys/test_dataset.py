import unittest
import subprocess
from dataset import Dataset

class DatasetTest(unittest.TestCase):

    def setUp(self):
        self.filename = 'testdata/ratings.csv'
        self.ds = Dataset(False)
        self.ds.load(self.filename)

    def test_count(self):
        self.assertTrue(len(self.ds.ratings) == self.__wccount())

    def test_read(self):
        head_ratings = [(1,809,4.0),(1,601,5.0),(1,238,5.0),(1,664,4.5),(1,3049,3.0)]
        self.assertTrue(self.ds.ratings[0:5] == [(self.ds.new_user_idx(u),self.ds.new_item_idx(i),r) for (u,i,r) in head_ratings])
        tail_ratings = [(5573,114,2.5),(5573,22,4.5),(5573,11,3.0),(5573,557,4.0),(5573,98,3.5)]
        self.assertTrue(self.ds.ratings[-5:] == [(self.ds.new_user_idx(u),self.ds.new_item_idx(i),r) for (u,i,r) in tail_ratings])

    def test_normalize(self):
        (user_col, item_col) = zip(*self.ds.ratings)[:2]
        self.assertTrue(len(set(user_col)) == self.ds.num_users())
        self.assertTrue(len(set(item_col)) == self.ds.num_items())
        self.assertTrue(range(self.ds.num_users()) == 
                [self.ds.new_user_idx(self.ds.old_user_idx(i)) for i in range(self.ds.num_users())])
        self.assertTrue(range(self.ds.num_items()) == 
                [self.ds.new_item_idx(self.ds.old_item_idx(i)) for i in range(self.ds.num_items())])

    #credits https://gist.github.com/zed/0ac760859e614cd03652
    def __wccount(self):
        out = subprocess.Popen(['wc', '-l', self.filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
        return int(out.strip().partition(b' ')[0])

if __name__ == '__main__':
    unittest.main()
