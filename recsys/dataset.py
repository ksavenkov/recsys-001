import csv
#for testing purposes
import unittest
import subprocess

class Dataset:
    '''Encapsulates all reading of the data from whatever source (currently CSV), 
       performs index normalization and provides new<->old index conversion tools'''
    
    def __init__(self, verbose = True):
        self.__verbose = verbose
    
    def load(self, filename):
        '''Loads the data from a proper source, and performs index normalization.'''
        self.__read(filename)
        self.__normalize()
        return

    def num_items(self):
        '''Number of different items in the dataset'''
        return len(self.__old_item_idx)

    def num_users(self):
        '''Number of different users in the dataset'''
        return len(self.__old_user_idx)

    def old_item_idx(self, idx):
        '''Old to new index conversion.'''
        return self.__old_item_idx[idx]

    def new_item_idx(self, idx):
        '''New to old index conversion.'''
        return self.__new_item_idx[idx]

    def old_user_idx(self, idx):
        '''Old to new index conversion.'''
        return self.__old_user_idx[idx]

    def new_user_idx(self, idx):
        '''New to old index conversion.'''
        return self.__new_user_idx[idx]
        pass

    def __log(self, msg):
        if self.__verbose:
            print msg

    def __read(self, filename):
        self.__log('Reading (user_id,item_id,rating) tuples from %s' % filename)
        with open(filename, 'rbU') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            self.ratings = [(int(r[0]),int(r[1]),float(r[2])) for r in csv_reader]
            self.__log('\tread %d entries' % len(self.ratings))
            csvfile.close()
            return

    def __normalize(self, ):
        '''Normalize the data by creating custom user and item indexes'''
        (user_col, item_col) = zip(*self.ratings)[:2]
        (self.__old_user_idx, self.__new_user_idx) = self.__normalize_idx(user_col)
        (self.__old_item_idx, self.__new_item_idx) = self.__normalize_idx(item_col)
        #TODO: inplace change may be more efficient
        self.ratings = [(self.__new_user_idx[u], self.__new_item_idx[i], r) for (u,i,r) in self.ratings]
    
    def __normalize_idx(self, idx_list):
        '''idx_list is a list of indexes with duplicates and gaps
           returns an array of old indexes and a dict old_idx -> new_idx
           two arrays is due to dict.keys() are unsorted
        '''
        # 1. remove duplicates and sort
        old_idx = sorted(set(idx_list))
        # 2. add new normalized index and put to a dict
        return old_idx, dict(zip(old_idx,range(len(old_idx))))

class DatasetTest(unittest.TestCase):

    def setUp(self):
        self.filename = '../data/ratings.csv'
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
