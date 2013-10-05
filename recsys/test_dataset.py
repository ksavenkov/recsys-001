import unittest
import subprocess
from itertools import groupby
import operator
import csv

from dataset import DataIO

class DatasetTest(unittest.TestCase):

    def setUp(self):
        self.ratings_file = 'testdata/ratings.csv'
        self.item_tags_file = 'testdata/movie-tags.csv'
        self.ds = DataIO(False)

    def test_ratings(self):
        self.ds.load(self.ratings_file)
        self.__ratings_norm_test()
        self.__ratings_test()
        self.__printer_test()

    def test_item_tags(self):
        self.ds.load(self.ratings_file, self.item_tags_file)
        self.__ratings_norm_test()
        self.__ratings_test()
        self.__tags_test()
        self.__tags_norm_test()

    def __printer_test(self):
        expected_users = '1: (11,9.00), (12,8.00), (13,7.00), (14,6.00), (22,5.00), (24,4.00), (38,3.00), (63,2.00), (77,1.00), (85,0.00)\n51: (11,9.00), (12,8.00), (13,7.00), (14,6.00), (22,5.00), (24,4.00), (38,3.00), (63,2.00), (77,1.00), (85,0.00)\n100: (11,9.00), (12,8.00), (13,7.00), (14,6.00), (22,5.00), (24,4.00), (38,3.00), (63,2.00), (77,1.00), (85,0.00)'
        expected_items = '11: (11,9.00), (12,8.00), (13,7.00), (14,6.00), (22,5.00), (24,4.00), (38,3.00), (63,2.00), (77,1.00), (85,0.00)\n603: (11,9.00), (12,8.00), (13,7.00), (14,6.00), (22,5.00), (24,4.00), (38,3.00), (63,2.00), (77,1.00), (85,0.00)\n36955: (11,9.00), (12,8.00), (13,7.00), (14,6.00), (22,5.00), (24,4.00), (38,3.00), (63,2.00), (77,1.00), (85,0.00)'
        recs = [zip(range(10), range(10)[::-1]),]*3
        ids = [0,50,99]
        self.assertTrue(self.ds.print_recs(recs, given_items = ids) == expected_items)
        self.assertTrue(self.ds.print_recs(recs, given_users = ids) == expected_users)

    def __ratings_test(self):
        # lines count
        self.assertTrue(len(self.ds.ratings) == self.__wccount(self.ratings_file))
        # values
        head_ratings = [(1,809,4.0),(1,601,5.0),(1,238,5.0),(1,664,4.5),(1,3049,3.0)]
        self.assertTrue(self.ds.ratings[0:5] == [(self.ds.new_user_idx(u),self.ds.new_item_idx(i),r) for (u,i,r) in head_ratings])
        tail_ratings = [(5573,114,2.5),(5573,22,4.5),(5573,11,3.0),(5573,557,4.0),(5573,98,3.5)]
        self.assertTrue(self.ds.ratings[-5:] == [(self.ds.new_user_idx(u),self.ds.new_item_idx(i),r) for (u,i,r) in tail_ratings])

    def __ratings_norm_test(self):
        (user_col, item_col) = zip(*self.ds.ratings)[:2]
        self.assertTrue(len(set(user_col)) == self.ds.num_users())
        self.assertTrue(len(set(item_col)) == self.ds.num_items())
        self.assertTrue(range(self.ds.num_users()) == 
                [self.ds.new_user_idx(self.ds.old_user_idx(i)) for i in range(self.ds.num_users())])
        self.assertTrue(range(self.ds.num_items()) == 
                [self.ds.new_item_idx(self.ds.old_item_idx(i)) for i in range(self.ds.num_items())])


    def __tags_test(self):
        # read tags file and check that all (item,tag) combinations appear in the dataset
        # get item-tag combinations from the original file
        file = open(self.item_tags_file, 'rbU')
        csv_reader = csv.reader(file, delimiter=',')
        item_tag_set_orig = set([(self.ds.new_item_idx(int(i)), self.ds.tag_idx(t)) for (i,t) in csv_reader]) 
        file.close()
        # item-tag combinations in the dataset
        item_tag_set = set(zip(*zip(*self.ds.item_tags)[:2]))
        self.assertTrue(len(item_tag_set_orig.symmetric_difference(item_tag_set)) == 0)

        # tag values
        tag_values = [(114,'afternoon section',1),
                      (114,'capitalism',4),
                      (114,"YOUNG WOMEN'S FAVORATE",1),
                      (10020,'18th century',2),
                      (581,'wolves',1)]

        self.assertTrue(all([self.ds.item_tags.index((self.ds.new_item_idx(i), self.ds.tag_idx(t), c )) for (i,t,c) in tag_values]))

        # tag count
        tag_count_expected = dict([(114,1),
                                   (680,1),
                                   (581,1)])
        # take list of unique (item,tag) pairs, replace tag with 1s and group-sum by the first argument
        item_tagcount = dict(self.__sum_group_by_first( zip(zip(*self.ds.item_tags)[0], [1,]*len(self.ds.item_tags)) ))
        self.assertTrue([item_tagcount[self.ds.new_item_idx(i)] == tag_count_expected[i] for i in [114,680,581]])

    def __tags_norm_test(self):
        # collect all users and items
        (item_col, tag_col) = zip(*self.ds.item_tags)[:2]
        # check that there are as many new indexes as different users and items
        self.assertTrue(len(set(item_col)) == self.ds.num_items())                  
        # actually, this may not hold, but let's keep for now
        self.assertTrue(len(set(tag_col)) == self.ds.num_tags())
        # for all tags, check that new(old(new) = new
        self.assertTrue(range(self.ds.num_tags()) == 
                [self.ds.tag_idx(self.ds.tags(i)) for i in range(self.ds.num_tags())])

    # takes a list of pairs
    # group by the first element and do summ aggregate of the second
    # credits http://stackoverflow.com/questions/11058001/python-group-by-and-sum-a-list-of-tuples
    def __sum_group_by_first(self, list_of_pairs):
        return [(x,sum([z[1] for z in y])) for (x,y)
                    in groupby(sorted(list_of_pairs, key = operator.itemgetter(0)),
                               key = operator.itemgetter(0))]
    
    #credits https://gist.github.com/zed/0ac760859e614cd03652
    def __wccount(self, filename):
        out = subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
        return int(out.strip().partition(b' ')[0])

if __name__ == '__main__':
    unittest.main()
