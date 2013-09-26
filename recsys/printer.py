import unittest
import dataset
import numpy as np

def coursera_pa1_printer(ds, item_idxs, scores, recs):
    '''Given a dataset, scores and array of recommended items, stringify the latter
        - ds is a dataset (see dataset.py)
        - item_idxs are *normalized* ids of requested items (should be denormalized before output)
        - scores is a dense np.matrix of scores (rows - requested items, cols - scores for other items)
        - recs is an array of recomended items as normalized indexes, one row per each requested item
    '''
    assert(len(scores) == len(item_idxs) and len(scores) == len(recs))

    return '\n'.join(['%d,' % ds.old_item_idx(item_idxs[i]) + 
                        ','.join(['%d,%.2f' % (ds.old_item_idx(j),scores[i,j]) for j in recs[i]]) 
                        for i in range(len(item_idxs))])

class PrinterTest(unittest.TestCase):
    
    def test(self):
        expected = '11,11,9.00,12,8.00,13,7.00,14,6.00,22,5.00,24,4.00,38,3.00,63,2.00,77,1.00,85,0.00\n603,11,9.00,12,8.00,13,7.00,14,6.00,22,5.00,24,4.00,38,3.00,63,2.00,77,1.00,85,0.00\n36955,11,9.00,12,8.00,13,7.00,14,6.00,22,5.00,24,4.00,38,3.00,63,2.00,77,1.00,85,0.00'
        ds = dataset.Dataset(False)
        ds.load('../data/ratings.csv')
        recs = [range(10),]*3
        idxs = [0,50,99]
        scores = np.dot(np.matrix(np.ones(len(idxs))).T, np.matrix(range(10)[::-1]))
        self.assertTrue(coursera_pa1_printer(ds, idxs, scores, recs) == expected)

if __name__ == '__main__':
    unittest.main()
