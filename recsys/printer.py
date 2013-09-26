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
