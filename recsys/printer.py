import dataset
import numpy as np

def coursera_pa1_printer(items, recs):
    '''Given a dataset, scores and array of recommended items, stringify the latter as a number of strings:
       "given item, item1, score1, ... item5, score5". The parameters:
        - items are ids of requested items (should be denormalized before output)
        - recs is a 2D list of pairs (recommended item, score), one row per given user
    '''
    assert(len(items) == len(recs))

    return '\n'.join(['%d,' % i + 
                        ','.join(['%d,%.2f' % (j,s) for j,s in r]) 
                        for i,r in zip(items, recs)])

def coursera_pa2_printer(users, recs):
    '''Print result of the 2nd assignment in a form of blocks (one per each given user):
        recommendations for user USER_ID:
          item1: score1 (as X.XXXX)
          ...
          item5: score5
       The parameters:
       - given_users is a list of original ids of given users
       - recs is a 2D list of pairs (recommended item, score), one row per given user
    '''
    assert(len(recs) == len(users))

    return '\n'.join(['recommendations for user %d:\n' % u + 
                        '\n'.join(['  %d: %.4f' % (i,s) for i,s in r]) 
                        for u,r in zip(users, recs)])
