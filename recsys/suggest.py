import numpy as np
import unittest

def top_n (score_matrix, n, excludes = []):
    '''Returns (index,score) list of top n elements of a given array
        - excludes is a list of ids not to suggest
    '''
    #set scores for excluded elements to 0
    score_matrix[:,excludes] = 0
    # get indexes of top items, then construct (id, prediction) pairs
    # in order to make [::-1] work, turn the matrix into an array 
    return np.squeeze(np.asarray(score_matrix)).argsort()[::-1][:n].tolist()

def top_ns (score_matrix, n):
    '''Returns indexes of top n elements for each of given arrays with metrics
        - score_matrix is a dense np.matrix
        - n is a number of elements to return
    '''
    # get indexes of top items, then construct (id, prediction) pairs
    return [top_n(sa,n) for sa in score_matrix]

class SuggestTest(unittest.TestCase):

    def test(self):
        n = 5
        input = np.array([10,1,9,2,8,3,7,4,6,5])
        output = np.array([0,2,4,6,8])
        print top_n(input,5)
        self.assertTrue(np.array_equal(top_n(input, n), output))
        self.assertTrue(np.array_equal(top_ns([input,]*3, n), [output,]*3))

if __name__ == '__main__':
    unittest.main()
