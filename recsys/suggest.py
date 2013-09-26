import numpy as np

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
