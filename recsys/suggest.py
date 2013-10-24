import numpy as np

def top_n (score_matrix, n, excludes = [], keep_order = False):
    '''Returns (index,score) list of top n elements of a given array
        - excludes is a list of ids not to suggest
        - keep_order means that despite of reverse sorting, indexes in ties will go in the original order
            (implements a different type of reverse sort)
    '''
    # get indexes of top items, then construct (id, prediction) pairs
    # in order to make [::-1] work, turn the matrix into an array 
    score_array = np.squeeze(np.asarray(score_matrix))
    #set scores for excluded elements to -999
    score_array[excludes] = -999

    if not keep_order:
        top_n_indexes = score_array.argsort()[::-1][:n].tolist()
    else:
        reversed_scores = (-1)*score_array
        # use 'sorted' as it preserved index order (unlike argsort)
        top_n_indexes = list(zip(*sorted(zip(reversed_scores,range(len(reversed_scores))), key = lambda a: a[0]))[1][:n])

    return zip(top_n_indexes, score_array[top_n_indexes])

def top_ns (score_matrix, n, excludes = None, keep_order = False):
    '''Returns indexes of top n elements for each of given arrays with metrics
        - score_matrix is a dense np.matrix
        - n is a number of elements to return
    '''
    excludes = excludes if excludes else [[],]*len(score_matrix)
    # get indexes of top items, then construct (id, prediction) pairs
    return [top_n(sa,n,excludes = ex, keep_order = keep_order) for (sa,ex) in zip(score_matrix, excludes)]
