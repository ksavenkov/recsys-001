import csv
import numpy as np
# division by zero is ok here
np.seterr(divide='ignore')
import argparse
from scipy import sparse

test_data = [
['simple', 5, [11,121,8587], '11,603,0.96,1892,0.94,1891,0.94,120,0.93,1894,0.93\n121,120,0.95,122,0.95,603,0.94,597,0.89,604,0.88\n8587,603,0.92,597,0.90,607,0.87,120,0.86,13,0.86'],
['advanced', 5, [11,121,8587], '11,1891,5.69,1892,5.65,243,5.00,1894,4.72,2164,4.11\n121,122,4.74,120,3.82,2164,3.40,243,3.26,1894,3.22\n8587,10020,4.18,812,4.03,7443,2.63,9331,2.46,786,2.39']
]

def top_n_predictions (metric_array, n):
    #get indexes of top items, then construct (id, prediction) pairs
    return [(i,metric_array[i]) for i in metric_array.argsort()[::-1][:n]]

def format_predictions(given_items, predictions):
    return '\n'.join(['%d,' % gi + ','.join(['%d,%.2f' % (a,b) for (a,b) in p]) for (gi,p) in zip(given_items, predictions)])

def predictions_to_csv(given_items, predictions, filename):
    #NB: the right way should be construct proper list
    #NB: output_list = [ [gi] + list(itertools.chain(*p)) for (gi,p) in zip(given_items, predictions)]
    #NB: and write it using csv writer.
    #However, I gave up trying to do proper formating of floats in csv writer and stick to the following
    file = open(filename, 'w')
    file.write(format_predictions(given_items,predictions)) 
    file.close()
    print 'Predictions are written to %s' % filename

def load_ratings(csvfilename):
    print 'Reading (user_id,item_id,rating) tuples from %s' % csvfilename
    with open(csvfilename, 'rbU') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        result = [(int(r[0]),int(r[1]),float(r[2])) for r in csv_reader]
        print '\tread %d entries' % len(result)
        csvfile.close()
        return result

def load_names(csvfilename):
    print 'Reading (id,name) pairs from %s' % csvfilename
    with open(csvfilename, 'rbU') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        result = dict([(int(r[0]),r[1]) for r in csv_reader])
        print '\tread %d entries' % len(result.keys())
        csvfile.close()
        return result

# idx_list is a list of indexes with duplicates and gaps
# ----
# returns a dict old_idx -> new_idx
def normalize_idx(idx_list):
    # 1. remove duplicates and sort
    used_idx = sorted(set(idx_list))
    # 2. add new normalized index and put to a dict
    return dict(zip(used_idx,range(len(used_idx))))

# ratings are ratings in (user,item,rating) form
# -----
# returns dicts for new indexes and sparse.csr_matrix with '1' for each fact when a user rated an item
def ratings2facts(ratings):
    # 1. normalize the data by creating custom user and item indexes
    (user_col, item_col) = zip(*ratings)[:2]
    user_idx = normalize_idx(user_col)
    item_idx = normalize_idx(item_col)
    # 2. create a sparse fact matrix from ratings
    # coo_matrix((V,(I,J)))
    R = sparse.coo_matrix(([1,]*len(user_col), ([user_idx[u] for u in user_col],[item_idx[i] for i in item_col])))
    print 'Ratings loaded to %dx%d matrix' % R.shape
    
    # convert to CSR for better performance
    return user_idx, item_idx, R.tocsr()

# R - rating facts as a binary sparse.csr_matrix
# given_items - array of given items
# n - how many similar items to return
# algo - algorithm, either 'simple' or 'advanced'
# -----
# returns |items2rec| x n array of similar items
def compute_cooccurence(R, given_items, n, algo='simple'):
    print 'Computing top-%d cooccurences for items %s' % (n, ','.join(['%d'%e for e in given_items]))

    if algo == 'simple':
        P = cooc_simple(R,given_items)

    elif algo == 'advanced':
        P = cooc_advanced(R,given_items)

    else:
        print 'Unknown algorithm, exiting'
        exit()
        
    #extract top n other coocuring movies
    #NB: since we need top-n except given item (as it will have the highest coocurence), 
    #NB: retrieve n+1 and skip the first one
    return [top_n_predictions(c.A.squeeze(), n+1)[1:] for c in P]

#returns dense matrix: [x,y] = number of users rated both item x and item y
def rated_X_and_Y(R, given_items):
    #highlight the fact the matrix is dense
    return ( R[:,given_items].T * R ).todense()

#returns dense matrix: [x,:] = number of users rated item x
def rated_X(R, given_items):
    return R[:,given_items].sum(0).T * np.ones((1,R.shape[1]))

def cooc_simple(R,given_items):
    #build matrix M1 given_items x num_items, where M1[X,Y] = (rated both X and Y) / rated X:
    # - rated both X and Y: computed by dot product of binary matrix
    # - rated X: computed by column-wise sum, then duplicating that as a column
    # - note that the matrix are int, so we use true_divide to get float results

    P = np.true_divide ( rated_X_and_Y(R,given_items) , 
                           rated_X(R,given_items) )
    return P

def cooc_advanced(R,given_items):
    # Build matrix M2 given_items x num_items, where 
    #   M2[X,Y] = ( (rated both X and Y) / 
    #                    rated X) / 
    #              ( (rated A not X) / 
    #                   not rated X )
    # Let's avoid some divisions:
    #           = ((rated both X and Y) * (not rated X)) / 
    #               ( (rated X) * (rated Y not X) )
    # Theoretically, both numerator and denominator can be computed using the same function cooc_simple
    #    and swapping 0s and 1s in X. However, it is not a good idea to do the swap in a sparse matrix ;-) 
    #    Instead, let's notice that 'not rated X = total users - rated X'
    #    In a similar fashion, 'rated Y but not X = rated Y - rated Y and X'
    #           = ((rated both X and Y) * (total users - rated X)) / 
    #               ( (rated X) * (rated Y - rated X and Y) )
    
    rated_x = rated_X(R, given_items)
    rated_x_and_y = rated_X_and_Y(R, given_items)
    rated_y = np.ones((len(given_items),1)) * R.sum(0)
    total_users = R.shape[0]

    # Also, for some y, there are no users rated y but not x (at least, for x = y). 
    # Masking seem to be a bit awkward here, just dump the exception :-)
    P = np.true_divide ( np.multiply( rated_x_and_y , total_users - rated_x ),
                           np.multiply( rated_x , rated_y - rated_x_and_y ) )

    return P

# replace normalized indexes with given ones
# [(new_idx, score)] -> [(old_idx, score)]
def denormalize_predictions(predictions, old_idx):
        denorm = sorted(old_idx.keys())
        return [[(denorm[i],p) for (i,p) in pred] for pred in predictions]

def run_tests(test_inputs):
    #read ratings
    ratings = load_ratings('../data/ratings.csv')
    # normalize ratings
    (user_idx, item_idx, R) = ratings2facts(ratings)
    print 'Running tests:'
    cnt = 1
    for (algorithm, n, given_items, output) in test_inputs:
        given_items_norm = [item_idx[i] for i in given_items]
        print 'Test %d: top-%d using algorithm "%s"...' % (cnt, n, algorithm)
        predictions_norm = compute_cooccurence(R, given_items_norm, n, algorithm)
        predictions = denormalize_predictions(predictions_norm, item_idx)
        print '\t...%s' % ('passed' if format_predictions(given_items,predictions) == output else 'failed')
        cnt += 1

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='number of top-N items to recommend', metavar='N', type=int, default=5)
    parser.add_argument('--items', help='comma-separated list of item ids', metavar='items', default='')
    parser.add_argument('--algo', help='algorithm name', metavar='algo', default='simple')
    parser.add_argument('--debug', help='write to console instead of files', action='store_true')
    parser.add_argument('--test', help='compute and check predictions for the test data', action='store_true')
    args = parser.parse_args()

    if args.test:
        run_tests(test_data)
        return

    #more convenient names for parameters
    n = args.n
    given_items = [int(i) for i in args.items.split(',')]
    debug = args.debug
    algo = args.algo

    print 'Starting to generate top-%d recommendations' % args.n

    #read the data
    #users = load_names('../data/users.csv')
    #items = load_names('../data/movie-titles.csv')
    ratings = load_ratings('../data/ratings.csv')

    #extract facts and convert to matrix
    (user_idx, item_idx, R) = ratings2facts(ratings)
    given_items_norm = [item_idx[i] for i in given_items]

    #compute predictions
    predictions_norm = compute_cooccurence(R, given_items_norm, n, algo)
    predictions = denormalize_predictions(predictions_norm, item_idx)

    if not debug:
        #write to the file
        predictions_to_csv(given_items, predictions, '%s.csv' % algo)
    else:
        print 'Computed predictions:\n%s' % format_predictions(given_items,predictions)

if __name__ == "__main__":
    main()
