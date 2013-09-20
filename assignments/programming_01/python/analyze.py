import csv
import numpy as np
import argparse

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

def compute_cooccurence(ratings, items2rec, n, algo='simple'):
    print 'Computing top-%d cooccurences for items %s' % (n, ','.join(['%d'%e for e in items2rec]))

    (num_users, num_items, max_grade) = map(max,*ratings)

    #for now, two major performance issues
    # - no normalization (excess rows/columns for gaps in ids)
    # - no sparse matrices used
    ## the following is proper processing of ratings, 
    ## ratings = np.empty((num_users, num_items))
    ## ratings.fill(np.nan)
    ##for (u,i,r) in ratings_raw:
    ##    ratings[int(u)][int(i)] = float(r)
    ## however for this tasks we need just to extract
    ##    binary data 'who rated what'
    R = np.zeros((num_users+1, num_items+1))
    for (u,i,r) in ratings:
        R[u][i] = 1

    print 'Ratings loaded to %dx%d matrix' % R.shape

    #let's compute the formulae only for given items, otherwise in dense form it will compute too long time
    G = R[:,items2rec]
    num_items = R.shape[1]

    if algo == 'simple':
        P = cooc_simple(R,G)
        #extract top n other coocuring movies
        #NB: since we need top-n except given item (as it will have the highest coocurence), 
        #NB: retrieve n+1 and skip the first one
        return [top_n_predictions(c, n+1)[1:] for c in P]
    elif algo == 'advanced':
        P = cooc_advanced(R,G)
        #extract top n other coocuring movies
        return [top_n_predictions(c, n) for c in P]
    else:
        print 'Unknown algorithm, exiting'
        exit()

# rated_X_and_Y[x,y] = number of users rated both x and y
def rated_X_and_Y(R,X):
    return np.dot(X.T, R)

# rated_X[x] = number of users rated x
def rated_X(R,X):
    return np.array([X.sum(0),] * R.shape[1]).T

# rated_Y_not_X[x,y] = number of users who haven't rated item x, but rated item y
def rated_Y_not_X(R,X):
    # yep, we just need to swap 0 and 1 in a raters of items X
    # (G + 1) % 2 is another trick to swap 0 and 1 in binary matrix
    return rated_X_and_Y(R, (X + 1) % 2)

# not_rated_X[x] = number of users who haven't rated item x
def not_rated_X(R,X):
    # the same tricks as above
    return rated_X(R, (X + 1) % 2)

def cooc_simple(R,X):
    #build matrix M1 given_items x num_items, where M1[X,Y] = (rated both X and Y) / rated X:
    # - rated both X and Y: computed by dot product of binary matrix
    # - rated X: computed by column-wise sum, then duplicating that as a column
    # - note that arrays are int, so we use true_divide to get float results

    P = np.true_divide ( rated_X_and_Y(R,X), 
                            rated_X(R,X) )
    return P

def cooc_advanced(R,X):
    # Build matrix M2 given_items x num_items, where 
    #   M2[X,Y] = ((rated both X and Y) / rated X) / (rated Y not X) / rated not X
    # Two points: 
    # 1. Theoretically, both numerator and denominator can be computed using the same function cooc_simple defined above:
    #    np.true_divide(cooc_simple(R,X) / cooc_simple(R, (X + 1) % 2))
    #    where (X + 1) % 2 is a simple trick to swap 0 and 1 in binary matrix
    #    However, in that case we have 3 divisions and loose a lot of precision. So pick a dumb approach:
    # 2. For some y, there are no users rated y but not X (at least, for x = y). 
    #    Let's mask them to avoid division by zero
    # 3. Guys who teach the course consider users who didn't rate any movies
    #    as idle, excluding them from this part of the formula. I would disagree, but fix this to get correct grading.
    #    TODO: make treatment of such users set by parameters
    
    num_idle_users = R.sum(1).tolist().count(0);

    rated_y_not_x = rated_Y_not_X(R,X)
    return np.true_divide( rated_X_and_Y(R,X) * ( not_rated_X(R,X) - num_idle_users ), 
                             rated_X(R,X) * np.ma.masked_array(rated_y_not_x, rated_y_not_x == 0).filled(999999) )

def run_tests(test_inputs):
    #read ratings
    ratings = load_ratings('../data/ratings.csv')
    print 'Running tests:'
    cnt = 1
    for (algorithm, n, given_items, output) in test_inputs:
        print 'Test %d: top-%d using algorithm "%s"...' % (cnt, n, algorithm)
        predictions = compute_cooccurence(ratings, given_items, n, algorithm)
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
    items2rec = [int(i) for i in args.items.split(',')]
    debug = args.debug
    algo = args.algo

    print 'Starting to generate top-%d recommendations' % args.n

    #read all the data
    users = load_names('../data/users.csv')
    items = load_names('../data/movie-titles.csv')
    ratings = load_ratings('../data/ratings.csv')

    #compute predictions
    predictions = compute_cooccurence(ratings, items2rec, n, algo)

    if not debug:
        #write to the file
        predictions_to_csv(items2rec, predictions, '%s.csv' % algo)
    else:
        print 'Computed predictions:\n%s' % format_predictions(items2rec,predictions)

if __name__ == "__main__":
    main()
