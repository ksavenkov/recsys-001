__all__ = ['cooc_simple', 'cooc_advanced', 
           'tfidf_unweighted', 'tfidf_unweighted',
           'cosine', 'pearson', 'leave_top_n']

import math
import numpy as np
from scipy import sparse

np.seterr(all='ignore',invalid='ignore')

#returns dense matrix: [x,y] = number of users rated both item x and item y
def rated_X_and_Y(R, given_items):
    #highlight the fact the matrix is dense
    return ( R[:,given_items].T * R ).todense()

#returns dense matrix: [x,:] = number of users rated item x
def rated_X(R, given_items):
    return R[:,given_items].sum(0).T * np.ones((1,R.shape[1]))

def cooc_simple(model, given_items):
    '''Builds matrix M1 given_items x num_items, where M1[X,Y] = (rated both X and Y) / rated X:
       - rated both X and Y: computed by dot product of binary matrix
       - rated X: computed by column-wise sum, then duplicating that as a column
       - note that the matrix are int, so we use true_divide to get float results
       parameter 'model' is a PreferenceModel
       given_items is a list of items to compute scores for
    '''
    # here we check that this is right type of model
    R = model.P()

    P = np.true_divide ( rated_X_and_Y(R,given_items) , 
                           rated_X(R,given_items) )

    # cooccurence algorithms assume the given items aren't scored
    # -1 is used because if there's less than n suggestions, 0s can also be recommended
    P[range(len(given_items)),given_items] = -1

    return P

def cooc_advanced(model, given_items):
    '''Builds matrix M2 given_items x num_items, where 
       M2[X,Y] = ( (rated both X and Y) / 
                        rated X) / 
                  ( (rated A not X) / 
                       not rated X )
     Let's avoid some divisions:
               = ((rated both X and Y) * (not rated X)) / 
                   ( (rated X) * (rated Y not X) )
     Theoretically, both numerator and denominator can be computed using the same function cooc_simple
        and swapping 0s and 1s in X. However, it is not a good idea to do the swap in a sparse matrix ;-) 
        Instead, let's notice that 'not rated X = total users - rated X'
        In a similar fashion, 'rated Y but not X = rated Y - rated Y and X'
               = ((rated both X and Y) * (total users - rated X)) / 
                   ( (rated X) * (rated Y - rated X and Y) )
    '''
    # here we check that this is right type of model
    R = model.P()

    rated_x = rated_X(R, given_items)
    rated_x_and_y = rated_X_and_Y(R, given_items)
    rated_y = np.ones((len(given_items),1)) * R.sum(0)
    total_users = R.shape[0]

    # extract here to handle division by zero
    cooc_x = np.multiply( rated_x_and_y , total_users - rated_x )
    cooc_not_x = np.multiply( rated_x , rated_y - rated_x_and_y )

    # For some y, there are no users rated y but not x (at least, for x = y). 
    # mask zero values in the denominator
    zero_mask = cooc_not_x == 0

    P = np.true_divide ( cooc_x,
               np.ma.masked_array(cooc_not_x, zero_mask) )
    
    # cooccurence algorithms assume the given items aren't scored
    # -1 is used because if there's less than n suggestions, 0s can also be recommended
    P[range(len(given_items)),given_items] = -1 

    # fill missing vlaues (x/0 and 0/0) with 0 
    return P.filled(0)

def tfidf_unweighted(model, given_users):
    '''For given user ids, return scores for all items we have.
       We assume that model has TFIDF scores for items (I) and binary user preferences (P).
       The preferences are used to compute unweughted user profiles.
    '''
    # get submatrix for given users (still sparse)
    U_given = __unweighted_user_profiles(model.P()[given_users], model.I())
    # having given user profiles in U_given and all item profiles, compute pairwise similarity (distance)
    # using cosine distance function
    scores = cosine(U_given, model.I())
    # now set to -999 scores of all items rated by user
    # as we're working from dense matrix given_users x items, 
    # user the corresponding part of ratings matrix to mask all cells with ratings and fill them with zeros
    # -999 is used because if there's less than n suggestions, 0s can also be recommended
    return np.ma.masked_array(scores.todense(), model.R()[given_users].todense()).filled(-999)

def tfidf_weighted(model, given_users):
    '''For given user ids, return scores for all items we have.
       We assume the model has TFIDF item profiles (I) and user ratings (R).
       First, we compute relative ratings for each user with the user median as a reference point.
       Then, proceed with computing weighted user profiles.
    '''
    # get submatrix for given users (still sparse)
    U_given = __weighted_user_profiles(model.R()[given_users], model.I())
    # having given user profiles in U_given and all item profiles, compute pairwise similarity (distance)
    # using cosine distance function
    scores = cosine(U_given, model.I())

    # now set to -999 scores of all items rated by user
    # as we're working from dense matrix given_users x items, 
    # use the corresponding part of ratings matrix to mask all cells with ratings and fill them with -999
    # -999 is used because if there's less than n item user likes, we'll recommend some with score < 0
    return np.ma.masked_array(scores.todense(), model.R()[given_users].todense()).filled(-999)

def user_based_knn(model, n, given_users, given_items, distance, promote_users = True, 
                                                    exclude_seen = False, normalize = 'none'):
    '''For each (user,item) pair, identify n nearest neighbours that rated this item, 
       using distance on rating vectors (already mean-centered from model), 
       then compute mean-centered average score for this item using those n nighbours.
       Parameter promote_users says if users from below n should be promoted to the nearest in case 
        some of the top n similar users have rated one of the items. 
       Parameter normalize reflects how we should normalize predicted scores:
        - 'none': no normalization, compute weighted sum of neighbour's scores
        - 'normalize': compute wighted sum of mean scores, add mean rating of the user
        - 'centered': using mean-centered ratings, thus no normalization in that final formula, but still add the mean
       The model should provide:
        - model.R() - a sparse CSR matrix of mean-centered user ratings,
        - model.mean() - a vector of user mean ratings. 
       NB: consider replacing the loops with sparse tensor operations.
    '''
    # 1. for each given user, calculate similarity with all users that co-rated at least 1 item
    #       (anyway we'll need them all when will calculate by-item neighbours)
    # We keep matrices sparse (e.g. only non-zero similarities are computed):
    #       S is a given_users x all_users sparse matrix
    S = distance(model.R()[given_users], model.R())                           # given_users x all_users (sparse)

    # prepare empty matrix to keep all the scores user x item
    scores = np.zeros((len(given_users), len(given_items)))                # given_users x given_items (will be dense)

    for u in range(len(given_users)):
        # 2. for each given user, identify n nearest neighbours for each given item
        neighbours = S[u,:]                                                 # 1 x all_users (sparse)
        # remove the user himself from the matrix
        neighbours[0,given_users[u]] = 0

        # 2.1 put distance to neighbours to diagonal matrix
        neighbours_diag = sparse.spdiags(neighbours.todense(), 0, 
                                         neighbours.shape[1], neighbours.shape[1])  # all_users x all_users (sparse)

        # 2.2 multiply neighbour similarities on a binary rating matrix
        # the result is all_users x given_items sparse matrix for user u, 
        # where (i,j) = similarity(u,i) iff u and i rated item j
        # NB: here we don't check R() == 0 as rating values may be mean-centered, 
        #     in which case 0 doesn't mean the user hasn't rated the item.
        #     Instead, we use P() binary matrix that was built based on the original ratings
        item_neighbours = neighbours_diag * model.P()[:,given_items]  # all_users x given_items (sparse)
        
        # if we promote users, do per-item maximum and wipe the rest
        if promote_users:
            # 2.3 turn into 0 everything but the first n values in each item column (inplace)
            leave_top_n(item_neighbours, n)                                      # all_users x given_items (sparse)
        else:
            # else identify top-n users (ids)
            top_neighbours = np.asarray(neighbours.todense()).reshape(-1).argsort()[::-1][:n]
            # then leave only their scores
            the_rest = np.setdiff1d(range(item_neighbours.shape[0]),top_neighbours)
            item_neighbours[the_rest,:] = 0
            item_neighbours.eliminate_zeros() 

        if normalize == 'normalize':
            # 2.4 having all_users x given_items matrix with similarities for top-n neighbours per each item
            #        use all_users x given_items matrix of their mean-centered scores
            #        to compute normalized average score for each of the items
            # The normalized average score p(u,i) for user u and item i is computed as 
            #       p(u,i) = mu(u) + sum(n_neighbours(u,i), sim(u,v) * (r(v,i) - mu(v))) /
            #                               sum(n_neighbours(u,i), |sim(u,v)|)
            # 3. put it to the proper position in the scores (given_users x given_items) matrix
            ratings = model.R() - (model.R() != 0).multiply(model.mean())
            weights_sum = np.asarray(item_neighbours.sum(0)).reshape(-1)
            scores[u,:] = (np.true_divide(item_neighbours.multiply(ratings[:,given_items]).sum(0), 
                            np.ma.masked_array(weights_sum, weights_sum == 0)) + model.mean()[given_users[u]]).filled(0)
        elif normalize == 'centered':
            weights_sum = np.asarray(item_neighbours.sum(0)).reshape(-1)
            scores[u,:] = (np.true_divide(item_neighbours.multiply(model.R()[:,given_items]).sum(0), 
                            np.ma.masked_array(weights_sum, weights_sum == 0)) + model.mean()[given_users[u]]).filled(0)
        elif normalize == 'none':
            weights_sum = np.asarray(item_neighbours.sum(0)).reshape(-1)
            scores[u,:] = np.true_divide(item_neighbours.multiply(model.R()[:,given_items]).sum(0),
                            np.ma.masked_array(weights_sum, weights_sum == 0)).filled(0)
        else:
            print 'No such normalization: ', normalize

    if exclude_seen:
        # now set to -999 scores of all items rated by user
        # as we're working from dense matrix given_users x items, 
        # use the corresponding part of ratings matrix to mask all cells with ratings and fill them with -999
        # -999 is used because if there's less than n item user likes, we'll recommend some with score < 0
        scores = np.ma.masked_array(scores, model.R()[given_users][:,given_items].todense()).filled(-999)

    return scores

def leave_top_n(M, n):
    '''For sparse CSR matrix M with float values, mask everything but up to n top elements in each column
    '''
    # get top n sorted indexes for each column (cols -> sort indexes -> revert -> take top n rows):
    # we're interested in non-0 correlations as 0s are for users w/o data to compute the correlation

    # FIXME ugly cycle, consider replace with more numpythonic idiom
    # iterate through columns / items
    for i in range(M.shape[1]):
        # for each column, attach original row indexes to data
        nonzero_elements = zip(M[:,i].data, __csr_row_indices(M[:,i].indptr))
        # then sort the data and get indexes of everything outside of top-n .data elements
        # reverse sort by 0th -> split to columns -> take 1th -> take after nth
        tail_indices = zip(*sorted(nonzero_elements, key = lambda a : a[0], reverse = True))[1][n:]
        # set those to zero
        M[tail_indices,i] = 0

    # eliminate zeros from the sparse matrix
    M.eliminate_zeros()

def cosine(U, I):
    '''Calculates the cosine distance between vectors in two sparse matrices U (a x b) and I (c x b).
       The result is written to sparse a x c matrix.
       cosine(u,v) = ( u .* v ) / (|u| * |v|)
    '''
    # u .* v numerator, assume that U() is |u| x |t| and I() is |i| x |t|, hence W is |u| x |i|
    # still sparse matrix if the rows don't intersect on t
    W = U * I.T
    # calculate vectors of sums for rows (dense)
    dU = np.sqrt(U.multiply(U).sum(1))
    dI = np.sqrt(I.multiply(I).sum(1))
    # elementwise divide W by user x item matrix of profile norm multiplications
    # in order to avoid making a huge dense matrix, we need perform the division in two steps
    __divide_csr_cols_by_vector(W, np.asarray(dU).reshape(-1))
    __divide_csr_rows_by_vector(W, np.asarray(dI).reshape(-1))
    # hooray, the u x v matrix is still sparse!
    return W

def pearson(U, V):
    '''Calculates the pearson correlation coeff. between vectors in two sparse matrices U (a x b) and V (c x b).
       The result is written to dense a x c matrix.
       This function mimics Excel CORREL function. To compute pearson(u,v), we should take elements that exist
       in both vectors, and then compute sum((u_i - u_mean)*(v_i - v_mean)) / sum(sqrt( (u_i - u_mean) * (v_i - v_mean) ))

       NB: b may be arbitrary big, avoid doing dense matrix with b as one of the dimensions
    '''
    # the thing is that mean values of u depends on elements of v and vice versa 
    # as zero elements don't count
    P = np.zeros((U.shape[0],V.shape[0]))
    for i in range(U.shape[0]):                                                         # u is sparse 1 x b
        u = U[i]
        # create a copy of I and remove elements that don't match u
        u_zero = np.setdiff1d(range(u.shape[1]),u.indices)
        V_u = V.copy()
        V_u[:,u_zero] = 0
        V_u.eliminate_zeros()                                           # c x b sparse
        Nz = V_u != 0

        # duplicate u for a number of rows in V and remove non-matched elements
        u_diag = sparse.spdiags(u.todense(), 0, u.shape[1], u.shape[1])
        u_V = Nz * u_diag 
        u_V.eliminate_zeros()                                           # c x b sparse

        # calculate mean for each of vectors in V (c x 1) and mean u for each of vectors in V (c x 1)
        V_u_mean = V_u.sum(1) / Nz.sum(1)                       # c x 1 dense
        u_V_mean = u_V.sum(1) / Nz.sum(1)                       # c x 1 dense

        # create mean-centered V_u and u_V
        V_u_centered = V_u - sparse.csr_matrix(Nz.multiply(V_u_mean))   # c x b sparse
        u_V_centered = u_V - sparse.csr_matrix(Nz.multiply(u_V_mean))   # c x b sparse

        # compute denominators using some low-level sparse magic
        # during centering, some values could turn into 0, thus making sparse structure different
        # between V_u_centered and u_V_centered
        denom = [math.sqrt(sum(u_V_centered.data[u_V_centered.indptr[j]:u_V_centered.indptr[j+1]]**2) * 
                      sum(V_u_centered.data[V_u_centered.indptr[j]:V_u_centered.indptr[j+1]]**2))
                        for j in range(len(u_V_centered.indptr)-1)]

        # now compute the pearson coefficients for all v
        P[i,:] = u_V_centered.multiply(V_u_centered).sum(1).T / denom

    # return sparse matrix to conform with cosine similarity function
    return sparse.csr_matrix(P)

def __divide_csr_rows_by_vector(M,v):
    '''Divide each CSR row by a value; values represented as an array or list v
       ! v is not a matrix !
    '''
    assert(M.shape[1] == len(v))
    # option 1:
    # decompress vector to fit data array in the CSR representation
    M.data = np.true_divide(M.data, [v[i] for i in M.indices])

def __divide_csr_cols_by_vector(M,v):
    '''Divide each CSR column by a value; values represenced as a vector v
       Consider doing M.tocsc and dividing by col
    '''
    assert(M.shape[0] == len(v))
    indices = __csr_row_indices(M.indptr)
    # then as in the provious function
    M.data = np.true_divide(M.data, [v[i] for i in indices])

def __csr_row_indices(indptr):
    '''Takes csr_matrix.indptr and returns row indexes for data elements
    '''
    # get row index for each M.data element. Hold on, thet'd be da fun!
    return [j for i in range(len(indptr)-1) for j in [i,]*(indptr[i+1] - indptr[i])]

def __unweighted_user_profiles(P,I):
    '''P - sparse CSR matrix of user preferences (user x item x {0,1})
       I - sparse CSR matrix of item profiles (item x feature x float)
       Returns CSR matrix of user profiles (users x feature x float), 
       built as a sum of profiles for all items user 'likes'
    '''
    return P * I

def __weighted_user_profiles(R,I):
    '''R - sparse CSR matrix of user preferences (user x item x float)
       I - sparse CSR matrix of item profiles (item x feature x float)
       Returns CSR matrix of user profiles (users x feature x float),
       built as a weighted sum of profiles for all items user rated, 
       with a weight correlated with the rating value.
       
    '''
    # NB: below the ratings are made dense. It is possible to avoid that, doing all operations
    # on csr_matrix.data

    # calculate mean ratings
    U_mean = R.sum(1) / (R != 0).sum(1)
    # subtract them from non-zero elements
    U_relative = R - sparse.csr_matrix((R != 0).multiply(U_mean))

    return U_relative * I
