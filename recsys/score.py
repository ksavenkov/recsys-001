__all__ = ['cooc_simple', 'cooc_advanced']

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
    P[range(len(given_items)),given_items] = 1

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
    scores = __cosine(U_given, model.I())

    # now set to -999 scores of all items rated by user
    # as we're working from dense matrix given_users x items, 
    # user the corresponding part of ratings matrix to mask all cells with ratings and fill them with zeros
    # -999 is used because if there's less than n suggestions, 0s can also be recommended
    return np.ma.masked_array(scores, model.R()[given_users].todense()).filled(-999)

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
    scores = __cosine(U_given, model.I())

    # now set to -999 scores of all items rated by user
    # as we're working from dense matrix given_users x items, 
    # user the corresponding part of ratings matrix to mask all cells with ratings and fill them with zeros
    # -999 is used because if there's less than n item user likes, we'll recommend some with score < 0
    return np.ma.masked_array(scores, model.R()[given_users].todense()).filled(-1)

def __cosine(U, I):
    '''Calculates the cosine distance between vectors in two sparse matrices U (a x b) and I (c x b).
       The result is written to dense a x x matrix.
       cosine(u,v) = ( u .* v ) / (|u| * |v|)
    '''
    # u .* v numerator, assume that U() is users x tags and I() is items x tags, hence W is users x items
    # should be dense matrix
    W = U * I.T
    # calculate vectors of sums for rows (dense)
    dU = np.sqrt(U.multiply(U).sum(1))
    dI = np.sqrt(I.multiply(I).sum(1))
    # elementwise divide W by user x item matrix of profile norm multiplications
    return np.true_divide(W.todense(), dU * dI.T)

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
