import dataset
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

class PreferenceModel():
    '''Model that contains user preferences as binary matrix users x items
       proper scoring functions: cooc_simple, cooc_advanced
    '''
    def __init__(self, verbose = True):
        self.__verbose = verbose

    def build(self, dataset):
        '''Processes dataset and builds a model for subsequent scoring'''
        # NB: here we check that get proper Dataset (ratings attribute available)
        self.__extract_facts(dataset.ratings)

    def P(self):
        '''Returns the model data (preferences), to be called from a proper scoring function'''
        return self.__R

    def __extract_facts(self, ratings):
        '''Converts dataset ratings to a sparse.csr_matrix with '1' for each fact when a user rated an item''' 
        # construct coo_matrix((V,(I,J))) and convert it to CSR for better performance
        self.__R = sparse.coo_matrix(([1,]*len(ratings), zip(*ratings)[:2] )).tocsr()
        self.__log('\tratings loaded to %dx%d matrix' % self.__R.shape)
    
    def __log(self, msg):
        if self.__verbose:
            print msg

class TFIDFModel:
    '''This model needs a dataset with item tags (item_tags) and user preferences (ratings).
       They are used to build TFIDF-normalized item profiles and then user profiles
    '''
    def __init__(self, verbose = True):
        self.__verbose = verbose
        self.threshold = 3.5

    def build(self, dataset):
        '''Processes a dataset with item tags and user-item ratings. Ratings are treated as binary facts,
           item-tag profiles are normalized using TF*IDF, then user-tag profiles are built.
        '''
        self.__build_item_profiles(dataset.item_tags)
        self.__build_user_preferences(dataset.ratings)

    def I(self):
        '''Return CSR matrix of TFIDF-normalized item profiles (items x tags x float)
        '''
        return self.__I

    def P(self):
        '''Return CSR matrix of user preferences (users x items x {0,1})
        '''
        return self.__P

    def R(self):
        '''Return CSR matrix of user ratings (users x items x float)
        '''
        return self.__R

    def __build_user_preferences(self, ratings):
        # put '1' for all ratings >= 3.5 as a users x items matrix of user preferences
        self.__R = self.__convert_tocsr(ratings)
        self.__P = self.__extract_facts(ratings, self.threshold)

    def __build_item_profiles(self, item_tags):
        '''Converts item_tags list of (item, tag, count) triples into a sparse CSR matrix Items x Tags
           log-TFIDF normalization applied.
        '''
        # 1. Create CSR matrix with term frequencies (TF)
        TF = self.__convert_tocsr(item_tags)

        # 2. Compute inverse document frequency (IDF) matrix
        #    - item-term usage facts are extracted by replacing values in TF with 1s
        #    - document frequency is constructed by summing the above for terms
        #    - total document number is taken from the matrix size
        #    - don't forget we should use log-normalized IDF
        # represent IDF with a diagonal matrix to perform column-wise multiplication later
        IDF = sparse.spdiags(np.log(float(TF.shape[0]) / self.__extract_facts(item_tags).sum(0)),
                            0,
                            TF.shape[1], TF.shape[1])

        #  Do unit-normalization for TFIDF (to penalty movies with lots of tags)
        # 3. Compute item profiles I as TF * IDF:
        self.__I = normalize(TF * IDF, norm='l2', axis=1)

    def __convert_tocsr(self, triples):
        '''Converts (id1, id2, n) triples to a sparse.csr_matrix of (id1, id2, 1)''' 
        # construct coo_matrix((V,(I,J))) and convert it to CSR for better performance
        R = sparse.coo_matrix((zip(*triples)[2], zip(*triples)[:2])).tocsr()
        self.__log('\tdata loaded to %dx%d matrix' % R.shape)
        return R

    def __extract_facts(self, triples, threshold = None):
        '''Converts (id1, id2, n) triples to a sparse.csr_matrix of (id1, id2, 1)
            for all n >= threshold
        ''' 
        # apply the threshold
        filtered_triples = [(a,b,c) for (a,b,c) in triples if c >= threshold] if threshold else triples
        # construct coo_matrix((V,(I,J))) and convert it to CSR for better performance
        R = sparse.coo_matrix(([1,]*len(filtered_triples), zip(*filtered_triples)[:2] )).tocsr()
        self.__log('\tdata converted to binary and loaded to %dx%d matrix' % R.shape)
        return R
    
    def __log(self, msg):
        if self.__verbose:
            print msg

class UserModel:
    '''This model needs a dataset with user preferences (ratings).
       When the model is built, the preferences are mean-normalized
       if the argument mormalize is provided, 
       the mean for each user is stored separately.
    '''
    def __init__(self, normalize = True, verbose = True):
        self.__verbose = verbose
        self.__normalize = normalize

    def build(self, dataset):
        '''Processes a dataset with user-item ratings.
        '''
        self.__build_user_preferences(dataset.ratings)

    def R(self):
        '''Return CSR matrix of user ratings (users x items x float)
        '''
        return self.__R

    def P(self):
        '''Return CSR matrix of user preferences (users x items x {0,1})
        '''
        return self.__P

    def normalized(self):
        return self.__normalize

    def mean(self):
        '''Return a vector of mean user ratings
        '''
        return self.__mean

    def __build_user_preferences(self, ratings):
        # put '1' for all ratings >= 3.5 as a users x items matrix of user preferences
        self.__R = self.__convert_tocsr(ratings)
        self.__P = self.__extract_facts(ratings)
        self.__mean = self.__R.sum(1) / (self.__R != 0).sum(1)
        if self.__normalize:
            self.__R = self.__R - sparse.csr_matrix((self.__R != 0).multiply(self.__mean))

    def __extract_facts(self, triples):
        '''Converts (id1, id2, n) triples to a sparse.csr_matrix of (id1, id2, 1)
        ''' 
        # construct coo_matrix((V,(I,J))) and convert it to CSR for better performance
        P = sparse.coo_matrix(([1,]*len(triples), zip(*triples)[:2] )).tocsr()
        self.__log('\tdata converted to binary and loaded to %dx%d matrix' % P.shape)
        return P
 
    def __convert_tocsr(self, triples):
        '''Converts (id1, id2, n) triples to a sparse.csr_matrix of (id1, id2, 1)''' 
        # construct coo_matrix((V,(I,J))) and convert it to CSR for better performance
        R = sparse.coo_matrix((zip(*triples)[2], zip(*triples)[:2])).tocsr()
        self.__log('\tdata loaded to %dx%d matrix' % R.shape)
        return R
   
    def __log(self, msg):
        if self.__verbose:
            print msg
