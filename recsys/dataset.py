import csv

class Dataset:
    '''Encapsulates all reading of the data from whatever source (currently CSV), 
       performs index normalization and provides new<->old index conversion tools'''
    
    def __init__(self, verbose = True):
        self.__verbose = verbose
    
    def load(self, filename):
        '''Loads the data from a proper source, and performs index normalization.'''
        self.__read(filename)
        self.__normalize()
        return

    def num_items(self):
        '''Number of different items in the dataset'''
        return len(self.__old_item_idx)

    def num_users(self):
        '''Number of different users in the dataset'''
        return len(self.__old_user_idx)

    def old_item_idx(self, idx):
        '''Old to new index conversion.'''
        return self.__old_item_idx[idx]

    def new_item_idx(self, idx):
        '''New to old index conversion.'''
        return self.__new_item_idx[idx]

    def old_user_idx(self, idx):
        '''Old to new index conversion.'''
        return self.__old_user_idx[idx]

    def new_user_idx(self, idx):
        '''New to old index conversion.'''
        return self.__new_user_idx[idx]
        pass

    def __log(self, msg):
        if self.__verbose:
            print msg

    def __read(self, filename):
        self.__log('Reading (user_id,item_id,rating) tuples from %s' % filename)
        with open(filename, 'rbU') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            self.ratings = [(int(r[0]),int(r[1]),float(r[2])) for r in csv_reader]
            self.__log('\tread %d entries' % len(self.ratings))
            csvfile.close()
            return

    def __normalize(self, ):
        '''Normalize the data by creating custom user and item indexes'''
        (user_col, item_col) = zip(*self.ratings)[:2]
        (self.__old_user_idx, self.__new_user_idx) = self.__normalize_idx(user_col)
        (self.__old_item_idx, self.__new_item_idx) = self.__normalize_idx(item_col)
        #TODO: inplace change may be more efficient
        self.ratings = [(self.__new_user_idx[u], self.__new_item_idx[i], r) for (u,i,r) in self.ratings]
    
    def __normalize_idx(self, idx_list):
        '''idx_list is a list of indexes with duplicates and gaps
           returns an array of old indexes and a dict old_idx -> new_idx
           two arrays is due to dict.keys() are unsorted
        '''
        # 1. remove duplicates and sort
        old_idx = sorted(set(idx_list))
        # 2. add new normalized index and put to a dict
        return old_idx, dict(zip(old_idx,range(len(old_idx))))

