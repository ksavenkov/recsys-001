import csv
from collections import defaultdict

class DataIO:
    '''Responsible for reading data from whatever source (CSV, DB, etc),
           normalizing indexes for processing and writing results in denormalized form.
           Also it provides an access to translation dictionaries between external and internal ids.
           Prettyprinting of the recommendation results is performed using special Printer classes
           supplied to the DataIO.
    '''
    
    def __init__(self, verbose = True):
        self.__verbose = verbose
        self.ratings = []           # [(user, item, rating)]
        self.item_tags = []         # {(user, tag, count)}
    
    def load(self, ratings_file, tags_file = None):
        '''Loads the data from a proper source, and performs index normalization.'''
        self.__read_ratings(ratings_file)
        if tags_file:
            self.__read_tags(tags_file)
        self.__normalize()
        return

    def translate_users(self, old_user_ids):
        '''Takes an array of original user ids and translates them into the normalized form
        '''
        return [self.new_user_idx(i) for i in old_user_ids]

    def translate_items(self, old_item_ids):
        '''Takes an array of original item ids and translates them into the normalized form
        '''
        return [self.new_item_idx(i) for i in old_item_ids]

    def num_items(self):
        '''Number of different items in the dataset'''
        return len(self.__old_item_idx)

    def num_users(self):
        '''Number of different users in the dataset'''
        return len(self.__old_user_idx)

    def num_tags(self):
        '''Number of different users in the dataset'''
        return len(self.__item_tags)

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

    def tags(self, idx):
        '''Get tag by index.'''
        return self.__item_tags[idx]

    def tag_idx(self, tag):
        '''Get tag index.'''
        return self.__item_tag_idx[tag]
        pass

    def __log(self, msg):
        if self.__verbose:
            print msg

    def __read_ratings(self, filename):
        self.__log('Reading (user_id,item_id,rating) tuples from %s' % filename)
        with open(filename, 'rbU') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            self.ratings = [(int(r[0]),int(r[1]),float(r[2])) for r in csv_reader]
            self.__log('\tread %d entries' % len(self.ratings))
            csvfile.close()
            return

    def __read_tags(self, filename):
        self.__log('Reading (item_id,tag) tuples from %s' % filename)
        with open(filename, 'rbU') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            tagdict = defaultdict(int)
            # aggregate item-tag pairs
            for (item, tag) in csv_reader:
                tagdict[(int(item),tag)] += 1
            # turn it into a list
            self.item_tags = [(i,t,c) for ((i,t),c) in tagdict.items()]
            self.__log('\tread %d entries' % len(self.item_tags))
            csvfile.close()
            return

    def __normalize(self):
        '''Normalize the data by creating custom user and item indexes'''
        # collect all user, item and tag values
        (users_rated, items_rated) = zip(*self.ratings)[:2]
        (items_tagged, tags) = zip(*self.item_tags)[:2] if self.item_tags else ([],[])
        # normalize all of them by creating conversion dictionaries
        (self.__old_user_idx, self.__new_user_idx) = self.__normalize_idx(users_rated)
        (self.__old_item_idx, self.__new_item_idx) = self.__normalize_idx(items_rated, items_tagged)
        (self.__item_tags, self.__item_tag_idx) = self.__normalize_idx(tags)
        # translate everything into normalized indexes
        #TODO: inplace change may be more efficient
        self.ratings = [(self.__new_user_idx[u], self.__new_item_idx[i], r) for (u,i,r) in self.ratings]
        self.item_tags = [(self.__new_item_idx[i], self.__item_tag_idx[t], c) for (i,t,c) in self.item_tags]
    
    def __normalize_idx(self, *idx_list):
        '''idx_list is a list of indexes with duplicates and gaps
           returns an array of old indexes and a dict old_idx -> new_idx
           two arrays is due to dict.keys() are unsorted
        '''
        # 1. merge value lists, remove duplicates and sort
        old_idx = sorted(set().union(*idx_list))
        # 2. add new normalized index and put to a dict
        return old_idx, dict(zip(old_idx,range(len(old_idx))))

    def print_recs(self, recs, given_items = None, given_users = None, printer = None):
        '''Stringifies recommendations along with given items or users, 
           after translating them into original index system.
           It takes recommendations and given_items or given users in normalized form,
            translate them to the original indexes and passes to the printer function.
           Depending of what was passed, either given_users, or given_items, or None
            is passed to the printer function as the first argument.
        '''
        # translate given objects to the original indexes
        given = [self.old_user_idx(u) for u in given_users] if given_users else [self.old_item_idx(i) for i in given_items] if given_items else None
        # translate recommended items to the original indexes
        recs = [[(self.old_item_idx(i),s) for i,s in r] for r in recs]

        return printer(given, recs) if printer else default_printer(given, recs)

def default_printer(given, recs):
    return '\n'.join(['%d: ' % i + ', '.join(['(%d,%.2f)' % (j,s) for j,s in r]) for i,r in zip(given,recs)])
