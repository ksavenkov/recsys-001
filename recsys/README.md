'recsys' is a small framework for recommender systems, originally developer to complete assignments 
of the University of Minnesota's Introduction to Recommender Systems course, taught by J. Konstan 
and M. Ekstrand, offered on Coursera Fall 2013

(c) Konstantin Savenkov, 2013 

The framework consists of the following parts:

Recommender -- provides the interface to the framework. It is not supposed to be changed, different 
            recommenders are created by passing proper parameters to the recommender object, 
            which is in fact a template.
        Recommender(DataIO, Model, Score, Suggest) - the recommender is created with a DataIO object
            already populated with data, an (empty) model to use, scoring function and suggest algorithm.
        build() - initiate the model building
        recommend(ids, n) - recommend n objects for each of the objects passed in ids. Objects may 
            refer either to items or to users depending on score function used. I.e. this way it can be
            item-item, user-item, item-user or user-user recommendation.

DataIO -- class that is responsible for reading data from whatever source (CSV, DB, etc), 
           normalizing indexes for processing and writing results in denormalized form.
           Also it provides an access to translation dictionaries between external and internal ids.
           Prettyprinting of the recommendation results is performed using special Printer classes 
           supplied to the DataIO.
        The DataIO may contain the following datasets, each defines a certain data relation:
            - ratings (user_id x item_id x rating) 
            - users (user_id x user), 
            - items (item_id x item),
            - tags (item_id x tag).

Model -- hosts the model computed from the data read via DataIO. This class encapsulates data
        that (1) shouldn't be recomputed to do a single recommendation and (2) should be recomputed
        when the dataset (event stream) is updated. Model should be serializeable to allow for 
        computing it on one server/cluster and replicating to another one that computes and server
        actual recommendations. All requests to data from the recommender internals (e.g. items seen
        by a user) should be served by Model.

Score -- score method uses data provided by Model to compute scores of candidate items (or users)
        for the recommendation. Score contains the computations that must be done in order to   
        serve the particular request for recommendations.

Suggest -- the suggest function select particular set of items to recommend to the user. Currently it is 
        plain top-N. In future, that may be used to combine different score results, e.g. to show 
        N recommended items, evenly distributed across the popularity scale.
        
