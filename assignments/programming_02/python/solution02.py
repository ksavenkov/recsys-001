# make python find our new modules
import sys
sys.path.append("../../../recsys")

import recsys
import dataset
import model
from score import tfidf_unweighted, tfidf_weighted
from suggest import top_ns
from printer import coursera_pa2_printer

n = 5
given_users = [4455, 764, 5063, 4925, 2797]

def generate_solution(algo, n, given_users, filename):
    # create a DataIO object
    data = dataset.DataIO(verbose = True)
    # load ratings and tags
    data.load('../data/ratings.csv','../data/movie-tags.csv')
    given_users = data.translate_users(given_users)

    # build the recommender, specifying model, scoring and suggest algorithms
    rs = recsys.Recommender(data, model.TFIDFModel(verbose = False), algo, top_ns, verbose = True)
    rs.build()
    # generate the recommendations as [(id, score)...]
    # note that we take given users from the DataIO as it performs the index normalization
    recs = rs.recommend(given_users, n)

    # generate predictions in the proper string format
    output = data.print_recs(recs, given_users = given_users, printer = coursera_pa2_printer)
    # save predictions to the file
    file = open(filename, 'w')
    file.write(output)
    file.close()
    print 'Predictions are written to %s' % filename
  
def main():
    generate_solution(tfidf_unweighted, n, given_users, 'unweighted.csv') 
    generate_solution(tfidf_weighted, n, given_users, 'weighted.csv') 

if __name__ == "__main__":
    main()
