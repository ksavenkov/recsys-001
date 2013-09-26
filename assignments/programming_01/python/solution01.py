# make python find our new modules
import sys
sys.path.append("../../../recsys")

import recsys
import dataset
from score import cooc_simple, cooc_advanced
from suggest import top_ns
from printer import coursera_pa1_printer

n = 5
given_items = [7443, 602, 280]

def generate_solution(algo, n, given_items, filename):
    # load the dataset
    ds = dataset.Dataset(verbose = True)
    ds.load('../data/ratings.csv')
    
    # build the recommender and generate predictions
    rs = recsys.Recommender(ds, algo, top_ns, coursera_pa1_printer, verbose = True)
    rs.build()
    predictions = rs.recommend(given_items, n)

    # save the predictions to file
    file = open(filename, 'w')
    file.write(predictions)
    file.close()
    print 'Predictions are written to %s' % filename
   
def main():
    generate_solution(cooc_simple, n, given_items, 'simple.csv') 
    generate_solution(cooc_advanced, n, given_items, 'advanced.csv') 

if __name__ == "__main__":
    main()
